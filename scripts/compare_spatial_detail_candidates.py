"""Generate v5 spatial-detail candidates from an existing registered v4 run.

This script reuses the harmonized low-resolution cube, material coefficients,
registration, and RGB ROI from a completed run.  It therefore evaluates only
the spatial-detail model and does not repeat registration or write a full
367-band high-resolution cube.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from geocorefusion.config import FusionConfig, load_config  # noqa: E402
from geocorefusion.dataset import discover_triplet, normalize_image, normalize_rgb  # noqa: E402
from geocorefusion.degradation import PsfModel, degrade_coefficients, degrade_spatial_map  # noqa: E402
from geocorefusion.envi import open_cube  # noqa: E402
from geocorefusion.fusion import (  # noqa: E402
    _rgb_weights,
    _spectral_shape_preserving_detail_gain,
    build_additive_spectral_scale,
    inject_coefficient_detail,
)
from geocorefusion.lowrank import SubspaceModel, fit_subspace  # noqa: E402
from geocorefusion.output import write_json  # noqa: E402
from geocorefusion.quality import sam_degrees  # noqa: E402


@dataclass(slots=True)
class Candidate:
    name: str
    label: str
    coefficient_strength: float
    gain_strength: float
    additive_strength: float
    clip_sigma: float
    support_floor: float


CANDIDATES = (
    Candidate("coefficient_only", "Coefficient detail", 0.42, 0.0, 0.0, 0.52, 0.34),
    Candidate("structure_balanced", "Structure-balanced v5", 0.46, 0.09, 0.18, 0.56, 0.36),
    Candidate("structure_strong", "Structure-strong v5", 0.60, 0.13, 0.26, 0.68, 0.40),
    Candidate("hybrid_sharp", "Hybrid-sharp v5", 0.26, 0.26, 0.18, 0.42, 0.34),
)


def _psf_from_json(path: Path) -> PsfModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PsfModel(
        sigma_x_highres=float(payload["sigma_x_highres"]),
        sigma_y_highres=float(payload["sigma_y_highres"]),
        score=float(payload["score"]),
        low_shape=tuple(int(v) for v in payload["low_shape"]),
        high_shape=tuple(int(v) for v in payload["high_shape"]),
        method=str(payload["method"]),
    )


def _resize_coefficients(coeff: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    out = np.empty((size[1], size[0], coeff.shape[2]), dtype=np.float32)
    for component in range(coeff.shape[2]):
        out[:, :, component] = cv2.resize(
            coeff[:, :, component],
            size,
            interpolation=cv2.INTER_AREA,
        )
    return out


def _reconstruct_selected_bands(
    coeff: np.ndarray,
    subspace: SubspaceModel,
    indices: np.ndarray,
    gain: np.ndarray,
    additive: np.ndarray,
    additive_scale: np.ndarray,
) -> np.ndarray:
    cube = np.einsum(
        "...k,kb->...b",
        coeff,
        subspace.basis[:, indices],
        optimize=True,
    ) + subspace.mean_spectrum[indices][None, None, :]
    cube = cube * gain[:, :, None]
    cube = cube + additive[:, :, None] * additive_scale[indices][None, None, :]
    return np.clip(
        cube,
        subspace.clip_min[indices][None, None, :],
        subspace.clip_max[indices][None, None, :],
    ).astype(np.float32)


def _edge_correlation(image: np.ndarray, rgb: np.ndarray) -> float:
    source = normalize_image(image)
    guide = cv2.cvtColor((normalize_rgb(rgb) * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_edge = cv2.magnitude(
        cv2.Scharr(source, cv2.CV_32F, 1, 0),
        cv2.Scharr(source, cv2.CV_32F, 0, 1),
    ).reshape(-1)
    guide_edge = cv2.magnitude(
        cv2.Scharr(guide, cv2.CV_32F, 1, 0),
        cv2.Scharr(guide, cv2.CV_32F, 0, 1),
    ).reshape(-1)
    valid = np.isfinite(source_edge) & np.isfinite(guide_edge)
    source_edge, guide_edge = source_edge[valid], guide_edge[valid]
    if source_edge.size < 32 or np.std(source_edge) < 1e-8 or np.std(guide_edge) < 1e-8:
        return float("nan")
    return float(np.corrcoef(source_edge, guide_edge)[0, 1])


def _normalized_high_frequency_energy(image: np.ndarray) -> float:
    normalized = normalize_image(image)
    high = normalized - cv2.GaussianBlur(normalized, (0, 0), 1.4)
    return float(np.percentile(np.abs(high[np.isfinite(high)]), 95.0))


def _observation_metrics(
    coeff: np.ndarray,
    gain: np.ndarray,
    additive: np.ndarray,
    additive_scale: np.ndarray,
    low_coeff: np.ndarray,
    low_cube: np.ndarray,
    subspace: SubspaceModel,
    psf: PsfModel,
) -> dict[str, float]:
    low_coeff_pred = degrade_coefficients(coeff, psf)
    low_gain = degrade_spatial_map(gain, psf)
    low_weighted_coeff = degrade_coefficients(coeff * gain[:, :, None], psf)
    prediction = (
        np.einsum("...k,kb->...b", low_weighted_coeff, subspace.basis, optimize=True)
        + low_gain[:, :, None] * subspace.mean_spectrum[None, None, :]
    )
    low_additive = degrade_spatial_map(additive, psf)
    prediction += low_additive[:, :, None] * additive_scale[None, None, :]
    prediction = np.clip(
        prediction,
        subspace.clip_min[None, None, :],
        subspace.clip_max[None, None, :],
    )
    return {
        "coefficient_rmse": float(np.sqrt(np.mean((low_coeff_pred - low_coeff) ** 2))),
        "continuous_cube_rmse": float(np.sqrt(np.mean((prediction - low_cube) ** 2))),
        "continuous_cube_sam_deg": sam_degrees(prediction, low_cube),
        "gain_lowres_rmse_from_one": float(np.sqrt(np.mean((low_gain - 1.0) ** 2))),
        "additive_lowres_rmse_from_zero": float(np.sqrt(np.mean(low_additive**2))),
    }


def _stretch_reference(images: list[np.ndarray], reference_index: int = 0) -> list[np.ndarray]:
    reference = images[reference_index]
    lows = np.percentile(reference.reshape(-1, reference.shape[2]), 1.0, axis=0)
    highs = np.percentile(reference.reshape(-1, reference.shape[2]), 99.0, axis=0)
    out: list[np.ndarray] = []
    for image in images:
        scaled = (image - lows[None, None, :]) / np.maximum(highs - lows, 1e-6)[None, None, :]
        out.append((np.clip(scaled, 0.0, 1.0) * 255.0).round().astype(np.uint8))
    return out


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
    width = min(panel.shape[1], max(210, 12 * len(label)))
    cv2.rectangle(panel, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _select_detail_patches(rgb: np.ndarray, patch: int = 300, count: int = 3) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor((normalize_rgb(rgb) * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    edge = cv2.magnitude(
        cv2.Scharr(gray, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray, cv2.CV_32F, 0, 1),
    )
    score = cv2.GaussianBlur(edge, (0, 0), max(8.0, patch / 8.0))
    h, w = gray.shape
    patch = min(patch, h - 2, w - 2)
    half = patch // 2
    candidates: list[tuple[float, int, int]] = []
    step = max(32, patch // 4)
    for cy in range(half, h - half, step):
        for cx in range(half, w - half, step):
            candidates.append((float(score[cy, cx]), cy, cx))
    candidates.sort(reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for _, cy, cx in candidates:
        if any((cy - (y0 + y1) / 2.0) ** 2 + (cx - (x0 + x1) / 2.0) ** 2 < (0.75 * patch) ** 2 for y0, x0, y1, x1 in selected):
            continue
        selected.append((cy - half, cx - half, cy - half + patch, cx - half + patch))
        if len(selected) >= count:
            break
    return selected


def _save_comparisons(
    output_dir: Path,
    rgb: np.ndarray,
    names: list[str],
    labels: list[str],
    false_colors: list[np.ndarray],
    band_images: list[np.ndarray],
    recommended_name: str,
) -> list[dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    false_u8 = _stretch_reference(false_colors)
    band_rgb = [np.repeat(image[:, :, None], 3, axis=2) for image in band_images]
    band_u8 = _stretch_reference(band_rgb)

    max_height = 900
    scale = min(1.0, max_height / float(rgb.shape[0]))
    size = (int(round(rgb.shape[1] * scale)), int(round(rgb.shape[0] * scale)))
    rgb_u8 = (normalize_rgb(rgb) * 255.0).round().astype(np.uint8)
    rgb_small = cv2.resize(rgb_u8, size, interpolation=cv2.INTER_AREA)
    false_small = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in false_u8]
    band_small = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in band_u8]
    Image.fromarray(np.concatenate([_label_panel(rgb_small, "RGB reference")] + [_label_panel(v, label) for v, label in zip(false_small, labels)], axis=1)).save(
        output_dir / "comparison_full_false_color.png"
    )
    Image.fromarray(np.concatenate([_label_panel(rgb_small, "RGB reference")] + [_label_panel(v, label) for v, label in zip(band_small, labels)], axis=1)).save(
        output_dir / "comparison_full_900nm.png"
    )
    recommended_index = names.index(recommended_name)
    Image.fromarray(
        np.concatenate(
            [
                _label_panel(rgb_small, "RGB reference"),
                _label_panel(false_small[0], labels[0]),
                _label_panel(false_small[recommended_index], labels[recommended_index]),
            ],
            axis=1,
        )
    ).save(output_dir / "recommended_review_false_color.png")
    Image.fromarray(
        np.concatenate(
            [
                _label_panel(rgb_small, "RGB reference"),
                _label_panel(band_small[0], labels[0]),
                _label_panel(band_small[recommended_index], labels[recommended_index]),
            ],
            axis=1,
        )
    ).save(output_dir / "recommended_review_900nm.png")

    patches = _select_detail_patches(rgb)
    false_rows: list[np.ndarray] = []
    band_rows: list[np.ndarray] = []
    for patch_index, (y0, x0, y1, x1) in enumerate(patches, start=1):
        rgb_patch = rgb_u8[y0:y1, x0:x1]
        false_panels = [_label_panel(rgb_patch, f"RGB zoom {patch_index}")]
        band_panels = [_label_panel(rgb_patch, f"RGB zoom {patch_index}")]
        for label, image, band in zip(labels, false_u8, band_u8):
            false_panels.append(_label_panel(image[y0:y1, x0:x1], label))
            band_panels.append(_label_panel(band[y0:y1, x0:x1], label))
        false_rows.append(np.concatenate(false_panels, axis=1))
        band_rows.append(np.concatenate(band_panels, axis=1))
    Image.fromarray(np.concatenate(false_rows, axis=0)).save(output_dir / "comparison_detail_zooms_false_color.png")
    Image.fromarray(np.concatenate(band_rows, axis=0)).save(output_dir / "comparison_detail_zooms_900nm.png")
    recommended_false_rows: list[np.ndarray] = []
    recommended_band_rows: list[np.ndarray] = []
    for patch_index, (y0, x0, y1, x1) in enumerate(patches, start=1):
        rgb_patch = rgb_u8[y0:y1, x0:x1]
        recommended_false_rows.append(
            np.concatenate(
                [
                    _label_panel(rgb_patch, f"RGB zoom {patch_index}"),
                    _label_panel(false_u8[0][y0:y1, x0:x1], labels[0]),
                    _label_panel(false_u8[recommended_index][y0:y1, x0:x1], labels[recommended_index]),
                ],
                axis=1,
            )
        )
        recommended_band_rows.append(
            np.concatenate(
                [
                    _label_panel(rgb_patch, f"RGB zoom {patch_index}"),
                    _label_panel(band_u8[0][y0:y1, x0:x1], labels[0]),
                    _label_panel(band_u8[recommended_index][y0:y1, x0:x1], labels[recommended_index]),
                ],
                axis=1,
            )
        )
    Image.fromarray(np.concatenate(recommended_false_rows, axis=0)).save(output_dir / "recommended_detail_zooms_false_color.png")
    Image.fromarray(np.concatenate(recommended_band_rows, axis=0)).save(output_dir / "recommended_detail_zooms_900nm.png")
    return [
        {"y": y0, "x": x0, "height": y1 - y0, "width": x1 - x0}
        for y0, x0, y1, x1 in patches
    ]


def run_scene(config_path: Path, run_dir: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    dataset = discover_triplet(config.data_dir)
    roi = config.roi
    if roi.x is None or roi.y is None:
        raise ValueError("Candidate comparison requires a fixed manual ROI")
    rgb = np.asarray(dataset.rgb.cube[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width, :3])
    low_cube_mem, _ = open_cube(run_dir / "analysis" / "harmonized_lowres.hdr")
    low_cube = np.asarray(low_cube_mem, dtype=np.float32)
    coeff_mem, _ = open_cube(run_dir / "coefficients" / "material_coefficients.hdr")
    base_coeff = np.asarray(coeff_mem, dtype=np.float32)
    gain_mem, _ = open_cube(run_dir / "metrics" / "spatial_detail_gain.hdr")
    base_gain = np.asarray(gain_mem[:, :, 0], dtype=np.float32)
    psf = _psf_from_json(run_dir / "metadata" / "psf_model.json")
    subspace, low_coeff = fit_subspace(
        low_cube,
        rank=config.fusion.rank,
        max_pixels=config.fusion.max_basis_pixels,
        random_seed=config.fusion.random_seed,
        clip_quantiles=config.fusion.clip_quantiles,
    )
    confidence = _rgb_weights(rgb, config.fusion.rgb_edge_sigma)[2]
    wavelengths = np.asarray(open_cube(run_dir / "analysis" / "harmonized_lowres.hdr")[1].wavelengths)
    selected_wavelengths = np.asarray([2200.0, 1650.0, 900.0])
    selected_indices = np.asarray([int(np.argmin(np.abs(wavelengths - value))) for value in selected_wavelengths])
    band_900_index = int(np.argmin(np.abs(wavelengths - 900.0)))

    names = ["v4_current"]
    labels = ["Current v4"]
    coefficient_sets = [base_coeff]
    gain_sets = [base_gain]
    additive_sets = [np.zeros(rgb.shape[:2], dtype=np.float32)]
    additive_scale_sets = [np.zeros(low_cube.shape[2], dtype=np.float32)]
    details: list[dict[str, Any]] = [{"source": "existing_v4_outputs"}]

    low_min = np.quantile(low_coeff, 0.005, axis=(0, 1)).astype(np.float32)
    low_max = np.quantile(low_coeff, 0.995, axis=(0, 1)).astype(np.float32)
    low_std = np.maximum(np.std(low_coeff, axis=(0, 1)).astype(np.float32), 1e-5)
    margin = float(config.fusion.coefficient_clip_margin) * np.maximum(low_max - low_min, low_std)
    for candidate in CANDIDATES:
        candidate_config: FusionConfig = copy.deepcopy(config.fusion)
        candidate_config.coefficient_detail_strength = candidate.coefficient_strength
        candidate_config.coefficient_detail_clip_sigma = candidate.clip_sigma
        candidate_config.coefficient_detail_support_floor = candidate.support_floor
        candidate_config.spatial_detail_strength = candidate.gain_strength
        candidate_config.spatial_detail_additive_strength = candidate.additive_strength
        candidate_config.spatial_detail_gain_limits = (0.68, 1.40)
        enhanced, coefficient_details = inject_coefficient_detail(
            base_coeff,
            low_coeff,
            rgb,
            psf,
            confidence,
            candidate_config,
            clip_min=low_min - margin,
            clip_max=low_max + margin,
        )
        gain, additive, spatial_details = _spectral_shape_preserving_detail_gain(
            rgb,
            psf,
            confidence,
            candidate_config,
        )
        additive_scale = build_additive_spectral_scale(low_cube, candidate_config)
        names.append(candidate.name)
        labels.append(candidate.label)
        coefficient_sets.append(enhanced)
        gain_sets.append(gain)
        additive_sets.append(additive)
        additive_scale_sets.append(additive_scale)
        details.append({"coefficient_detail": coefficient_details, "spatial_detail": spatial_details})

    preview_scale = min(1.0, 1024.0 / float(max(rgb.shape[:2])))
    preview_size = (int(round(rgb.shape[1] * preview_scale)), int(round(rgb.shape[0] * preview_scale)))
    rgb_small = cv2.resize(normalize_rgb(rgb), preview_size, interpolation=cv2.INTER_AREA)
    rgb_luminance = cv2.cvtColor((rgb_small * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    false_colors: list[np.ndarray] = []
    band_images: list[np.ndarray] = []
    metrics: dict[str, Any] = {}
    full_false_colors: list[np.ndarray] = []
    full_band_images: list[np.ndarray] = []
    for name, coeff, gain, additive, additive_scale, detail in zip(
        names,
        coefficient_sets,
        gain_sets,
        additive_sets,
        additive_scale_sets,
        details,
    ):
        coeff_small = _resize_coefficients(coeff, preview_size)
        gain_small = cv2.resize(gain, preview_size, interpolation=cv2.INTER_AREA)
        additive_small = cv2.resize(additive, preview_size, interpolation=cv2.INTER_AREA)
        selected = _reconstruct_selected_bands(
            coeff_small,
            subspace,
            selected_indices,
            gain_small,
            additive_small,
            additive_scale,
        )
        band_900 = _reconstruct_selected_bands(
            coeff_small,
            subspace,
            np.asarray([band_900_index]),
            gain_small,
            additive_small,
            additive_scale,
        )[:, :, 0]
        false_colors.append(selected)
        band_images.append(band_900)

        full_selected = _reconstruct_selected_bands(
            coeff,
            subspace,
            selected_indices,
            gain,
            additive,
            additive_scale,
        )
        full_band = _reconstruct_selected_bands(
            coeff,
            subspace,
            np.asarray([band_900_index]),
            gain,
            additive,
            additive_scale,
        )[:, :, 0]
        full_false_colors.append(full_selected)
        full_band_images.append(full_band)
        gray_false = np.mean(selected, axis=2)
        observation = _observation_metrics(
            coeff,
            gain,
            additive,
            additive_scale,
            low_coeff,
            low_cube,
            subspace,
            psf,
        )
        metrics[name] = {
            **observation,
            "rgb_edge_correlation_false_color": _edge_correlation(gray_false, rgb_small),
            "rgb_edge_correlation_900nm": _edge_correlation(band_900, rgb_small),
            "false_color_high_frequency_energy": _normalized_high_frequency_energy(gray_false),
            "band_900_high_frequency_energy": _normalized_high_frequency_energy(band_900),
            "rgb_high_frequency_energy": _normalized_high_frequency_energy(rgb_luminance),
            "mean_abs_gain_from_one": float(np.mean(np.abs(gain - 1.0))),
            "coefficient_detail": detail,
        }

    borehole = str(config.project.borehole_id or "").lower()
    recommended_name = "structure_strong" if "3dssz" in borehole else "hybrid_sharp"
    patches = _save_comparisons(
        output_dir,
        rgb,
        names,
        labels,
        full_false_colors,
        full_band_images,
        recommended_name,
    )
    for name, image, band in zip(names, full_false_colors, full_band_images):
        candidate_dir = output_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        false_u8 = _stretch_reference([full_false_colors[0], image], reference_index=0)[1]
        band_u8 = _stretch_reference(
            [np.repeat(full_band_images[0][:, :, None], 3, axis=2), np.repeat(band[:, :, None], 3, axis=2)],
            reference_index=0,
        )[1]
        Image.fromarray(false_u8).save(candidate_dir / "false_color_2200_1650_900.png")
        Image.fromarray(band_u8[:, :, 0]).save(candidate_dir / "band_900nm.png")
    write_json(
        output_dir / "candidate_metrics.json",
        {
            "source_config": str(config_path),
            "source_run": str(run_dir),
            "roi": {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
            "selected_wavelengths_nm": selected_wavelengths.tolist(),
            "detail_patches": patches,
            "candidate_order": names,
            "recommended_candidate": recommended_name,
            "metrics": metrics,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=("3dssz", "zkh3", "both"),
        default="both",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs" / "spatial_detail_v5_candidates",
    )
    args = parser.parse_args()
    scenes = {
        "3dssz": (
            ROOT / "configs" / "3dssz_roi_fusion_v4.yaml",
            ROOT / "runs" / "3dssz_roi_fusion_v4",
        ),
        "zkh3": (
            ROOT / "configs" / "zkh3_roi_fusion_v4.yaml",
            ROOT / "runs" / "zkh3_roi_fusion_v4",
        ),
    }
    selected = scenes.keys() if args.scene == "both" else (args.scene,)
    for scene in selected:
        config_path, run_dir = scenes[scene]
        run_scene(config_path, run_dir, args.output_root / scene)


if __name__ == "__main__":
    main()

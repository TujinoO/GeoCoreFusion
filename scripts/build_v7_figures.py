"""Build publication-ready V7 fusion figures from completed run artifacts.

The real-scene comparison does not reuse the pipeline PNG previews.  It
reconstructs the requested 901/1651/2201 nm bands from the saved coefficient
fields, refits the deterministic spectral basis from ``harmonized_lowres``,
reapplies the saved gain/additive fields, and then uses one shared linear
display mapping for V5/V6/V7 within each scene.

Outputs are deterministic for fixed run artifacts.  Every figure is written as
300-dpi PNG and vector PDF.  No input artifact is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# Okabe-Ito-derived, colour-blind-friendly palette.
NAVY = "#16324F"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
DARK_GRAY = "#374151"
LIGHT = "#F5F7FA"
PALE_BLUE = "#E5F1F8"
PALE_GREEN = "#E5F5EF"
PALE_ORANGE = "#FFF1D6"
PALE_RED = "#FCE8E2"
GRID = "#D8DEE8"
WHITE = "#FFFFFF"

METHOD_COLORS = {"V5": GRAY, "V6": VERMILION, "V7": BLUE}
SCENE_COLORS = {"3dssz": BLUE, "zkh3": ORANGE}
DISPLAY_WAVELENGTHS = np.asarray([2200.0, 1650.0, 900.0], dtype=np.float32)
# Fixed, native-grid crops chosen on RGB only.  They avoid printed labels,
# tray junctions, and saturated glints, and instead sample dark fractured core.
DARK_CROP_OVERRIDES = {
    "3dssz": (520, 180, 260, 320),
    "zkh3": (95, 455, 260, 320),
}

SCENES: dict[str, dict[str, Any]] = {
    "3dssz": {
        "label": "3DSSZ",
        "runs": {
            "V5": "3dssz_roi_fusion_v5_matched_v7eval_ampfix",
            "V6": "3dssz_roi_fusion_v6_v7eval_ampfix",
            "V7": "3dssz_roi_fusion_v7_final_ampfix",
        },
    },
    "zkh3": {
        "label": "ZKH3",
        "runs": {
            "V5": "zkh3_roi_fusion_v5_matched_v7eval_ampfix",
            "V6": "zkh3_roi_fusion_v6_v7eval_ampfix",
            "V7": "zkh3_roi_fusion_v7_final_ampfix",
        },
    },
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": DARK_GRAY,
            "axes.edgecolor": GRID,
            "xtick.color": DARK_GRAY,
            "ytick.color": DARK_GRAY,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [png.name, pdf.name]


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    face: str = PALE_BLUE,
    edge: str = BLUE,
    title_color: str = NAVY,
    body_size: float = 8.8,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        color=title_color,
        weight="bold",
        fontsize=10.5,
        transform=ax.transAxes,
    )
    ax.text(
        x + width / 2,
        y + height * 0.35,
        body,
        ha="center",
        va="center",
        color=DARK_GRAY,
        fontsize=body_size,
        linespacing=1.35,
        transform=ax.transAxes,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BLUE,
    width: float = 1.6,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def import_project(repo: Path) -> dict[str, Any]:
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from geocorefusion.config import FusionConfig
    from geocorefusion.degradation import PsfModel
    from geocorefusion.envi import open_cube
    from geocorefusion.fusion import (
        _dark_texture_confidence,
        _rgb_texture_coherence,
        _rgb_weights,
        build_additive_spectral_scale,
        build_band_adaptive_mtf_detail,
    )
    from geocorefusion.lowrank import fit_subspace

    return {
        "FusionConfig": FusionConfig,
        "PsfModel": PsfModel,
        "open_cube": open_cube,
        "fit_subspace": fit_subspace,
        "build_additive_spectral_scale": build_additive_spectral_scale,
        "build_band_adaptive_mtf_detail": build_band_adaptive_mtf_detail,
        "_rgb_texture_coherence": _rgb_texture_coherence,
        "_dark_texture_confidence": _dark_texture_confidence,
        "_rgb_weights": _rgb_weights,
    }


def fusion_config_from_payload(payload: dict[str, Any], FusionConfig: Any) -> Any:
    config = FusionConfig()
    allowed = {field.name for field in fields(config)}
    for key, value in payload.items():
        if key not in allowed:
            continue
        current = getattr(config, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(config, key, value)
    return config


def find_rgb_header(repo: Path, run_dir: Path) -> Path:
    meta = read_json(run_dir / "metadata" / "input_metadata.json")["rgb"]
    stated = Path(meta["hdr_path"])
    if stated.exists():
        return stated
    matches = sorted((repo / "data").glob(f"*/{stated.name}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Cannot resolve RGB header {stated.name!r}; matches={len(matches)}"
        )
    return matches[0]


def load_rgb_crop(repo: Path, run_dir: Path, api: dict[str, Any]) -> np.ndarray:
    processing = read_json(run_dir / "metadata" / "processing_config.json")
    roi = processing["roi"]
    cube, _ = api["open_cube"](find_rgb_header(repo, run_dir))
    crop = np.asarray(
        cube[
            int(roi["y"]) : int(roi["y"] + roi["height"]),
            int(roi["x"]) : int(roi["x"] + roi["width"]),
            :3,
        ]
    ).copy()
    if np.issubdtype(crop.dtype, np.integer):
        crop = crop.astype(np.float32) / float(np.iinfo(crop.dtype).max)
    else:
        crop = crop.astype(np.float32)
        finite = crop[np.isfinite(crop)]
        if finite.size and float(np.percentile(finite, 99.9)) > 2.0:
            crop /= 255.0
    return np.clip(crop, 0.0, 1.0).astype(np.float32)


def load_scalar_map(path: Path, api: dict[str, Any]) -> np.ndarray:
    cube, _ = api["open_cube"](path)
    return np.asarray(cube[:, :, 0], dtype=np.float32).copy()


def validate_against_pipeline_preview(
    fused: np.ndarray, run_dir: Path
) -> tuple[float, float]:
    """Replay the pipeline's per-band preview stretch as a reconstruction check.

    This validation is not used for the publication comparison (which uses a
    shared scene mapping).  It only confirms that the reconstructed reflectance
    bands reproduce the run's saved, independently stretched diagnostic PNG.
    """

    target_path = run_dir / "previews" / "fused_false_color_2200_1650_900.png"
    target_bgr = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if target_bgr is None:
        raise FileNotFoundError(target_path)
    target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    scale = min(1.0, 1024.0 / float(max(fused.shape[:2])))
    size = (
        max(1, int(round(fused.shape[1] * scale))),
        max(1, int(round(fused.shape[0] * scale))),
    )
    small = cv2.resize(
        fused,
        size,
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    replay = np.empty_like(target, dtype=np.uint8)
    for channel in range(3):
        values = small[:, :, channel]
        valid = np.isfinite(values)
        lo, hi = np.percentile(values[valid], [2.0, 98.0])
        hi = max(float(hi), float(lo) + 1e-6)
        replay[:, :, channel] = np.round(
            np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    absolute = np.abs(replay.astype(np.int16) - target.astype(np.int16))
    return float(np.mean(absolute)), float(np.percentile(absolute, 99.0))


def reconstruct_display_bands(
    repo: Path,
    run_dir: Path,
    api: dict[str, Any],
    *,
    rgb_cache: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    processing = read_json(run_dir / "metadata" / "processing_config.json")
    fusion_config = fusion_config_from_payload(
        processing["fusion"], api["FusionConfig"]
    )
    low_cube_mmap, low_meta = api["open_cube"](
        run_dir / "analysis" / "harmonized_lowres.hdr"
    )
    low_cube = np.asarray(low_cube_mmap, dtype=np.float32).copy()
    model, _ = api["fit_subspace"](
        low_cube,
        rank=int(fusion_config.rank),
        max_pixels=int(fusion_config.max_basis_pixels),
        random_seed=int(fusion_config.random_seed),
        clip_quantiles=tuple(fusion_config.clip_quantiles),
    )
    stored_subspace = read_json(run_dir / "metadata" / "subspace_model.json")
    stored_mean = np.asarray(stored_subspace["mean_spectrum"], dtype=np.float32)
    mean_error = float(np.max(np.abs(model.mean_spectrum - stored_mean)))
    if mean_error > 2e-6:
        raise RuntimeError(f"Subspace refit mismatch for {run_dir.name}: {mean_error:.3g}")

    indices = np.asarray(
        [int(np.argmin(np.abs(low_meta.wavelengths - w))) for w in DISPLAY_WAVELENGTHS],
        dtype=np.int64,
    )
    wavelengths = np.asarray(low_meta.wavelengths[indices], dtype=np.float32)
    coeff_mmap, _ = api["open_cube"](
        run_dir / "coefficients" / "material_coefficients.hdr"
    )
    coeff = np.asarray(coeff_mmap, dtype=np.float32)
    gain = load_scalar_map(
        run_dir / "metrics" / "spatial_detail_gain.hdr", api
    )
    additive = load_scalar_map(
        run_dir / "metrics" / "spatial_additive_detail.hdr", api
    )
    rgb = rgb_cache if rgb_cache is not None else load_rgb_crop(repo, run_dir, api)

    mode = str(fusion_config.spatial_detail_additive_mode).strip().lower()
    additive_strength = float(fusion_config.spatial_detail_additive_strength)
    detail_recompute_error = 0.0
    if mode in {"band_adaptive_mtf_gsa", "mtf_gsa", "band_adaptive"}:
        psf_payload = read_json(run_dir / "metadata" / "psf_model.json")
        psf = api["PsfModel"](
            sigma_x_highres=float(psf_payload["sigma_x_highres"]),
            sigma_y_highres=float(psf_payload["sigma_y_highres"]),
            score=float(psf_payload["score"]),
            low_shape=tuple(int(v) for v in psf_payload["low_shape"]),
            high_shape=tuple(int(v) for v in psf_payload["high_shape"]),
            method=str(psf_payload.get("method", "anisotropic_gaussian_grid_search")),
        )
        if additive_strength > 0.0:
            recomputed_detail, additive_scale, _ = api[
                "build_band_adaptive_mtf_detail"
            ](low_cube, rgb, psf, fusion_config)
            detail_recompute_error = float(
                np.max(np.abs(recomputed_detail - additive))
            )
            # The saved map is float32 ENVI while the deterministic replay
            # crosses a few BLAS/OpenCV reduction boundaries; sub-1e-4
            # differences are numerically immaterial for the display bands.
            if detail_recompute_error > 1e-4:
                raise RuntimeError(
                    f"Additive detail recompute mismatch for {run_dir.name}: "
                    f"{detail_recompute_error:.3g}"
                )
        else:
            additive_scale = np.zeros(low_cube.shape[2], dtype=np.float32)
    else:
        additive_scale = api["build_additive_spectral_scale"](
            low_cube, fusion_config
        )

    selected_basis = np.asarray(model.basis[:, indices], dtype=np.float32)
    selected_mean = np.asarray(model.mean_spectrum[indices], dtype=np.float32)
    selected_scale = np.asarray(additive_scale[indices], dtype=np.float32)
    height, width = coeff.shape[:2]
    fused = np.empty((height, width, len(indices)), dtype=np.float32)
    tile = 192
    for y0 in range(0, height, tile):
        y1 = min(height, y0 + tile)
        for x0 in range(0, width, tile):
            x1 = min(width, x0 + tile)
            values = np.einsum(
                "hwk,kc->hwc",
                coeff[y0:y1, x0:x1, :],
                selected_basis,
                optimize=True,
            )
            values += selected_mean[None, None, :]
            values *= gain[y0:y1, x0:x1, None]
            values += additive[y0:y1, x0:x1, None] * selected_scale[None, None, :]
            np.maximum(values, 0.0, out=values)
            fused[y0:y1, x0:x1, :] = values

    diagnostics = {
        "actual_wavelengths_nm": [float(v) for v in wavelengths],
        "subspace_mean_refit_max_abs_error": mean_error,
        "additive_detail_recompute_max_abs_error": detail_recompute_error,
        "additive_selected_scale": [float(v) for v in selected_scale],
    }
    preview_mae, preview_p99 = validate_against_pipeline_preview(fused, run_dir)
    diagnostics["pipeline_preview_replay_mae_8bit"] = preview_mae
    diagnostics["pipeline_preview_replay_p99_abs_8bit"] = preview_p99
    if preview_mae > 0.25 or preview_p99 > 1.0:
        raise RuntimeError(
            f"Reconstructed preview mismatch for {run_dir.name}: "
            f"MAE={preview_mae:.3g}, P99={preview_p99:.3g}"
        )
    return fused, rgb, diagnostics


def percentile_bounds(images: list[np.ndarray]) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for channel in range(3):
        values = np.concatenate(
            [image[:, :, channel][::3, ::3].reshape(-1) for image in images]
        )
        values = values[np.isfinite(values)]
        lo, hi = np.percentile(values, [2.0, 98.0])
        if hi <= lo:
            hi = lo + 1e-6
        bounds.append((float(lo), float(hi)))
    return bounds


def apply_bounds(image: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    out = np.empty_like(image, dtype=np.float32)
    for channel, (lo, hi) in enumerate(bounds):
        out[:, :, channel] = np.clip(
            (image[:, :, channel] - lo) / (hi - lo), 0.0, 1.0
        )
    return out


def select_dark_texture_crop(
    rgb: np.ndarray,
    *,
    crop_width: int = 260,
    crop_height: int = 320,
) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(
        (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    best: tuple[float, int, int] | None = None
    x_stop = max(21, gray.shape[1] - crop_width - 20)
    y_stop = max(21, gray.shape[0] - crop_height - 20)
    for y in range(20, y_stop, 40):
        for x in range(20, x_stop, 36):
            patch = gray[y : y + crop_height, x : x + crop_width]
            mean = float(np.mean(patch))
            if not 0.025 <= mean <= 0.42:
                continue
            texture = float(
                np.percentile(
                    gradient[y : y + crop_height, x : x + crop_width], 86.0
                )
            )
            saturation = float(np.mean((patch < 0.01) | (patch > 0.99)))
            score = texture * (0.60 + max(0.0, 0.42 - mean)) * (1.0 - saturation)
            if best is None or score > best[0]:
                best = (score, x, y)
    if best is None:
        return (
            max(0, (gray.shape[1] - crop_width) // 2),
            max(0, (gray.shape[0] - crop_height) // 2),
            crop_width,
            crop_height,
        )
    return best[1], best[2], crop_width, crop_height


def extract_metrics(run_dir: Path) -> dict[str, float]:
    report = read_json(run_dir / "metrics" / "quality_report.json")
    band = report["spatial"]["band_detail_by_brightness"]["bands"]["2201.0nm"]
    reliable = band["multiscale_log_high_frequency"]["sigma_2.4px"][
        "reliable_rgb_detail"
    ]
    dark = band["multiscale_log_high_frequency"]["sigma_2.4px"][
        "dark_reliable_rgb_detail"
    ]
    edge = band["gradient_and_edge"]["reliable_rgb_detail"]
    dark_edge = band["gradient_and_edge"]["dark_reliable_rgb_detail"]
    observation = report["final_hr_product_observation"]
    return {
        "rho": float(reliable["rho"]),
        "beta": float(reliable["beta"]),
        "A": float(reliable["energy_ratio_A"]),
        "R_perp": float(reliable["orthogonal_residual_ratio_R_perp"]),
        "dark_rho": float(dark["rho"]),
        "dark_beta": float(dark["beta"]),
        "dark_A": float(dark["energy_ratio_A"]),
        "dark_R_perp": float(dark["orthogonal_residual_ratio_R_perp"]),
        "edge_f1": float(edge["edge_f1_1px"]),
        "dark_edge_f1": float(dark_edge["edge_f1_1px"]),
        "forward_rmse": float(observation["rmse"]),
        "forward_sam_deg": float(observation["sam_mean_deg"]),
        "boundary_correlation": float(
            report["spatial"]["rgb_material_boundary_correlation"]
        ),
    }


def collect_data(repo: Path, api: dict[str, Any]) -> dict[str, Any]:
    runs_root = repo / "runs"
    result: dict[str, Any] = {}
    for scene, scene_spec in SCENES.items():
        scene_result: dict[str, Any] = {"methods": {}}
        first_run = runs_root / scene_spec["runs"]["V7"]
        rgb = load_rgb_crop(repo, first_run, api)
        for method, run_name in scene_spec["runs"].items():
            run_dir = runs_root / run_name
            required = [
                run_dir / "metrics" / "quality_report.json",
                run_dir / "analysis" / "harmonized_lowres.hdr",
                run_dir / "coefficients" / "material_coefficients.hdr",
                run_dir / "metrics" / "spatial_detail_gain.hdr",
                run_dir / "metrics" / "spatial_additive_detail.hdr",
            ]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete source run {run_name}; missing: {missing}"
                )
            image, _, diagnostics = reconstruct_display_bands(
                repo, run_dir, api, rgb_cache=rgb
            )
            scene_result["methods"][method] = {
                "run": run_name,
                "image": image,
                "metrics": extract_metrics(run_dir),
                "diagnostics": diagnostics,
            }
        images = [
            scene_result["methods"][method]["image"]
            for method in ("V5", "V6", "V7")
        ]
        scene_result["rgb"] = rgb
        scene_result["display_bounds"] = percentile_bounds(images)
        scene_result["dark_crop"] = DARK_CROP_OVERRIDES.get(
            scene, select_dark_texture_crop(rgb)
        )
        result[scene] = scene_result
    return result


def root_cause_and_mechanism(data: dict[str, Any], output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(18.2, 9.4))
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "从 V6 的‘共享乘性增益’到 V7 的‘材料条件化相干细节注入’",
        transform=ax.transAxes,
        fontsize=20,
        weight="bold",
        color=NAVY,
        va="top",
    )
    ax.text(
        0.02,
        0.915,
        "根因不是增益不够大，而是细节通道、局部材料响应和噪声判别方式不完整",
        transform=ax.transAxes,
        fontsize=11,
        color=DARK_GRAY,
        va="top",
    )
    ax.plot([0.495, 0.495], [0.09, 0.88], color=GRID, lw=1.5, transform=ax.transAxes)
    ax.text(0.035, 0.855, "V6 根因审计", color=VERMILION, fontsize=15, weight="bold", transform=ax.transAxes)
    ax.text(0.525, 0.855, "V7 机制链", color=BLUE, fontsize=15, weight="bold", transform=ax.transAxes)

    add_box(
        ax,
        (0.035, 0.63),
        0.19,
        0.16,
        "单一 RGB 亮度细节",
        "$d_{RGB}=\\log Y-G_{MTF}(\\log Y)$\n无局部材料方向/符号",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
    )
    add_box(
        ax,
        (0.275, 0.63),
        0.18,
        0.16,
        "367 波段共享乘法",
        "$\\hat X_\\lambda^{V6}=X_\\lambda\\exp(\\alpha d_{RGB})$\n"
        "暗波段接近 0 时，绝对细节仍接近 0",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
        body_size=8.4,
    )
    add_arrow(ax, (0.225, 0.71), (0.275, 0.71), color=VERMILION)

    v6_gain_text = (
        "增益 P05–P95：3DSSZ 0.874–1.134；ZKH3 0.889–1.118\n"
        "边界相关 V5→V6：3DSSZ 0.373→0.332；ZKH3 0.310→0.246"
    )
    add_box(
        ax,
        (0.035, 0.405),
        0.42,
        0.15,
        "真实运行证据：幅度放大并未等价为有效细节",
        v6_gain_text,
        face=LIGHT,
        edge=GRAY,
        title_color=NAVY,
        body_size=8.8,
    )
    v6_3 = data["3dssz"]["methods"]["V6"]["metrics"]
    v6_z = data["zkh3"]["methods"]["V6"]["metrics"]
    add_box(
        ax,
        (0.035, 0.19),
        0.42,
        0.145,
        "2201 nm：非相干高频仍占主导",
        (
            f"3DSSZ：A={v6_3['A']:.2f}，R⊥={v6_3['R_perp']:.2f}；"
            f"ZKH3：A={v6_z['A']:.2f}，R⊥={v6_z['R_perp']:.2f}\n"
            "继续扩大 α 会同时扩大颗粒、振铃与 halo，不能解决根因"
        ),
        face=PALE_ORANGE,
        edge=ORANGE,
        title_color=VERMILION,
        body_size=8.7,
    )

    x_positions = [0.515, 0.609, 0.703, 0.797, 0.891]
    titles = [
        "① MTF 对齐",
        "② 相干门控",
        "③ 局部材料响应",
        "④ 双通道注入",
        "⑤ 最终闭环",
    ]
    bodies = [
        "线性/log 分解\n真实 PSF/MTF\n定义高频",
        "max |通道相关|\n+局部能量\n+黑电平可靠度",
        "局部 $\\beta_k(x)$\n亮度 + R−G + B−G\n有符号材料响应",
        "系数 MTF-GSA\nlog-HPM\n+$a_\\lambda d$ 暗区支路",
        "单次物理裁剪\n最终 HR→PSF→LR\nforward 一致性",
    ]
    faces = [PALE_BLUE, PALE_GREEN, PALE_BLUE, PALE_GREEN, LIGHT]
    edges = [BLUE, GREEN, BLUE, GREEN, NAVY]
    for x, title, body, face, edge in zip(
        x_positions, titles, bodies, faces, edges, strict=True
    ):
        add_box(
            ax,
            (x, 0.61),
            0.080,
            0.19,
            title,
            body,
            face=face,
            edge=edge,
            title_color=edge,
            body_size=7.1,
        )
    for first, second in zip(x_positions[:-1], x_positions[1:], strict=True):
        add_arrow(ax, (first + 0.080, 0.705), (second, 0.705), color=BLUE, width=1.2)

    for row, scene in enumerate(("3dssz", "zkh3")):
        y = 0.43 - 0.17 * row
        label = SCENES[scene]["label"]
        v6 = data[scene]["methods"]["V6"]["metrics"]
        v7 = data[scene]["methods"]["V7"]["metrics"]
        add_box(
            ax,
            (0.54, y),
            0.43,
            0.12,
            f"{label}：V6 → V7（2201 nm，可靠 RGB 细节区）",
            (
                f"ρ {v6['rho']:.3f}→{v7['rho']:.3f} ｜ "
                f"A {v6['A']:.2f}→{v7['A']:.2f} ｜ "
                f"R⊥ {v6['R_perp']:.2f}→{v7['R_perp']:.2f} ｜ "
                f"forward RMSE {v6['forward_rmse']:.5f}→{v7['forward_rmse']:.5f}"
            ),
            face=PALE_GREEN if row == 0 else PALE_BLUE,
            edge=GREEN if row == 0 else BLUE,
            title_color=NAVY,
            body_size=8.5,
        )
    ax.text(
        0.525,
        0.115,
        "优化目标：ρ↑，β≈1，A≈1，R⊥↓，edge F1↑；由幅度—噪声—光谱一致性 Pareto 选择参数。",
        transform=ax.transAxes,
        fontsize=10.3,
        weight="bold",
        color=NAVY,
    )
    ax.text(
        0.02,
        0.055,
        "科学声明：V7 改善的是同源 RGB 引导下的等效空间结构与低分辨率观测一致性；没有独立 HR-SWIR 真值时，不宣称 SWIR 辐射细节‘无损恢复’。",
        transform=ax.transAxes,
        fontsize=10.0,
        weight="bold",
        color=VERMILION,
    )
    return save_figure(fig, output_dir, "01_v6_root_cause_v7_mechanism")


def visual_comparison(
    scene: str, data: dict[str, Any], output_dir: Path, number: int
) -> list[str]:
    scene_data = data[scene]
    rgb = scene_data["rgb"]
    bounds = scene_data["display_bounds"]
    x, y, width, height = scene_data["dark_crop"]
    displays: dict[str, np.ndarray] = {"RGB": np.clip(rgb, 0.0, 1.0)}
    for method in ("V5", "V6", "V7"):
        displays[method] = apply_bounds(
            scene_data["methods"][method]["image"], bounds
        )

    fig, axes = plt.subplots(2, 4, figsize=(16.4, 10.0))
    titles = {
        "RGB": "RGB 空间参照",
        "V5": "V5 matched",
        "V6": "V6 共享乘性增益",
        "V7": "V7 材料条件化相干注入",
    }
    for column, method in enumerate(("RGB", "V5", "V6", "V7")):
        image = displays[method]
        axes[0, column].imshow(image)
        axes[0, column].add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                edgecolor=ORANGE,
                facecolor="none",
                linewidth=2.0,
            )
        )
        axes[0, column].set_title(titles[method], color=NAVY)
        axes[0, column].axis("off")
        axes[1, column].imshow(image[y : y + height, x : x + width])
        axes[1, column].axis("off")
        if method == "RGB":
            subtitle = "暗区纹理 ROI（仅作几何参照）"
        else:
            metrics = scene_data["methods"][method]["metrics"]
            subtitle = (
                f"暗区 2201 nm：ρ={metrics['dark_rho']:.3f}  "
                f"β={metrics['dark_beta']:.3f}  A={metrics['dark_A']:.2f}"
            )
        axes[1, column].set_title(
            subtitle,
            fontsize=9.5,
            color=METHOD_COLORS.get(method, GREEN),
            weight="bold" if method == "V7" else "normal",
        )
    label = SCENES[scene]["label"]
    actual = scene_data["methods"]["V7"]["diagnostics"]["actual_wavelengths_nm"]
    bounds_text = "; ".join(
        f"{actual[index]:.0f} nm [{lo:.4f}, {hi:.4f}]"
        for index, (lo, hi) in enumerate(bounds)
    )
    fig.suptitle(
        f"{label}：RGB / V5 / V6 / V7 同裁剪、同显示域对比",
        fontsize=17,
        weight="bold",
        color=NAVY,
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "HSI 三种方法共享同一场景、同一通道的 P2–P98 线性映射（无逐方法自适应拉伸）："
        + bounds_text,
        ha="center",
        fontsize=8.8,
        color=DARK_GRAY,
    )
    fig.text(
        0.5,
        0.012,
        "RGB 与假彩色 HSI 不是同一辐射量；图像与 ρ/β/A 均为同源结构诊断，不是独立 HR-SWIR 真值。",
        ha="center",
        fontsize=9.2,
        color=VERMILION,
        weight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.065, 0.99, 0.95), h_pad=1.4, w_pad=0.5)
    return save_figure(
        fig,
        output_dir,
        f"{number:02d}_{scene}_v5_v6_v7_unified_visual_comparison",
    )


def detail_metric_small_multiples(data: dict[str, Any], output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.8))
    methods = ("V5", "V6", "V7")
    x = np.arange(len(methods), dtype=np.float32)
    panels = [
        ("rho", "ρ：相干性（高为优）", (0.0, 0.9), (0.8, 0.9)),
        ("beta", "β：相干幅度斜率（≈1）", (0.0, 1.65), (0.8, 1.2)),
        ("A", "A：高频能量比（≈1）", (0.0, 5.0), (0.8, 1.25)),
        ("R_perp", "R⊥：非相干残差（低为优）", (0.0, 4.8), (0.0, 0.35)),
    ]
    markers = {"3dssz": "o", "zkh3": "s"}
    for ax, (metric, title, ylim, target) in zip(axes, panels, strict=True):
        ax.axhspan(target[0], target[1], color=GREEN, alpha=0.10, zorder=0)
        for scene in ("3dssz", "zkh3"):
            values = [
                data[scene]["methods"][method]["metrics"][metric]
                for method in methods
            ]
            ax.plot(
                x,
                values,
                color=SCENE_COLORS[scene],
                marker=markers[scene],
                markersize=6.5,
                linewidth=1.8,
                label=SCENES[scene]["label"],
            )
            for index, value in enumerate(values):
                ax.annotate(
                    f"{value:.2f}",
                    (x[index], value),
                    xytext=(0, 6 if scene == "3dssz" else -13),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.6,
                    color=SCENE_COLORS[scene],
                )
        ax.set_xticks(x, methods)
        ax.set_ylim(*ylim)
        ax.set_title(title, color=NAVY, fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "2201 nm 细节 Pareto 诊断：V7 提高相干性并压低过量高频，但尚未满足全部保守界限",
        fontsize=15.5,
        weight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.03,
        "绿色带为预注册的保守筛选区；统计范围为 sigma=2.4 px、可靠 RGB 细节像元。ρ/β/A/R⊥ 是同源结构传递诊断，不是独立 SWIR 真值。",
        ha="center",
        color=VERMILION,
        fontsize=9.2,
        weight="bold",
    )
    fig.tight_layout(w_pad=1.2)
    return save_figure(fig, output_dir, "04_rho_beta_A_Rperp_small_multiples")


def forward_and_edge_metrics(data: dict[str, Any], output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0))
    scenes = ("3dssz", "zkh3")
    methods = ("V5", "V6", "V7")
    x = np.arange(len(scenes), dtype=np.float32)
    width = 0.23
    panels = [
        ("forward_rmse", "最终 HR 产品 forward RMSE", "反射率 RMSE", None),
        ("forward_sam_deg", "最终 HR 产品 forward SAM", "SAM（°）", None),
        ("edge_f1", "2201 nm edge F1（1 px）", "F1", (0.85, 1.0)),
    ]
    for ax, (metric, title, ylabel, target) in zip(axes, panels, strict=True):
        if target is not None:
            ax.axhspan(*target, color=GREEN, alpha=0.10, zorder=0)
        for method_index, method in enumerate(methods):
            values = [
                data[scene]["methods"][method]["metrics"][metric]
                for scene in scenes
            ]
            positions = x + (method_index - 1) * width
            bars = ax.bar(
                positions,
                values,
                width=width,
                color=METHOD_COLORS[method],
                label=method,
                zorder=2,
            )
            for bar, value in zip(bars, values, strict=True):
                fmt = f"{value:.5f}" if metric == "forward_rmse" else f"{value:.3f}"
                ax.annotate(
                    fmt,
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    rotation=0,
                )
        ax.set_xticks(x, [SCENES[scene]["label"] for scene in scenes])
        ax.set_title(title, color=NAVY, fontsize=11.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "edge_f1":
            ax.set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle(
        "V5 / V6 / V7：观测闭环与边缘定位的联合验证",
        fontsize=15.5,
        weight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.035,
        "forward RMSE/SAM 由最终 HR 立方体经实际 PSF 降质后与 LR 观测比较，只证明观测一致性；edge F1 以同源 RGB 边缘为结构参照，不是独立 HR-SWIR 真值。",
        ha="center",
        fontsize=9.0,
        color=VERMILION,
        weight="bold",
    )
    fig.tight_layout(w_pad=1.5)
    return save_figure(fig, output_dir, "05_forward_rmse_sam_edge_f1")


def coherence_components(rgb: np.ndarray, config: Any) -> dict[str, np.ndarray]:
    epsilon = max(float(config.intrinsic_log_epsilon), 1e-4)
    log_rgb = np.log(np.asarray(rgb, dtype=np.float32) + epsilon)
    residual = np.empty_like(log_rgb, dtype=np.float32)
    for channel in range(3):
        base = cv2.GaussianBlur(
            log_rgb[:, :, channel],
            (0, 0),
            sigmaX=1.25,
            sigmaY=1.25,
            borderType=cv2.BORDER_REFLECT101,
        )
        residual[:, :, channel] = log_rgb[:, :, channel] - base
    radius = max(1, int(config.dark_texture_window_radius))
    size = 2 * radius + 1

    def local_mean(arr: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(
            arr.astype(np.float32),
            ddepth=-1,
            ksize=(size, size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT101,
        )

    means = [local_mean(residual[:, :, channel]) for channel in range(3)]
    variances = [
        np.maximum(local_mean(residual[:, :, channel] ** 2) - means[channel] ** 2, 0.0)
        for channel in range(3)
    ]
    correlations = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        covariance = local_mean(residual[:, :, first] * residual[:, :, second]) - means[first] * means[second]
        denominator = np.sqrt(np.maximum(variances[first] * variances[second], 1e-12))
        correlations.append(
            np.divide(
                covariance,
                denominator,
                out=np.zeros_like(covariance),
                where=denominator > 1e-6,
            )
        )
    coherence = np.max(np.abs(np.stack(correlations, axis=2)), axis=2)
    energy = np.sqrt(np.mean(np.stack(variances, axis=2), axis=2))
    luminance_log = np.mean(log_rgb, axis=2)
    luminance_detail = luminance_log - cv2.GaussianBlur(
        luminance_log,
        (0, 0),
        sigmaX=1.25,
        sigmaY=1.25,
        borderType=cv2.BORDER_REFLECT101,
    )
    return {
        "residual": residual,
        "coherence": coherence,
        "energy": energy,
        "luminance_detail": luminance_detail,
    }


def dark_coherence_gate(
    repo: Path, data: dict[str, Any], api: dict[str, Any], output_dir: Path
) -> list[str]:
    scene = "zkh3"
    scene_data = data[scene]
    rgb = scene_data["rgb"]
    x, y, width, height = scene_data["dark_crop"]
    run_dir = repo / "runs" / SCENES[scene]["runs"]["V7"]
    processing = read_json(run_dir / "metadata" / "processing_config.json")
    config = fusion_config_from_payload(processing["fusion"], api["FusionConfig"])
    components = coherence_components(rgb, config)
    texture_gate = api["_rgb_texture_coherence"](rgb, config)
    dark_gate = api["_dark_texture_confidence"](rgb, config)
    rgb_confidence = api["_rgb_weights"](rgb, config.rgb_edge_sigma, config)[2]
    sl = np.s_[y : y + height, x : x + width]
    rgb_patch = rgb[sl]
    residual_patch = components["residual"][sl]
    coherence_patch = components["coherence"][sl]
    energy_patch = components["energy"][sl]
    luminance_patch = components["luminance_detail"][sl]
    texture_patch = texture_gate[sl]
    dark_patch = dark_gate[sl]
    confidence_patch = rgb_confidence[sl]
    accepted = luminance_patch * confidence_patch

    residual_limit = max(float(np.percentile(np.abs(residual_patch), 99.0)), 1e-6)
    residual_rgb = np.clip(0.5 + residual_patch / (2.0 * residual_limit), 0.0, 1.0)
    detail_limit = max(float(np.percentile(np.abs(accepted), 99.0)), 1e-6)
    correlation_floor = float(config.dark_texture_correlation_floor)
    noise_floor = float(config.dark_texture_noise_floor)

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.8))
    axes[0, 0].imshow(rgb_patch)
    axes[0, 0].set_title("a  暗区 RGB 纹理", color=NAVY)
    axes[0, 1].imshow(residual_rgb)
    axes[0, 1].set_title("b  三通道 log-HF（零值=中灰）", color=NAVY)
    image = axes[0, 2].imshow(coherence_patch, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title(r"c  $C=\max |corr(R,G),corr(R,B),corr(G,B)|$", color=NAVY)
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046, pad=0.03)
    image = axes[1, 0].imshow(confidence_patch, cmap="cividis", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("d  曝光 + 暗信号 + 跨通道相干可靠度 q", color=NAVY)
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.03)
    image = axes[1, 1].imshow(
        accepted,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-detail_limit, vcenter=0.0, vmax=detail_limit),
    )
    axes[1, 1].set_title(r"e  通过门控的细节 $q\,d_{RGB}$", color=NAVY)
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.03)

    ax = axes[1, 2]
    step = max(1, int(np.sqrt(coherence_patch.size / 4500)))
    scatter = ax.scatter(
        coherence_patch[::step, ::step].reshape(-1),
        (energy_patch[::step, ::step] / max(noise_floor, 1e-6)).reshape(-1),
        c=texture_patch[::step, ::step].reshape(-1),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=6,
        alpha=0.45,
        linewidths=0,
    )
    ax.axvline(correlation_floor, color=VERMILION, ls="--", lw=1.3)
    ax.axhline(1.0, color=ORANGE, ls="--", lw=1.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, min(8.0, float(np.percentile(energy_patch / max(noise_floor, 1e-6), 99.0))))
    ax.set_xlabel("跨通道相关 C")
    ax.set_ylabel(r"局部残差能量 $E/\tau$")
    ax.set_title("f  相干—能量联合门控", color=NAVY)
    ax.grid(color=GRID, lw=0.6)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, label="纹理可靠度")
    for axis in axes.flat[:5]:
        axis.axis("off")

    fig.suptitle(
        "暗区跨通道相干门控：保留可重复结构，抑制独立读出/量化噪声",
        fontsize=16,
        color=NAVY,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.025,
        (
            f"ZKH3 实际暗 ROI；C0={correlation_floor:.2f}，噪声能量阈值 τ={noise_floor:.3f}。"
            "使用三对通道中最大绝对相关，可保留只有两通道响应的等亮度彩色纹理；黑电平以下仍被信号可靠度抑制。"
        ),
        ha="center",
        fontsize=9.0,
        color=DARK_GRAY,
    )
    fig.text(
        0.5,
        0.006,
        "门控只说明 RGB 细节是否可信，并不证明该细节在 2201 nm 具有独立辐射真值。",
        ha="center",
        fontsize=9.2,
        color=VERMILION,
        weight="bold",
    )
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.95), h_pad=1.6, w_pad=1.0)
    return save_figure(fig, output_dir, "06_dark_cross_channel_coherence_gate")


def evidence_ladder(data: dict[str, Any], output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(13.8, 8.0))
    ax.axis("off")
    ax.text(
        0.035,
        0.95,
        "证据阶梯：从‘看起来更清晰’到‘可声称真实 SWIR 细节恢复’",
        transform=ax.transAxes,
        fontsize=18,
        weight="bold",
        color=NAVY,
    )
    levels = [
        (
            0.05,
            0.10,
            0.76,
            "L1  工程与单元证据｜已具备",
            "正弦/斜边/暗纹理/平坦区测试；单次裁剪；最终 HR forward degradation；可复现配置与脚本",
            PALE_GREEN,
            GREEN,
        ),
        (
            0.11,
            0.28,
            0.70,
            "L2  同源真实 ROI 诊断｜已具备但不能当真值",
            "3DSSZ + ZKH3；统一显示域；ρ/β/A/R⊥、edge F1、halo、forward RMSE/SAM；V5/V6/V7 对照",
            PALE_BLUE,
            BLUE,
        ),
        (
            0.17,
            0.46,
            0.64,
            "L3  独立合成 HR-HSI / Wald 真值｜投稿前必补",
            "已知 PSF/SRF/噪声；公开数据 + 岩心风格合成；PSNR/SSIM/SAM/ERGAS/Q2ⁿ；盲测参数冻结",
            PALE_ORANGE,
            ORANGE,
        ),
        (
            0.23,
            0.64,
            0.58,
            "L4  独立真实 HR-NIR/SWIR 真值｜‘无损/恢复’声明门槛",
            "重复近景扫描、显微/高分辨率 SWIR、标定斜边与点目标；独立于 RGB 引导的空间真值",
            PALE_RED,
            VERMILION,
        ),
        (
            0.29,
            0.82,
            0.52,
            "L5  跨钻孔地质效用｜高水平论文主证据",
            "≥3 钻孔；XRD/Raman/专家边界；矿物/岩性/裂隙下游任务；置信区间、效应量与失败案例",
            LIGHT,
            NAVY,
        ),
    ]
    for x, y, width, title, body, face, edge in levels:
        add_box(
            ax,
            (x, y),
            width,
            0.12,
            title,
            body,
            face=face,
            edge=edge,
            title_color=edge,
            body_size=8.6,
        )
    add_box(
        ax,
        (0.835, 0.245),
        0.15,
        0.18,
        "当前证据上限：L2",
        "可支持：两场景同源结构传递\n与低分辨率观测一致性改善",
        face=PALE_GREEN,
        edge=GREEN,
        title_color=GREEN,
        body_size=8.2,
    )
    add_box(
        ax,
        (0.835, 0.54),
        0.15,
        0.22,
        "L3–L5 仍缺",
        "不可支持：\n‘SWIR 细节无损’\n‘真实 HR-SWIR 已恢复’",
        face=PALE_RED,
        edge=VERMILION,
        title_color=VERMILION,
        body_size=8.2,
    )
    add_arrow(ax, (0.91, 0.44), (0.91, 0.54), color=GRAY, width=1.4)
    v7_3 = data["3dssz"]["methods"]["V7"]["metrics"]
    v7_z = data["zkh3"]["methods"]["V7"]["metrics"]
    ax.text(
        0.035,
        0.035,
        (
            f"当前锚点：V7 forward RMSE = {v7_3['forward_rmse']:.5f} / {v7_z['forward_rmse']:.5f}，"
            f"2201 nm edge F1 = {v7_3['edge_f1']:.3f} / {v7_z['edge_f1']:.3f}（3DSSZ / ZKH3）。"
            "这些仍属于 L2 同源诊断。"
        ),
        transform=ax.transAxes,
        fontsize=9.3,
        color=VERMILION,
        weight="bold",
    )
    return save_figure(fig, output_dir, "07_evidence_ladder_and_claim_boundary")


def experiment_matrix(output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(19.2, 10.5))
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "V7 高水平论文实验矩阵：每一项同时回答‘更清晰、是否真实、是否稳健、是否有地质价值’",
        transform=ax.transAxes,
        fontsize=18,
        weight="bold",
        color=NAVY,
        va="top",
    )
    columns = ["实验层", "数据与分层", "强基线", "预注册主指标", "关键消融", "通过/否决条件"]
    rows = [
        [
            "E1 合成真值",
            "公开 HR-HSI + 岩心风格\n已知 PSF/SRF/噪声；盲测",
            "Bicubic、GS/GSA\nMTF-GLP、MTF-GLP-HPM\nCNMF/HySure + V5/V6",
            "PSNR、SSIM、SAM\nERGAS、Q2ⁿ、MTF50\n暗区 PSNR/边缘 F1",
            "局部 β；MTF 匹配\nlog-HPM；加性暗区支路\n相干门控；单次裁剪",
            "显著优于最强基线\nSAM/吸收带不退化\nhalo 与平坦区噪声不过线",
        ],
        [
            "E2 真实 ROI",
            "3DSSZ + ZKH3\n统一裁剪/统一拉伸\n明/中/暗三级分层",
            "V5 matched、V6、V7\nGS/GSA、MTF-GLP-HPM\n所有方法同一配准几何",
            "ρ、β、A、R⊥\nedge F1、方向一致性\nhalo、clipping、耗时",
            "亮度 vs R−G/B−G\n全局 vs 局部回归\n共享 vs 按波段加性",
            "ρ↑且 β/A→1、R⊥↓\n两个场景方向一致\n失败 ROI 必须公开",
        ],
        [
            "E3 观测闭环",
            "最终原生 HR 立方体\n逐波段 PSF 降质到 LR",
            "未调制基线、V5/V6\n无回投影与代理 D(g) 版本",
            "forward RMSE/SAM/CC\n逐波段残差谱\n低频漂移与裁剪能量",
            "产品级回投影次数/权重\n单次裁剪 vs 双裁剪",
            "细节提升时 forward 不恶化\n不能用一致性替代 HR 真值",
        ],
        [
            "E4 暗区与噪声",
            "暗 20% + RGB 平坦区\n黑电平附近 + 等亮度彩纹",
            "无门控、曝光门控\n单通道 SNR、跨通道相干",
            "暗区 ρ/β/A/R⊥\n平坦区 HF RMS、噪声谱\n伪纹理率、局部 SNR",
            "C=max|corr| vs 平均相关\n能量阈值 τ；暗信号可靠度",
            "暗细节提高且平坦区不增噪\n黑电平以下强制否决",
        ],
        [
            "E5 边缘/halo",
            "标定斜边 + 点目标\n天然强边缘仅作补充",
            "锐化核、HPM、经典 PAN\nV5/V6/V7",
            "MTF50、10–90% 宽度\novershoot/undershoot\n1 px edge F1、相位偏差",
            "PSF/MTF σ；增益上限\n幅度恢复；回投影 clip",
            "无可见双边/振铃\n标定目标通过后才做 halo 声明",
        ],
        [
            "E6 光谱与地质",
            "独立点光谱/XRD/Raman\n≥3 钻孔、按孔留一验证",
            "原始 HSI、V5/V6/V7\n专家编录与强学习基线",
            "吸收中心/带深误差\n矿物 F1/mIoU、裂隙召回\n跨孔均值±95% CI",
            "按波段加性平滑 σλ\n材料系数秩；不确定度门控",
            "下游效用稳定提高\n无单钻孔调参依赖\n光谱伪影不过预注册界限",
        ],
    ]
    left, right, bottom, top = 0.02, 0.985, 0.055, 0.90
    table_width = right - left
    table_height = top - bottom
    column_widths = np.asarray([0.10, 0.17, 0.19, 0.18, 0.18, 0.18], dtype=np.float32)
    column_widths /= column_widths.sum()
    header_height = 0.065
    row_height = (table_height - header_height) / len(rows)
    x_positions = left + table_width * np.concatenate(([0.0], np.cumsum(column_widths[:-1])))
    widths = table_width * column_widths
    for column, (label, x, width) in enumerate(zip(columns, x_positions, widths, strict=True)):
        ax.add_patch(
            Rectangle(
                (x, top - header_height),
                width,
                header_height,
                transform=ax.transAxes,
                facecolor=NAVY,
                edgecolor=WHITE,
                linewidth=1.0,
            )
        )
        ax.text(
            x + width / 2,
            top - header_height / 2,
            label,
            transform=ax.transAxes,
            color=WHITE,
            ha="center",
            va="center",
            weight="bold",
            fontsize=10.2,
        )
    row_colors = [PALE_BLUE, LIGHT, PALE_GREEN, PALE_ORANGE, LIGHT, PALE_BLUE]
    row_edges = [BLUE, GRAY, GREEN, ORANGE, GRAY, BLUE]
    for row_index, row in enumerate(rows):
        y = top - header_height - (row_index + 1) * row_height
        for column, (value, x, width) in enumerate(zip(row, x_positions, widths, strict=True)):
            face = row_colors[row_index] if column == 0 else (WHITE if row_index % 2 == 0 else "#F8FAFC")
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    row_height,
                    transform=ax.transAxes,
                    facecolor=face,
                    edgecolor=WHITE,
                    linewidth=1.0,
                )
            )
            ax.text(
                x + width / 2,
                y + row_height / 2,
                value,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=row_edges[row_index] if column == 0 else DARK_GRAY,
                weight="bold" if column == 0 else "normal",
                fontsize=8.0 if column else 9.0,
                linespacing=1.35,
            )
    ax.text(
        0.02,
        0.018,
        "统计原则：场景/钻孔级划分，参数在验证集冻结；报告均值、95% CI、效应量和失败案例。真实 ROI 的 RGB 引导指标不得替代独立 HR-SWIR 真值。",
        transform=ax.transAxes,
        fontsize=9.4,
        color=VERMILION,
        weight="bold",
    )
    return save_figure(fig, output_dir, "08_paper_experiment_matrix")


def manifest_payload(data: dict[str, Any], generated_files: list[str]) -> dict[str, Any]:
    scenes: dict[str, Any] = {}
    for scene in ("3dssz", "zkh3"):
        scene_data = data[scene]
        scenes[scene] = {
            "source_runs": {
                method: scene_data["methods"][method]["run"]
                for method in ("V5", "V6", "V7")
            },
            "display_policy": "shared_scene_channelwise_linear_P2_P98_over_V5_V6_V7",
            "display_bounds_2201_1651_901nm": [
                [float(lo), float(hi)] for lo, hi in scene_data["display_bounds"]
            ],
            "dark_crop_xywh_native_rgb": [int(v) for v in scene_data["dark_crop"]],
            "metrics_2201nm_sigma2p4_reliable": {
                method: scene_data["methods"][method]["metrics"]
                for method in ("V5", "V6", "V7")
            },
            "reconstruction_diagnostics": {
                method: scene_data["methods"][method]["diagnostics"]
                for method in ("V5", "V6", "V7")
            },
        }
    return {
        "schema": "GeoCoreFusion-V7-figure-evidence-v1",
        "figure_files": generated_files,
        "scenes": scenes,
        "truth_scope": (
            "same-data RGB-guided structural diagnostics and low-resolution observation "
            "consistency; no independent HR-SWIR truth"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="GeoCoreFusion repository root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory (default: artifacts/v7_research/figures)",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo / "artifacts" / "v7_research" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    api = import_project(repo)
    data = collect_data(repo, api)
    generated: list[str] = []
    generated += root_cause_and_mechanism(data, output_dir)
    generated += visual_comparison("3dssz", data, output_dir, 2)
    generated += visual_comparison("zkh3", data, output_dir, 3)
    generated += detail_metric_small_multiples(data, output_dir)
    generated += forward_and_edge_metrics(data, output_dir)
    generated += dark_coherence_gate(repo, data, api, output_dir)
    generated += evidence_ladder(data, output_dir)
    generated += experiment_matrix(output_dir)
    manifest = manifest_payload(data, generated)
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"figures={len(generated) // 2}; files={len(generated) + 1}")


if __name__ == "__main__":
    main()

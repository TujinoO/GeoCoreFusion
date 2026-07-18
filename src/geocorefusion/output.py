"""Reconstruction writers, previews, metadata, and run manifests."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .dataset import normalize_image, normalize_rgb
from .envi import create_bip_writer
from .lowrank import SubspaceModel, reconstruct


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_band_metadata(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_to_envi(
    coeff: np.ndarray,
    subspace: SubspaceModel,
    wavelengths: np.ndarray,
    hdr_path: str | Path,
    *,
    tile_size: int,
    dtype: str,
    detail_gain: np.ndarray | None = None,
    additive_detail: np.ndarray | None = None,
    additive_spectral_scale: np.ndarray | None = None,
) -> tuple[Path, Path]:
    writer, hdr, dat = create_bip_writer(
        hdr_path,
        (coeff.shape[0], coeff.shape[1], subspace.basis.shape[1]),
        dtype=dtype,
        wavelengths=wavelengths,
        description="GeoCoreFusion continuous 691-2518 nm cube on RGB grid; RGB supplies co-registered observation-nullspace spatial detail but no additional spectral bands",
    )
    for y0 in range(0, coeff.shape[0], tile_size):
        y1 = min(coeff.shape[0], y0 + tile_size)
        for x0 in range(0, coeff.shape[1], tile_size):
            x1 = min(coeff.shape[1], x0 + tile_size)
            tile = reconstruct(coeff[y0:y1, x0:x1, :], subspace)
            if detail_gain is not None:
                tile = tile * np.asarray(detail_gain[y0:y1, x0:x1], dtype=np.float32)[:, :, None]
            if additive_detail is not None and additive_spectral_scale is not None:
                tile = tile + (
                    np.asarray(additive_detail[y0:y1, x0:x1], dtype=np.float32)[:, :, None]
                    * np.asarray(additive_spectral_scale, dtype=np.float32)[None, None, :]
                )
            if detail_gain is not None or additive_detail is not None:
                tile = np.clip(tile, subspace.clip_min[None, None, :], subspace.clip_max[None, None, :])
            writer[y0:y1, x0:x1, :] = tile.astype(dtype)
    writer.flush()
    del writer
    return hdr, dat


def write_coefficients_envi(coeff: np.ndarray, hdr_path: str | Path) -> tuple[Path, Path]:
    writer, hdr, dat = create_bip_writer(hdr_path, coeff.shape, dtype="float32", description="GeoCoreFusion high-resolution material coefficients")
    writer[:] = coeff
    writer.flush()
    del writer
    return hdr, dat


def _stretch(image: np.ndarray) -> np.ndarray:
    return (normalize_image(image) * 255.0).round().astype(np.uint8)


def _preview_coefficients(coeff: np.ndarray, max_size: int = 1024) -> np.ndarray:
    scale = min(1.0, max_size / float(max(coeff.shape[:2])))
    size = (max(1, int(round(coeff.shape[1] * scale))), max(1, int(round(coeff.shape[0] * scale))))
    out = np.empty((size[1], size[0], coeff.shape[2]), dtype=np.float32)
    for k in range(coeff.shape[2]):
        out[:, :, k] = cv2.resize(coeff[:, :, k], size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return out


def write_previews(
    preview_dir: Path,
    rgb: np.ndarray,
    coeff: np.ndarray,
    subspace: SubspaceModel,
    wavelengths: np.ndarray,
    uncertainty: np.ndarray,
    detail_gain: np.ndarray | None = None,
    additive_detail: np.ndarray | None = None,
    additive_spectral_scale: np.ndarray | None = None,
) -> dict[str, str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    rgb01 = normalize_rgb(rgb)
    scale = min(1.0, 1024 / float(max(rgb01.shape[:2])))
    size = (max(1, int(round(rgb01.shape[1] * scale))), max(1, int(round(rgb01.shape[0] * scale))))
    rgb_small = cv2.resize(rgb01, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    Image.fromarray((rgb_small * 255).round().astype(np.uint8)).save(preview_dir / "rgb_reference.png")

    coeff_small = _preview_coefficients(coeff)
    cube_base_small = reconstruct(coeff_small, subspace)
    cube_small = cube_base_small
    if detail_gain is not None:
        gain_small = cv2.resize(
            np.asarray(detail_gain, dtype=np.float32),
            (cube_small.shape[1], cube_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        cube_small = np.clip(
            cube_small * gain_small[:, :, None],
            subspace.clip_min[None, None, :],
            subspace.clip_max[None, None, :],
        )
    if additive_detail is not None and additive_spectral_scale is not None:
        additive_small = cv2.resize(
            np.asarray(additive_detail, dtype=np.float32),
            (cube_small.shape[1], cube_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        cube_small = np.clip(
            cube_small
            + additive_small[:, :, None]
            * np.asarray(additive_spectral_scale, dtype=np.float32)[None, None, :],
            subspace.clip_min[None, None, :],
            subspace.clip_max[None, None, :],
        )
    def band_from(cube: np.ndarray, wavelength: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(wavelengths - wavelength)))
        return _stretch(cube[:, :, idx])
    def band(wavelength: float) -> np.ndarray:
        return band_from(cube_small, wavelength)
    false_color_base = np.stack([
        band_from(cube_base_small, 2200),
        band_from(cube_base_small, 1650),
        band_from(cube_base_small, 900),
    ], axis=2)
    false_color = np.stack([band(2200), band(1650), band(900)], axis=2)
    Image.fromarray(false_color_base).save(preview_dir / "fused_false_color_base_2200_1650_900.png")
    Image.fromarray(false_color).save(preview_dir / "fused_false_color_2200_1650_900.png")
    Image.fromarray(band(2200)).save(preview_dir / "fused_2200nm.png")
    Image.fromarray(band(2350)).save(preview_dir / "fused_2350nm.png")
    Image.fromarray(_stretch(np.mean(cube_small, axis=2))).save(preview_dir / "fused_mean_reflectance.png")
    uncertainty_small = cv2.resize(uncertainty, (cube_small.shape[1], cube_small.shape[0]), interpolation=cv2.INTER_AREA)
    heat = cv2.applyColorMap((_stretch(uncertainty_small)), cv2.COLORMAP_TURBO)
    Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).save(preview_dir / "uncertainty.png")
    comparison = np.concatenate([
        (rgb_small * 255).round().astype(np.uint8),
        false_color_base,
        false_color,
    ], axis=1)
    labels = ("RGB reference", "HSI base", "RGB-detail HSI")
    panel_width = false_color.shape[1]
    for index, label in enumerate(labels):
        x0 = index * panel_width
        cv2.rectangle(comparison, (x0, 0), (x0 + 185, 28), (0, 0, 0), thickness=-1)
        cv2.putText(comparison, label, (x0 + 7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    Image.fromarray(comparison).save(preview_dir / "spatial_detail_before_after.png")
    outputs = {
        "rgb_reference": "previews/rgb_reference.png",
        "fused_false_color_base": "previews/fused_false_color_base_2200_1650_900.png",
        "fused_false_color": "previews/fused_false_color_2200_1650_900.png",
        "fused_2200nm": "previews/fused_2200nm.png",
        "fused_2350nm": "previews/fused_2350nm.png",
        "fused_mean_reflectance": "previews/fused_mean_reflectance.png",
        "uncertainty": "previews/uncertainty.png",
        "spatial_detail_before_after": "previews/spatial_detail_before_after.png",
    }
    if detail_gain is not None:
        gain_preview = cv2.resize(
            np.asarray(detail_gain, dtype=np.float32),
            (cube_small.shape[1], cube_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        Image.fromarray(_stretch(gain_preview)).save(preview_dir / "spatial_detail_gain.png")
        outputs["spatial_detail_gain"] = "previews/spatial_detail_gain.png"
    if additive_detail is not None:
        additive_preview = cv2.resize(
            np.asarray(additive_detail, dtype=np.float32),
            (cube_small.shape[1], cube_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        Image.fromarray(_stretch(additive_preview)).save(preview_dir / "spatial_additive_detail.png")
        outputs["spatial_additive_detail"] = "previews/spatial_additive_detail.png"
    return outputs


def build_manifest(
    config: PipelineConfig,
    roi: dict[str, int],
    wavelengths: np.ndarray,
    outputs: dict[str, str],
    previews: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "geocorefusion.run.v1",
        "software_version": "0.5.0",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "scientific_scope": {
            "continuous_hyperspectral_product_nm": [float(wavelengths[0]), float(wavelengths[-1])],
            "rgb_role": "co-registered broad-band reference for signed material-coefficient detail, a common multiplicative gain, and conservative band-adaptive additive detail",
            "rgb_detail_constraint": "RGB detail is transferred only through coefficient-specific regression and observation-near-nullspace maps; low-resolution HSI back-projection remains the spectral constraint.",
            "explicit_non_claim": "The software does not claim continuous hyperspectral recovery below the NIR start wavelength from RGB alone.",
        },
        "project": json_safe(config.project),
        "input_data_dir": str(config.data_dir),
        "rgb_roi": roi,
        "output_grid": {"height": roi["height"], "width": roi["width"], "bands": int(wavelengths.size), "axis_order": "y,x,band"},
        "outputs": outputs,
        "previews": previews,
    }

"""Reconstruction writers, previews, metadata, and run manifests."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import cv2
import numpy as np
from PIL import Image

from ._version import __version__
from .config import PipelineConfig
from .dataset import normalize_image, normalize_rgb
from .envi import create_bip_writer
from .lowrank import (
    SubspaceModel,
    checksum_float32_arrays,
    describe_float32_array,
    reconstruct,
    validate_float32_array,
)


PhysicalClipLimits = tuple[float | None, float | None] | None


_DUAL_PRODUCT_CONTRACT_VERSION = "geocorefusion.dual-product.v1"
_FUSED_CUBE_OUTPUT_KEYS = (
    "fused_cube_hdr",
    "fused_cube_dat",
)
_RECONSTRUCTION_FACTOR_OUTPUT_KEYS = (
    "material_coefficients_hdr",
    "material_coefficients_dat",
    "subspace_model_json",
    "spatial_detail_gain_hdr",
    "spatial_detail_gain_dat",
    "spatial_additive_detail_hdr",
    "spatial_additive_detail_dat",
    "additive_spectral_scale_json",
)
_QUALITY_ARTIFACT_OUTPUT_KEYS = (
    "quality_report_json",
    "spatial_uncertainty_hdr",
    "spatial_uncertainty_dat",
)
_OBSERVED_LOWRES_OUTPUT_KEYS = (
    "harmonized_lowres_hdr",
    "harmonized_lowres_dat",
)
_PREVIEW_QUANTITATIVE_METRIC_EXCLUSIONS = ("RMSE", "SAM")


def _selected_output_paths(
    outputs: Mapping[str, str], keys: tuple[str, ...]
) -> dict[str, str]:
    return {
        key: value
        for key in keys
        if isinstance((value := outputs.get(key)), str) and value
    }


def _build_dual_product_contract(
    outputs: Mapping[str, str], previews: Mapping[str, str]
) -> dict[str, Any]:
    """Describe scientific files separately from display-only RGB/PNG products.

    The run manifest remains ``geocorefusion.run.v1``.  This contract is an
    additive extension so existing readers can continue to consume the legacy
    ``outputs`` and ``previews`` mappings unchanged.
    """

    fused_cube = _selected_output_paths(outputs, _FUSED_CUBE_OUTPUT_KEYS)
    reconstruction_factors = _selected_output_paths(
        outputs, _RECONSTRUCTION_FACTOR_OUTPUT_KEYS
    )
    quality_artifacts = _selected_output_paths(
        outputs, _QUALITY_ARTIFACT_OUTPUT_KEYS
    )
    observed_lowres = _selected_output_paths(outputs, _OBSERVED_LOWRES_OUTPUT_KEYS)
    explicitly_grouped = {
        *_FUSED_CUBE_OUTPUT_KEYS,
        *_RECONSTRUCTION_FACTOR_OUTPUT_KEYS,
        *_QUALITY_ARTIFACT_OUTPUT_KEYS,
        *_OBSERVED_LOWRES_OUTPUT_KEYS,
    }
    provenance_and_metadata = {
        key: value
        for key, value in outputs.items()
        if key not in explicitly_grouped and isinstance(value, str) and value
    }
    preview_members = {
        key: {
            "path": value,
            "product_class": "visualization_only",
            "not_for_quantitative_spectroscopy": True,
            "excluded_from_quantitative_metrics": list(
                _PREVIEW_QUANTITATIVE_METRIC_EXCLUSIONS
            ),
        }
        for key, value in previews.items()
        if isinstance(value, str) and value
    }
    return {
        "contract_version": _DUAL_PRODUCT_CONTRACT_VERSION,
        "schema_compatibility": {
            "run_schema_version": "geocorefusion.run.v1",
            "extension_mode": "additive",
            "legacy_missing_contract_policy": "accepted_as_legacy_unchecked",
        },
        "scientific_product": {
            "product_class": "scientific",
            "bundle_role": (
                "Quantitative hyperspectral arrays plus the reconstruction, "
                "quality, observation, and provenance artifacts needed to audit them."
            ),
            "members": {
                "fused_cube": fused_cube,
                "reconstruction_factors": reconstruction_factors,
                "quality_artifacts": quality_artifacts,
                "observed_low_resolution_cube": observed_lowres,
                "provenance_and_metadata": provenance_and_metadata,
            },
            "quantitative_evaluation_policy": (
                "RMSE or SAM may use only numeric scientific arrays with an explicitly "
                "declared reference or forward-observation model; bundle membership alone "
                "does not establish independent high-resolution spectral truth."
            ),
        },
        "visualization_only": {
            "product_class": "visualization_only",
            "not_for_quantitative_spectroscopy": True,
            "excluded_from_quantitative_metrics": list(
                _PREVIEW_QUANTITATIVE_METRIC_EXCLUSIONS
            ),
            "policy": (
                "RGB references and rendered previews are display and registration-QA "
                "products. Their pixels must not enter quantitative spectroscopy, RMSE, "
                "or SAM calculations."
            ),
            "members": preview_members,
        },
    }


def _build_visual_full_detail_contract(
    outputs: Mapping[str, str], previews: Mapping[str, str]
) -> dict[str, Any]:
    """Describe a single V6.1 RGB-textured fusion product and its evidence."""

    return {
        "contract_version": "geocorefusion.visual-full-detail.v1",
        "product_mode": "visual_full_detail",
        "primary_product": {
            "product_class": "rgb_textured_hyperspectral_visualization",
            "bundle_role": (
                "NIR/SWIR low-frequency radiometry with deliberately transferred, "
                "denoised RGB spatial detail for close-range core-image interpretation."
            ),
            "members": {
                "outputs": {
                    key: value
                    for key, value in outputs.items()
                    if isinstance(value, str) and value
                },
                "previews": {
                    key: value
                    for key, value in previews.items()
                    if isinstance(value, str) and value
                },
            },
            "detail_policy": "transfer_all_denoised_registered_rgb_spatial_detail",
            "spectral_retention_policy": (
                "The final-product observation cycle preserves compatibility with "
                "the measured low-resolution NIR/SWIR cube. High-resolution per-pixel "
                "spectra contain deliberate RGB texture and are not independent HR-SWIR truth."
            ),
        },
    }


def _validated_physical_clip_limits(limits: PhysicalClipLimits) -> tuple[float | None, float | None]:
    """Validate final output limits without consulting data quantiles."""

    if limits is None:
        return None, None
    if len(limits) != 2:
        raise ValueError("physical_clip_limits must contain exactly (minimum, maximum)")
    lower = None if limits[0] is None else float(limits[0])
    upper = None if limits[1] is None else float(limits[1])
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("physical_clip_limits minimum cannot exceed maximum")
    return lower, upper


def _new_clip_accumulator(limits: PhysicalClipLimits) -> dict[str, Any]:
    lower, upper = _validated_physical_clip_limits(limits)
    return {
        "lower_bound": lower,
        "upper_bound": upper,
        "total_value_count": 0,
        "finite_value_count": 0,
        "nonfinite_value_count": 0,
        "below_lower_bound_count": 0,
        "above_upper_bound_count": 0,
        "minimum_before_clip": float("inf"),
        "maximum_before_clip": float("-inf"),
        "minimum_after_clip": float("inf"),
        "maximum_after_clip": float("-inf"),
    }


def _apply_physical_output_constraints(
    values: np.ndarray,
    limits: PhysicalClipLimits,
    accumulator: MutableMapping[str, Any] | None = None,
) -> np.ndarray:
    """Apply one final physical/configured clip to an already modulated product.

    This deliberately does not use ``SubspaceModel.clip_min`` or ``clip_max``:
    those fields are fitted data quantiles, not physical reflectance limits.
    ``values`` is modified in place when it already is a float32 array.
    """

    lower, upper = _validated_physical_clip_limits(limits)
    result = np.asarray(values, dtype=np.float32)
    if accumulator is not None:
        finite = np.isfinite(result)
        finite_values = result[finite]
        accumulator["total_value_count"] += int(result.size)
        accumulator["finite_value_count"] += int(finite_values.size)
        accumulator["nonfinite_value_count"] += int(result.size - finite_values.size)
        if finite_values.size:
            accumulator["minimum_before_clip"] = min(
                float(accumulator["minimum_before_clip"]),
                float(np.min(finite_values)),
            )
            accumulator["maximum_before_clip"] = max(
                float(accumulator["maximum_before_clip"]),
                float(np.max(finite_values)),
            )
            if lower is not None:
                accumulator["below_lower_bound_count"] += int(np.sum(finite_values < lower))
            if upper is not None:
                accumulator["above_upper_bound_count"] += int(np.sum(finite_values > upper))
    if lower is not None:
        np.maximum(result, lower, out=result)
    if upper is not None:
        np.minimum(result, upper, out=result)
    if accumulator is not None:
        finite_after = result[np.isfinite(result)]
        if finite_after.size:
            accumulator["minimum_after_clip"] = min(
                float(accumulator["minimum_after_clip"]),
                float(np.min(finite_after)),
            )
            accumulator["maximum_after_clip"] = max(
                float(accumulator["maximum_after_clip"]),
                float(np.max(finite_after)),
            )
    return result


def _finalize_clip_statistics(
    accumulator: MutableMapping[str, Any],
    destination: MutableMapping[str, Any],
) -> None:
    finite_count = int(accumulator["finite_value_count"])
    lower_count = int(accumulator["below_lower_bound_count"])
    upper_count = int(accumulator["above_upper_bound_count"])
    clipped_count = lower_count + upper_count
    destination.clear()
    destination.update(
        {
            "constraint_kind": "single_final_physical_or_configured_clip",
            "quantile_limits_used": False,
            "lower_bound": accumulator["lower_bound"],
            "upper_bound": accumulator["upper_bound"],
            "total_value_count": int(accumulator["total_value_count"]),
            "finite_value_count": finite_count,
            "nonfinite_value_count": int(accumulator["nonfinite_value_count"]),
            "below_lower_bound_count": lower_count,
            "above_upper_bound_count": upper_count,
            "clipped_value_count": clipped_count,
            "clipped_fraction_of_finite": float(clipped_count / max(finite_count, 1)),
            "minimum_before_clip": (
                None if not np.isfinite(accumulator["minimum_before_clip"])
                else float(accumulator["minimum_before_clip"])
            ),
            "maximum_before_clip": (
                None if not np.isfinite(accumulator["maximum_before_clip"])
                else float(accumulator["maximum_before_clip"])
            ),
            "minimum_after_clip": (
                None if not np.isfinite(accumulator["minimum_after_clip"])
                else float(accumulator["minimum_after_clip"])
            ),
            "maximum_after_clip": (
                None if not np.isfinite(accumulator["maximum_after_clip"])
                else float(accumulator["maximum_after_clip"])
            ),
        }
    )


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


def additive_spectral_scale_payload(
    wavelengths_nm: np.ndarray,
    additive_spectral_scale: np.ndarray,
    *,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a transparent, checksum-verifiable per-band additive scale table."""

    wavelengths = np.asarray(wavelengths_nm, dtype=np.float32)
    scales = np.asarray(additive_spectral_scale, dtype=np.float32)
    if wavelengths.ndim != 1 or scales.ndim != 1:
        raise ValueError("wavelengths_nm and additive_spectral_scale must be one-dimensional")
    if wavelengths.shape != scales.shape:
        raise ValueError(
            "wavelengths_nm and additive_spectral_scale must contain the same number of bands"
        )
    if not np.isfinite(wavelengths).all() or not np.isfinite(scales).all():
        raise ValueError("additive spectral scale metadata cannot contain non-finite values")
    arrays = (
        ("wavelengths_nm", wavelengths),
        ("additive_spectral_scale", scales),
    )
    return {
        "schema_version": "geocorefusion.additive-spectral-scale.v1",
        "band_count": int(scales.size),
        "axis": "spectral_band",
        "association": (
            "additive_spectral_scale[i] applies to wavelengths_nm[i] and to "
            "spectral_band i of the reconstructed cube"
        ),
        "wavelengths_nm": wavelengths.tolist(),
        "additive_spectral_scale": scales.tolist(),
        "array_metadata": {
            name: describe_float32_array(values, ("spectral_band",))
            for name, values in arrays
        },
        "pair_checksum": {
            "algorithm": "sha256",
            "canonical_representation": "ordered_named_little_endian_float32_arrays_v1",
            "array_order": [name for name, _ in arrays],
            "value": checksum_float32_arrays(arrays),
        },
        "source_metadata": json_safe(source_metadata or {}),
    }


def write_additive_spectral_scale(
    path: str | Path,
    wavelengths_nm: np.ndarray,
    additive_spectral_scale: np.ndarray,
    *,
    source_metadata: Mapping[str, Any] | None = None,
) -> None:
    write_json(
        path,
        additive_spectral_scale_payload(
            wavelengths_nm,
            additive_spectral_scale,
            source_metadata=source_metadata,
        ),
    )


def load_additive_spectral_scale(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load wavelength/scale vectors after shape, axis, and checksum validation."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Additive spectral scale JSON root must be an object")
    if payload.get("schema_version") != "geocorefusion.additive-spectral-scale.v1":
        raise ValueError("Unsupported additive spectral scale schema")
    metadata = payload.get("array_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Additive spectral scale payload is missing array_metadata")
    arrays: dict[str, np.ndarray] = {}
    for name in ("wavelengths_nm", "additive_spectral_scale"):
        if name not in payload or name not in metadata:
            raise ValueError(f"Additive spectral scale payload is missing {name}")
        arrays[name] = validate_float32_array(
            payload[name],
            metadata[name],
            name=name,
            expected_axes=("spectral_band",),
        )
    if arrays["wavelengths_nm"].shape != arrays["additive_spectral_scale"].shape:
        raise ValueError("Persisted wavelength and additive scale shapes differ")
    if int(payload.get("band_count", -1)) != arrays["wavelengths_nm"].size:
        raise ValueError("Persisted additive scale band_count is inconsistent")
    pair_checksum = payload.get("pair_checksum")
    expected_order = ["wavelengths_nm", "additive_spectral_scale"]
    if (
        not isinstance(pair_checksum, Mapping)
        or pair_checksum.get("algorithm") != "sha256"
        or pair_checksum.get("canonical_representation")
        != "ordered_named_little_endian_float32_arrays_v1"
        or pair_checksum.get("array_order") != expected_order
    ):
        raise ValueError("Additive spectral scale pair checksum metadata is invalid")
    actual_checksum = checksum_float32_arrays(
        (name, arrays[name]) for name in expected_order
    )
    if actual_checksum != pair_checksum.get("value"):
        raise ValueError("Additive spectral scale pair checksum mismatch")
    return arrays["wavelengths_nm"], arrays["additive_spectral_scale"]


def _modulate_reconstructed(
    cube: np.ndarray,
    *,
    detail_gain: np.ndarray | None = None,
    additive_detail: np.ndarray | None = None,
    additive_spectral_scale: np.ndarray | None = None,
    physical_clip_limits: PhysicalClipLimits = (0.0, None),
    clip_accumulator: MutableMapping[str, Any] | None = None,
) -> np.ndarray:
    """Apply all spatial modulation to an unconstrained cube and clip once."""

    result = np.asarray(cube, dtype=np.float32)
    if detail_gain is not None:
        result *= np.asarray(detail_gain, dtype=np.float32)[:, :, None]
    if additive_detail is not None and additive_spectral_scale is not None:
        result += (
            np.asarray(additive_detail, dtype=np.float32)[:, :, None]
            * np.asarray(additive_spectral_scale, dtype=np.float32)[None, None, :]
        )
    return _apply_physical_output_constraints(result, physical_clip_limits, clip_accumulator)


def reconstruct_modulated(
    coeff: np.ndarray,
    subspace: SubspaceModel,
    *,
    detail_gain: np.ndarray | None = None,
    additive_detail: np.ndarray | None = None,
    additive_spectral_scale: np.ndarray | None = None,
    physical_clip_limits: PhysicalClipLimits = (0.0, None),
    clip_accumulator: MutableMapping[str, Any] | None = None,
) -> np.ndarray:
    """Reconstruct and modulate on the supplied native grid, then clip once.

    The subspace is reconstructed without its fitted quantile bounds. Shared
    multiplicative and additive detail are applied to that unconstrained
    reconstruction before a single optional physical/configured clip.
    """

    cube = reconstruct(np.asarray(coeff, dtype=np.float32), subspace, clip=False)
    return _modulate_reconstructed(
        cube,
        detail_gain=detail_gain,
        additive_detail=additive_detail,
        additive_spectral_scale=additive_spectral_scale,
        physical_clip_limits=physical_clip_limits,
        clip_accumulator=clip_accumulator,
    )


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
    physical_clip_limits: PhysicalClipLimits = (0.0, None),
    clip_statistics: MutableMapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    clip_accumulator = _new_clip_accumulator(physical_clip_limits) if clip_statistics is not None else None
    writer, hdr, dat = create_bip_writer(
        hdr_path,
        (coeff.shape[0], coeff.shape[1], subspace.basis.shape[1]),
        dtype=dtype,
        wavelengths=wavelengths,
        description="GeoCoreFusion continuous 691-2518 nm cube on RGB grid; unconstrained subspace reconstruction is spatially modulated before one final physical/configured clip",
    )
    for y0 in range(0, coeff.shape[0], tile_size):
        y1 = min(coeff.shape[0], y0 + tile_size)
        for x0 in range(0, coeff.shape[1], tile_size):
            x1 = min(coeff.shape[1], x0 + tile_size)
            tile = reconstruct_modulated(
                coeff[y0:y1, x0:x1, :],
                subspace,
                detail_gain=(None if detail_gain is None else detail_gain[y0:y1, x0:x1]),
                additive_detail=(None if additive_detail is None else additive_detail[y0:y1, x0:x1]),
                additive_spectral_scale=additive_spectral_scale,
                physical_clip_limits=physical_clip_limits,
                clip_accumulator=clip_accumulator,
            )
            writer[y0:y1, x0:x1, :] = tile.astype(dtype)
    writer.flush()
    del writer
    if clip_statistics is not None and clip_accumulator is not None:
        _finalize_clip_statistics(clip_accumulator, clip_statistics)
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


def _native_preview_products(
    coeff: np.ndarray,
    subspace: SubspaceModel,
    selected_indices: np.ndarray,
    *,
    detail_gain: np.ndarray | None,
    additive_detail: np.ndarray | None,
    additive_spectral_scale: np.ndarray | None,
    physical_clip_limits: PhysicalClipLimits,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build selected bands and the fused mean on the native coefficient grid."""

    indices = np.asarray(selected_indices, dtype=np.int64)
    height, width = coeff.shape[:2]
    base_selected = np.empty((height, width, indices.size), dtype=np.float32)
    fused_selected = np.empty_like(base_selected)
    fused_mean = np.empty((height, width), dtype=np.float32)
    step = max(1, int(tile_size))
    for y0 in range(0, height, step):
        y1 = min(height, y0 + step)
        for x0 in range(0, width, step):
            x1 = min(width, x0 + step)
            raw_tile = reconstruct(coeff[y0:y1, x0:x1, :], subspace, clip=False)
            fused_tile = _modulate_reconstructed(
                raw_tile.copy(),
                detail_gain=(None if detail_gain is None else detail_gain[y0:y1, x0:x1]),
                additive_detail=(None if additive_detail is None else additive_detail[y0:y1, x0:x1]),
                additive_spectral_scale=additive_spectral_scale,
                physical_clip_limits=physical_clip_limits,
            )
            base_tile = _apply_physical_output_constraints(raw_tile, physical_clip_limits)
            base_selected[y0:y1, x0:x1, :] = base_tile[:, :, indices]
            fused_selected[y0:y1, x0:x1, :] = fused_tile[:, :, indices]
            fused_mean[y0:y1, x0:x1] = np.mean(fused_tile, axis=2)
    return base_selected, fused_selected, fused_mean


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
    *,
    tile_size: int = 256,
    max_size: int = 1024,
    physical_clip_limits: PhysicalClipLimits = (0.0, None),
) -> dict[str, str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    rgb01 = normalize_rgb(rgb)
    scale = min(1.0, max(1, int(max_size)) / float(max(rgb01.shape[:2])))
    size = (max(1, int(round(rgb01.shape[1] * scale))), max(1, int(round(rgb01.shape[0] * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    rgb_small = cv2.resize(rgb01, size, interpolation=interpolation)
    Image.fromarray((rgb_small * 255).round().astype(np.uint8)).save(preview_dir / "rgb_reference.png")

    preview_wavelengths = np.asarray((2200.0, 1650.0, 900.0, 2350.0), dtype=np.float32)
    selected_indices = np.asarray(
        [int(np.argmin(np.abs(np.asarray(wavelengths, dtype=np.float32) - value))) for value in preview_wavelengths],
        dtype=np.int64,
    )
    base_native, fused_native, fused_mean_native = _native_preview_products(
        coeff,
        subspace,
        selected_indices,
        detail_gain=detail_gain,
        additive_detail=additive_detail,
        additive_spectral_scale=additive_spectral_scale,
        physical_clip_limits=physical_clip_limits,
        tile_size=tile_size,
    )
    base_small = cv2.resize(base_native, size, interpolation=interpolation)
    fused_small = cv2.resize(fused_native, size, interpolation=interpolation)
    fused_mean_small = cv2.resize(fused_mean_native, size, interpolation=interpolation)

    false_color_base = np.stack([
        _stretch(base_small[:, :, 0]),
        _stretch(base_small[:, :, 1]),
        _stretch(base_small[:, :, 2]),
    ], axis=2)
    false_color = np.stack([
        _stretch(fused_small[:, :, 0]),
        _stretch(fused_small[:, :, 1]),
        _stretch(fused_small[:, :, 2]),
    ], axis=2)
    band_2200 = _stretch(fused_small[:, :, 0])
    band_2350 = _stretch(fused_small[:, :, 3])
    Image.fromarray(false_color_base).save(preview_dir / "fused_false_color_base_2200_1650_900.png")
    Image.fromarray(false_color).save(preview_dir / "fused_false_color_2200_1650_900.png")
    Image.fromarray(band_2200).save(preview_dir / "fused_2200nm.png")
    Image.fromarray(band_2350).save(preview_dir / "fused_2350nm.png")
    Image.fromarray(_stretch(fused_mean_small)).save(preview_dir / "fused_mean_reflectance.png")
    uncertainty_small = cv2.resize(uncertainty, size, interpolation=interpolation)
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
            size,
            interpolation=interpolation,
        )
        Image.fromarray(_stretch(gain_preview)).save(preview_dir / "spatial_detail_gain.png")
        outputs["spatial_detail_gain"] = "previews/spatial_detail_gain.png"
    if additive_detail is not None:
        additive_preview = cv2.resize(
            np.asarray(additive_detail, dtype=np.float32),
            size,
            interpolation=interpolation,
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
    factor_paths = {
        "material_coefficients_hdr": outputs.get("material_coefficients_hdr"),
        "material_coefficients_dat": outputs.get("material_coefficients_dat"),
        "subspace_model_json": outputs.get("subspace_model_json"),
        "spatial_detail_gain_hdr": outputs.get("spatial_detail_gain_hdr"),
        "spatial_detail_gain_dat": outputs.get("spatial_detail_gain_dat"),
        "spatial_additive_detail_hdr": outputs.get("spatial_additive_detail_hdr"),
        "spatial_additive_detail_dat": outputs.get("spatial_additive_detail_dat"),
        "additive_spectral_scale_json": outputs.get("additive_spectral_scale_json"),
    }
    required_factor_keys = (
        "material_coefficients_hdr",
        "material_coefficients_dat",
        "subspace_model_json",
        "spatial_detail_gain_hdr",
        "spatial_detail_gain_dat",
        "spatial_additive_detail_hdr",
        "spatial_additive_detail_dat",
        "additive_spectral_scale_json",
    )
    product_mode = str(config.output.product_mode).strip().lower()
    visual_full_detail = product_mode in {
        "visual_full_detail",
        "v61_visual_full_detail",
        "single_visual_full_detail",
    }
    if visual_full_detail:
        scope = {
            "continuous_hyperspectral_product_nm": [
                float(wavelengths[0]),
                float(wavelengths[-1]),
            ],
            "primary_goal": "maximum_registered_rgb_spatial_detail_transfer",
            "rgb_role": (
                "All denoised, co-registered RGB spatial detail is intentionally "
                "transferred into NIR/SWIR visualization bands."
            ),
            "spectral_retention": (
                "Low-resolution measured NIR/SWIR response is retained by the final "
                "observation cycle; high-resolution pixel spectra are RGB-textured."
            ),
            "registration_claim_boundary": (
                "Same-data structure scores do not prove subpixel accuracy; an independent "
                "landmark TRE audit remains required for a formal production claim."
            ),
            "explicit_non_claim": (
                "The RGB-textured high-resolution pixels are not independent HR-NIR/SWIR "
                "radiometric truth."
            ),
        }
        product_contract = _build_visual_full_detail_contract(outputs, previews)
    else:
        scope = {
            "continuous_hyperspectral_product_nm": [
                float(wavelengths[0]),
                float(wavelengths[-1]),
            ],
            "rgb_role": "Co-registered broad-band structural guide for configured coefficient-specific detail and shared spatial modulation; RGB is not NIR/SWIR high-resolution radiometric truth.",
            "rgb_detail_constraint": "Transferred RGB detail is constrained by observation-near-nullspace projection and low-resolution HSI back-projection; a common gain preserves the reconstructed per-pixel spectral shape before clipping.",
            "registration_claim_boundary": "Same-data structure correlations do not prove subpixel accuracy. Subpixel claims require known-truth synthetic TRE/EPE or independent held-out real landmarks.",
            "fusion_claim_boundary": "RGB/fused-band detail correlation measures geometric detail transfer, not independent NIR/SWIR high-frequency truth.",
            "explicit_non_claim": "The software does not claim RGB-only hyperspectral recovery below the NIR start wavelength or independently verified high-resolution NIR/SWIR radiometry.",
        }
        product_contract = _build_dual_product_contract(outputs, previews)
    return {
        "schema_version": "geocorefusion.run.v1",
        "software_version": __version__,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "scientific_scope": scope,
        "project": json_safe(config.project),
        "input_data_dir": str(config.data_dir),
        "rgb_roi": roi,
        "output_grid": {"height": roi["height"], "width": roi["width"], "bands": int(wavelengths.size), "axis_order": "y,x,band"},
        "reconstruction_factors": {
            "formula": (
                "clip((coefficients @ basis + mean_spectrum) * spatial_detail_gain "
                "+ spatial_additive_detail * additive_spectral_scale)"
            ),
            "final_clip_policy": {
                "constraint_kind": "single_final_physical_or_configured_clip",
                "lower_bound": 0.0,
                "upper_bound": None,
                "subspace_quantile_clip_min_max_used": False,
            },
            "axis_contract": {
                "material_coefficients": ["y", "x", "component"],
                "subspace_basis": ["component", "spectral_band"],
                "spatial_detail_maps": ["y", "x"],
                "additive_spectral_scale": ["spectral_band"],
                "output_cube": ["y", "x", "spectral_band"],
            },
            "complete_factor_set_persisted": all(factor_paths[key] for key in required_factor_keys),
            "paths": factor_paths,
            "source_metadata": {
                "input": outputs.get("input_metadata_json"),
                "processing_config": outputs.get("processing_config_json"),
                "spectral_harmonization": outputs.get("spectral_harmonization_json"),
                "band_metadata": outputs.get("band_metadata_csv"),
                "fusion_model": outputs.get("fusion_model_json"),
            },
        },
        "outputs": outputs,
        "previews": previews,
        "product_contract": product_contract,
    }

"""Run seeded V6 component checks and summarize matched real-ROI results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from geocorefusion.config import FusionConfig, RegistrationConfig
from geocorefusion.degradation import PsfModel, degrade_coefficients, degrade_spatial_map
from geocorefusion.fusion import refine_coefficients
from geocorefusion.registration import (
    _corr,
    _estimate_one,
    _estimate_roi_affine,
    _estimate_tiepoint_field,
    _modality_feature,
    _warp_aligned_residual,
)


def texture(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape)
    image = 0.3 * np.sin(x / 5.0) + 0.2 * np.cos(y / 11.0)
    image += ((x - 42) ** 2 + (y - 118) ** 2 < 18**2) * 1.2
    image += ((x > 75) & (x < 88) & (y > 20) & (y < 205)) * 0.8
    rng = np.random.default_rng(20260720)
    image += cv2.GaussianBlur(rng.normal(0.0, 0.22, shape).astype(np.float32), (0, 0), 1.0)
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), 0.7)


def affine_errors(estimated: np.ndarray, truth: np.ndarray, shape: tuple[int, int], margin: int) -> np.ndarray:
    ys, xs = np.meshgrid(
        np.linspace(margin, shape[0] - margin - 1, 11),
        np.linspace(margin, shape[1] - margin - 1, 9),
        indexing="ij",
    )
    points = np.stack([xs.reshape(-1), ys.reshape(-1), np.ones(xs.size)], axis=0)
    return np.linalg.norm(estimated[:2] @ points - truth[:2] @ points, axis=0)


def summarize_error(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_px": float(np.mean(arr)),
        "median_px": float(np.median(arr)),
        "p95_px": float(np.percentile(arr, 95.0)),
        "max_px": float(np.max(arr)),
    }


def registration_benchmark() -> dict:
    shape = (240, 128)
    reference = texture(shape)
    truth = np.asarray([[1.0, 0.006, 4.0], [-0.004, 1.0, -3.0]], dtype=np.float32)
    moving = cv2.warpAffine(
        reference,
        truth,
        (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    coarse_cfg = RegistrationConfig(
        preview_width=shape[1],
        preview_max_height=shape[0],
        ecc_iterations=500,
        enable_strip_drift=False,
    )
    coarse, _ = _estimate_one(
        reference,
        moving,
        rgb_shape=shape,
        sensor_shape=shape,
        sensor_name="SYNTHETIC",
        config=coarse_cfg,
    )

    roi_shape = (260, 170)
    roi_reference = texture(roi_shape)
    roi_truth = np.asarray([[1.012, 0.009, 5.2], [-0.006, 0.987, -4.1]], dtype=np.float32)
    roi_moving = cv2.warpAffine(
        roi_reference,
        roi_truth,
        (roi_shape[1], roi_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    valid = np.isfinite(roi_moving)
    roi_moving[valid] = np.sqrt(
        np.clip(
            (roi_moving[valid] - np.nanmin(roi_moving))
            / (np.nanmax(roi_moving) - np.nanmin(roi_moving)),
            0,
            1,
        )
    )
    roi_matrix, _, _ = _estimate_roi_affine(
        roi_reference,
        roi_moving,
        RegistrationConfig(roi_ecc_iterations=900, enable_roi_row_refinement=False),
    )

    dense_shape = (280, 176)
    dense_reference = texture(dense_shape)
    yy, xx = np.indices(dense_shape, dtype=np.float32)
    displacement_y = lambda y, x: 1.2 * np.sin(y / 48.0) + 0.45 * np.sin(x / 21.0)
    displacement_x = lambda y, x: 1.8 * np.sin(x / 38.0) - 0.55 * np.cos(y / 33.0)
    true_dy = displacement_y(yy, xx)
    true_dx = displacement_x(yy, xx)
    inverse_y, inverse_x = yy.copy(), xx.copy()
    for _ in range(12):
        next_y = yy - displacement_y(inverse_y, inverse_x)
        next_x = xx - displacement_x(inverse_y, inverse_x)
        inverse_y, inverse_x = next_y, next_x
    dense_moving = cv2.remap(
        dense_reference,
        inverse_x,
        inverse_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    valid = np.isfinite(dense_moving)
    dense_moving[valid] = np.sqrt(
        np.clip(
            (dense_moving[valid] - np.nanmin(dense_moving))
            / (np.nanmax(dense_moving) - np.nanmin(dense_moving)),
            0,
            1,
        )
    )
    dense_cfg = RegistrationConfig(
        roi_tiepoint_grid_rows=13,
        roi_tiepoint_grid_cols=8,
        roi_tiepoint_template_radius=7,
        roi_tiepoint_search_radius=6,
        roi_tiepoint_min_score=0.20,
        roi_tiepoint_min_margin=0.004,
        roi_tiepoint_max_backward_error=2.0,
        roi_tiepoint_min_points=10,
    )
    shift_y, shift_x, details = _estimate_tiepoint_field(dense_reference, dense_moving, dense_cfg)
    aligned = _warp_aligned_residual(dense_moving, shift_y, shift_x, 1.0)
    interior = np.s_[20:-20, 20:-20]
    epe = np.hypot(shift_y[interior] - true_dy[interior], shift_x[interior] - true_dx[interior])
    return {
        "units": "analysis-grid pixels",
        "truth_scope": "seeded controlled synthetic overlap only; not a production-configuration replay",
        "coarse_affine_tre": summarize_error(affine_errors(coarse.rgb_to_sensor_matrix, truth, shape, 16)),
        "roi_affine_tre": summarize_error(affine_errors(roi_matrix, roi_truth, roi_shape, 20)),
        "dense_residual_epe": summarize_error(epe),
        "dense_tiepoint_count": int(details["tie_point_count"]),
        "dense_structure_correlation_before": _corr(
            _modality_feature(dense_reference), _modality_feature(dense_moving)
        ),
        "dense_structure_correlation_after": _corr(
            _modality_feature(dense_reference), _modality_feature(aligned)
        ),
    }


def fusion_benchmark() -> dict:
    high_shape = (160, 128)
    low_shape = (30, 20)
    y, x = np.indices(high_shape, dtype=np.float32)
    texture_field = 0.55 * np.sin(x / 2.8) + 0.45 * np.cos(y / 3.7)
    dark = x < 32
    illumination = np.where(dark, 0.012, 0.48).astype(np.float32)
    rgb_detail = np.where(dark, 0.0035, 0.045) * texture_field
    rgb = np.stack(
        [
            illumination + rgb_detail,
            0.92 * illumination + 0.78 * rgb_detail,
            0.84 * illumination - 0.55 * rgb_detail,
        ],
        axis=2,
    ).astype(np.float32)
    rgb = np.clip(rgb, 0.002, 1.0)
    coarse = 0.35 + 0.18 * (x > 72) + 0.10 * np.sin(y / 28.0)
    truth = np.stack(
        [
            coarse + 0.055 * texture_field,
            0.55 * coarse - 0.025 * texture_field,
            0.20 * np.cos(x / 32.0) + 0.018 * texture_field,
        ],
        axis=2,
    ).astype(np.float32)
    psf = PsfModel(2.2, 3.0, 0.8, low_shape, high_shape)
    low = degrade_coefficients(truth, psf)
    common = dict(
        rank=3,
        refiner="bicubic",
        coefficient_detail_strength=0.28,
        coefficient_detail_min_correlation=0.01,
        coefficient_detail_clip_sigma=0.55,
        coefficient_detail_support_floor=0.45,
        spatial_detail_strength=0.28,
        spatial_detail_additive_strength=0.0,
        spatial_detail_nullspace_iterations=3,
    )
    legacy = refine_coefficients(low, rgb, psf, FusionConfig(**common))
    proposed = refine_coefficients(
        low,
        rgb,
        psf,
        FusionConfig(
            **common,
            intrinsic_detail_enabled=True,
            intrinsic_log_epsilon=1.0 / 255.0,
            dark_detail_boost=1.0,
            dark_detail_percentile=30.0,
            dark_texture_noise_floor=0.008,
            spatial_detail_log_gain=True,
            spatial_detail_confidence_mode="none",
            spatial_detail_back_projection_iterations=4,
        ),
    )

    def dark_correlation(result) -> float:
        predicted = np.maximum(result.coefficients[:, :, 0] * result.detail_gain_map, 1e-4)
        target = np.maximum(truth[:, :, 0], 1e-4)
        predicted_detail = np.log(predicted) - cv2.GaussianBlur(np.log(predicted), (0, 0), 2.4)
        target_detail = np.log(target) - cv2.GaussianBlur(np.log(target), (0, 0), 2.4)
        mask = dark & (y > 8) & (y < high_shape[0] - 9) & (x > 8)
        return float(np.corrcoef(predicted_detail[mask], target_detail[mask])[0, 1])

    def consistency(result) -> dict[str, float]:
        return {
            "dark_log_detail_correlation": dark_correlation(result),
            "coefficient_observation_rmse": float(
                np.sqrt(np.mean((degrade_coefficients(result.coefficients, psf) - low) ** 2))
            ),
            "gain_observation_rmse": float(
                np.sqrt(np.mean((degrade_spatial_map(result.detail_gain_map, psf) - 1.0) ** 2))
            ),
        }

    return {
        "truth_scope": "seeded controlled synthetic HR coefficient field; not a production-configuration replay",
        "v5_style": consistency(legacy),
        "v6_intrinsic": consistency(proposed),
    }


def nested_get(payload: dict, *keys):
    value = payload
    for key in keys:
        value = value[key]
    return value


def real_results(repo: Path) -> dict:
    results = {}
    for scene in ("3dssz", "zkh3"):
        scene_results = {}
        for version in ("v5_matched", "v6"):
            path = repo / "runs" / f"{scene}_roi_fusion_{version}" / "metrics" / "quality_report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            dark = nested_get(
                report,
                "spatial",
                "band_detail_by_brightness",
                "log_high_frequency_correlation",
            )
            scene_results[version] = {
                "continuous_cube_rmse": nested_get(report, "continuous_cube_observation", "rmse"),
                "sam_mean_deg": nested_get(report, "continuous_cube_observation", "sam_mean_deg"),
                "band_cc_mean": nested_get(report, "continuous_cube_observation", "band_cc_mean"),
                "coefficient_rmse": nested_get(report, "coefficient_observation", "rmse"),
                "gain_lowres_rmse": nested_get(report, "spatial", "detail_gain_lowres_rmse_from_one"),
                "dark_log_detail_correlation": {
                    band: values["darkest_20pct"] for band, values in dark.items()
                },
                "registration": {
                    "nir_rgb": nested_get(report, "registration", "roi_refinement", "nir", "score_after"),
                    "swir_rgb": nested_get(report, "registration", "roi_refinement", "swir", "score_after"),
                    "nir_swir": nested_get(
                        report, "registration", "roi_refinement", "nir_swir_overlap", "score_after"
                    ),
                },
            }
        results[scene] = scene_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # ECC and related reductions can vary slightly with OpenCV threading and
    # optimized kernels.  Pin them for repeatability within a given build; the
    # JSON still does not promise bitwise identity across OpenCV versions.
    cv2.setNumThreads(1)
    cv2.setUseOptimized(False)
    cv2.setRNGSeed(20260720)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registration_synthetic": registration_benchmark(),
        "fusion_synthetic": fusion_benchmark(),
        "real_roi_matched": real_results(args.repo),
        "evidence_limit": (
            "Real ROI metrics use corrected geometry and observation consistency; they are not independent HR-HSI truth."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""End-to-end RGB-NIR-SWIR fusion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import PipelineConfig
from .dataset import DatasetTriplet, discover_triplet, normalize_image, normalize_rgb
from .degradation import degrade_coefficients, estimate_psf
from .envi import create_bip_writer, metadata_dict
from .fusion import (
    back_project_modulated_product,
    build_additive_spectral_scale,
    build_band_adaptive_mtf_detail,
    refine_coefficients,
)
from .lowrank import fit_hybrid_simplex_subspace, fit_simplex_subspace, fit_subspace
from .output import (
    build_manifest,
    reconstruct_to_envi,
    write_additive_spectral_scale,
    write_band_metadata,
    write_coefficients_envi,
    write_json,
    write_previews,
)
from .quality import build_quality_report
from .registration import (
    RegistrationBundle,
    RoiRegistrationBundle,
    analysis_rgb_grid,
    estimate_registration,
    estimate_roi_registration,
    refined_analysis_rgb_grid,
    sample_cube_on_rgb_grid,
)
from .roi import choose_roi
from .spectral import harmonize_sensors


@dataclass(slots=True)
class PipelineResult:
    output_dir: Path
    manifest: dict[str, Any]
    quality_report: dict[str, Any]
    roi: dict[str, int]


@dataclass(slots=True)
class RegistrationReviewResult:
    output_dir: Path
    roi: dict[str, int]
    status: str
    report: dict[str, Any]


def _analysis_shape(roi: dict[str, int], registration: RegistrationBundle) -> tuple[int, int]:
    def scales(matrix: np.ndarray) -> tuple[float, float]:
        return float(np.linalg.norm(matrix[:2, 0])), float(np.linalg.norm(matrix[:2, 1]))
    nir_x, nir_y = scales(registration.nir.rgb_to_sensor_matrix)
    swir_x, swir_y = scales(registration.swir.rgb_to_sensor_matrix)
    width = max(24, int(round(roi["width"] * min(nir_x, swir_x))))
    height = max(24, int(round(roi["height"] * min(nir_y, swir_y))))
    width = min(width, registration.nir.sensor_shape[1], registration.swir.sensor_shape[1])
    height = min(height, registration.nir.sensor_shape[0], registration.swir.sensor_shape[0])
    return height, width


def _prepare_dirs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("cube", "coefficients", "metadata", "metrics", "previews", "analysis"):
        (out / name).mkdir(exist_ok=True)


def _planned_pipeline_outputs(config: PipelineConfig) -> tuple[Path, ...]:
    """Enumerate deterministic files that this pipeline run may replace."""

    out = config.output_dir
    relative_paths = [
        "manifest.json",
        "analysis/harmonized_lowres.hdr",
        "analysis/harmonized_lowres.dat",
        "metadata/input_metadata.json",
        "metadata/registration_model.json",
        "metadata/spectral_harmonization.json",
        "metadata/band_metadata.csv",
        "metadata/psf_model.json",
        "metadata/subspace_model.json",
        "metadata/additive_spectral_scale.json",
        "metadata/fusion_model.json",
        "metadata/processing_config.json",
        "metrics/quality_report.json",
    ]
    if config.output.write_envi:
        relative_paths.extend(
            (
                "cube/fused_continuous_691_2518nm.hdr",
                "cube/fused_continuous_691_2518nm.dat",
            )
        )
    if config.output.write_coefficients:
        relative_paths.extend(
            (
                "coefficients/material_coefficients.hdr",
                "coefficients/material_coefficients.dat",
            )
        )
    if config.output.write_uncertainty:
        relative_paths.extend(
            (
                "metrics/spatial_uncertainty.hdr",
                "metrics/spatial_uncertainty.dat",
                "metrics/spatial_detail_gain.hdr",
                "metrics/spatial_detail_gain.dat",
                "metrics/spatial_additive_detail.hdr",
                "metrics/spatial_additive_detail.dat",
            )
        )
    if config.output.write_previews:
        relative_paths.extend(
            (
                "previews/rgb_reference.png",
                "previews/fused_false_color_base_2200_1650_900.png",
                "previews/fused_false_color_2200_1650_900.png",
                "previews/fused_2200nm.png",
                "previews/fused_2350nm.png",
                "previews/fused_mean_reflectance.png",
                "previews/uncertainty.png",
                "previews/spatial_detail_before_after.png",
                "previews/spatial_detail_gain.png",
                "previews/spatial_additive_detail.png",
                "previews/registration_rgb_structure.png",
                "previews/registration_nir_overlay.png",
                "previews/registration_swir_overlay.png",
            )
        )
        if config.registration.enable_roi_refinement:
            relative_paths.extend(
                (
                    "previews/registration_nir_roi_before.png",
                    "previews/registration_nir_roi_after.png",
                    "previews/registration_swir_roi_before.png",
                    "previews/registration_swir_roi_after.png",
                    "previews/registration_nir_swir_overlap_after.png",
                    "previews/registration_nir_roi_checkerboard.png",
                    "previews/registration_swir_roi_checkerboard.png",
                    "previews/registration_nir_tiepoints.png",
                    "previews/registration_swir_tiepoints.png",
                    "previews/registration_nir_to_swir_tiepoints.png",
                    "previews/registration_joint_hsi_tiepoints.png",
                )
            )
    return tuple(out / relative for relative in relative_paths)


def _protect_pipeline_outputs(config: PipelineConfig) -> None:
    if config.output.overwrite_files:
        return
    existing = [path for path in _planned_pipeline_outputs(config) if path.exists()]
    if existing:
        shown = ", ".join(str(path) for path in existing[:5])
        suffix = "" if len(existing) <= 5 else f" (and {len(existing) - 5} more)"
        raise FileExistsError(
            f"Output file(s) already exist: {shown}{suffix}; enable "
            "output.overwrite_files to replace explicit files"
        )


def _preview_uint8(image: np.ndarray) -> np.ndarray:
    normalized = normalize_image(image)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(normalized, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _save_registration_previews(bundle: RegistrationBundle, preview_dir: Path) -> None:
    rgb = _preview_uint8(bundle.preview_rgb)
    nir = _preview_uint8(bundle.preview_nir_aligned)
    swir = _preview_uint8(bundle.preview_swir_aligned)
    Image.fromarray(rgb).save(preview_dir / "registration_rgb_structure.png")
    Image.fromarray(np.stack([rgb, nir, ((rgb.astype(np.float32) + nir) * 0.5).astype(np.uint8)], axis=2)).save(preview_dir / "registration_nir_overlay.png")
    Image.fromarray(np.stack([rgb, swir, ((rgb.astype(np.float32) + swir) * 0.5).astype(np.uint8)], axis=2)).save(preview_dir / "registration_swir_overlay.png")


def _registration_edge(image: np.ndarray) -> np.ndarray:
    base = normalize_image(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0))
    gx = cv2.Scharr(base, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(base, cv2.CV_32F, 0, 1)
    return normalize_image(cv2.magnitude(gx, gy))


def _preview_resize(image: np.ndarray, max_size: int = 1400) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_size / float(max(height, width)), 5.0)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA)


def _save_pair_overlay(path: Path, reference: np.ndarray, moving: np.ndarray, *, edges: bool) -> None:
    ref = _registration_edge(reference) if edges else normalize_image(reference)
    mov = _registration_edge(moving) if edges else normalize_image(moving)
    overlay = np.stack([ref, mov, 0.5 * (ref + mov)], axis=2)
    Image.fromarray(_preview_uint8(_preview_resize(np.clip(overlay, 0, 1)))).save(path)


def _save_checkerboard(path: Path, reference: np.ndarray, moving: np.ndarray, block: int = 24) -> None:
    ref = normalize_image(reference)
    mov = normalize_image(moving)
    yy, xx = np.indices(ref.shape)
    mask = ((yy // block + xx // block) % 2).astype(bool)
    board = np.where(mask, ref, mov)
    Image.fromarray(_preview_uint8(_preview_resize(board))).save(path)


def _save_tiepoint_vectors(path: Path, reference: np.ndarray, details: dict[str, Any]) -> None:
    base = _preview_uint8(_preview_resize(reference))
    canvas = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    scale_x = canvas.width / float(reference.shape[1])
    scale_y = canvas.height / float(reference.shape[0])
    selection = details.get("selection") or {}
    factor = float(selection.get("selected_factor", 0.0))
    for point in details.get("tie_points", []):
        x0 = float(point["ref_x"]) * scale_x
        y0 = float(point["ref_y"]) * scale_y
        x1 = (float(point["ref_x"]) + factor * float(point["shift_x"])) * scale_x
        y1 = (float(point["ref_y"]) + factor * float(point["shift_y"])) * scale_y
        score = float(point.get("score", 0.0))
        color = (40, int(np.clip(score, 0.0, 1.0) * 255), 255)
        draw.line((x0, y0, x1, y1), fill=color, width=max(1, int(round(min(scale_x, scale_y)))))
        radius = max(2, int(round(0.8 * min(scale_x, scale_y))))
        draw.ellipse((x0 - radius, y0 - radius, x0 + radius, y0 + radius), outline=(255, 220, 0), width=1)
    draw.rectangle((0, 0, min(canvas.width, 390), 25), fill=(0, 0, 0))
    draw.text(
        (6, 6),
        f"points={len(details.get('tie_points', []))} selected_factor={factor:.2f}",
        fill=(255, 255, 255),
    )
    canvas.save(path)


def _save_roi_registration_previews(bundle: RoiRegistrationBundle, preview_dir: Path) -> dict[str, str]:
    _save_pair_overlay(preview_dir / "registration_nir_roi_before.png", bundle.reference_structure, bundle.nir_initial, edges=True)
    _save_pair_overlay(preview_dir / "registration_nir_roi_after.png", bundle.reference_structure, bundle.nir_aligned, edges=True)
    _save_pair_overlay(preview_dir / "registration_swir_roi_before.png", bundle.reference_structure, bundle.swir_initial, edges=True)
    _save_pair_overlay(preview_dir / "registration_swir_roi_after.png", bundle.reference_structure, bundle.swir_aligned, edges=True)
    _save_pair_overlay(
        preview_dir / "registration_nir_swir_overlap_after.png",
        bundle.swir_overlap_aligned,
        bundle.nir_overlap_aligned,
        edges=True,
    )
    _save_checkerboard(preview_dir / "registration_nir_roi_checkerboard.png", bundle.reference_structure, bundle.nir_aligned)
    _save_checkerboard(preview_dir / "registration_swir_roi_checkerboard.png", bundle.reference_structure, bundle.swir_aligned)
    nir_details = (bundle.nir.details or {}).get("tiepoint_refinement", {})
    swir_details = (bundle.swir.details or {}).get("tiepoint_refinement", {})
    pair_details = (bundle.nir.details or {}).get("nir_to_swir_tiepoint_refinement", {})
    joint_details = (bundle.nir.details or {}).get("joint_hsi_to_rgb_tiepoint_refinement", {})
    _save_tiepoint_vectors(preview_dir / "registration_nir_tiepoints.png", bundle.reference_structure, nir_details)
    _save_tiepoint_vectors(preview_dir / "registration_swir_tiepoints.png", bundle.reference_structure, swir_details)
    _save_tiepoint_vectors(preview_dir / "registration_nir_to_swir_tiepoints.png", bundle.swir_overlap_aligned, pair_details)
    _save_tiepoint_vectors(preview_dir / "registration_joint_hsi_tiepoints.png", bundle.reference_structure, joint_details)
    return {
        "registration_nir_roi_before": "previews/registration_nir_roi_before.png",
        "registration_nir_roi_after": "previews/registration_nir_roi_after.png",
        "registration_swir_roi_before": "previews/registration_swir_roi_before.png",
        "registration_swir_roi_after": "previews/registration_swir_roi_after.png",
        "registration_nir_swir_overlap_after": "previews/registration_nir_swir_overlap_after.png",
        "registration_nir_roi_checkerboard": "previews/registration_nir_roi_checkerboard.png",
        "registration_swir_roi_checkerboard": "previews/registration_swir_roi_checkerboard.png",
        "registration_nir_tiepoints": "previews/registration_nir_tiepoints.png",
        "registration_swir_tiepoints": "previews/registration_swir_tiepoints.png",
        "registration_nir_to_swir_tiepoints": "previews/registration_nir_to_swir_tiepoints.png",
        "registration_joint_hsi_tiepoints": "previews/registration_joint_hsi_tiepoints.png",
    }


def _write_uncertainty(path: Path, uncertainty: np.ndarray) -> tuple[Path, Path]:
    writer, hdr, dat = create_bip_writer(path, uncertainty.shape + (1,), dtype="float32", description="GeoCoreFusion normalized spatial uncertainty")
    writer[:, :, 0] = uncertainty
    writer.flush()
    del writer
    return hdr, dat


def _write_detail_gain(path: Path, gain: np.ndarray) -> tuple[Path, Path]:
    writer, hdr, dat = create_bip_writer(
        path,
        gain.shape + (1,),
        dtype="float32",
        description="GeoCoreFusion spectral-shape-preserving RGB spatial detail gain",
    )
    writer[:, :, 0] = gain
    writer.flush()
    del writer
    return hdr, dat


def _write_additive_detail(path: Path, detail: np.ndarray) -> tuple[Path, Path]:
    writer, hdr, dat = create_bip_writer(
        path,
        detail.shape + (1,),
        dtype="float32",
        description="GeoCoreFusion observation-near-nullspace RGB additive spatial detail map",
    )
    writer[:, :, 0] = detail
    writer.flush()
    del writer
    return hdr, dat


def _write_lowres_cube(path: Path, cube: np.ndarray, wavelengths: np.ndarray) -> tuple[Path, Path]:
    writer, hdr, dat = create_bip_writer(path, cube.shape, dtype="float32", wavelengths=wavelengths, description="GeoCoreFusion aligned and harmonized low-resolution observation cube")
    writer[:] = cube
    writer.flush()
    del writer
    return hdr, dat


def input_summary(dataset: DatasetTriplet) -> dict[str, Any]:
    return {
        "root": str(dataset.root),
        "rgb": metadata_dict(dataset.rgb.meta),
        "nir": metadata_dict(dataset.nir.meta),
        "swir": metadata_dict(dataset.swir.meta),
    }


def run_registration_review(config: PipelineConfig) -> RegistrationReviewResult:
    """Run registration only and persist exact ROI QA images for manual review."""

    dataset = discover_triplet(config.data_dir)
    out = config.output_dir
    _prepare_dirs(out)
    registration = estimate_registration(dataset, config.registration)
    roi = choose_roi(config.roi, registration, dataset.rgb.meta.shape[:2])
    analysis_shape = _analysis_shape(roi, registration)
    roi_registration = estimate_roi_registration(
        dataset,
        registration,
        roi,
        analysis_shape,
        config.registration,
    ) if config.registration.enable_roi_refinement else None
    _save_registration_previews(registration, out / "previews")
    preview_paths = {
        "registration_nir_global_proxy": "previews/registration_nir_overlay.png",
        "registration_swir_global_proxy": "previews/registration_swir_overlay.png",
    }
    if roi_registration is not None:
        preview_paths.update(_save_roi_registration_previews(roi_registration, out / "previews"))
        status = roi_registration.status
    else:
        status = "not_refined"
    registration_metadata: dict[str, Any] = {"full_scan_coarse": registration.to_dict()}
    if roi_registration is not None:
        registration_metadata["roi_refinement"] = roi_registration.to_dict()
    report = {
        "status": status,
        "input_data_dir": str(config.data_dir),
        "output_dir": str(out),
        "rgb_roi": roi,
        "analysis_shape": list(analysis_shape),
        "previews": preview_paths,
        "registration": registration_metadata,
        "manual_review_note": "Use the ROI after/ checkerboard images for acceptance. The global proxy images are only for full-scan localization.",
    }
    write_json(out / "metadata" / "input_metadata.json", input_summary(dataset))
    write_json(out / "metadata" / "registration_model.json", registration_metadata)
    write_json(out / "metadata" / "processing_config.json", config.to_dict())
    write_json(out / "metrics" / "registration_quality.json", report)
    write_json(out / "registration_review.json", report)
    return RegistrationReviewResult(output_dir=out, roi=roi, status=status, report=report)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    np.random.seed(config.fusion.random_seed)
    out = config.output_dir
    _protect_pipeline_outputs(config)
    _prepare_dirs(out)
    dataset = discover_triplet(config.data_dir)
    fused_hdr = out / "cube" / "fused_continuous_691_2518nm.hdr"

    registration = estimate_registration(dataset, config.registration)
    roi = choose_roi(config.roi, registration, dataset.rgb.meta.shape[:2])
    rgb_crop = np.asarray(
        dataset.rgb.cube[roi["y"] : roi["y"] + roi["height"], roi["x"] : roi["x"] + roi["width"], :3]
    )
    analysis_shape = _analysis_shape(roi, registration)
    roi_registration: RoiRegistrationBundle | None = None
    if config.registration.enable_roi_refinement:
        roi_registration = estimate_roi_registration(
            dataset,
            registration,
            roi,
            analysis_shape,
            config.registration,
        )
        nir_grid_y, nir_grid_x = refined_analysis_rgb_grid(roi, roi_registration.nir)
        swir_grid_y, swir_grid_x = refined_analysis_rgb_grid(roi, roi_registration.swir)
    else:
        nir_grid_y, nir_grid_x = analysis_rgb_grid(roi, *analysis_shape)
        swir_grid_y, swir_grid_x = nir_grid_y, nir_grid_x
    nir_aligned = sample_cube_on_rgb_grid(dataset.nir.cube, registration.nir, nir_grid_y, nir_grid_x)
    swir_aligned = sample_cube_on_rgb_grid(dataset.swir.cube, registration.swir, swir_grid_y, swir_grid_x)
    spectral = harmonize_sensors(
        nir_aligned,
        swir_aligned,
        dataset.nir.meta.wavelengths,
        dataset.swir.meta.wavelengths,
        config.spectral,
    )
    hsi_structure = normalize_image(np.nanmean(spectral.cube, axis=2))
    psf = estimate_psf(rgb_crop, hsi_structure, config.degradation)
    factorization_method = str(config.fusion.factorization_method).strip().lower()
    if factorization_method in {"pca", "randomized_svd", "svd"}:
        subspace, low_coeff = fit_subspace(
            spectral.cube,
            rank=config.fusion.rank,
            max_pixels=config.fusion.max_basis_pixels,
            random_seed=config.fusion.random_seed,
            clip_quantiles=config.fusion.clip_quantiles,
        )
    elif factorization_method in {
        "simplex_nmf",
        "simplex",
        "nonnegative_simplex",
    }:
        subspace, low_coeff = fit_simplex_subspace(
            spectral.cube,
            rank=config.fusion.rank,
            max_pixels=config.fusion.max_basis_pixels,
            random_seed=config.fusion.random_seed,
            clip_quantiles=config.fusion.clip_quantiles,
            max_iterations=config.fusion.simplex_factorization_iterations,
            tolerance=config.fusion.simplex_factorization_tolerance,
        )
        config.fusion.coefficient_constraint = "simplex"
    elif factorization_method in {
        "hybrid_simplex_residual",
        "simplex_plus_residual",
        "uarf_hybrid",
    }:
        subspace, low_coeff = fit_hybrid_simplex_subspace(
            spectral.cube,
            rank=config.fusion.rank,
            residual_rank=config.fusion.simplex_residual_rank,
            max_pixels=config.fusion.max_basis_pixels,
            random_seed=config.fusion.random_seed,
            clip_quantiles=config.fusion.clip_quantiles,
            max_iterations=config.fusion.simplex_factorization_iterations,
            tolerance=config.fusion.simplex_factorization_tolerance,
        )
        config.fusion.coefficient_constraint = "hybrid_simplex"
    else:
        raise ValueError(
            f"Unknown fusion factorization_method "
            f"{config.fusion.factorization_method!r}"
        )
    fusion = refine_coefficients(low_coeff, rgb_crop, psf, config.fusion)
    fusion.details["factorization"] = {
        "configured_method": factorization_method,
        "representation": str(subspace.representation),
        "fit_metadata": dict(subspace.fit_metadata),
    }
    selected_rmse = float(np.sqrt(np.mean((degrade_coefficients(fusion.coefficients, psf) - low_coeff) ** 2)))
    safety_trials = [{"psf_factor": 1.0, "coefficient_rmse": selected_rmse}]
    if selected_rmse > config.fusion.safety_observation_rmse:
        best = (selected_rmse, psf, fusion)
        for factor in config.fusion.psf_backoff_factors:
            factor = float(factor)
            if factor >= 0.999:
                continue
            candidate_psf = replace(
                psf,
                sigma_x_highres=psf.sigma_x_highres * factor,
                sigma_y_highres=psf.sigma_y_highres * factor,
                method=f"{psf.method}_self_consistency_backoff_{factor:.2f}",
            )
            candidate_fusion = refine_coefficients(low_coeff, rgb_crop, candidate_psf, config.fusion)
            candidate_rmse = float(np.sqrt(np.mean((degrade_coefficients(candidate_fusion.coefficients, candidate_psf) - low_coeff) ** 2)))
            safety_trials.append({"psf_factor": factor, "coefficient_rmse": candidate_rmse})
            if candidate_rmse < best[0]:
                best = (candidate_rmse, candidate_psf, candidate_fusion)
            if candidate_rmse <= config.fusion.safety_observation_rmse:
                break
        selected_rmse, psf, fusion = best
    fusion.details["psf_safety_trials"] = safety_trials
    fusion.details["selected_coefficient_rmse_pre_product_cycle"] = selected_rmse
    additive_mode = str(config.fusion.spatial_detail_additive_mode).strip().lower()
    if additive_mode in {"band_adaptive_mtf_gsa", "mtf_gsa", "band_adaptive"}:
        (
            fusion.additive_detail_map,
            additive_spectral_scale,
            additive_details,
        ) = build_band_adaptive_mtf_detail(
            spectral.cube,
            rgb_crop,
            psf,
            config.fusion,
        )
        fusion.details["spatial_detail"]["additive_model"] = additive_details
    elif additive_mode in {"legacy", "shared", "none"}:
        additive_spectral_scale = build_additive_spectral_scale(
            spectral.cube, config.fusion
        )
    else:
        raise ValueError(
            f"Unknown spatial_detail_additive_mode "
            f"{config.fusion.spatial_detail_additive_mode!r}"
        )
    fusion.details["spatial_detail"]["additive_spectral_scale_min"] = float(np.min(additive_spectral_scale))
    fusion.details["spatial_detail"]["additive_spectral_scale_max"] = float(np.max(additive_spectral_scale))
    fusion.details["spatial_detail"]["additive_spectral_scale_mean"] = float(np.mean(additive_spectral_scale))
    fusion.coefficients, product_cycle = back_project_modulated_product(
        fusion.coefficients,
        low_coeff,
        spectral.cube,
        subspace.basis,
        subspace.mean_spectrum,
        fusion.detail_gain_map,
        fusion.additive_detail_map,
        additive_spectral_scale,
        psf,
        config.fusion,
    )
    fusion.details["spatial_detail"]["final_product_observation_cycle"] = product_cycle
    final_coefficient_rmse = float(
        np.sqrt(
            np.mean(
                (degrade_coefficients(fusion.coefficients, psf) - low_coeff) ** 2
            )
        )
    )
    fusion.details["selected_coefficient_rmse"] = final_coefficient_rmse
    quality = build_quality_report(
        fusion.coefficients,
        low_coeff,
        subspace,
        psf,
        rgb_crop,
        spectral,
        nir_aligned,
        swir_aligned,
        dataset.nir.meta.wavelengths,
        dataset.swir.meta.wavelengths,
        fusion.uncertainty_map,
        fusion.detail_gain_map,
        fusion.additive_detail_map,
        additive_spectral_scale,
    )
    if roi_registration is not None:
        quality["registration"] = {
            "status": roi_registration.status,
            "full_scan_coarse": {
                "nir_ecc_score": registration.nir.ecc_score,
                "nir_edge_correlation": registration.nir.edge_correlation,
                "swir_ecc_score": registration.swir.ecc_score,
                "swir_edge_correlation": registration.swir.edge_correlation,
            },
            "roi_refinement": roi_registration.to_dict(),
            "interpretation": "ROI scores are measured on the exact low-resolution grids sampled for fusion. NIR/SWIR overlap is evaluated independently of RGB.",
        }
        quality_rank = {"passed": 0, "warning": 1, "failed": 2}
        current_status = str(quality["summary"]["status"])
        if quality_rank[roi_registration.status] > quality_rank.get(current_status, 0):
            quality["summary"]["status"] = roi_registration.status
        quality["summary"]["interpretation"] += " Registration status is gated by final ROI alignment, not by the full-scan proxy preview."
    else:
        quality["registration"] = {
            "status": "not_refined",
            "nir_ecc_score": registration.nir.ecc_score,
            "nir_edge_correlation": registration.nir.edge_correlation,
            "swir_ecc_score": registration.swir.ecc_score,
            "swir_edge_correlation": registration.swir.edge_correlation,
        }
    quality["degradation"] = psf.to_dict()
    quality["subspace"] = {
        "rank": int(subspace.basis.shape[0]),
        "explained_variance_total": float(subspace.explained_variance_ratio.sum()),
        "representation": str(subspace.representation),
        "fit_metadata": dict(subspace.fit_metadata),
    }

    outputs: dict[str, str] = {}
    if config.output.write_envi:
        hdr, dat = reconstruct_to_envi(
            fusion.coefficients,
            subspace,
            spectral.wavelengths_nm,
            fused_hdr,
            tile_size=config.fusion.reconstruct_tile,
            dtype=config.fusion.output_dtype,
            detail_gain=fusion.detail_gain_map,
            additive_detail=fusion.additive_detail_map,
            additive_spectral_scale=additive_spectral_scale,
        )
        outputs["fused_cube_hdr"] = str(hdr.relative_to(out)).replace("\\", "/")
        outputs["fused_cube_dat"] = str(dat.relative_to(out)).replace("\\", "/")
    if config.output.write_coefficients:
        hdr, dat = write_coefficients_envi(fusion.coefficients, out / "coefficients" / "material_coefficients.hdr")
        outputs["material_coefficients_hdr"] = str(hdr.relative_to(out)).replace("\\", "/")
        outputs["material_coefficients_dat"] = str(dat.relative_to(out)).replace("\\", "/")
    if config.output.write_uncertainty:
        hdr, dat = _write_uncertainty(out / "metrics" / "spatial_uncertainty.hdr", fusion.uncertainty_map)
        outputs["spatial_uncertainty_hdr"] = str(hdr.relative_to(out)).replace("\\", "/")
        outputs["spatial_uncertainty_dat"] = str(dat.relative_to(out)).replace("\\", "/")
        hdr, dat = _write_detail_gain(out / "metrics" / "spatial_detail_gain.hdr", fusion.detail_gain_map)
        outputs["spatial_detail_gain_hdr"] = str(hdr.relative_to(out)).replace("\\", "/")
        outputs["spatial_detail_gain_dat"] = str(dat.relative_to(out)).replace("\\", "/")
        hdr, dat = _write_additive_detail(out / "metrics" / "spatial_additive_detail.hdr", fusion.additive_detail_map)
        outputs["spatial_additive_detail_hdr"] = str(hdr.relative_to(out)).replace("\\", "/")
        outputs["spatial_additive_detail_dat"] = str(dat.relative_to(out)).replace("\\", "/")
    low_hdr, low_dat = _write_lowres_cube(out / "analysis" / "harmonized_lowres.hdr", spectral.cube, spectral.wavelengths_nm)
    outputs["harmonized_lowres_hdr"] = str(low_hdr.relative_to(out)).replace("\\", "/")
    outputs["harmonized_lowres_dat"] = str(low_dat.relative_to(out)).replace("\\", "/")

    previews: dict[str, str] = {}
    if config.output.write_previews:
        previews = write_previews(
            out / "previews",
            rgb_crop,
            fusion.coefficients,
            subspace,
            spectral.wavelengths_nm,
            fusion.uncertainty_map,
            fusion.detail_gain_map,
            fusion.additive_detail_map,
            additive_spectral_scale,
        )
        _save_registration_previews(registration, out / "previews")
        previews.update({
            "registration_nir_global_proxy": "previews/registration_nir_overlay.png",
            "registration_swir_global_proxy": "previews/registration_swir_overlay.png",
        })
        if roi_registration is not None:
            previews.update(_save_roi_registration_previews(roi_registration, out / "previews"))

    write_json(out / "metadata" / "input_metadata.json", input_summary(dataset))
    outputs["input_metadata_json"] = "metadata/input_metadata.json"
    registration_metadata: dict[str, Any] = {"full_scan_coarse": registration.to_dict()}
    if roi_registration is not None:
        registration_metadata["roi_refinement"] = roi_registration.to_dict()
    write_json(out / "metadata" / "registration_model.json", registration_metadata)
    outputs["registration_model_json"] = "metadata/registration_model.json"
    write_json(out / "metadata" / "spectral_harmonization.json", spectral.model)
    outputs["spectral_harmonization_json"] = "metadata/spectral_harmonization.json"
    write_band_metadata(out / "metadata" / "band_metadata.csv", spectral.band_metadata)
    outputs["band_metadata_csv"] = "metadata/band_metadata.csv"
    write_json(out / "metadata" / "psf_model.json", psf.to_dict())
    outputs["psf_model_json"] = "metadata/psf_model.json"
    subspace_payload = subspace.to_dict()
    subspace_payload["source_metadata"] = {
        "derived_from": [
            "analysis/harmonized_lowres.hdr",
            "metadata/spectral_harmonization.json",
            "metadata/band_metadata.csv",
        ],
        "processing_config": "metadata/processing_config.json",
        "algorithm": str(
            subspace.fit_metadata.get(
                "algorithm",
                "randomized_svd"
                if factorization_method in {"pca", "randomized_svd", "svd"}
                else factorization_method,
            )
        ),
        "representation": str(subspace.representation),
        "random_seed": int(config.fusion.random_seed),
    }
    write_json(out / "metadata" / "subspace_model.json", subspace_payload)
    outputs["subspace_model_json"] = "metadata/subspace_model.json"
    write_additive_spectral_scale(
        out / "metadata" / "additive_spectral_scale.json",
        spectral.wavelengths_nm,
        additive_spectral_scale,
        source_metadata={
            "derived_from": [
                "analysis/harmonized_lowres.hdr",
                "metadata/spectral_harmonization.json",
            ],
            "fusion_model": "metadata/fusion_model.json",
            "method": additive_mode,
        },
    )
    outputs["additive_spectral_scale_json"] = "metadata/additive_spectral_scale.json"
    write_json(out / "metadata" / "fusion_model.json", {"details": fusion.details, "history": fusion.history})
    outputs["fusion_model_json"] = "metadata/fusion_model.json"
    write_json(out / "metadata" / "processing_config.json", config.to_dict())
    outputs["processing_config_json"] = "metadata/processing_config.json"
    write_json(out / "metrics" / "quality_report.json", quality)
    outputs["quality_report_json"] = "metrics/quality_report.json"
    outputs["run_manifest_json"] = "manifest.json"
    manifest = build_manifest(config, roi, spectral.wavelengths_nm, outputs, previews)
    write_json(out / "manifest.json", manifest)
    return PipelineResult(output_dir=out, manifest=manifest, quality_report=quality, roi=roi)

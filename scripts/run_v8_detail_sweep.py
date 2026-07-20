"""Run a small, non-overwriting 3DSSZ detail-recovery Pareto sweep.

The sweep changes only detail-transfer parameters.  Registration, spectral
harmonisation, the estimated sensor PSF, ROI, subspace rank, and quality
evaluation stay identical to the frozen V7 configuration.  Each variant is
written to its own run directory so historical evidence is never replaced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


VARIANTS: dict[str, dict[str, Any]] = {
    "c020": {
        "coefficient_detail_strength": 0.020,
        "spatial_detail_additive_strength": 0.0,
    },
    "c050": {
        "coefficient_detail_strength": 0.050,
        "spatial_detail_additive_strength": 0.0,
    },
    "c100": {
        "coefficient_detail_strength": 0.100,
        "spatial_detail_additive_strength": 0.0,
    },
    "c150": {
        "coefficient_detail_strength": 0.150,
        "spatial_detail_additive_strength": 0.0,
    },
    "c050_g125": {
        "coefficient_detail_strength": 0.050,
        "spatial_detail_strength": 1.25,
        "spatial_detail_additive_strength": 0.0,
    },
    "c050_a010": {
        "coefficient_detail_strength": 0.050,
        "spatial_detail_additive_strength": 0.10,
        "spatial_detail_additive_std_fraction": 0.20,
        "spatial_detail_additive_mean_fraction": 0.02,
        "spatial_detail_additive_correlation_floor": 0.15,
    },
    "ungated_g105": {
        "coefficient_detail_strength": 0.001,
        "spatial_detail_strength": 1.05,
        "spatial_detail_confidence_mode": "none",
        "spatial_detail_additive_strength": 0.0,
    },
    "plateau_g105": {
        "coefficient_detail_strength": 0.001,
        "spatial_detail_strength": 1.05,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_confidence_gate_low": 0.08,
        "spatial_detail_confidence_gate_high": 0.28,
        "spatial_detail_additive_strength": 0.0,
    },
    "plateau_g140": {
        "coefficient_detail_strength": 0.001,
        "spatial_detail_strength": 1.40,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_confidence_gate_low": 0.08,
        "spatial_detail_confidence_gate_high": 0.28,
        "spatial_detail_additive_strength": 0.0,
    },
    "plateau_np0_g105": {
        "coefficient_detail_strength": 0.001,
        "spatial_detail_strength": 1.05,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_confidence_gate_low": 0.08,
        "spatial_detail_confidence_gate_high": 0.28,
        # _mtf_matched_rgb_features already forms R - U(D(R)).  This
        # ablation removes the second I-U(D(.)) projection while retaining
        # the final modulated-product observation back-projection.
        "spatial_detail_nullspace_iterations": 0,
        "spatial_detail_additive_strength": 0.0,
    },
    "ungated_np0_g100": {
        "coefficient_detail_strength": 0.001,
        "spatial_detail_strength": 1.00,
        "spatial_detail_confidence_mode": "none",
        "spatial_detail_nullspace_iterations": 0,
        "spatial_detail_additive_strength": 0.0,
    },
    "simplex_k6_s000": {
        "rank": 6,
        "factorization_method": "simplex_nmf",
        "simplex_factorization_iterations": 20,
        "simplex_factorization_tolerance": 1e-4,
        "coefficient_constraint": "simplex",
        "refiner": "bicubic",
        "coefficient_detail_method": "simplex_abundance_rgb_residual",
        "coefficient_detail_strength": 0.0,
        "coefficient_detail_nullspace_iterations": 1,
        "coefficient_detail_back_projection_iterations": 2,
        "simplex_abundance_detail_l1_limit": 0.25,
        "spatial_detail_strength": 0.0,
        "spatial_detail_additive_strength": 0.0,
        "spatial_detail_product_back_projection_iterations": 3,
        "spatial_detail_product_back_projection_weight": 0.50,
    },
    "simplex_k6_s050": {
        "rank": 6,
        "factorization_method": "simplex_nmf",
        "simplex_factorization_iterations": 20,
        "simplex_factorization_tolerance": 1e-4,
        "coefficient_constraint": "simplex",
        "refiner": "bicubic",
        "coefficient_detail_method": "simplex_abundance_rgb_residual",
        "coefficient_detail_strength": 0.50,
        "coefficient_detail_nullspace_iterations": 1,
        "coefficient_detail_back_projection_iterations": 2,
        "simplex_abundance_rgb_ridge": 0.02,
        "simplex_abundance_min_r2": 0.05,
        "simplex_abundance_detail_l1_limit": 0.25,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_confidence_gate_low": 0.08,
        "spatial_detail_confidence_gate_high": 0.28,
        "spatial_detail_strength": 0.0,
        "spatial_detail_additive_strength": 0.0,
        "spatial_detail_product_back_projection_iterations": 3,
        "spatial_detail_product_back_projection_weight": 0.50,
    },
    "hybrid_k6r6_s000": {
        "rank": 6,
        "factorization_method": "hybrid_simplex_residual",
        "simplex_residual_rank": 6,
        "simplex_factorization_iterations": 20,
        "simplex_factorization_tolerance": 1e-4,
        "coefficient_constraint": "hybrid_simplex",
        "refiner": "bicubic",
        "coefficient_detail_method": "simplex_abundance_rgb_residual",
        "coefficient_detail_strength": 0.0,
        "coefficient_detail_nullspace_iterations": 1,
        "coefficient_detail_back_projection_iterations": 2,
        "simplex_abundance_detail_l1_limit": 0.25,
        "spatial_detail_strength": 0.0,
        "spatial_detail_additive_strength": 0.0,
        "spatial_detail_product_back_projection_iterations": 3,
        "spatial_detail_product_back_projection_weight": 0.50,
    },
    "hybrid_k6r6_s050": {
        "rank": 6,
        "factorization_method": "hybrid_simplex_residual",
        "simplex_residual_rank": 6,
        "simplex_factorization_iterations": 20,
        "simplex_factorization_tolerance": 1e-4,
        "coefficient_constraint": "hybrid_simplex",
        "refiner": "bicubic",
        "coefficient_detail_method": "simplex_abundance_rgb_residual",
        "coefficient_detail_strength": 0.50,
        "coefficient_detail_nullspace_iterations": 1,
        "coefficient_detail_back_projection_iterations": 2,
        "simplex_abundance_rgb_ridge": 0.02,
        "simplex_abundance_min_r2": 0.05,
        "simplex_abundance_detail_l1_limit": 0.25,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_confidence_gate_low": 0.08,
        "spatial_detail_confidence_gate_high": 0.28,
        "spatial_detail_strength": 0.0,
        "spatial_detail_additive_strength": 0.0,
        "spatial_detail_product_back_projection_iterations": 3,
        "spatial_detail_product_back_projection_weight": 0.50,
    },
    "bridge_r1_s050": {
        "factorization_method": "pca",
        "coefficient_constraint": "unconstrained",
        "coefficient_detail_method": "lowrank_coefficient_bridge",
        "coefficient_detail_strength": 0.50,
        "coefficient_detail_bridge_rank": 1,
        "coefficient_detail_bridge_cv_r2_floor": 0.0,
        "coefficient_detail_ridge": 0.03,
        "coefficient_detail_nullspace_iterations": 0,
        "coefficient_detail_back_projection_iterations": 1,
        "coefficient_detail_base_residual_keep": 0.0,
        "coefficient_detail_clip_sigma": 1.5,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_strength": 0.0,
        "spatial_detail_additive_strength": 0.0,
        "spatial_detail_product_back_projection_iterations": 2,
    },
    "bridge_r2_s050": {
        "factorization_method": "pca",
        "coefficient_constraint": "unconstrained",
        "coefficient_detail_method": "lowrank_coefficient_bridge",
        "coefficient_detail_strength": 0.50,
        "coefficient_detail_bridge_rank": 2,
        "coefficient_detail_bridge_cv_r2_floor": 0.0,
        "coefficient_detail_ridge": 0.03,
        "coefficient_detail_nullspace_iterations": 0,
        "coefficient_detail_back_projection_iterations": 1,
        "coefficient_detail_base_residual_keep": 0.0,
        "coefficient_detail_clip_sigma": 1.5,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_strength": 0.0,
        "spatial_detail_additive_strength": 0.0,
        "spatial_detail_product_back_projection_iterations": 2,
    },
    "bridge_r2_s100": {
        "factorization_method": "pca",
        "coefficient_constraint": "unconstrained",
        "coefficient_detail_method": "lowrank_coefficient_bridge",
        "coefficient_detail_strength": 1.00,
        "coefficient_detail_bridge_rank": 2,
        "coefficient_detail_bridge_cv_r2_floor": 0.0,
        "coefficient_detail_ridge": 0.03,
        "coefficient_detail_nullspace_iterations": 0,
        "coefficient_detail_back_projection_iterations": 1,
        "coefficient_detail_base_residual_keep": 0.0,
        "coefficient_detail_clip_sigma": 1.5,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_strength": 0.0,
        "spatial_detail_additive_strength": 0.0,
        "spatial_detail_product_back_projection_iterations": 2,
    },
    "local_bridge_r1_s050": {
        "factorization_method": "pca",
        "coefficient_constraint": "unconstrained",
        "coefficient_detail_method": "local_lowrank_coefficient_bridge",
        "coefficient_detail_strength": 0.50,
        "coefficient_detail_bridge_rank": 1,
        "coefficient_detail_bridge_cv_r2_floor": 0.0,
        "coefficient_detail_bridge_local_radius": 4,
        "coefficient_detail_bridge_local_correlation_floor": 0.10,
        "coefficient_detail_ridge": 0.03,
        "coefficient_detail_nullspace_iterations": 0,
        "coefficient_detail_back_projection_iterations": 1,
        "coefficient_detail_base_residual_keep": 0.0,
        "coefficient_detail_clip_sigma": 1.5,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_strength": 0.0,
        "spatial_detail_additive_strength": 0.0,
        "spatial_detail_product_back_projection_iterations": 2,
    },
    "local_bridge_r2_s050": {
        "factorization_method": "pca",
        "coefficient_constraint": "unconstrained",
        "coefficient_detail_method": "local_lowrank_coefficient_bridge",
        "coefficient_detail_strength": 0.50,
        "coefficient_detail_bridge_rank": 2,
        "coefficient_detail_bridge_cv_r2_floor": 0.0,
        "coefficient_detail_bridge_local_radius": 4,
        "coefficient_detail_bridge_local_correlation_floor": 0.10,
        "coefficient_detail_ridge": 0.03,
        "coefficient_detail_nullspace_iterations": 0,
        "coefficient_detail_back_projection_iterations": 1,
        "coefficient_detail_base_residual_keep": 0.0,
        "coefficient_detail_clip_sigma": 1.5,
        "spatial_detail_confidence_mode": "amplitude_preserving_snr",
        "spatial_detail_strength": 0.0,
        "spatial_detail_additive_strength": 0.0,
        "spatial_detail_product_back_projection_iterations": 2,
    },
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--scene", choices=("3dssz", "zkh3"), default="3dssz")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=["c020", "c050", "c100"],
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Explicitly refresh only this script's named sweep directories.",
    )
    return parser.parse_args()


def detail_metrics(quality: dict[str, Any]) -> dict[str, Any]:
    report = quality["spatial"]["band_detail_by_brightness"]
    metrics: dict[str, Any] = {}
    for band_name, band_report in report["bands"].items():
        scale = band_report["multiscale_log_high_frequency"]["sigma_2.4px"]
        reliable = scale["reliable_rgb_detail"]
        dark = scale["dark_reliable_rgb_detail"]
        flat = scale["rgb_flat"]
        edge = band_report["gradient_and_edge"]["reliable_rgb_detail"]
        metrics[band_name] = {
            "rho": reliable["rho"],
            "beta": reliable["beta"],
            "energy_ratio_A": reliable["energy_ratio_A"],
            "orthogonal_residual_ratio_R_perp": reliable[
                "orthogonal_residual_ratio_R_perp"
            ],
            "dark_rho": dark["rho"],
            "dark_beta": dark["beta"],
            "dark_energy_ratio_A": dark["energy_ratio_A"],
            "dark_orthogonal_residual_ratio_R_perp": dark[
                "orthogonal_residual_ratio_R_perp"
            ],
            "flat_energy_ratio_A": flat["energy_ratio_A"],
            "edge_f1_1px": edge["edge_f1_1px"],
            "gradient_orientation_coherence": edge[
                "gradient_orientation_coherence_abs_cosine"
            ],
        }
    return metrics


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from geocorefusion.config import load_config
    from geocorefusion.pipeline import run_pipeline

    results: list[dict[str, Any]] = []
    config_name = (
        "3dssz_roi_fusion_v7.yaml"
        if args.scene == "3dssz"
        else "zkh3_roi_fusion_v7.yaml"
    )
    for variant in args.variants:
        config = load_config(repo / "configs" / config_name)
        overrides = VARIANTS[variant]
        for name, value in overrides.items():
            setattr(config.fusion, name, value)
        config.output_dir = (
            repo / "runs" / f"{args.scene}_roi_fusion_v8_sweep_{variant}"
        )
        config.output.write_envi = False
        config.output.write_coefficients = True
        config.output.overwrite_files = bool(args.overwrite_existing)
        config.project.notes = (
            "V8 Pareto sweep: reliability-gated local MTF coefficient detail; "
            f"variant={variant}; overrides={overrides}"
        )

        started = time.perf_counter()
        run = run_pipeline(config)
        elapsed = time.perf_counter() - started
        forward = run.quality_report["final_hr_product_observation"]
        item = {
            "scene": args.scene,
            "variant": variant,
            "overrides": overrides,
            "output_dir": str(run.output_dir),
            "elapsed_seconds": elapsed,
            "summary_status": run.quality_report["summary"]["status"],
            "forward_rmse": forward["rmse"],
            "forward_sam_mean_deg": forward["sam_mean_deg"],
            "detail": detail_metrics(run.quality_report),
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    print(json.dumps({"runs": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

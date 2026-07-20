"""Sweep active-set simplex abundance detail without refitting frozen factors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


VARIANTS = {
    "pair_s050_a000": (0.50, 2, 0.00),
    "pair_s050_a015": (0.50, 2, 0.15),
    "pair_s100_a015": (1.00, 2, 0.15),
    "pair_s200_a015": (2.00, 2, 0.15),
    "pair_s100_a035": (1.00, 2, 0.35),
    "global_s050": (0.50, 0, 0.00),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=["pair_s050_a015", "pair_s100_a015", "pair_s200_a015"],
    )
    parser.add_argument(
        "--base-run",
        default="3dssz_roi_fusion_v8_sweep_hybrid_k6r6_s000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from geocorefusion.config import load_config
    from geocorefusion.dataset import discover_triplet
    from geocorefusion.degradation import PsfModel, degrade_coefficients
    from geocorefusion.envi import open_cube
    from geocorefusion.fusion import _rgb_weights, inject_coefficient_detail
    from geocorefusion.lowrank import load_subspace_model
    from geocorefusion.quality import (
        _degrade_final_modulated_cube,
        _selected_band_detail_metrics,
        sam_degrees,
    )

    run = repo / "runs" / args.base_run
    coefficients, _ = open_cube(run / "coefficients" / "material_coefficients.hdr")
    base_coefficients = np.asarray(coefficients, dtype=np.float32).copy()
    observed, observed_meta = open_cube(run / "analysis" / "harmonized_lowres.hdr")
    observed_cube = np.asarray(observed, dtype=np.float32)
    model = load_subspace_model(run / "metadata" / "subspace_model.json")
    psf_payload = json.loads((run / "metadata" / "psf_model.json").read_text())
    psf = PsfModel(
        sigma_x_highres=float(psf_payload["sigma_x_highres"]),
        sigma_y_highres=float(psf_payload["sigma_y_highres"]),
        score=float(psf_payload["score"]),
        low_shape=tuple(int(v) for v in psf_payload["low_shape"]),
        high_shape=tuple(int(v) for v in psf_payload["high_shape"]),
        method=str(psf_payload["method"]),
    )
    pipeline_config = load_config(repo / "configs" / "3dssz_roi_fusion_v7.yaml")
    dataset = discover_triplet(pipeline_config.data_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    roi = manifest["rgb_roi"]
    rgb = np.asarray(
        dataset.rgb.cube[
            int(roi["y"]) : int(roi["y"]) + int(roi["height"]),
            int(roi["x"]) : int(roi["x"]) + int(roi["width"]),
            :3,
        ]
    )
    wavelengths = np.asarray(observed_meta.wavelengths, dtype=np.float32)
    low_coefficients = degrade_coefficients(base_coefficients, psf)

    results = []
    for name in args.variants:
        strength, active_components, alignment_floor = VARIANTS[name]
        config = load_config(repo / "configs" / "3dssz_roi_fusion_v7.yaml").fusion
        config.rank = 6
        config.coefficient_constraint = "hybrid_simplex"
        config.coefficient_detail_method = "simplex_abundance_rgb_residual"
        config.coefficient_detail_strength = strength
        config.coefficient_detail_nullspace_iterations = 1
        config.coefficient_detail_back_projection_iterations = 2
        config.simplex_abundance_rgb_ridge = 0.02
        config.simplex_abundance_min_r2 = 0.05
        config.simplex_abundance_detail_l1_limit = 0.25
        config.simplex_abundance_active_components = active_components
        config.simplex_abundance_rgb_alignment_floor = alignment_floor
        config.spatial_detail_confidence_gate_low = 0.08
        config.spatial_detail_confidence_gate_high = 0.28
        config.spatial_detail_strength = 0.0
        config.spatial_detail_additive_strength = 0.0
        config.spatial_detail_product_back_projection_iterations = 0
        confidence = _rgb_weights(rgb, config.rgb_edge_sigma, config)[2]
        started = time.perf_counter()
        refined, detail_model = inject_coefficient_detail(
            base_coefficients,
            low_coefficients,
            rgb,
            psf,
            confidence,
            config,
        )
        gain = np.ones(psf.high_shape, dtype=np.float32)
        additive = np.zeros(psf.high_shape, dtype=np.float32)
        additive_scale = np.zeros(model.basis.shape[1], dtype=np.float32)
        selected = {
            int(np.argmin(np.abs(wavelengths - requested)))
            for requested in (900.0, 1650.0, 2200.0)
        }
        low_prediction, final_bands = _degrade_final_modulated_cube(
            refined,
            model,
            psf,
            gain,
            additive,
            additive_scale,
            retained_band_indices=selected,
        )
        detail = _selected_band_detail_metrics(
            refined,
            model,
            rgb,
            wavelengths,
            gain,
            additive,
            additive_scale,
            final_hr_bands=final_bands,
            observed_low_cube=observed_cube,
            psf=psf,
        )
        compact = {}
        for band, report in detail["bands"].items():
            scale = report["multiscale_log_high_frequency"]["sigma_2.4px"]
            reliable = scale["reliable_rgb_detail"]
            dark = scale["dark_reliable_rgb_detail"]
            compact[band] = {
                "rho": reliable["rho"],
                "beta": reliable["beta"],
                "A": reliable["energy_ratio_A"],
                "R_perp": reliable["orthogonal_residual_ratio_R_perp"],
                "dark_beta": dark["beta"],
                "dark_R_perp": dark["orthogonal_residual_ratio_R_perp"],
                "flat_A": scale["rgb_flat"]["energy_ratio_A"],
                "edge_f1": report["gradient_and_edge"]["reliable_rgb_detail"][
                    "edge_f1_1px"
                ],
            }
        item = {
            "variant": name,
            "strength": strength,
            "active_components": active_components,
            "alignment_floor": alignment_floor,
            "elapsed_seconds": time.perf_counter() - started,
            "forward_rmse": float(
                np.sqrt(np.mean((low_prediction - observed_cube) ** 2))
            ),
            "forward_sam_deg": sam_degrees(low_prediction, observed_cube),
            "detail_model": detail_model,
            "detail": compact,
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        del refined, low_prediction, final_bands

    output = repo / "artifacts" / "v7_research" / "evidence" / "v8_cached_abundance_sweep.json"
    output.write_text(json.dumps({"base_run": args.base_run, "runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(results)}))


if __name__ == "__main__":
    main()

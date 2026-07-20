"""Build the V7 real-ROI and synthetic-control evidence package.

Only the six explicitly matched/evaluated run directories are read for the
real-ROI comparison.  The script does not reconstruct imagery or modify any
run.  Same-data RGB/fused-band metrics are kept separate from independent
truth claims, and natural-edge halo/edge-width measurements are labelled as
diagnostic proxies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "geocorefusion.v7_benchmark_summary.v1"
REQUESTED_WAVELENGTHS_NM = (900.0, 1650.0, 2200.0)
VERSIONS = ("v5", "v6", "v7")
RUNS: Mapping[str, Mapping[str, str]] = {
    "3dssz": {
        "v5": "3dssz_roi_fusion_v5_matched_v7eval_ampfix",
        "v6": "3dssz_roi_fusion_v6_v7eval_ampfix",
        "v7": "3dssz_roi_fusion_v7_final_ampfix",
    },
    "zkh3": {
        "v5": "zkh3_roi_fusion_v5_matched_v7eval_ampfix",
        "v6": "zkh3_roi_fusion_v6_v7eval_ampfix",
        "v7": "zkh3_roi_fusion_v7_final_ampfix",
    },
}
VERSION_LABELS = {
    "v5": "V5 matched (V7 evaluator)",
    "v6": "V6 (V7 evaluator)",
    "v7": "V7 final",
}


class EvidenceError(RuntimeError):
    """Raised when an expected evidence field is absent or invalid."""


@dataclass(frozen=True)
class MetricSpec:
    key: str
    group: str
    scope: str
    unit: str
    preferred_direction: str
    target_value: float | None
    definition: str


GLOBAL_METRICS = (
    MetricSpec(
        "final_hr_forward_rmse",
        "final_hr_forward_observation",
        "global_all_bands",
        "normalized_reflectance",
        "lower",
        0.0,
        "RMSE after the complete final HR product is PSF-degraded to the low-resolution observation grid.",
    ),
    MetricSpec(
        "final_hr_forward_sam_mean_deg",
        "final_hr_forward_observation",
        "global_all_bands",
        "degree",
        "lower",
        0.0,
        "Mean spectral angle after the complete final HR product is PSF-degraded to the observation grid.",
    ),
    MetricSpec(
        "final_hr_forward_band_cc_mean",
        "final_hr_forward_observation",
        "global_all_bands",
        "correlation",
        "higher",
        1.0,
        "Mean per-band correlation on the degraded observation grid.",
    ),
)

BAND_METRICS = (
    MetricSpec(
        "rho_reliable",
        "coherent_log_high_frequency",
        "reliable_rgb_detail_sigma_2.4px",
        "correlation",
        "higher",
        1.0,
        "Pearson correlation of signed native-log high-frequency detail in reliable RGB-detail pixels; no candidate-specific percentile stretch.",
    ),
    MetricSpec(
        "beta_reliable",
        "coherent_log_high_frequency",
        "reliable_rgb_detail_sigma_2.4px",
        "ratio",
        "target",
        1.0,
        "Least-squares native-log band-detail amplitude relative to RGB detail; unity is the RGB-equivalent relative-contrast target. RGB and the candidate are not independently stretched.",
    ),
    MetricSpec(
        "energy_ratio_A_reliable",
        "coherent_log_high_frequency",
        "reliable_rgb_detail_sigma_2.4px",
        "ratio",
        "target",
        1.0,
        "Native-log band/RGB high-frequency standard-deviation ratio; unity is the RGB-equivalent energy target and candidate-specific stretching is forbidden.",
    ),
    MetricSpec(
        "orthogonal_residual_ratio_R_perp_reliable",
        "coherent_log_high_frequency",
        "reliable_rgb_detail_sigma_2.4px",
        "ratio",
        "lower",
        0.0,
        "Band-detail residual after removing the RGB-aligned component, normalized by RGB-detail deviation.",
    ),
    MetricSpec(
        "rho_dark_reliable",
        "dark_region_log_high_frequency",
        "dark_reliable_rgb_detail_sigma_2.4px",
        "correlation",
        "higher",
        1.0,
        "Signed log-detail correlation in pixels that are both dark and reliable in the RGB guide.",
    ),
    MetricSpec(
        "beta_dark_reliable",
        "dark_region_log_high_frequency",
        "dark_reliable_rgb_detail_sigma_2.4px",
        "ratio",
        "target",
        1.0,
        "Dark-region least-squares detail amplitude relative to RGB detail.",
    ),
    MetricSpec(
        "energy_ratio_A_dark_reliable",
        "dark_region_log_high_frequency",
        "dark_reliable_rgb_detail_sigma_2.4px",
        "ratio",
        "target",
        1.0,
        "Dark-region band/RGB high-frequency energy ratio.",
    ),
    MetricSpec(
        "orthogonal_residual_ratio_R_perp_dark_reliable",
        "dark_region_log_high_frequency",
        "dark_reliable_rgb_detail_sigma_2.4px",
        "ratio",
        "lower",
        0.0,
        "Dark-region non-RGB-aligned high-frequency residual ratio.",
    ),
    MetricSpec(
        "energy_ratio_A_rgb_flat",
        "flat_region_artifact_screen",
        "rgb_flat_sigma_2.4px",
        "ratio",
        "lower",
        0.0,
        "Band/RGB high-frequency energy ratio where the RGB guide is flat; lower indicates less unsupported texture.",
    ),
    MetricSpec(
        "edge_f1_1px_reliable",
        "gradient_and_edge",
        "reliable_rgb_detail",
        "score",
        "higher",
        1.0,
        "Symmetric top-decile edge overlap after one-pixel dilation.",
    ),
    MetricSpec(
        "gradient_orientation_coherence_reliable",
        "gradient_and_edge",
        "reliable_rgb_detail",
        "absolute_cosine",
        "higher",
        1.0,
        "Absolute gradient-direction cosine on strong RGB edges, allowing wavelength-dependent contrast reversal.",
    ),
    MetricSpec(
        "edge_f1_1px_dark_reliable",
        "dark_region_gradient_and_edge",
        "dark_reliable_rgb_detail",
        "score",
        "higher",
        1.0,
        "One-pixel edge F1 restricted to dark, reliable RGB-detail pixels.",
    ),
    MetricSpec(
        "gradient_orientation_coherence_dark_reliable",
        "dark_region_gradient_and_edge",
        "dark_reliable_rgb_detail",
        "absolute_cosine",
        "higher",
        1.0,
        "Gradient orientation coherence in dark, reliable RGB-detail pixels.",
    ),
    MetricSpec(
        "halo_overshoot_p50_edge_step",
        "natural_edge_proxy",
        "reliable_rgb_detail_natural_edges",
        "edge_step_fraction",
        "lower",
        0.0,
        "Median normalized overshoot plus undershoot on natural edge profiles; diagnostic proxy only.",
    ),
    MetricSpec(
        "halo_overshoot_p95_edge_step",
        "natural_edge_proxy",
        "reliable_rgb_detail_natural_edges",
        "edge_step_fraction",
        "lower",
        0.0,
        "95th-percentile normalized overshoot plus undershoot on natural edge profiles; diagnostic proxy only.",
    ),
    MetricSpec(
        "reference_edge_width_10_90_px",
        "natural_edge_proxy",
        "reliable_rgb_detail_natural_edges",
        "pixel",
        "none",
        None,
        "RGB-reference 10-90% natural-edge width; not a calibrated slanted-edge MTF measurement.",
    ),
    MetricSpec(
        "candidate_edge_width_10_90_px",
        "natural_edge_proxy",
        "reliable_rgb_detail_natural_edges",
        "pixel",
        "none",
        None,
        "Fused-band 10-90% natural-edge width; interpret jointly with the reference width and halo proxy.",
    ),
    MetricSpec(
        "edge_width_ratio",
        "natural_edge_proxy",
        "reliable_rgb_detail_natural_edges",
        "ratio",
        "target",
        1.0,
        "Candidate/reference natural-edge width ratio; unity is the matched-width target, but this is only a proxy.",
    ),
    MetricSpec(
        "edge_profile_count",
        "natural_edge_proxy",
        "reliable_rgb_detail_natural_edges",
        "count",
        "none",
        None,
        "Number of natural edge profiles contributing to the edge-width/halo diagnostic.",
    ),
)

METRIC_BY_KEY = {spec.key: spec for spec in (*GLOBAL_METRICS, *BAND_METRICS)}

REAL_TRUTH_SCOPE = (
    "Low-resolution observation consistency plus same-data RGB structural-transfer diagnostics; "
    "not independent high-resolution NIR/SWIR truth."
)
HALO_LIMIT = (
    "Natural-edge halo and edge-width values are diagnostic proxies; publishable MTF/halo claims "
    "require a calibrated slanted-edge or equivalent controlled target."
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"Required JSON does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"Expected a JSON object in {path}")
    return value


def nested(payload: Mapping[str, Any], path: Sequence[str], source: Path) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise EvidenceError(f"Missing {'.'.join(path)} in {source}")
        value = value[key]
    return value


def finite_float(value: Any, label: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not numeric in {source}: {value!r}") from exc
    if not math.isfinite(result):
        raise EvidenceError(f"{label} is not finite in {source}: {result!r}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def parse_wavelength_label(label: str, source: Path) -> float:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)nm\s*", label)
    if not match:
        raise EvidenceError(f"Invalid wavelength label {label!r} in {source}")
    return float(match.group(1))


def choose_band_label(
    bands: Mapping[str, Any], requested_nm: float, source: Path, tolerance_nm: float = 5.0
) -> tuple[str, float]:
    candidates = [(label, parse_wavelength_label(str(label), source)) for label in bands]
    if not candidates:
        raise EvidenceError(f"No selected-band metrics in {source}")
    label, actual = min(candidates, key=lambda item: abs(item[1] - requested_nm))
    if abs(actual - requested_nm) > tolerance_nm:
        raise EvidenceError(
            f"Nearest selected band to {requested_nm:g} nm is {actual:g} nm in {source}"
        )
    return str(label), actual


def extract_band_metrics(
    band: Mapping[str, Any], selected_scale: str, source: Path
) -> dict[str, float]:
    scale_metrics = nested(
        band, ("multiscale_log_high_frequency", selected_scale), source
    )
    reliable = nested(scale_metrics, ("reliable_rgb_detail",), source)
    dark = nested(scale_metrics, ("dark_reliable_rgb_detail",), source)
    flat = nested(scale_metrics, ("rgb_flat",), source)
    edges = nested(band, ("gradient_and_edge",), source)
    reliable_edge = nested(edges, ("reliable_rgb_detail",), source)
    dark_edge = nested(edges, ("dark_reliable_rgb_detail",), source)
    halo = nested(band, ("halo_and_edge_spread_proxy",), source)
    raw = {
        "rho_reliable": nested(reliable, ("rho",), source),
        "beta_reliable": nested(reliable, ("beta",), source),
        "energy_ratio_A_reliable": nested(reliable, ("energy_ratio_A",), source),
        "orthogonal_residual_ratio_R_perp_reliable": nested(
            reliable, ("orthogonal_residual_ratio_R_perp",), source
        ),
        "rho_dark_reliable": nested(dark, ("rho",), source),
        "beta_dark_reliable": nested(dark, ("beta",), source),
        "energy_ratio_A_dark_reliable": nested(dark, ("energy_ratio_A",), source),
        "orthogonal_residual_ratio_R_perp_dark_reliable": nested(
            dark, ("orthogonal_residual_ratio_R_perp",), source
        ),
        "energy_ratio_A_rgb_flat": nested(flat, ("energy_ratio_A",), source),
        "edge_f1_1px_reliable": nested(reliable_edge, ("edge_f1_1px",), source),
        "gradient_orientation_coherence_reliable": nested(
            reliable_edge, ("gradient_orientation_coherence_abs_cosine",), source
        ),
        "edge_f1_1px_dark_reliable": nested(dark_edge, ("edge_f1_1px",), source),
        "gradient_orientation_coherence_dark_reliable": nested(
            dark_edge, ("gradient_orientation_coherence_abs_cosine",), source
        ),
        "halo_overshoot_p50_edge_step": nested(
            halo, ("overshoot_plus_undershoot_p50_edge_step",), source
        ),
        "halo_overshoot_p95_edge_step": nested(
            halo, ("overshoot_plus_undershoot_p95_edge_step",), source
        ),
        "reference_edge_width_10_90_px": nested(
            halo, ("reference_edge_width_10_90_px",), source
        ),
        "candidate_edge_width_10_90_px": nested(
            halo, ("candidate_edge_width_10_90_px",), source
        ),
        "edge_width_ratio": nested(halo, ("edge_width_ratio",), source),
        "edge_profile_count": nested(halo, ("edge_profile_count",), source),
    }
    return {
        key: finite_float(value, key, source)
        for key, value in raw.items()
    }


def extract_run(repo: Path, scene: str, version: str, run_name: str) -> dict[str, Any]:
    run_dir = repo / "runs" / run_name
    report_path = run_dir / "metrics" / "quality_report.json"
    manifest_path = run_dir / "manifest.json"
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    summary = nested(report, ("summary",), report_path)
    forward = nested(report, ("final_hr_product_observation",), report_path)
    detail = nested(report, ("spatial", "band_detail_by_brightness"), report_path)
    selected_scale = str(nested(detail, ("selected_scale_for_screening",), report_path))
    bands = nested(detail, ("bands",), report_path)
    if not isinstance(bands, Mapping):
        raise EvidenceError(f"Selected band metrics are not an object in {report_path}")

    global_values = {
        "final_hr_forward_rmse": finite_float(
            nested(forward, ("rmse",), report_path), "forward RMSE", report_path
        ),
        "final_hr_forward_sam_mean_deg": finite_float(
            nested(forward, ("sam_mean_deg",), report_path), "forward SAM", report_path
        ),
        "final_hr_forward_band_cc_mean": finite_float(
            nested(forward, ("band_cc_mean",), report_path), "forward band CC", report_path
        ),
    }
    selected_bands: dict[str, Any] = {}
    for requested in REQUESTED_WAVELENGTHS_NM:
        label, actual = choose_band_label(bands, requested, report_path)
        band = bands[label]
        if not isinstance(band, Mapping):
            raise EvidenceError(f"Band {label} metrics are not an object in {report_path}")
        selected_bands[f"{requested:g}"] = {
            "requested_wavelength_nm": requested,
            "actual_wavelength_nm": actual,
            "report_label": label,
            "selected_scale": selected_scale,
            "metrics": extract_band_metrics(band, selected_scale, report_path),
            "screening_status": str(
                nested(band, ("conservative_screening", "screening_status"), report_path)
            ),
            "claim_status": str(
                nested(band, ("conservative_screening", "claim_status"), report_path)
            ),
            "screening_checks": nested(
                band, ("conservative_screening", "checks"), report_path
            ),
            "halo_status": str(
                nested(band, ("halo_and_edge_spread_proxy", "status"), report_path)
            ),
        }

    output_grid = nested(manifest, ("output_grid",), manifest_path)
    rgb_roi = nested(manifest, ("rgb_roi",), manifest_path)
    project = nested(manifest, ("project",), manifest_path)
    return {
        "scene": scene,
        "version": version,
        "version_label": VERSION_LABELS[version],
        "run_name": run_name,
        "run_directory": relative_path(run_dir, repo),
        "created_at": manifest.get("created_at"),
        "software_version": manifest.get("software_version"),
        "project": {
            "borehole_id": project.get("borehole_id"),
            "core_interval": project.get("core_interval"),
            "notes": project.get("notes"),
        },
        "rgb_roi": rgb_roi,
        "output_grid": output_grid,
        "sources": {
            "quality_report": {
                "path": relative_path(report_path, repo),
                "sha256": sha256(report_path),
            },
            "manifest": {
                "path": relative_path(manifest_path, repo),
                "sha256": sha256(manifest_path),
            },
        },
        "summary_status": nested(summary, ("status",), report_path),
        # The V5/V6 controls were evaluated with the exact final-product
        # forward path but were written immediately before these two summary
        # aliases were added.  Keep an absent field absent instead of
        # rewriting historical evidence; the forward.truth_scope field below
        # remains the authoritative scope for all six runs.
        "summary_status_scope": summary.get("status_scope"),
        "independent_hr_hsi_truth_status": summary.get(
            "independent_hr_hsi_truth_status"
        ),
        "forward": {
            **global_values,
            "method": nested(forward, ("method",), report_path),
            "truth_scope": nested(forward, ("truth_scope",), report_path),
        },
        "selected_scale_for_screening": selected_scale,
        "bands": selected_bands,
    }


def raw_percent_change(new: float, baseline: float) -> float | None:
    if abs(baseline) <= 1e-12:
        return None
    return 100.0 * (new - baseline) / abs(baseline)


def directional_improvement(
    new: float, baseline: float, spec: MetricSpec
) -> float | None:
    raw = raw_percent_change(new, baseline)
    if spec.preferred_direction == "higher":
        return raw
    if spec.preferred_direction == "lower":
        return None if raw is None else -raw
    if spec.preferred_direction == "target":
        if spec.target_value is None:
            return None
        baseline_error = abs(baseline - spec.target_value)
        if baseline_error <= 1e-12:
            return None
        new_error = abs(new - spec.target_value)
        return 100.0 * (baseline_error - new_error) / baseline_error
    return None


def comparison(new: float, baseline: float, spec: MetricSpec) -> dict[str, Any]:
    return {
        "baseline_value": baseline,
        "v7_value": new,
        "raw_percent_change": raw_percent_change(new, baseline),
        "directional_improvement_percent": directional_improvement(new, baseline, spec),
        "preferred_direction": spec.preferred_direction,
        "target_value": spec.target_value,
    }


def build_comparisons(scene_runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    v7 = scene_runs["v7"]
    result: dict[str, Any] = {}
    for baseline_version in ("v5", "v6"):
        baseline = scene_runs[baseline_version]
        forward: dict[str, Any] = {}
        for spec in GLOBAL_METRICS:
            forward[spec.key] = comparison(
                float(v7["forward"][spec.key]),
                float(baseline["forward"][spec.key]),
                spec,
            )
        bands: dict[str, Any] = {}
        for requested in REQUESTED_WAVELENGTHS_NM:
            key = f"{requested:g}"
            band_result: dict[str, Any] = {}
            for spec in BAND_METRICS:
                band_result[spec.key] = comparison(
                    float(v7["bands"][key]["metrics"][spec.key]),
                    float(baseline["bands"][key]["metrics"][spec.key]),
                    spec,
                )
            bands[key] = band_result
        result[baseline_version] = {"forward": forward, "bands": bands}
    return result


def validate_real_roi(real_roi: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected_method_prefix = "final_hr_unbounded_subspace_then_gain_then_additive_then_single_clip"
    for scene, scene_payload in real_roi.items():
        runs = scene_payload["runs"]
        reference_grid = runs["v7"]["output_grid"]
        reference_roi = runs["v7"]["rgb_roi"]
        checks.append(
            {
                "check": f"{scene}: matched output grid",
                "passed": all(runs[v]["output_grid"] == reference_grid for v in VERSIONS),
            }
        )
        checks.append(
            {
                "check": f"{scene}: matched RGB ROI",
                "passed": all(runs[v]["rgb_roi"] == reference_roi for v in VERSIONS),
            }
        )
        checks.append(
            {
                "check": f"{scene}: exact final-product forward path",
                "passed": all(
                    str(runs[v]["forward"]["method"]).startswith(expected_method_prefix)
                    for v in VERSIONS
                ),
            }
        )
        checks.append(
            {
                "check": f"{scene}: independent HR-HSI truth is not claimed",
                "passed": all(
                    "not_independent_hr_hsi_truth"
                    in str(runs[v]["forward"]["truth_scope"])
                    for v in VERSIONS
                ),
            }
        )
        checks.append(
            {
                "check": f"{scene}: identical selected metric scale",
                "passed": len(
                    {runs[v]["selected_scale_for_screening"] for v in VERSIONS}
                )
                == 1,
            }
        )
        for requested in REQUESTED_WAVELENGTHS_NM:
            key = f"{requested:g}"
            actuals = {runs[v]["bands"][key]["actual_wavelength_nm"] for v in VERSIONS}
            checks.append(
                {
                    "check": f"{scene}/{requested:g} nm: matched actual band",
                    "passed": len(actuals) == 1,
                    "values": sorted(actuals),
                }
            )
    failed = [item for item in checks if not item["passed"]]
    if failed:
        raise EvidenceError(f"Real-ROI evidence validation failed: {failed}")
    return checks


def run_synthetic_contract_tests(repo: Path) -> dict[str, Any]:
    targets = [
        "tests/test_fusion.py",
        "tests/test_quality_detail.py",
        "tests/test_output.py",
    ]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-q",
        "-p",
        "no:cacheprovider",
        *targets,
    ]
    env = os.environ.copy()
    source_path = str(repo / "src")
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration = time.perf_counter() - started
    output = completed.stdout.strip()
    match = re.search(r"(?P<count>[0-9]+) passed", output)
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "passed_count": int(match.group("count")) if match else None,
        "duration_seconds": duration,
        "command": command,
        "targets": targets,
        "stdout": output[-6000:],
        "truth_scope": (
            "Controlled synthetic/unit contracts for local material signs, isoluminant colour texture, "
            "dark-noise rejection, single clipping, native-grid preview ordering, exact final-product "
            "degradation, detail-statistic closure, and halo-proxy response. Test passage is not "
            "independent real-scene HR-SWIR validation."
        ),
    }


def load_synthetic_evidence(repo: Path, run_tests: bool) -> dict[str, Any]:
    legacy_path = repo / "artifacts" / "v6_research" / "experiments" / "benchmark_summary.json"
    if legacy_path.is_file():
        legacy = read_json(legacy_path)
        legacy_payload: dict[str, Any] = {
            "status": "available",
            "source": {
                "path": relative_path(legacy_path, repo),
                "sha256": sha256(legacy_path),
                "generated_at": legacy.get("generated_at"),
            },
            "registration_synthetic": legacy.get("registration_synthetic"),
            "fusion_synthetic": legacy.get("fusion_synthetic"),
            "scope_note": legacy.get("evidence_limit"),
            "v7_numeric_entry_status": "not_present_in_legacy_v5_v6_benchmark",
        }
    else:
        legacy_payload = {
            "status": "not_found",
            "expected_path": relative_path(legacy_path, repo),
        }
    contract_tests = (
        run_synthetic_contract_tests(repo)
        if run_tests
        else {
            "status": "not_run",
            "note": "Use --run-synthetic-tests to execute the existing V7 synthetic/unit contracts.",
        }
    )
    return {
        "legacy_seeded_numeric_benchmark": legacy_payload,
        "v7_synthetic_contract_tests": contract_tests,
    }


def metric_row(
    *,
    evidence_type: str,
    scene_or_experiment: str,
    method: str,
    run_name: str,
    requested_wavelength_nm: float | None,
    actual_wavelength_nm: float | None,
    spec: MetricSpec,
    value: float,
    truth_scope: str,
    source_path: str,
    note: str,
    v7_vs_v5: Mapping[str, Any] | None = None,
    v7_vs_v6: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_type": evidence_type,
        "scene_or_experiment": scene_or_experiment,
        "method": method,
        "run_name": run_name,
        "requested_wavelength_nm": requested_wavelength_nm,
        "actual_wavelength_nm": actual_wavelength_nm,
        "metric_group": spec.group,
        "metric": spec.key,
        "scope": spec.scope,
        "value": value,
        "unit": spec.unit,
        "preferred_direction": spec.preferred_direction,
        "target_value": spec.target_value,
        "v7_vs_v5_raw_percent_change": (
            None if v7_vs_v5 is None else v7_vs_v5["raw_percent_change"]
        ),
        "v7_vs_v5_directional_improvement_percent": (
            None if v7_vs_v5 is None else v7_vs_v5["directional_improvement_percent"]
        ),
        "v7_vs_v6_raw_percent_change": (
            None if v7_vs_v6 is None else v7_vs_v6["raw_percent_change"]
        ),
        "v7_vs_v6_directional_improvement_percent": (
            None if v7_vs_v6 is None else v7_vs_v6["directional_improvement_percent"]
        ),
        "truth_scope": truth_scope,
        "source_path": source_path,
        "note": note,
    }


def real_csv_rows(real_roi: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene, scene_payload in real_roi.items():
        runs = scene_payload["runs"]
        comparisons = scene_payload["v7_relative_to"]
        for version in VERSIONS:
            run = runs[version]
            source = run["sources"]["quality_report"]["path"]
            for spec in GLOBAL_METRICS:
                rows.append(
                    metric_row(
                        evidence_type="real_roi_same_data",
                        scene_or_experiment=scene,
                        method=VERSION_LABELS[version],
                        run_name=run["run_name"],
                        requested_wavelength_nm=None,
                        actual_wavelength_nm=None,
                        spec=spec,
                        value=float(run["forward"][spec.key]),
                        truth_scope=str(run["forward"]["truth_scope"]),
                        source_path=source,
                        note="Global all-band forward metric; not wavelength-specific.",
                        v7_vs_v5=(
                            comparisons["v5"]["forward"][spec.key]
                            if version == "v7"
                            else None
                        ),
                        v7_vs_v6=(
                            comparisons["v6"]["forward"][spec.key]
                            if version == "v7"
                            else None
                        ),
                    )
                )
            for requested in REQUESTED_WAVELENGTHS_NM:
                key = f"{requested:g}"
                band = run["bands"][key]
                for spec in BAND_METRICS:
                    rows.append(
                        metric_row(
                            evidence_type="real_roi_same_data",
                            scene_or_experiment=scene,
                            method=VERSION_LABELS[version],
                            run_name=run["run_name"],
                            requested_wavelength_nm=requested,
                            actual_wavelength_nm=float(band["actual_wavelength_nm"]),
                            spec=spec,
                            value=float(band["metrics"][spec.key]),
                            truth_scope=REAL_TRUTH_SCOPE,
                            source_path=source,
                            note=(
                                HALO_LIMIT
                                if spec.group == "natural_edge_proxy"
                                else f"Selected scale: {band['selected_scale']}."
                            ),
                            v7_vs_v5=(
                                comparisons["v5"]["bands"][key][spec.key]
                                if version == "v7"
                                else None
                            ),
                            v7_vs_v6=(
                                comparisons["v6"]["bands"][key][spec.key]
                                if version == "v7"
                                else None
                            ),
                        )
                    )
    return rows


def synthetic_csv_rows(synthetic: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    legacy = synthetic["legacy_seeded_numeric_benchmark"]
    if legacy.get("status") == "available":
        source = legacy["source"]["path"]
        registration = legacy.get("registration_synthetic") or {}
        registration_scope = str(registration.get("truth_scope", "seeded synthetic"))
        for block_name in ("coarse_affine_tre", "roi_affine_tre", "dense_residual_epe"):
            block = registration.get(block_name) or {}
            for statistic in ("mean_px", "median_px", "p95_px", "max_px"):
                if statistic not in block:
                    continue
                spec = MetricSpec(
                    f"{block_name}_{statistic}",
                    "legacy_seeded_registration",
                    block_name,
                    "analysis_grid_pixel",
                    "lower",
                    0.0,
                    "Known-truth synthetic registration error from the legacy seeded V6 benchmark.",
                )
                rows.append(
                    metric_row(
                        evidence_type="synthetic_seeded_numeric",
                        scene_or_experiment="registration_synthetic",
                        method="legacy V6 benchmark",
                        run_name="",
                        requested_wavelength_nm=None,
                        actual_wavelength_nm=None,
                        spec=spec,
                        value=float(block[statistic]),
                        truth_scope=registration_scope,
                        source_path=source,
                        note="This source predates V7 and is not a production-configuration replay.",
                    )
                )
        fusion = legacy.get("fusion_synthetic") or {}
        fusion_scope = str(fusion.get("truth_scope", "seeded synthetic"))
        fusion_specs = {
            "dark_log_detail_correlation": MetricSpec(
                "dark_log_detail_correlation",
                "legacy_seeded_fusion",
                "synthetic_dark_region",
                "correlation",
                "higher",
                1.0,
                "Known-truth dark log-detail correlation in the legacy V5/V6 synthetic benchmark.",
            ),
            "coefficient_observation_rmse": MetricSpec(
                "coefficient_observation_rmse",
                "legacy_seeded_fusion",
                "synthetic_low_resolution_observation",
                "coefficient_unit",
                "lower",
                0.0,
                "Coefficient-domain observation RMSE in the legacy V5/V6 synthetic benchmark.",
            ),
            "gain_observation_rmse": MetricSpec(
                "gain_observation_rmse",
                "legacy_seeded_fusion",
                "synthetic_low_resolution_observation",
                "gain",
                "lower",
                0.0,
                "Low-resolution gain residual in the legacy V5/V6 synthetic benchmark.",
            ),
        }
        for method_key in ("v5_style", "v6_intrinsic"):
            method_payload = fusion.get(method_key) or {}
            for metric_key, spec in fusion_specs.items():
                if metric_key not in method_payload:
                    continue
                rows.append(
                    metric_row(
                        evidence_type="synthetic_seeded_numeric",
                        scene_or_experiment="fusion_synthetic",
                        method=method_key,
                        run_name="",
                        requested_wavelength_nm=None,
                        actual_wavelength_nm=None,
                        spec=spec,
                        value=float(method_payload[metric_key]),
                        truth_scope=fusion_scope,
                        source_path=source,
                        note="Legacy V5/V6 numeric control; no V7 numeric entry is present in this file.",
                    )
                )
    tests = synthetic["v7_synthetic_contract_tests"]
    if tests.get("status") in {"passed", "failed"} and tests.get("passed_count") is not None:
        spec = MetricSpec(
            "passed_test_count",
            "v7_synthetic_contract_tests",
            "targeted_pytest_files",
            "count",
            "none",
            None,
            "Number of passing existing V7 synthetic/unit contract tests in the targeted files.",
        )
        rows.append(
            metric_row(
                evidence_type="synthetic_contract_test",
                scene_or_experiment="v7_contract_tests",
                method="V7 current code",
                run_name="",
                requested_wavelength_nm=None,
                actual_wavelength_nm=None,
                spec=spec,
                value=float(tests["passed_count"]),
                truth_scope=str(tests["truth_scope"]),
                source_path=";".join(tests["targets"]),
                note=f"pytest status={tests['status']}; return_code={tests['return_code']}.",
            )
        )
    return rows


def format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvidenceError("No rows were produced for the CSV summary")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key)) for key in fieldnames})


def md_number(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def md_percent(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):+.1f}%"


def improvement_counts(
    real_roi: Mapping[str, Mapping[str, Any]], metric_key: str, baseline: str
) -> tuple[int, int]:
    improved = 0
    evaluated = 0
    for scene_payload in real_roi.values():
        for requested in REQUESTED_WAVELENGTHS_NM:
            comparison_payload = scene_payload["v7_relative_to"][baseline]["bands"][
                f"{requested:g}"
            ][metric_key]
            value = comparison_payload["directional_improvement_percent"]
            if value is None:
                continue
            evaluated += 1
            improved += int(float(value) > 0.0)
    return improved, evaluated


def summarize_v7_screening(
    real_roi: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    evaluated_bands = 0
    within_bounds = 0
    failed_check_counts: dict[str, int] = {}
    evaluated_check_counts: dict[str, int] = {}
    for scene_payload in real_roi.values():
        run = scene_payload["runs"]["v7"]
        for requested in REQUESTED_WAVELENGTHS_NM:
            band = run["bands"][f"{requested:g}"]
            evaluated_bands += 1
            within_bounds += int(
                band["screening_status"] != "outside_conservative_screening_bounds"
            )
            for check_name, check in band["screening_checks"].items():
                if not bool(check.get("evaluated")):
                    continue
                evaluated_check_counts[check_name] = (
                    evaluated_check_counts.get(check_name, 0) + 1
                )
                if not bool(check.get("within_bound")):
                    failed_check_counts[check_name] = (
                        failed_check_counts.get(check_name, 0) + 1
                    )
    return {
        "status": (
            "within_conservative_bounds"
            if within_bounds == evaluated_bands
            else "outside_conservative_screening_bounds"
        ),
        "bands_within_all_bounds": within_bounds,
        "bands_evaluated": evaluated_bands,
        "failed_check_counts": dict(sorted(failed_check_counts.items())),
        "evaluated_check_counts": dict(sorted(evaluated_check_counts.items())),
        "interpretation": (
            "A conservative screening warning does not negate low-resolution forward consistency, "
            "but it prevents an RGB-equivalent/no-loss detail claim."
        ),
    }


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_interpretation(payload: Mapping[str, Any]) -> str:
    real_roi = payload["real_roi"]
    lines = [
        "# V7 融合实验证据解释",
        "",
        "## 结论先行",
        "",
        (
            "V7 在两个真实 ROI 的最终产品 forward consistency 上均优于 V5 matched 和 V6，"
            "尤其 ZKH3 的 RMSE/SAM 改善明显；但 RGB—融合波段细节闭环是混合结果，不能据此声称"
            "已经实现‘真实 SWIR 细节无损’。这些 RGB 指标使用同一幅 RGB 作为结构引导和结构参考，"
            "没有独立高分辨率 NIR/SWIR 真值。"
        ),
        "",
        "## 最终 HR 产品回观测网格的一致性",
        "",
        (
            "RMSE 和 SAM 均对完整最终 HR 产品执行 PSF 退化后计算，覆盖增益、加性细节和单次物理裁剪。"
            "它们是全波段、逐场景指标，并不是 900/1650/2200 nm 的独立真值误差。方向性改善率为正表示更好。"
        ),
        "",
    ]
    forward_rows: list[list[str]] = []
    for scene, scene_payload in real_roi.items():
        runs = scene_payload["runs"]
        comparisons = scene_payload["v7_relative_to"]
        forward_rows.append(
            [
                scene.upper(),
                md_number(runs["v5"]["forward"]["final_hr_forward_rmse"], 6),
                md_number(runs["v6"]["forward"]["final_hr_forward_rmse"], 6),
                md_number(runs["v7"]["forward"]["final_hr_forward_rmse"], 6),
                md_percent(
                    comparisons["v5"]["forward"]["final_hr_forward_rmse"][
                        "directional_improvement_percent"
                    ]
                ),
                md_percent(
                    comparisons["v6"]["forward"]["final_hr_forward_rmse"][
                        "directional_improvement_percent"
                    ]
                ),
                md_number(runs["v5"]["forward"]["final_hr_forward_sam_mean_deg"], 4),
                md_number(runs["v6"]["forward"]["final_hr_forward_sam_mean_deg"], 4),
                md_number(runs["v7"]["forward"]["final_hr_forward_sam_mean_deg"], 4),
                md_percent(
                    comparisons["v5"]["forward"]["final_hr_forward_sam_mean_deg"][
                        "directional_improvement_percent"
                    ]
                ),
                md_percent(
                    comparisons["v6"]["forward"]["final_hr_forward_sam_mean_deg"][
                        "directional_improvement_percent"
                    ]
                ),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "场景",
                    "RMSE V5",
                    "RMSE V6",
                    "RMSE V7",
                    "V7 vs V5",
                    "V7 vs V6",
                    "SAM° V5",
                    "SAM° V6",
                    "SAM° V7",
                    "V7 vs V5",
                    "V7 vs V6",
                ],
                forward_rows,
            ),
            "",
            "## 900/1650/2200 nm 的 V7 细节闭环",
            "",
            (
                "下表使用每个报告指定的 `sigma_2.4px`。ρ、β、A、R⊥、flat A 分别表示相干性、"
                "回归幅度、能量比、非相干残差和 RGB 平坦区高频能量比。RGB 与融合波段均保留"
                "原生归一化/log 反射率单位，禁止分别做候选相关的百分位拉伸。β/A/边宽比的目标是 1，"
                "R⊥ 与 flat A 越低越好；自然边缘 halo 只作风险代理，不进入真实数据硬筛查。"
            ),
            "",
        ]
    )
    detail_rows: list[list[str]] = []
    for scene, scene_payload in real_roi.items():
        run = scene_payload["runs"]["v7"]
        for requested in REQUESTED_WAVELENGTHS_NM:
            band = run["bands"][f"{requested:g}"]
            values = band["metrics"]
            detail_rows.append(
                [
                    scene.upper(),
                    f"{band['actual_wavelength_nm']:.0f}",
                    md_number(values["rho_reliable"], 3),
                    md_number(values["beta_reliable"], 3),
                    md_number(values["energy_ratio_A_reliable"], 3),
                    md_number(values["orthogonal_residual_ratio_R_perp_reliable"], 3),
                    md_number(values["rho_dark_reliable"], 3),
                    md_number(values["energy_ratio_A_rgb_flat"], 3),
                    md_number(values["edge_f1_1px_reliable"], 3),
                    md_number(values["gradient_orientation_coherence_reliable"], 3),
                    md_number(values["edge_width_ratio"], 3),
                    md_number(values["halo_overshoot_p95_edge_step"], 3),
                ]
            )
    lines.extend(
        [
            markdown_table(
                [
                    "场景",
                    "实际 nm",
                    "ρ",
                    "β",
                    "A",
                    "R⊥",
                    "暗区 ρ",
                    "flat A",
                    "Edge F1",
                    "梯度方向",
                    "边宽比",
                    "Halo P95",
                ],
                detail_rows,
            ),
            "",
            (
                "保守筛查结果：V7 仅有 "
                f"{payload['validation']['v7_conservative_screening']['bands_within_all_bounds']}/"
                f"{payload['validation']['v7_conservative_screening']['bands_evaluated']} 个场景—波段组合"
                "同时落入全部预设边界。该警告不否定低分辨率 forward consistency，"
                "但明确阻止‘RGB 等幅无损/无伪影’结论。"
            ),
            "",
            "## V7 相对基线的逐波段改善覆盖率",
            "",
            (
                "覆盖率统计两个场景 × 三个波段，共 6 个比较点；只判断方向性改善率是否大于 0，"
                "不把小幅数值变化自动解释为统计显著。完整逐点百分比位于 CSV/JSON。"
            ),
            "",
        ]
    )
    coverage_metrics = (
        ("rho_reliable", "ρ 更高"),
        ("beta_reliable", "β 更接近 1"),
        ("energy_ratio_A_reliable", "A 更接近 1"),
        ("orthogonal_residual_ratio_R_perp_reliable", "R⊥ 更低"),
        ("rho_dark_reliable", "暗区 ρ 更高"),
        ("energy_ratio_A_rgb_flat", "flat A 更低"),
        ("edge_f1_1px_reliable", "Edge F1 更高"),
        ("gradient_orientation_coherence_reliable", "梯度方向更高"),
        ("edge_width_ratio", "边宽比更接近 1"),
        ("halo_overshoot_p95_edge_step", "Halo P95 更低"),
    )
    coverage_rows: list[list[str]] = []
    for metric_key, label in coverage_metrics:
        v5_improved, v5_total = improvement_counts(real_roi, metric_key, "v5")
        v6_improved, v6_total = improvement_counts(real_roi, metric_key, "v6")
        coverage_rows.append(
            [label, f"{v5_improved}/{v5_total}", f"{v6_improved}/{v6_total}"]
        )
    lines.extend(
        [
            markdown_table(["判据", "V7 优于 V5", "V7 优于 V6"], coverage_rows),
            "",
            "## 合成证据",
            "",
        ]
    )
    synthetic = payload["synthetic_evidence"]
    legacy = synthetic["legacy_seeded_numeric_benchmark"]
    if legacy.get("status") == "available":
        registration = legacy.get("registration_synthetic") or {}
        fusion = legacy.get("fusion_synthetic") or {}
        lines.extend(
            [
                (
                    "既有 seeded registration 控制给出：coarse affine TRE P95 = "
                    f"{md_number((registration.get('coarse_affine_tre') or {}).get('p95_px'), 3)} px，"
                    "ROI affine TRE P95 = "
                    f"{md_number((registration.get('roi_affine_tre') or {}).get('p95_px'), 3)} px，"
                    "dense residual EPE P95 = "
                    f"{md_number((registration.get('dense_residual_epe') or {}).get('p95_px'), 3)} px。"
                ),
                "",
            ]
        )
        v5 = fusion.get("v5_style") or {}
        v6 = fusion.get("v6_intrinsic") or {}
        if v5 and v6:
            lines.extend(
                [
                    (
                        "既有 seeded fusion 数值只覆盖 V5-style/V6-intrinsic：暗区 log-detail 相关由 "
                        f"{md_number(v5.get('dark_log_detail_correlation'), 4)} 提升到 "
                        f"{md_number(v6.get('dark_log_detail_correlation'), 4)}。该旧文件没有 V7 数值项，"
                        "因此不能把它当作 V7 的合成真值优势。"
                    ),
                    "",
                ]
            )
    tests = synthetic["v7_synthetic_contract_tests"]
    if tests.get("status") == "passed":
        lines.extend(
            [
                f"现有 V7 定向合成/单元契约：{tests.get('passed_count', '—')} 项通过。",
                "",
            ]
        )
    elif tests.get("status") == "failed":
        lines.extend(
            [
                "现有 V7 定向合成/单元契约未全部通过；详见 JSON 中保存的 pytest 输出。",
                "",
            ]
        )
    else:
        lines.extend(["本次未执行 V7 合成/单元契约。", ""])
    lines.extend(
        [
            "## 科学边界与使用建议",
            "",
            "- RGB/fused-band ρ、β、A、R⊥、Edge F1 和梯度方向均为同数据结构转移诊断，不是独立 SWIR 高频真值。",
            "- 自然边缘的 halo 与 10–90% 边宽受材料对比、局部归一化和边缘选择影响，只能作为风险代理；正式 MTF/halo 结论需要校准斜边或等效靶标。",
            "- 百分比变化不附带重复样本置信区间；当前两个 ROI 是案例证据，不应写成总体显著性结论。",
            "- 正确论文口径是“V7 改善了观测一致性，并在部分波段/场景改善 RGB 等效空间结构”；不能写成“已证明 SWIR 细节无损”。",
            "",
            "## 百分比定义",
            "",
            "- 原始变化率：`100 × (V7 − baseline) / |baseline|`。",
            "- higher 指标的方向性改善率等于原始变化率；lower 指标取相反数。",
            "- target 指标（β、A、边宽比）按距目标 1 的绝对误差缩减比例计算。",
            "- `none` 指标（参考/候选绝对边宽、边缘样本数）不计算方向性改善率。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            "python scripts\\summarize_v7_results.py --run-synthetic-tests",
            "```",
            "",
            f"生成时间（UTC）：{payload['generated_at_utc']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_payload(repo: Path, run_synthetic_tests: bool) -> dict[str, Any]:
    real_roi: dict[str, Any] = {}
    for scene, version_runs in RUNS.items():
        runs = {
            version: extract_run(repo, scene, version, version_runs[version])
            for version in VERSIONS
        }
        real_roi[scene] = {
            "runs": runs,
            "v7_relative_to": build_comparisons(runs),
        }
    validation_checks = validate_real_roi(real_roi)
    v7_screening = summarize_v7_screening(real_roi)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo.resolve()),
        "methodology": {
            "real_roi_run_scope": RUNS,
            "requested_wavelengths_nm": list(REQUESTED_WAVELENGTHS_NM),
            "forward_metrics_scope": (
                "Global all-band error after the complete final HR product is degraded through the sensor PSF "
                "to the low-resolution observation grid."
            ),
            "band_metrics_scope": (
                "Nearest available bands to 900/1650/2200 nm at the report-selected sigma_2.4px scale; "
                "main statistics use reliable_rgb_detail, with dark and RGB-flat diagnostics reported separately. "
                "RGB and fused bands remain in native normalized/log-reflectance units with a fixed epsilon; "
                "no per-image or candidate-specific percentile stretch is applied before beta/A/R_perp."
            ),
            "raw_percent_change_formula": "100 * (V7 - baseline) / abs(baseline)",
            "directional_improvement": {
                "higher": "raw percent change",
                "lower": "negative raw percent change",
                "target": "100 * (abs(baseline-target) - abs(V7-target)) / abs(baseline-target)",
                "none": None,
            },
            "real_roi_truth_limit": REAL_TRUTH_SCOPE,
            "natural_edge_proxy_limit": HALO_LIMIT,
        },
        "metric_definitions": {
            spec.key: {
                "group": spec.group,
                "scope": spec.scope,
                "unit": spec.unit,
                "preferred_direction": spec.preferred_direction,
                "target_value": spec.target_value,
                "definition": spec.definition,
            }
            for spec in (*GLOBAL_METRICS, *BAND_METRICS)
        },
        "real_roi": real_roi,
        "synthetic_evidence": load_synthetic_evidence(repo, run_synthetic_tests),
        "validation": {
            "status": (
                "source_integrity_passed_with_quality_screening_warning"
                if v7_screening["status"] != "within_conservative_bounds"
                else "passed"
            ),
            "source_integrity_checks": validation_checks,
            "v7_conservative_screening": v7_screening,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="GeoCoreFusion repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/artifacts/v7_research/evidence).",
    )
    parser.add_argument(
        "--run-synthetic-tests",
        action="store_true",
        help="Run the existing V7 synthetic/unit contract files and record their pass status.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo / "artifacts" / "v7_research" / "evidence"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(repo, args.run_synthetic_tests)
    json_path = output_dir / "benchmark_summary.json"
    csv_path = output_dir / "benchmark_summary.csv"
    markdown_path = output_dir / "benchmark_interpretation.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = real_csv_rows(payload["real_roi"]) + synthetic_csv_rows(
        payload["synthetic_evidence"]
    )
    write_csv(csv_path, rows)
    markdown_path.write_text(build_interpretation(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["validation"]["status"],
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
                "csv_row_count": len(rows),
                "synthetic_test_status": payload["synthetic_evidence"][
                    "v7_synthetic_contract_tests"
                ]["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

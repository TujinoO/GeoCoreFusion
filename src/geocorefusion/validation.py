"""Structural validation of completed fusion runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .envi import open_cube, parse_header


_DUAL_PRODUCT_CONTRACT_VERSION = "geocorefusion.dual-product.v1"
_SCIENTIFIC_OUTPUT_GROUPS = {
    "fused_cube": ("fused_cube_hdr", "fused_cube_dat"),
    "reconstruction_factors": (
        "material_coefficients_hdr",
        "material_coefficients_dat",
        "subspace_model_json",
        "spatial_detail_gain_hdr",
        "spatial_detail_gain_dat",
        "spatial_additive_detail_hdr",
        "spatial_additive_detail_dat",
        "additive_spectral_scale_json",
    ),
    "quality_artifacts": (
        "quality_report_json",
        "spatial_uncertainty_hdr",
        "spatial_uncertainty_dat",
    ),
    "observed_low_resolution_cube": (
        "harmonized_lowres_hdr",
        "harmonized_lowres_dat",
    ),
}
_REQUIRED_PREVIEW_METRIC_EXCLUSIONS = {"RMSE", "SAM"}


def _has_required_metric_exclusions(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return _REQUIRED_PREVIEW_METRIC_EXCLUSIONS.issubset(
        {str(item).upper() for item in value}
    )


def _validate_dual_product_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the additive dual-product extension without rejecting v1 legacy runs."""

    compatibility_policy = (
        "A missing product_contract is accepted as a legacy manifest but remains "
        "unchecked; once product_contract is present, all dual-product checks are required."
    )
    if "product_contract" not in manifest:
        return {
            "present": False,
            "valid": None,
            "status": "legacy_compatible_missing_contract",
            "compatibility_policy": compatibility_policy,
            "errors": [],
        }

    errors: list[str] = []
    contract = manifest.get("product_contract")
    if not isinstance(contract, dict):
        return {
            "present": True,
            "valid": False,
            "status": "invalid",
            "compatibility_policy": compatibility_policy,
            "errors": ["product_contract must be an object"],
        }

    if contract.get("contract_version") != _DUAL_PRODUCT_CONTRACT_VERSION:
        errors.append(
            "product_contract.contract_version must be "
            f"{_DUAL_PRODUCT_CONTRACT_VERSION!r}"
        )
    schema_compatibility = contract.get("schema_compatibility")
    if not isinstance(schema_compatibility, dict):
        errors.append("product_contract.schema_compatibility must be an object")
    else:
        if schema_compatibility.get("run_schema_version") != "geocorefusion.run.v1":
            errors.append(
                "product_contract.schema_compatibility.run_schema_version must preserve "
                "'geocorefusion.run.v1'"
            )
        if schema_compatibility.get("extension_mode") != "additive":
            errors.append(
                "product_contract.schema_compatibility.extension_mode must be 'additive'"
            )
        if (
            schema_compatibility.get("legacy_missing_contract_policy")
            != "accepted_as_legacy_unchecked"
        ):
            errors.append(
                "product_contract.schema_compatibility.legacy_missing_contract_policy "
                "must be 'accepted_as_legacy_unchecked'"
            )

    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
        errors.append("manifest.outputs must be an object")
    scientific = contract.get("scientific_product")
    scientific_paths: dict[str, str] = {}
    if not isinstance(scientific, dict):
        errors.append("product_contract.scientific_product must be an object")
    else:
        if scientific.get("product_class") != "scientific":
            errors.append(
                "product_contract.scientific_product.product_class must be 'scientific'"
            )
        members = scientific.get("members")
        if not isinstance(members, dict):
            errors.append(
                "product_contract.scientific_product.members must be an object"
            )
        else:
            for group_name, group in members.items():
                if not isinstance(group, dict):
                    errors.append(
                        "product_contract.scientific_product.members."
                        f"{group_name} must be an object"
                    )
                    continue
                for key, path in group.items():
                    if not isinstance(path, str) or not path:
                        errors.append(
                            f"scientific member {group_name}.{key} must declare a path"
                        )
                        continue
                    if key in scientific_paths:
                        errors.append(
                            f"scientific output {key!r} is classified more than once"
                        )
                    scientific_paths[key] = path
                    if outputs.get(key) != path:
                        errors.append(
                            f"scientific member {group_name}.{key} does not match outputs"
                        )

            for group_name, output_keys in _SCIENTIFIC_OUTPUT_GROUPS.items():
                group = members.get(group_name)
                if not isinstance(group, dict):
                    errors.append(
                        "product_contract.scientific_product.members."
                        f"{group_name} must be an object"
                    )
                    continue
                for key in output_keys:
                    path = outputs.get(key)
                    if isinstance(path, str) and path and group.get(key) != path:
                        errors.append(
                            f"output {key!r} must be classified under {group_name!r}"
                        )

            for key, path in outputs.items():
                if (
                    isinstance(path, str)
                    and path
                    and scientific_paths.get(key) != path
                ):
                    errors.append(
                        f"declared output {key!r} is not classified as a scientific product"
                    )

    previews = manifest.get("previews", {})
    if not isinstance(previews, dict):
        previews = {}
        errors.append("manifest.previews must be an object")
    expected_previews = {
        key: path
        for key, path in previews.items()
        if isinstance(path, str) and path
    }
    visualization = contract.get("visualization_only")
    if not isinstance(visualization, dict):
        errors.append("product_contract.visualization_only must be an object")
    else:
        if visualization.get("product_class") != "visualization_only":
            errors.append(
                "product_contract.visualization_only.product_class must be "
                "'visualization_only'"
            )
        if visualization.get("not_for_quantitative_spectroscopy") is not True:
            errors.append(
                "product_contract.visualization_only must set "
                "not_for_quantitative_spectroscopy=true"
            )
        if not _has_required_metric_exclusions(
            visualization.get("excluded_from_quantitative_metrics")
        ):
            errors.append(
                "product_contract.visualization_only must exclude RMSE and SAM"
            )
        preview_members = visualization.get("members")
        if not isinstance(preview_members, dict):
            errors.append("product_contract.visualization_only.members must be an object")
        else:
            if set(preview_members) != set(expected_previews):
                errors.append(
                    "visualization-only member names must exactly match manifest.previews"
                )
            for key, path in expected_previews.items():
                member = preview_members.get(key)
                if not isinstance(member, dict):
                    errors.append(
                        f"preview {key!r} must have a visualization-only contract entry"
                    )
                    continue
                if member.get("path") != path:
                    errors.append(f"preview {key!r} contract path does not match previews")
                if member.get("product_class") != "visualization_only":
                    errors.append(
                        f"preview {key!r} product_class must be 'visualization_only'"
                    )
                if member.get("not_for_quantitative_spectroscopy") is not True:
                    errors.append(
                        f"preview {key!r} must set "
                        "not_for_quantitative_spectroscopy=true"
                    )
                if not _has_required_metric_exclusions(
                    member.get("excluded_from_quantitative_metrics")
                ):
                    errors.append(f"preview {key!r} must exclude RMSE and SAM")
                if path in scientific_paths.values():
                    errors.append(
                        f"preview {key!r} cannot also be a scientific product path"
                    )

    return {
        "present": True,
        "valid": not errors,
        "status": "validated" if not errors else "invalid",
        "compatibility_policy": compatibility_policy,
        "errors": errors,
    }


def validate_run(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    product_contract = _validate_dual_product_contract(manifest)
    outputs = manifest.get("outputs", {})
    fused_relative = outputs.get("fused_cube_hdr")
    if fused_relative:
        validation_mode = "full_cube"
        primary_hdr = root / fused_relative
    else:
        validation_mode = "metrics_only"
        harmonized_relative = outputs.get("harmonized_lowres_hdr")
        if not harmonized_relative:
            raise KeyError(
                "manifest outputs must contain either 'fused_cube_hdr' for a full run "
                "or 'harmonized_lowres_hdr' for a metrics-only run"
            )
        primary_hdr = root / harmonized_relative

    meta = parse_header(primary_hdr)
    cube, _ = open_cube(meta)
    ys = np.linspace(0, meta.lines - 1, min(16, meta.lines)).round().astype(int)
    xs = np.linspace(0, meta.samples - 1, min(12, meta.samples)).round().astype(int)
    bs = np.linspace(0, meta.bands - 1, min(24, meta.bands)).round().astype(int)
    sample = np.asarray(cube[np.ix_(ys, xs, bs)], dtype=np.float32)
    wavelengths = meta.wavelengths

    declared_output_paths = [root / value for value in outputs.values() if isinstance(value, str)]
    preview_paths = [root / value for value in manifest.get("previews", {}).values() if isinstance(value, str)]
    declared_outputs_exist = bool(declared_output_paths) and all(path.exists() for path in declared_output_paths)
    declared_previews_exist = all(path.exists() for path in preview_paths)

    quality_path = root / "metrics" / "quality_report.json"
    quality_report: dict[str, Any] | None = None
    if quality_path.exists():
        quality_report = json.loads(quality_path.read_text(encoding="utf-8"))

    output_grid = manifest["output_grid"]
    if validation_mode == "full_cube":
        expected_shape = [output_grid["height"], output_grid["width"], output_grid["bands"]]
        shape_matches_manifest: bool | None = list(meta.shape) == expected_shape
        shape_matches_mode = bool(shape_matches_manifest)
    else:
        low_shape = None
        if quality_report is not None:
            low_shape = quality_report.get("degradation", {}).get("low_shape")
        expected_shape = (
            [int(low_shape[0]), int(low_shape[1]), int(output_grid["bands"])]
            if isinstance(low_shape, list) and len(low_shape) == 2
            else None
        )
        shape_matches_manifest = None
        shape_matches_mode = (
            list(meta.shape) == expected_shape
            if expected_shape is not None
            else meta.bands == int(output_grid["bands"])
        )

    checks = {
        "validation_mode": validation_mode,
        "shape_matches_manifest": shape_matches_manifest,
        "shape_matches_declared_mode": bool(shape_matches_mode),
        "expected_primary_shape": expected_shape,
        "primary_shape": list(meta.shape),
        "data_size_validated": True,
        "sample_all_finite": bool(np.isfinite(sample).all()),
        "wavelength_count_matches": int(wavelengths.size) == meta.bands,
        "wavelength_strictly_increasing": bool(np.all(np.diff(wavelengths) > 0)),
        "declared_outputs_exist": declared_outputs_exist,
        "declared_previews_exist": declared_previews_exist,
        "product_contract_present": product_contract["present"],
        "product_contract_valid": product_contract["valid"],
        "product_contract_status": product_contract["status"],
        "product_contract_compatibility_policy": product_contract[
            "compatibility_policy"
        ],
        "product_contract_errors": product_contract["errors"],
        "quality_report_exists": quality_report is not None,
        "quality_status_passed": (
            quality_report.get("summary", {}).get("status") == "passed"
            if quality_report is not None
            else None
        ),
        "wavelength_start_nm": float(wavelengths[0]) if wavelengths.size else None,
        "wavelength_end_nm": float(wavelengths[-1]) if wavelengths.size else None,
        "sample_min": float(np.min(sample)),
        "sample_max": float(np.max(sample)),
    }
    checks["passed"] = all(
        checks[key]
        for key in (
            "shape_matches_declared_mode",
            "data_size_validated",
            "sample_all_finite",
            "wavelength_count_matches",
            "wavelength_strictly_increasing",
            "declared_outputs_exist",
            "declared_previews_exist",
        )
    )
    if quality_report is not None:
        checks["passed"] = checks["passed"] and bool(checks["quality_status_passed"])
    if product_contract["present"]:
        checks["passed"] = checks["passed"] and bool(
            checks["product_contract_valid"]
        )
    return {
        "output_dir": str(root),
        "validation_mode": validation_mode,
        "primary_cube": str(primary_hdr),
        "fused_cube": str(primary_hdr) if validation_mode == "full_cube" else None,
        "checks": checks,
    }

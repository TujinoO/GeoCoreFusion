import json
from pathlib import Path

import numpy as np

from geocorefusion.config import PipelineConfig
from geocorefusion.envi import create_bip_writer
from geocorefusion.output import build_manifest
from geocorefusion.validation import validate_run


def _write_metrics_only_inputs(tmp_path: Path) -> tuple[dict[str, str], np.ndarray]:
    analysis_dir = tmp_path / "analysis"
    metrics_dir = tmp_path / "metrics"
    analysis_dir.mkdir()
    metrics_dir.mkdir()

    expected = np.arange(6 * 5 * 4, dtype=np.float32).reshape(6, 5, 4) / 100.0
    wavelengths = np.asarray([700.0, 710.0, 720.0, 730.0])
    writer, hdr, dat = create_bip_writer(
        analysis_dir / "harmonized_lowres.hdr",
        expected.shape,
        wavelengths=wavelengths,
    )
    writer[:] = expected
    writer.flush()
    del writer

    quality = {
        "summary": {"status": "passed"},
        "degradation": {"low_shape": [6, 5]},
    }
    (metrics_dir / "quality_report.json").write_text(json.dumps(quality), encoding="utf-8")
    return (
        {
            "harmonized_lowres_hdr": str(hdr.relative_to(tmp_path)).replace(
                "\\", "/"
            ),
            "harmonized_lowres_dat": str(dat.relative_to(tmp_path)).replace(
                "\\", "/"
            ),
            "quality_report_json": "metrics/quality_report.json",
        },
        wavelengths,
    )


def test_validate_metrics_only_run_accepts_legacy_manifest(tmp_path: Path) -> None:
    outputs, _ = _write_metrics_only_inputs(tmp_path)
    manifest = {
        "output_grid": {"height": 60, "width": 50, "bands": 4},
        "outputs": outputs,
        "previews": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run(tmp_path)

    assert result["validation_mode"] == "metrics_only"
    assert result["fused_cube"] is None
    assert result["checks"]["shape_matches_manifest"] is None
    assert result["checks"]["shape_matches_declared_mode"]
    assert result["checks"]["product_contract_present"] is False
    assert result["checks"]["product_contract_valid"] is None
    assert (
        result["checks"]["product_contract_status"]
        == "legacy_compatible_missing_contract"
    )
    assert "accepted as a legacy manifest" in result["checks"][
        "product_contract_compatibility_policy"
    ]
    assert result["checks"]["passed"]


def test_validate_run_checks_dual_product_contract(tmp_path: Path) -> None:
    outputs, wavelengths = _write_metrics_only_inputs(tmp_path)
    preview_dir = tmp_path / "previews"
    preview_dir.mkdir()
    (preview_dir / "rgb_reference.png").write_bytes(b"display-only")
    previews = {"rgb_reference": "previews/rgb_reference.png"}
    config = PipelineConfig(data_dir=tmp_path / "inputs", output_dir=tmp_path)
    manifest = build_manifest(
        config,
        {"x": 0, "y": 0, "width": 50, "height": 60},
        wavelengths,
        outputs,
        previews,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run(tmp_path)

    assert result["checks"]["product_contract_present"] is True
    assert result["checks"]["product_contract_valid"] is True
    assert result["checks"]["product_contract_status"] == "validated"
    assert result["checks"]["product_contract_errors"] == []
    assert result["checks"]["passed"]


def test_validate_run_rejects_preview_without_quantitative_exclusions(
    tmp_path: Path,
) -> None:
    outputs, wavelengths = _write_metrics_only_inputs(tmp_path)
    preview_dir = tmp_path / "previews"
    preview_dir.mkdir()
    (preview_dir / "rgb_reference.png").write_bytes(b"display-only")
    previews = {"rgb_reference": "previews/rgb_reference.png"}
    config = PipelineConfig(data_dir=tmp_path / "inputs", output_dir=tmp_path)
    manifest = build_manifest(
        config,
        {"x": 0, "y": 0, "width": 50, "height": 60},
        wavelengths,
        outputs,
        previews,
    )
    member = manifest["product_contract"]["visualization_only"]["members"][
        "rgb_reference"
    ]
    member["not_for_quantitative_spectroscopy"] = False
    member["excluded_from_quantitative_metrics"] = ["RMSE"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run(tmp_path)

    assert result["checks"]["product_contract_valid"] is False
    assert result["checks"]["product_contract_status"] == "invalid"
    assert any(
        "not_for_quantitative_spectroscopy=true" in error
        for error in result["checks"]["product_contract_errors"]
    )
    assert any(
        "must exclude RMSE and SAM" in error
        for error in result["checks"]["product_contract_errors"]
    )
    assert result["checks"]["passed"] is False

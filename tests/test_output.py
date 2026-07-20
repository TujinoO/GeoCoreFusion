import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import geocorefusion.output as output_module
import geocorefusion.pipeline as pipeline_module
from geocorefusion.config import PipelineConfig
from geocorefusion.envi import create_bip_writer, open_cube
from geocorefusion.lowrank import SubspaceModel, load_subspace_model
from geocorefusion.output import (
    build_manifest,
    load_additive_spectral_scale,
    reconstruct_modulated,
    reconstruct_to_envi,
    write_additive_spectral_scale,
    write_coefficients_envi,
    write_json,
    write_previews,
)


def _subspace(basis: np.ndarray, mean: np.ndarray) -> SubspaceModel:
    bands = int(mean.size)
    return SubspaceModel(
        mean_spectrum=np.asarray(mean, dtype=np.float32),
        basis=np.asarray(basis, dtype=np.float32),
        explained_variance_ratio=np.ones(basis.shape[0], dtype=np.float32),
        clip_min=np.full(bands, 0.25, dtype=np.float32),
        clip_max=np.full(bands, 0.35, dtype=np.float32),
    )


def test_reconstruct_to_envi_modulates_before_one_physical_clip(tmp_path: Path) -> None:
    coeff = np.asarray(
        [
            [[0.4], [-1.0]],
            [[0.1], [0.2]],
        ],
        dtype=np.float32,
    )
    subspace = _subspace(
        basis=np.asarray([[1.0, 0.5]], dtype=np.float32),
        mean=np.asarray([0.2, 0.3], dtype=np.float32),
    )
    gain = np.asarray([[2.0, 1.0], [1.5, 0.5]], dtype=np.float32)
    additive = np.asarray([[0.1, 0.0], [0.2, -0.1]], dtype=np.float32)
    additive_scale = np.asarray([0.2, 0.4], dtype=np.float32)
    expected_unclipped = (
        np.einsum("...k,kb->...b", coeff, subspace.basis, optimize=True)
        + subspace.mean_spectrum[None, None, :]
    )
    expected_unclipped = expected_unclipped * gain[:, :, None]
    expected_unclipped = expected_unclipped + additive[:, :, None] * additive_scale[None, None, :]
    expected = np.clip(expected_unclipped, 0.0, 1.0)

    statistics: dict[str, object] = {}
    hdr, _ = reconstruct_to_envi(
        coeff,
        subspace,
        np.asarray([900.0, 2200.0], dtype=np.float32),
        tmp_path / "fused.hdr",
        tile_size=1,
        dtype="float32",
        detail_gain=gain,
        additive_detail=additive,
        additive_spectral_scale=additive_scale,
        physical_clip_limits=(0.0, 1.0),
        clip_statistics=statistics,
    )
    cube, _ = open_cube(hdr)
    actual = np.asarray(cube).copy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    assert float(actual[1, 0, 0]) > float(subspace.clip_max[0])
    assert float(actual[1, 0, 1]) > float(subspace.clip_max[1])
    assert statistics["quantile_limits_used"] is False
    assert statistics["below_lower_bound_count"] == 2
    assert statistics["above_upper_bound_count"] == 2
    assert statistics["clipped_value_count"] == 4
    assert statistics["clipped_fraction_of_finite"] == 0.5


def test_previews_modulate_on_native_grid_before_downsampling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    y, x = np.indices((4, 4))
    coefficient = (((x + y) % 2) * 100.0).astype(np.float32)
    coeff = coefficient[:, :, None]
    gain = np.where(coefficient > 0.0, 2.0, 0.5).astype(np.float32)
    subspace = _subspace(
        basis=np.ones((1, 4), dtype=np.float32),
        mean=np.zeros(4, dtype=np.float32),
    )
    wavelengths = np.asarray([900.0, 1650.0, 2200.0, 2350.0], dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    uncertainty = np.zeros((4, 4), dtype=np.float32)

    monkeypatch.setattr(
        output_module,
        "_stretch",
        lambda image: np.clip(np.rint(image), 0.0, 255.0).astype(np.uint8),
    )
    write_previews(
        tmp_path,
        rgb,
        coeff,
        subspace,
        wavelengths,
        uncertainty,
        gain,
        tile_size=2,
        max_size=2,
        physical_clip_limits=(0.0, None),
    )

    fused_2200 = np.asarray(Image.open(tmp_path / "fused_2200nm.png"))
    fused_mean = np.asarray(Image.open(tmp_path / "fused_mean_reflectance.png"))
    base_false_color = np.asarray(Image.open(tmp_path / "fused_false_color_base_2200_1650_900.png"))
    fused_false_color = np.asarray(Image.open(tmp_path / "fused_false_color_2200_1650_900.png"))

    # Native products are 0/200 and therefore average to 100 in every 2x2
    # block. Downsampling coefficient and gain first would incorrectly yield
    # mean(coeff) * mean(gain) = 50 * 1.25 = 62.5.
    np.testing.assert_array_equal(fused_2200, np.full((2, 2), 100, dtype=np.uint8))
    np.testing.assert_array_equal(fused_mean, np.full((2, 2), 100, dtype=np.uint8))
    np.testing.assert_array_equal(base_false_color, np.full((2, 2, 3), 50, dtype=np.uint8))
    np.testing.assert_array_equal(fused_false_color, np.full((2, 2, 3), 100, dtype=np.uint8))


def test_manifest_adds_compatible_dual_product_contract(tmp_path: Path) -> None:
    config = PipelineConfig(
        data_dir=tmp_path / "inputs",
        output_dir=tmp_path / "run",
    )
    outputs = {
        "fused_cube_hdr": "cube/fused.hdr",
        "fused_cube_dat": "cube/fused.dat",
        "material_coefficients_hdr": "coefficients/material_coefficients.hdr",
        "material_coefficients_dat": "coefficients/material_coefficients.dat",
        "subspace_model_json": "metadata/subspace_model.json",
        "spatial_detail_gain_hdr": "metrics/spatial_detail_gain.hdr",
        "spatial_detail_gain_dat": "metrics/spatial_detail_gain.dat",
        "spatial_additive_detail_hdr": "metrics/spatial_additive_detail.hdr",
        "spatial_additive_detail_dat": "metrics/spatial_additive_detail.dat",
        "additive_spectral_scale_json": "metadata/additive_spectral_scale.json",
        "quality_report_json": "metrics/quality_report.json",
        "spatial_uncertainty_hdr": "metrics/spatial_uncertainty.hdr",
        "spatial_uncertainty_dat": "metrics/spatial_uncertainty.dat",
        "harmonized_lowres_hdr": "analysis/harmonized_lowres.hdr",
        "harmonized_lowres_dat": "analysis/harmonized_lowres.dat",
        "processing_config_json": "metadata/processing_config.json",
    }
    previews = {
        "rgb_reference": "previews/rgb_reference.png",
        "fused_2200nm": "previews/fused_2200nm.png",
    }

    manifest = build_manifest(
        config,
        {"x": 1, "y": 2, "width": 20, "height": 30},
        np.asarray([900.0, 2200.0], dtype=np.float32),
        outputs,
        previews,
    )

    assert manifest["schema_version"] == "geocorefusion.run.v1"
    assert manifest["outputs"] == outputs
    assert manifest["previews"] == previews
    contract = manifest["product_contract"]
    assert contract["contract_version"] == "geocorefusion.dual-product.v1"
    assert contract["schema_compatibility"] == {
        "run_schema_version": "geocorefusion.run.v1",
        "extension_mode": "additive",
        "legacy_missing_contract_policy": "accepted_as_legacy_unchecked",
    }
    scientific = contract["scientific_product"]
    assert scientific["product_class"] == "scientific"
    assert scientific["members"]["fused_cube"] == {
        "fused_cube_hdr": outputs["fused_cube_hdr"],
        "fused_cube_dat": outputs["fused_cube_dat"],
    }
    assert (
        scientific["members"]["reconstruction_factors"][
            "material_coefficients_hdr"
        ]
        == outputs["material_coefficients_hdr"]
    )
    assert scientific["members"]["quality_artifacts"]["quality_report_json"] == (
        outputs["quality_report_json"]
    )
    assert scientific["members"]["observed_low_resolution_cube"] == {
        "harmonized_lowres_hdr": outputs["harmonized_lowres_hdr"],
        "harmonized_lowres_dat": outputs["harmonized_lowres_dat"],
    }
    assert scientific["members"]["provenance_and_metadata"] == {
        "processing_config_json": outputs["processing_config_json"]
    }
    visualization = contract["visualization_only"]
    assert visualization["product_class"] == "visualization_only"
    assert visualization["not_for_quantitative_spectroscopy"] is True
    assert set(visualization["excluded_from_quantitative_metrics"]) == {
        "RMSE",
        "SAM",
    }
    for name, path in previews.items():
        member = visualization["members"][name]
        assert member["path"] == path
        assert member["product_class"] == "visualization_only"
        assert member["not_for_quantitative_spectroscopy"] is True
        assert set(member["excluded_from_quantitative_metrics"]) == {
            "RMSE",
            "SAM",
        }


def test_manifest_supports_single_v61_visual_full_detail_product(tmp_path: Path) -> None:
    config = PipelineConfig(
        data_dir=tmp_path / "inputs",
        output_dir=tmp_path / "run",
    )
    config.output.product_mode = "visual_full_detail"
    outputs = {
        "fused_cube_hdr": "cube/fused.hdr",
        "quality_report_json": "metrics/quality_report.json",
    }
    previews = {"fused_2200nm": "previews/fused_2200nm.png"}

    manifest = build_manifest(
        config,
        {"x": 0, "y": 0, "width": 32, "height": 48},
        np.asarray([900.0, 2200.0], dtype=np.float32),
        outputs,
        previews,
    )

    contract = manifest["product_contract"]
    assert contract["contract_version"] == "geocorefusion.visual-full-detail.v1"
    assert contract["product_mode"] == "visual_full_detail"
    assert "excluded_modes" not in contract
    assert "scientific_product" not in contract
    assert manifest["scientific_scope"]["primary_goal"] == (
        "maximum_registered_rgb_spatial_detail_transfer"
    )


def test_persisted_v7_factors_reconstruct_367_band_target_exactly(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    band_count = 367
    rank = 3
    wavelengths = np.linspace(691.0, 2521.0, band_count, dtype=np.float32)
    subspace = SubspaceModel(
        mean_spectrum=np.linspace(0.12, 0.42, band_count, dtype=np.float32),
        basis=rng.normal(0.0, 0.025, size=(rank, band_count)).astype(np.float32),
        explained_variance_ratio=np.asarray([0.73, 0.18, 0.06], dtype=np.float32),
        clip_min=np.linspace(0.01, 0.04, band_count, dtype=np.float32),
        clip_max=np.linspace(0.65, 0.85, band_count, dtype=np.float32),
    )
    coefficients = rng.normal(0.0, 0.3, size=(4, 5, rank)).astype(np.float32)
    detail_gain = rng.uniform(0.92, 1.08, size=(4, 5)).astype(np.float32)
    additive_detail = rng.normal(0.0, 0.02, size=(4, 5)).astype(np.float32)
    additive_scale = np.linspace(-0.04, 0.07, band_count, dtype=np.float32)
    target_index = 302

    expected_target = reconstruct_modulated(
        coefficients,
        subspace,
        detail_gain=detail_gain,
        additive_detail=additive_detail,
        additive_spectral_scale=additive_scale,
        physical_clip_limits=None,
    )[:, :, target_index]

    subspace_path = tmp_path / "subspace_model.json"
    scale_path = tmp_path / "additive_spectral_scale.json"
    coefficient_hdr, _ = write_coefficients_envi(coefficients, tmp_path / "coefficients.hdr")
    gain_writer, gain_hdr, _ = create_bip_writer(
        tmp_path / "detail_gain.hdr",
        detail_gain.shape + (1,),
        dtype="float32",
    )
    gain_writer[:, :, 0] = detail_gain
    gain_writer.flush()
    del gain_writer
    additive_writer, additive_hdr, _ = create_bip_writer(
        tmp_path / "additive_detail.hdr",
        additive_detail.shape + (1,),
        dtype="float32",
    )
    additive_writer[:, :, 0] = additive_detail
    additive_writer.flush()
    del additive_writer
    write_json(subspace_path, subspace.to_dict())
    write_additive_spectral_scale(
        scale_path,
        wavelengths,
        additive_scale,
        source_metadata={"derived_from": "harmonized_lowres.hdr"},
    )
    loaded_subspace = load_subspace_model(subspace_path)
    loaded_wavelengths, loaded_scale = load_additive_spectral_scale(scale_path)
    loaded_coefficients = np.asarray(open_cube(coefficient_hdr)[0]).copy()
    loaded_detail_gain = np.asarray(open_cube(gain_hdr)[0][:, :, 0]).copy()
    loaded_additive_detail = np.asarray(open_cube(additive_hdr)[0][:, :, 0]).copy()

    actual_target = reconstruct_modulated(
        loaded_coefficients,
        loaded_subspace,
        detail_gain=loaded_detail_gain,
        additive_detail=loaded_additive_detail,
        additive_spectral_scale=loaded_scale,
        physical_clip_limits=None,
    )[:, :, target_index]

    np.testing.assert_array_equal(loaded_subspace.basis, subspace.basis)
    np.testing.assert_array_equal(loaded_subspace.mean_spectrum, subspace.mean_spectrum)
    np.testing.assert_array_equal(loaded_subspace.clip_min, subspace.clip_min)
    np.testing.assert_array_equal(loaded_subspace.clip_max, subspace.clip_max)
    np.testing.assert_array_equal(
        loaded_subspace.explained_variance_ratio,
        subspace.explained_variance_ratio,
    )
    np.testing.assert_array_equal(loaded_wavelengths, wavelengths)
    np.testing.assert_array_equal(loaded_scale, additive_scale)
    np.testing.assert_array_equal(actual_target, expected_target)

    subspace_payload = json.loads(subspace_path.read_text(encoding="utf-8"))
    assert subspace_payload["array_metadata"]["basis"]["shape"] == [rank, band_count]
    assert subspace_payload["array_metadata"]["basis"]["dtype"] == "float32"
    assert subspace_payload["array_metadata"]["basis"]["axes"] == [
        "component",
        "spectral_band",
    ]
    assert len(subspace_payload["array_metadata"]["basis"]["checksum"]["value"]) == 64
    scale_payload = json.loads(scale_path.read_text(encoding="utf-8"))
    assert scale_payload["band_count"] == band_count
    assert len(scale_payload["additive_spectral_scale"]) == band_count
    assert scale_payload["source_metadata"]["derived_from"] == "harmonized_lowres.hdr"


def test_subspace_loader_rejects_checksum_mismatch() -> None:
    subspace = _subspace(
        basis=np.asarray([[0.1, 0.2]], dtype=np.float32),
        mean=np.asarray([0.3, 0.4], dtype=np.float32),
    )
    payload = subspace.to_dict()
    payload["basis"][0][0] = float(payload["basis"][0][0]) + 0.01

    with pytest.raises(ValueError, match="basis checksum mismatch"):
        SubspaceModel.from_dict(payload)


@pytest.mark.parametrize(
    "relative_path",
    (Path("manifest.json"), Path("metadata/subspace_model.json")),
)
def test_metrics_only_pipeline_protects_existing_core_outputs_before_discovery(
    tmp_path: Path,
    monkeypatch,
    relative_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    existing = output_dir / relative_path
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("do not replace", encoding="utf-8")
    config = PipelineConfig(data_dir=tmp_path / "unused-data", output_dir=output_dir)
    config.output.write_envi = False
    config.output.overwrite_files = False

    def fail_if_discovered(*_args, **_kwargs):
        raise AssertionError("input discovery must not run before overwrite protection")

    monkeypatch.setattr(pipeline_module, "discover_triplet", fail_if_discovered)
    with pytest.raises(FileExistsError, match=relative_path.name):
        pipeline_module.run_pipeline(config)
    assert existing.read_text(encoding="utf-8") == "do not replace"

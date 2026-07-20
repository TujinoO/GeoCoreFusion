import itertools
import json

import numpy as np

from geocorefusion.lowrank import (
    SubspaceModel,
    fit_hybrid_simplex_subspace,
    fit_simplex_subspace,
    project_simplex,
    reconstruct,
)


def _synthetic_mixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    wavelength = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    endmembers = np.stack(
        [
            0.10 + 0.24 * wavelength,
            0.37
            - 0.17 * wavelength
            - 0.075 * np.exp(-0.5 * ((wavelength - 0.68) / 0.10) ** 2),
            0.16
            + 0.16 * np.exp(-0.5 * ((wavelength - 0.43) / 0.14) ** 2)
            + 0.025 * np.sin(4.0 * np.pi * wavelength),
        ],
        axis=0,
    ).astype(np.float32)
    abundance = rng.dirichlet(np.asarray([0.55, 0.70, 0.60]), size=20 * 15)
    abundance[:3] = np.eye(3, dtype=np.float64)
    cube = (abundance @ endmembers).reshape(20, 15, endmembers.shape[1])
    return cube.astype(np.float32), endmembers, abundance.reshape(20, 15, 3).astype(np.float32)


def _best_endmember_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    values = []
    for permutation in itertools.permutations(range(expected.shape[0])):
        difference = actual[np.asarray(permutation)] - expected
        values.append(float(np.sqrt(np.mean(difference * difference))))
    return min(values)


def test_simplex_subspace_recovers_synthetic_mixture() -> None:
    cube, expected_endmembers, _ = _synthetic_mixture()

    model, abundance = fit_simplex_subspace(
        cube,
        rank=3,
        max_pixels=cube.shape[0] * cube.shape[1],
        max_iterations=160,
        abundance_steps=12,
        endmember_steps=6,
        tolerance=1e-8,
        random_seed=19,
        clip_quantiles=(0.0, 1.0),
    )
    reconstructed = reconstruct(abundance, model, clip=False)
    relative_error = float(
        np.linalg.norm(reconstructed - cube) / np.linalg.norm(cube)
    )

    assert model.representation == "nonnegative_endmember_simplex_abundance"
    np.testing.assert_array_equal(model.mean_spectrum, np.zeros(cube.shape[2], dtype=np.float32))
    assert float(np.min(model.basis)) >= 0.0
    assert float(np.min(abundance)) >= 0.0
    np.testing.assert_allclose(np.sum(abundance, axis=2), 1.0, rtol=0.0, atol=2e-6)
    assert relative_error < 0.015
    assert _best_endmember_rmse(model.basis, expected_endmembers) < 0.025
    assert model.fit_metadata["final_full_rmse"] < model.fit_metadata["initial_fit_rmse"]
    assert model.fit_metadata["simplex_max_sum_error"] < 1e-10
    assert model.fit_metadata["abundance_minimum"] >= 0.0
    assert len(model.fit_metadata["objective_history"]) >= 2


def test_simplex_subspace_is_deterministic_for_fixed_seed() -> None:
    cube, _, _ = _synthetic_mixture()
    options = {
        "rank": 3,
        "max_pixels": 180,
        "max_iterations": 40,
        "abundance_steps": 8,
        "random_seed": 7,
        "clip_quantiles": (0.0, 1.0),
    }

    first_model, first_abundance = fit_simplex_subspace(cube, **options)
    second_model, second_abundance = fit_simplex_subspace(cube, **options)

    np.testing.assert_array_equal(first_model.basis, second_model.basis)
    np.testing.assert_array_equal(first_abundance, second_abundance)
    assert first_model.fit_metadata == second_model.fit_metadata


def test_simplex_subspace_v2_persistence_round_trip_and_legacy_defaults() -> None:
    cube, _, _ = _synthetic_mixture()
    model, abundance = fit_simplex_subspace(
        cube,
        rank=3,
        max_pixels=160,
        max_iterations=35,
        random_seed=11,
        clip_quantiles=(0.0, 1.0),
    )
    payload = json.loads(json.dumps(model.to_dict()))
    loaded = SubspaceModel.from_dict(payload)

    assert loaded.representation == model.representation
    assert loaded.fit_metadata == model.fit_metadata
    np.testing.assert_array_equal(loaded.mean_spectrum, model.mean_spectrum)
    np.testing.assert_array_equal(loaded.basis, model.basis)
    np.testing.assert_array_equal(
        reconstruct(abundance, loaded, clip=False),
        reconstruct(abundance, model, clip=False),
    )

    payload.pop("representation")
    payload.pop("fit_metadata")
    legacy_loaded = SubspaceModel.from_dict(payload)
    assert legacy_loaded.representation == "affine_pca_subspace"
    assert legacy_loaded.fit_metadata == {}


def test_project_simplex_supports_arbitrary_axis() -> None:
    values = np.asarray(
        [
            [[-1.0, 0.2], [2.0, 0.4], [0.5, -0.3]],
            [[0.2, 1.5], [0.2, -0.1], [0.2, 0.7]],
        ],
        dtype=np.float32,
    )

    projected = project_simplex(values, axis=1)

    assert projected.dtype == np.float32
    assert projected.shape == values.shape
    assert float(np.min(projected)) >= 0.0
    np.testing.assert_allclose(np.sum(projected, axis=1), 1.0, rtol=0.0, atol=1e-6)


def test_hybrid_simplex_residual_preserves_abundance_and_models_signed_residual() -> None:
    cube, _, _ = _synthetic_mixture()
    y, x = np.indices(cube.shape[:2], dtype=np.float32)
    wavelength = np.linspace(0.0, 1.0, cube.shape[2], dtype=np.float32)
    residual_spectrum = 0.018 * np.sin(5.0 * np.pi * wavelength)
    residual_field = np.sin(x / 3.7) * np.cos(y / 4.9)
    observed = cube + residual_field[:, :, None] * residual_spectrum[None, None, :]
    assert float(np.min(observed)) > 0.0

    simplex_model, simplex_coeff = fit_simplex_subspace(
        observed,
        rank=3,
        max_pixels=observed.shape[0] * observed.shape[1],
        max_iterations=60,
        random_seed=23,
        clip_quantiles=(0.0, 1.0),
    )
    hybrid_model, hybrid_coeff = fit_hybrid_simplex_subspace(
        observed,
        rank=3,
        residual_rank=2,
        max_pixels=observed.shape[0] * observed.shape[1],
        max_iterations=60,
        random_seed=23,
        clip_quantiles=(0.0, 1.0),
    )
    simplex_rmse = float(
        np.sqrt(np.mean((reconstruct(simplex_coeff, simplex_model, clip=False) - observed) ** 2))
    )
    hybrid_rmse = float(
        np.sqrt(np.mean((reconstruct(hybrid_coeff, hybrid_model, clip=False) - observed) ** 2))
    )

    assert hybrid_model.representation == "simplex_abundance_plus_pca_observation_residual"
    assert hybrid_coeff.shape[2] == 5
    assert float(np.min(hybrid_coeff[:, :, :3])) >= 0.0
    np.testing.assert_allclose(
        np.sum(hybrid_coeff[:, :, :3], axis=2), 1.0, rtol=0.0, atol=2e-6
    )
    assert hybrid_rmse < 0.55 * simplex_rmse
    assert hybrid_model.fit_metadata["simplex_component_count"] == 3
    assert hybrid_model.fit_metadata["residual_component_count"] == 2

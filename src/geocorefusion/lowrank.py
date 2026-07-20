"""Low-rank spectral basis and material-coefficient representation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.utils.extmath import randomized_svd


_SUBSPACE_ARRAY_AXES: dict[str, tuple[str, ...]] = {
    "mean_spectrum": ("spectral_band",),
    "basis": ("component", "spectral_band"),
    "explained_variance_ratio": ("component",),
    "clip_min": ("spectral_band",),
    "clip_max": ("spectral_band",),
}


def _canonical_float32(array: np.ndarray) -> np.ndarray:
    """Return the checksum representation used by persisted factor arrays."""

    return np.ascontiguousarray(np.asarray(array, dtype=np.dtype("<f4")))


def _float32_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(_canonical_float32(array).tobytes(order="C")).hexdigest()


def describe_float32_array(array: np.ndarray, axes: Iterable[str]) -> dict[str, Any]:
    """Describe a float32 factor array and its canonical byte checksum."""

    values = np.asarray(array, dtype=np.float32)
    axis_names = [str(axis) for axis in axes]
    if values.ndim != len(axis_names):
        raise ValueError(
            f"Array has {values.ndim} dimensions but {len(axis_names)} axis names were supplied"
        )
    return {
        "shape": [int(value) for value in values.shape],
        "dtype": "float32",
        "axes": axis_names,
        "storage_order": "C",
        "checksum": {
            "algorithm": "sha256",
            "canonical_dtype": "<f4",
            "value": _float32_sha256(values),
        },
    }


def validate_float32_array(
    values: Any,
    metadata: Mapping[str, Any],
    *,
    name: str,
    expected_axes: Iterable[str] | None = None,
) -> np.ndarray:
    """Decode and checksum one transparent JSON float32 array."""

    if metadata.get("dtype") != "float32":
        raise ValueError(f"{name} declares unsupported dtype {metadata.get('dtype')!r}")
    if metadata.get("storage_order") != "C":
        raise ValueError(f"{name} declares unsupported storage order {metadata.get('storage_order')!r}")
    checksum = metadata.get("checksum")
    if not isinstance(checksum, Mapping):
        raise ValueError(f"{name} is missing checksum metadata")
    if checksum.get("algorithm") != "sha256" or checksum.get("canonical_dtype") != "<f4":
        raise ValueError(f"{name} declares an unsupported checksum representation")

    array = np.asarray(values, dtype=np.float32)
    declared_shape = tuple(int(value) for value in metadata.get("shape", ()))
    if array.shape != declared_shape:
        raise ValueError(f"{name} shape mismatch: values {array.shape}, metadata {declared_shape}")
    declared_axes = tuple(str(value) for value in metadata.get("axes", ()))
    if len(declared_axes) != array.ndim:
        raise ValueError(f"{name} axis count does not match its rank")
    if expected_axes is not None and declared_axes != tuple(expected_axes):
        raise ValueError(
            f"{name} axis mismatch: {declared_axes!r} != {tuple(expected_axes)!r}"
        )
    actual_checksum = _float32_sha256(array)
    if actual_checksum != checksum.get("value"):
        raise ValueError(f"{name} checksum mismatch")
    return np.ascontiguousarray(array)


def checksum_float32_arrays(arrays: Iterable[tuple[str, np.ndarray]]) -> str:
    """Checksum an ordered collection of named float32 factor arrays."""

    digest = hashlib.sha256()
    digest.update(b"GeoCoreFusion float32 factor collection v1\0")
    for name, array in arrays:
        canonical = _canonical_float32(array)
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(slots=True)
class SubspaceModel:
    mean_spectrum: np.ndarray
    basis: np.ndarray
    explained_variance_ratio: np.ndarray
    clip_min: np.ndarray
    clip_max: np.ndarray
    representation: str = "affine_pca_subspace"
    fit_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        arrays = {
            "mean_spectrum": np.asarray(self.mean_spectrum, dtype=np.float32),
            "basis": np.asarray(self.basis, dtype=np.float32),
            "explained_variance_ratio": np.asarray(
                self.explained_variance_ratio, dtype=np.float32
            ),
            "clip_min": np.asarray(self.clip_min, dtype=np.float32),
            "clip_max": np.asarray(self.clip_max, dtype=np.float32),
        }
        self._validate_shapes(arrays)
        return {
            "schema_version": "geocorefusion.subspace-model.v2",
            "representation": str(self.representation),
            "fit_metadata": dict(self.fit_metadata),
            "rank": int(self.basis.shape[0]),
            "band_count": int(self.basis.shape[1]),
            "axis_convention": {
                "coefficient_cube": ["y", "x", "component"],
                "reconstructed_cube": ["y", "x", "spectral_band"],
                "basis_multiplication": "spectra = coefficients @ basis + mean_spectrum",
            },
            "clip_policy": {
                "clip_min_clip_max_role": "fitted quantile bounds retained for audit",
                "applied_during_v7_final_reconstruction": False,
                "final_output_contract": (
                    "reconstruct the unbounded subspace, apply spatial factors, then apply "
                    "the manifest final_clip_policy exactly once"
                ),
            },
            "mean_spectrum": arrays["mean_spectrum"].tolist(),
            "basis": arrays["basis"].tolist(),
            "explained_variance_ratio": arrays["explained_variance_ratio"].tolist(),
            "explained_variance_total": float(arrays["explained_variance_ratio"].sum()),
            "clip_min": arrays["clip_min"].tolist(),
            "clip_max": arrays["clip_max"].tolist(),
            "array_metadata": {
                name: describe_float32_array(values, _SUBSPACE_ARRAY_AXES[name])
                for name, values in arrays.items()
            },
            "model_checksum": {
                "algorithm": "sha256",
                "canonical_representation": "ordered_named_little_endian_float32_arrays_v1",
                "array_order": list(arrays),
                "value": checksum_float32_arrays(arrays.items()),
            },
        }

    @staticmethod
    def _validate_shapes(arrays: Mapping[str, np.ndarray]) -> None:
        basis = arrays["basis"]
        if basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] < 1:
            raise ValueError(f"basis must have shape (component, spectral_band), got {basis.shape}")
        rank, bands = basis.shape
        expected = {
            "mean_spectrum": (bands,),
            "explained_variance_ratio": (rank,),
            "clip_min": (bands,),
            "clip_max": (bands,),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {arrays[name].shape}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubspaceModel":
        """Load a fully reconstructable, checksum-verified subspace model."""

        if payload.get("schema_version") != "geocorefusion.subspace-model.v2":
            raise ValueError(
                "Subspace payload is not reconstructable schema "
                "geocorefusion.subspace-model.v2"
            )
        metadata = payload.get("array_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Subspace payload is missing array_metadata")
        arrays: dict[str, np.ndarray] = {}
        for name, axes in _SUBSPACE_ARRAY_AXES.items():
            if name not in payload or name not in metadata:
                raise ValueError(f"Subspace payload is missing {name}")
            arrays[name] = validate_float32_array(
                payload[name], metadata[name], name=name, expected_axes=axes
            )
        cls._validate_shapes(arrays)
        rank, bands = arrays["basis"].shape
        if int(payload.get("rank", -1)) != rank or int(payload.get("band_count", -1)) != bands:
            raise ValueError("Subspace rank/band_count does not match the persisted arrays")

        model_checksum = payload.get("model_checksum")
        if not isinstance(model_checksum, Mapping):
            raise ValueError("Subspace payload is missing model_checksum")
        expected_order = list(arrays)
        if (
            model_checksum.get("algorithm") != "sha256"
            or model_checksum.get("canonical_representation")
            != "ordered_named_little_endian_float32_arrays_v1"
            or model_checksum.get("array_order") != expected_order
        ):
            raise ValueError("Subspace model checksum metadata is invalid")
        actual_checksum = checksum_float32_arrays(arrays.items())
        if actual_checksum != model_checksum.get("value"):
            raise ValueError("Subspace model checksum mismatch")
        return cls(
            mean_spectrum=arrays["mean_spectrum"],
            basis=arrays["basis"],
            explained_variance_ratio=arrays["explained_variance_ratio"],
            clip_min=arrays["clip_min"],
            clip_max=arrays["clip_max"],
            representation=str(payload.get("representation", "affine_pca_subspace")),
            fit_metadata=(
                dict(payload["fit_metadata"])
                if isinstance(payload.get("fit_metadata"), Mapping)
                else {}
            ),
        )


def load_subspace_model(path: str | Path) -> SubspaceModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Subspace model JSON root must be an object")
    return SubspaceModel.from_dict(payload)


def audit_subspace_model(path: str | Path) -> dict[str, Any]:
    """Return a non-throwing integrity audit for a persisted subspace model."""

    source = Path(path)
    try:
        model = load_subspace_model(source)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"path": str(source), "passed": False, "error": str(exc)}
    return {
        "path": str(source),
        "passed": True,
        "rank": int(model.basis.shape[0]),
        "band_count": int(model.basis.shape[1]),
        "representation": str(model.representation),
        "dtype": "float32",
        "checksums_verified": True,
    }


def _project_rows_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection of each matrix row onto the probability simplex."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("Simplex projection requires a non-empty two-dimensional matrix")
    ordered = np.sort(matrix, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    divisor = np.arange(1, matrix.shape[1] + 1, dtype=np.float64)[None, :]
    support = ordered - cumulative / divisor > 0.0
    rho = np.maximum(np.sum(support, axis=1) - 1, 0)
    theta = cumulative[np.arange(matrix.shape[0]), rho] / (rho + 1.0)
    projected = np.maximum(matrix - theta[:, None], 0.0)
    projected /= np.maximum(np.sum(projected, axis=1, keepdims=True), 1e-15)
    return projected


def project_simplex(array: np.ndarray, axis: int = -1) -> np.ndarray:
    """Project vectors along ``axis`` onto the nonnegative unit simplex.

    Float32 inputs remain float32; other input dtypes return float64 so the
    operation never truncates simplex values to an integer dtype.
    """

    source = np.asarray(array)
    if source.ndim < 1:
        raise ValueError("Simplex projection requires an array with at least one axis")
    normalized_axis = int(axis)
    if normalized_axis < 0:
        normalized_axis += source.ndim
    if normalized_axis < 0 or normalized_axis >= source.ndim:
        raise np.AxisError(axis, ndim=source.ndim)
    moved = np.moveaxis(source, normalized_axis, -1)
    projected = _project_rows_to_simplex(
        np.asarray(moved, dtype=np.float64).reshape(-1, moved.shape[-1])
    ).reshape(moved.shape)
    restored = np.moveaxis(projected, -1, normalized_axis)
    output_dtype = np.float32 if source.dtype == np.dtype("float32") else np.float64
    return restored.astype(output_dtype, copy=False)


def _farthest_spectra_initialization(
    spectra: np.ndarray,
    rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select stable convex-hull anchors with seeded tie breaking."""

    values = np.asarray(spectra, dtype=np.float64)
    center = np.mean(values, axis=0)
    distance = np.sum((values - center[None, :]) ** 2, axis=1)
    jitter_scale = max(float(np.max(distance)), 1.0) * np.finfo(np.float64).eps
    first = int(np.argmax(distance + jitter_scale * rng.random(values.shape[0])))
    selected = [first]
    nearest = np.sum((values - values[first][None, :]) ** 2, axis=1)
    for _ in range(1, int(rank)):
        score = nearest + jitter_scale * rng.random(values.shape[0])
        score[np.asarray(selected, dtype=np.int64)] = -np.inf
        index = int(np.argmax(score))
        if not np.isfinite(score[index]):
            raise ValueError("Cannot initialize the requested number of distinct endmembers")
        selected.append(index)
        candidate_distance = np.sum((values - values[index][None, :]) ** 2, axis=1)
        nearest = np.minimum(nearest, candidate_distance)
    return np.maximum(values[np.asarray(selected, dtype=np.int64)], 0.0)


def _solve_simplex_abundances(
    spectra: np.ndarray,
    endmembers: np.ndarray,
    *,
    initial: np.ndarray | None,
    steps: int,
    ridge: float,
) -> np.ndarray:
    """Projected least-squares abundance update for fixed endmembers."""

    values = np.asarray(spectra, dtype=np.float64)
    basis = np.asarray(endmembers, dtype=np.float64)
    rank = basis.shape[0]
    gram = basis @ basis.T
    if initial is None:
        normal = gram + max(float(ridge), 1e-12) * np.eye(rank, dtype=np.float64)
        unconstrained = np.linalg.solve(normal, basis @ values.T).T
        abundance = _project_rows_to_simplex(unconstrained)
    else:
        abundance = _project_rows_to_simplex(initial)
    lipschitz = max(float(np.linalg.norm(gram, ord=2)), 1e-12)
    for _ in range(max(1, int(steps))):
        gradient = (abundance @ basis - values) @ basis.T
        abundance = _project_rows_to_simplex(abundance - gradient / lipschitz)
    return abundance


def _simplex_objective(
    spectra: np.ndarray,
    abundance: np.ndarray,
    endmembers: np.ndarray,
    ridge: float,
) -> float:
    residual = abundance @ endmembers - spectra
    return float(
        0.5 * np.sum(residual * residual)
        + 0.5 * max(float(ridge), 0.0) * np.sum(endmembers * endmembers)
    )


def _update_nonnegative_endmembers(
    spectra: np.ndarray,
    abundance: np.ndarray,
    endmembers: np.ndarray,
    *,
    ridge: float,
    steps: int,
) -> np.ndarray:
    """Projected alternating least-squares update for fixed abundances."""

    values = np.asarray(spectra, dtype=np.float64)
    weights = np.asarray(abundance, dtype=np.float64)
    current = np.asarray(endmembers, dtype=np.float64)
    rank = current.shape[0]
    regularization = max(float(ridge), 0.0)
    gram = weights.T @ weights
    normal = gram + max(regularization, 1e-12) * np.eye(rank, dtype=np.float64)
    candidate = np.maximum(np.linalg.solve(normal, weights.T @ values), 0.0)
    if _simplex_objective(values, weights, candidate, regularization) <= _simplex_objective(
        values, weights, current, regularization
    ):
        current = candidate
    lipschitz = max(float(np.linalg.norm(normal, ord=2)), 1e-12)
    for _ in range(max(1, int(steps))):
        gradient = weights.T @ (weights @ current - values) + regularization * current
        proposal = np.maximum(current - gradient / lipschitz, 0.0)
        if _simplex_objective(values, weights, proposal, regularization) <= _simplex_objective(
            values, weights, current, regularization
        ) + 1e-12:
            current = proposal
        else:
            break
    return current


def fit_simplex_subspace(
    cube: np.ndarray,
    *,
    rank: int,
    max_pixels: int,
    random_seed: int,
    clip_quantiles: tuple[float, float],
    max_iterations: int = 100,
    abundance_steps: int = 8,
    endmember_steps: int = 4,
    tolerance: float = 1e-6,
    ridge: float = 1e-8,
) -> tuple[SubspaceModel, np.ndarray]:
    """Fit a nonnegative endmember/simplex-abundance prototype.

    The result is compatible with :func:`reconstruct`: the model mean is
    exactly zero, ``basis`` stores endmembers in ``(component, band)`` order,
    and the returned coefficient cube contains per-pixel abundances.  This is
    an independent prototype and does not replace the PCA path used by the
    current pipeline.
    """

    values = np.asarray(cube, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 1:
        raise ValueError("cube must have shape (height, width, spectral_band)")
    if not np.isfinite(values).all():
        raise ValueError("simplex factorization requires a finite cube")
    minimum = float(np.min(values))
    if minimum < -1e-7:
        raise ValueError(f"simplex factorization requires nonnegative input, minimum={minimum}")
    values = np.maximum(values, 0.0)
    height, width, bands = values.shape
    flat = values.reshape(-1, bands)
    pixel_count = flat.shape[0]
    requested_rank = int(rank)
    if requested_rank < 1 or requested_rank > min(pixel_count, bands):
        raise ValueError(
            f"rank must lie in [1, {min(pixel_count, bands)}], got {requested_rank}"
        )
    if int(max_pixels) < requested_rank:
        raise ValueError("max_pixels must be at least rank")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance cannot be negative")
    if len(clip_quantiles) != 2:
        raise ValueError("clip_quantiles must contain exactly two values")
    q_low, q_high = map(float, clip_quantiles)
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("clip_quantiles must satisfy 0 <= low < high <= 1")
    if not np.any(flat > 0.0):
        raise ValueError("simplex factorization cannot identify endmembers in an all-zero cube")

    rng = np.random.default_rng(int(random_seed))
    fit_count = min(int(max_pixels), pixel_count)
    if fit_count < pixel_count:
        fit_indices = np.sort(rng.choice(pixel_count, size=fit_count, replace=False))
        fit_spectra = flat[fit_indices]
    else:
        fit_spectra = flat
    endmembers = _farthest_spectra_initialization(fit_spectra, requested_rank, rng)
    abundance = _solve_simplex_abundances(
        fit_spectra,
        endmembers,
        initial=None,
        steps=max(12, int(abundance_steps)),
        ridge=ridge,
    )
    initial_rmse = float(np.sqrt(np.mean((abundance @ endmembers - fit_spectra) ** 2)))
    objective_history = [
        _simplex_objective(fit_spectra, abundance, endmembers, ridge)
    ]
    converged = False
    completed_iterations = 0
    for iteration in range(int(max_iterations)):
        endmembers = _update_nonnegative_endmembers(
            fit_spectra,
            abundance,
            endmembers,
            ridge=ridge,
            steps=endmember_steps,
        )
        abundance = _solve_simplex_abundances(
            fit_spectra,
            endmembers,
            initial=abundance,
            steps=abundance_steps,
            ridge=ridge,
        )
        objective = _simplex_objective(fit_spectra, abundance, endmembers, ridge)
        objective_history.append(objective)
        completed_iterations = iteration + 1
        previous = objective_history[-2]
        relative_improvement = (previous - objective) / max(abs(previous), 1e-15)
        if completed_iterations >= 5 and 0.0 <= relative_improvement <= float(tolerance):
            converged = True
            break

    full_abundance = _solve_simplex_abundances(
        flat,
        endmembers,
        initial=(abundance if fit_count == pixel_count else None),
        steps=max(24, 3 * int(abundance_steps)),
        ridge=ridge,
    )
    reconstruction = full_abundance @ endmembers
    residual = reconstruction - flat
    final_rmse = float(np.sqrt(np.mean(residual * residual)))
    relative_rmse = float(
        np.linalg.norm(residual) / max(float(np.linalg.norm(flat)), 1e-15)
    )
    centered = flat - np.mean(flat, axis=0, keepdims=True)
    total_variance = float(np.sum(centered * centered))
    explained_total = float(
        np.clip(1.0 - np.sum(residual * residual) / max(total_variance, 1e-15), 0.0, 1.0)
    )
    component_energy = (
        np.var(full_abundance, axis=0) * np.sum(endmembers * endmembers, axis=1)
    )
    if float(np.sum(component_energy)) <= 1e-15:
        component_fraction = np.full(requested_rank, 1.0 / requested_rank)
    else:
        component_fraction = component_energy / np.sum(component_energy)
    explained = (component_fraction * explained_total).astype(np.float32)
    endmembers32 = np.asarray(endmembers, dtype=np.float32)
    clip_min = np.maximum(0.0, np.quantile(flat, q_low, axis=0)).astype(np.float32)
    clip_max = np.maximum(
        clip_min,
        np.quantile(flat, q_high, axis=0).astype(np.float32),
    )
    model = SubspaceModel(
        mean_spectrum=np.zeros(bands, dtype=np.float32),
        basis=endmembers32,
        explained_variance_ratio=explained,
        clip_min=clip_min,
        clip_max=clip_max,
        representation="nonnegative_endmember_simplex_abundance",
        fit_metadata={
            "algorithm": "projected_alternating_least_squares",
            "random_seed": int(random_seed),
            "rank": requested_rank,
            "total_pixel_count": int(pixel_count),
            "fit_pixel_count": int(fit_count),
            "max_iterations": int(max_iterations),
            "completed_iterations": int(completed_iterations),
            "converged": bool(converged),
            "tolerance": float(tolerance),
            "ridge": float(ridge),
            "clip_quantiles": [q_low, q_high],
            "abundance_steps": int(abundance_steps),
            "endmember_steps": int(endmember_steps),
            "initial_fit_rmse": initial_rmse,
            "final_full_rmse": final_rmse,
            "relative_full_rmse": relative_rmse,
            "explained_variance_total": explained_total,
            "objective_history": [float(value) for value in objective_history],
            "simplex_max_sum_error": float(
                np.max(np.abs(np.sum(full_abundance, axis=1) - 1.0))
            ),
            "abundance_minimum": float(np.min(full_abundance)),
            "endmember_minimum": float(np.min(endmembers)),
        },
    )
    return model, full_abundance.reshape(height, width, requested_rank).astype(np.float32)


def fit_hybrid_simplex_subspace(
    cube: np.ndarray,
    *,
    rank: int,
    residual_rank: int,
    max_pixels: int,
    random_seed: int,
    clip_quantiles: tuple[float, float],
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[SubspaceModel, np.ndarray]:
    """Fit simplex abundances plus an observation-supported PCA residual.

    Only the first ``rank`` coefficients are physical simplex abundances.  The
    appended PCA coefficients explain spectral structure that the linear
    endmember convex hull cannot represent.  Fusion code must never inject RGB
    detail into those appended residual coefficients.
    """

    values = np.asarray(cube, dtype=np.float32)
    simplex_model, abundance = fit_simplex_subspace(
        values,
        rank=rank,
        max_pixels=max_pixels,
        random_seed=random_seed,
        clip_quantiles=clip_quantiles,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    simplex_reconstruction = reconstruct(abundance, simplex_model, clip=False)
    residual_cube = (values - simplex_reconstruction).astype(np.float32)
    residual_model, residual_coefficients = fit_subspace(
        residual_cube,
        rank=residual_rank,
        max_pixels=max_pixels,
        random_seed=random_seed + 104729,
        clip_quantiles=clip_quantiles,
    )
    coefficients = np.concatenate(
        [abundance, residual_coefficients], axis=2
    ).astype(np.float32)
    basis = np.concatenate(
        [simplex_model.basis, residual_model.basis], axis=0
    ).astype(np.float32)
    mean_spectrum = residual_model.mean_spectrum.astype(np.float32)
    reconstructed = (
        np.einsum("...k,kb->...b", coefficients, basis, optimize=True)
        + mean_spectrum[None, None, :]
    ).astype(np.float32)
    error = reconstructed - values
    centered = values - np.mean(values, axis=(0, 1), keepdims=True)
    explained_total = float(
        np.clip(
            1.0
            - np.sum(error.astype(np.float64) ** 2)
            / max(float(np.sum(centered.astype(np.float64) ** 2)), 1e-15),
            0.0,
            1.0,
        )
    )
    component_energy = np.var(
        coefficients.reshape(-1, coefficients.shape[2]).astype(np.float64),
        axis=0,
    ) * np.sum(basis.astype(np.float64) ** 2, axis=1)
    if float(np.sum(component_energy)) <= 1e-15:
        component_fraction = np.full(
            coefficients.shape[2], 1.0 / coefficients.shape[2]
        )
    else:
        component_fraction = component_energy / np.sum(component_energy)
    explained = (component_fraction * explained_total).astype(np.float32)
    model = SubspaceModel(
        mean_spectrum=mean_spectrum,
        basis=basis,
        explained_variance_ratio=explained,
        clip_min=simplex_model.clip_min.copy(),
        clip_max=simplex_model.clip_max.copy(),
        representation="simplex_abundance_plus_pca_observation_residual",
        fit_metadata={
            "algorithm": "simplex_als_plus_randomized_svd_residual",
            "simplex_component_count": int(rank),
            "residual_component_count": int(residual_coefficients.shape[2]),
            "total_component_count": int(coefficients.shape[2]),
            "simplex_fit": dict(simplex_model.fit_metadata),
            "residual_pca_explained_variance_total": float(
                np.sum(residual_model.explained_variance_ratio)
            ),
            "simplex_only_rmse": float(
                np.sqrt(np.mean((simplex_reconstruction - values) ** 2))
            ),
            "final_full_rmse": float(np.sqrt(np.mean(error * error))),
            "explained_variance_total": explained_total,
            "simplex_max_sum_error": float(
                np.max(np.abs(np.sum(abundance, axis=2) - 1.0))
            ),
            "abundance_minimum": float(np.min(abundance)),
            "rgb_detail_allowed_components": [0, int(rank)],
            "rgb_detail_forbidden_components": [
                int(rank),
                int(coefficients.shape[2]),
            ],
        },
    )
    return model, coefficients


def fit_subspace(
    cube: np.ndarray,
    *,
    rank: int,
    max_pixels: int,
    random_seed: int,
    clip_quantiles: tuple[float, float],
) -> tuple[SubspaceModel, np.ndarray]:
    arr = np.asarray(cube, dtype=np.float32)
    h, w, bands = arr.shape
    flat = arr.reshape(-1, bands)
    valid = np.isfinite(flat).all(axis=1)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 4:
        raise ValueError("Not enough finite spectra for low-rank fitting")
    rng = np.random.default_rng(random_seed)
    sample_idx = rng.choice(valid_idx, size=min(max_pixels, valid_idx.size), replace=False)
    sample = flat[sample_idx].astype(np.float64)
    mean = np.mean(sample, axis=0)
    centered = sample - mean
    k = max(1, min(int(rank), bands, sample.shape[0] - 1))
    _, singular, vt = randomized_svd(centered, n_components=k, random_state=random_seed, n_iter=5)
    variance = singular**2
    total_variance = float(np.sum(np.var(centered, axis=0, ddof=1)) * max(centered.shape[0] - 1, 1))
    explained = variance / max(total_variance, 1e-12)
    basis = vt.astype(np.float32)
    filled = np.nan_to_num(flat, nan=mean).astype(np.float32)
    coeff = (filled - mean.astype(np.float32)) @ basis.T
    q_low, q_high = clip_quantiles
    clip_min = np.maximum(0.0, np.quantile(sample, q_low, axis=0)).astype(np.float32)
    clip_max = np.quantile(sample, q_high, axis=0).astype(np.float32)
    model = SubspaceModel(
        mean_spectrum=mean.astype(np.float32),
        basis=basis,
        explained_variance_ratio=explained.astype(np.float32),
        clip_min=clip_min,
        clip_max=clip_max,
    )
    return model, coeff.reshape(h, w, k).astype(np.float32)


def reconstruct(coeff: np.ndarray, model: SubspaceModel, *, clip: bool = True) -> np.ndarray:
    arr = np.asarray(coeff, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[2])
    spectra = flat @ model.basis + model.mean_spectrum
    if clip:
        spectra = np.clip(spectra, model.clip_min, model.clip_max)
    return spectra.reshape(arr.shape[0], arr.shape[1], model.basis.shape[1]).astype(np.float32)

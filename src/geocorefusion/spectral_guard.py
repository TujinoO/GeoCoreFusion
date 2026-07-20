"""Candidate-independent spectral safety projections.

This module contains geometry only.  It does not inspect high-resolution
truth, RGB imagery, fusion configuration, or candidate provenance, and it
never enables spatial-detail injection.  Callers must decide independently
whether a candidate should be generated or accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, overload

import numpy as np


@dataclass(frozen=True, slots=True)
class SpectralConeDiagnostics:
    """Aggregate diagnostics for :func:`project_spectral_cone`."""

    max_angle_deg: float
    spectrum_count: int
    valid_reference_count: int
    zero_reference_count: int
    cone_clipped_count: int
    cone_clip_fraction: float
    mean_orthogonal_scale_all_valid: float
    mean_orthogonal_scale_clipped: float
    angle_before_median_deg: float
    angle_before_p95_deg: float
    angle_after_max_deg: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@overload
def project_spectral_cone(
    candidate: np.ndarray,
    reference: np.ndarray,
    max_angle_deg: float,
    *,
    return_diagnostics: Literal[False] = False,
) -> np.ndarray: ...


@overload
def project_spectral_cone(
    candidate: np.ndarray,
    reference: np.ndarray,
    max_angle_deg: float,
    *,
    return_diagnostics: Literal[True],
) -> tuple[np.ndarray, SpectralConeDiagnostics]: ...


def project_spectral_cone(
    candidate: np.ndarray,
    reference: np.ndarray,
    max_angle_deg: float,
    *,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, SpectralConeDiagnostics]:
    """Limit spectral-axis deviation while preserving parallel detail.

    Parameters
    ----------
    candidate, reference:
        Same-shape arrays whose last dimension contains spectral bands.  Any
        leading dimensions are treated as independent spectra, so a single
        ``(bands,)`` vector, an image ``(height, width, bands)``, or a batched
        image array are all accepted.
    max_angle_deg:
        Closed cone half-angle in degrees.  Zero is allowed and removes every
        orthogonal component; values must be finite and lie in ``[0, 90)``.
    return_diagnostics:
        If true, return ``(projected, diagnostics)``.  The default returns only
        the projected array.

    Notes
    -----
    Let ``r`` be a reference spectrum and ``d = candidate - r``.  The function
    decomposes ``d = d_parallel + d_orthogonal`` and returns

    ``r + d_parallel + scale * d_orthogonal``.

    Thus the candidate-minus-reference parallel component, including common
    shading, is retained.  ``scale`` is reduced only when needed to make the
    resulting spectrum lie within ``max_angle_deg`` of the positive reference
    ray.  For a zero reference the axis is undefined, so the function fails
    closed to the reference vector.  A candidate with a negative projection
    on a nonzero reference cannot simultaneously keep its parallel component
    and lie in a cone narrower than 90 degrees; such input raises ``ValueError``.

    The operation is a pure safety guard.  It does not use truth and does not
    create or authorize RGB-derived detail.
    """

    source = np.asarray(candidate)
    axis = np.asarray(reference)
    if source.shape != axis.shape:
        raise ValueError(
            "candidate and reference must have identical shapes; "
            f"got {source.shape} and {axis.shape}"
        )
    if source.ndim < 1 or source.shape[-1] < 1 or source.size == 0:
        raise ValueError("candidate and reference must contain at least one spectrum")
    try:
        angle = float(max_angle_deg)
    except (TypeError, ValueError) as error:
        raise ValueError("max_angle_deg must be a finite real number") from error
    if not math.isfinite(angle) or not 0.0 <= angle < 90.0:
        raise ValueError("max_angle_deg must be finite and lie in [0, 90)")
    if not np.issubdtype(source.dtype, np.number) or not np.issubdtype(
        axis.dtype, np.number
    ):
        raise TypeError("candidate and reference must be numeric arrays")
    if np.issubdtype(source.dtype, np.complexfloating) or np.issubdtype(
        axis.dtype, np.complexfloating
    ):
        raise TypeError("candidate and reference must be real-valued arrays")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(axis)):
        raise ValueError("candidate and reference must contain only finite values")

    output_dtype = np.result_type(source.dtype, axis.dtype, np.float32)
    flat_source = source.astype(output_dtype, copy=False).reshape(
        -1, source.shape[-1]
    )
    flat_axis = axis.astype(output_dtype, copy=False).reshape(-1, axis.shape[-1])
    projected = flat_source.copy()
    tangent = math.tan(math.radians(angle))
    dtype_epsilon = float(np.finfo(output_dtype).eps)
    # Pull a clipped vector very slightly inside the boundary so casting to
    # float32 cannot move it materially outside the requested cone.
    boundary_safety = max(0.0, 1.0 - 8.0 * dtype_epsilon)

    clipped_count = 0
    valid_reference_count = 0
    zero_reference_count = 0
    valid_scale_sum = 0.0
    clipped_scale_sum = 0.0
    before_angles: list[np.ndarray] = []
    after_max = 0.0

    for start in range(0, flat_source.shape[0], 8192):
        stop = min(flat_source.shape[0], start + 8192)
        current = flat_source[start:stop].astype(np.float64)
        base = flat_axis[start:stop].astype(np.float64)
        delta = current - base
        base_norm_sq = np.sum(base * base, axis=1)
        valid = base_norm_sq > np.finfo(np.float64).tiny
        zero = ~valid
        valid_reference_count += int(np.sum(valid))
        zero_reference_count += int(np.sum(zero))
        projected[start:stop][zero] = flat_axis[start:stop][zero]
        if not np.any(valid):
            continue

        delta_parallel_scale = np.divide(
            np.sum(delta * base, axis=1),
            base_norm_sq,
            out=np.zeros_like(base_norm_sq),
            where=valid,
        )
        axial_scale = 1.0 + delta_parallel_scale
        negative_projection = valid & (axial_scale < -32.0 * dtype_epsilon)
        if np.any(negative_projection):
            raise ValueError(
                "candidate has a negative projection on a nonzero reference; "
                "parallel preservation is incompatible with a positive spectral cone"
            )
        axial_scale = np.maximum(axial_scale, 0.0)
        parallel = axial_scale[:, None] * base
        orthogonal = delta - delta_parallel_scale[:, None] * base
        orthogonal_norm = np.linalg.norm(orthogonal, axis=1)
        axial_norm = axial_scale * np.sqrt(np.maximum(base_norm_sq, 0.0))
        allowed = axial_norm * tangent
        needs_clip = valid & (orthogonal_norm > allowed)
        scale = np.ones_like(orthogonal_norm)
        scale[needs_clip] = np.divide(
            allowed[needs_clip] * boundary_safety,
            orthogonal_norm[needs_clip],
            out=np.zeros(int(np.sum(needs_clip)), dtype=np.float64),
            where=orthogonal_norm[needs_clip] > 0.0,
        )
        # Leave already-valid spectra bit-for-bit unchanged.  Only clipped
        # spectra and zero-reference fallbacks are written.
        clipped = parallel[needs_clip] + scale[needs_clip, None] * orthogonal[
            needs_clip
        ]
        projected[start:stop][needs_clip] = clipped.astype(output_dtype)

        clipped_count += int(np.sum(needs_clip))
        valid_scale_sum += float(np.sum(scale[valid]))
        clipped_scale_sum += float(np.sum(scale[needs_clip]))
        before_angle = np.degrees(
            np.arctan2(
                orthogonal_norm[valid],
                np.maximum(axial_norm[valid], np.finfo(np.float64).tiny),
            )
        )
        before_angles.append(before_angle.astype(np.float32))

        actual = projected[start:stop].astype(np.float64)
        actual_axial_scale = np.divide(
            np.sum(actual * base, axis=1),
            base_norm_sq,
            out=np.zeros_like(base_norm_sq),
            where=valid,
        )
        actual_parallel = actual_axial_scale[:, None] * base
        actual_orthogonal = actual - actual_parallel
        actual_angle = np.degrees(
            np.arctan2(
                np.linalg.norm(actual_orthogonal[valid], axis=1),
                np.maximum(
                    actual_axial_scale[valid]
                    * np.sqrt(base_norm_sq[valid]),
                    np.finfo(np.float64).tiny,
                ),
            )
        )
        if actual_angle.size:
            after_max = max(after_max, float(np.max(actual_angle)))

    all_before = (
        np.concatenate(before_angles)
        if before_angles
        else np.asarray([], dtype=np.float32)
    )
    diagnostics = SpectralConeDiagnostics(
        max_angle_deg=angle,
        spectrum_count=int(flat_source.shape[0]),
        valid_reference_count=int(valid_reference_count),
        zero_reference_count=int(zero_reference_count),
        cone_clipped_count=int(clipped_count),
        cone_clip_fraction=float(
            clipped_count / max(valid_reference_count, 1)
        ),
        mean_orthogonal_scale_all_valid=float(
            valid_scale_sum / max(valid_reference_count, 1)
        ),
        mean_orthogonal_scale_clipped=float(
            clipped_scale_sum / max(clipped_count, 1)
        ),
        angle_before_median_deg=(
            float(np.median(all_before)) if all_before.size else 0.0
        ),
        angle_before_p95_deg=(
            float(np.percentile(all_before, 95.0)) if all_before.size else 0.0
        ),
        angle_after_max_deg=float(after_max),
    )
    restored = projected.reshape(source.shape)
    if return_diagnostics:
        return restored, diagnostics
    return restored


__all__ = ["SpectralConeDiagnostics", "project_spectral_cone"]

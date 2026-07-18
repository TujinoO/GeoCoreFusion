"""Low-rank spectral basis and material-coefficient representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.utils.extmath import randomized_svd


@dataclass(slots=True)
class SubspaceModel:
    mean_spectrum: np.ndarray
    basis: np.ndarray
    explained_variance_ratio: np.ndarray
    clip_min: np.ndarray
    clip_max: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.basis.shape[0]),
            "band_count": int(self.basis.shape[1]),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "explained_variance_total": float(self.explained_variance_ratio.sum()),
            "mean_spectrum": self.mean_spectrum.tolist(),
        }


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

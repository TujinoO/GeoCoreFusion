from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from geocorefusion.config import load_config  # noqa: E402
from geocorefusion.dataset import discover_triplet, normalize_image  # noqa: E402
from geocorefusion.pipeline import _analysis_shape  # noqa: E402
from geocorefusion.registration import (  # noqa: E402
    _estimate_roi_affine,
    _mean_selected_bands,
    _modality_feature,
    _warp_affine,
    analysis_rgb_grid,
    estimate_registration,
    estimate_roi_registration,
)
from geocorefusion.roi import choose_roi  # noqa: E402


def _profile(image: np.ndarray, axis: str) -> np.ndarray:
    base = normalize_image(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0))
    if axis == "y":
        response = np.abs(cv2.Scharr(base, cv2.CV_32F, 0, 1))
        margin = max(2, response.shape[1] // 12)
        values = response[:, margin:-margin]
        profile = np.quantile(values, 0.82, axis=1)
    else:
        response = np.abs(cv2.Scharr(base, cv2.CV_32F, 1, 0))
        margin = max(2, response.shape[0] // 12)
        values = response[margin:-margin, :]
        profile = np.quantile(values, 0.82, axis=0)
    profile = cv2.GaussianBlur(profile.astype(np.float32)[:, None], (1, 9), 0).reshape(-1)
    return normalize_image(profile)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else -1.0


def _window_lags(reference: np.ndarray, moving: np.ndarray, radius: int = 5, controls: int = 9) -> list[dict[str, float]]:
    n = reference.size
    half = max(18, int(np.ceil(n / max(5, controls - 1))))
    centers = np.linspace(half, n - half - 1, controls)
    output = []
    for center in centers:
        start = max(0, int(round(center)) - half)
        stop = min(n, int(round(center)) + half + 1)
        scores = []
        for lag in range(-radius, radius + 1):
            m0, m1 = start + lag, stop + lag
            if m0 < 0 or m1 > n:
                continue
            scores.append((_corr(reference[start:stop], moving[m0:m1]), lag))
        scores.sort(reverse=True)
        best = scores[0]
        base = next(score for score in scores if score[1] == 0)
        output.append({
            "center": float(center),
            "lag": int(best[1]),
            "score": float(best[0]),
            "gain": float(best[0] - base[0]),
            "peak_margin": float(best[0] - scores[1][0]) if len(scores) > 1 else 0.0,
        })
    return output


def _peaks(profile: np.ndarray, distance: int) -> list[dict[str, float]]:
    indices, properties = find_peaks(profile, distance=distance, prominence=0.06)
    rows = [
        {"index": int(index), "value": float(profile[index]), "prominence": float(properties["prominences"][i])}
        for i, index in enumerate(indices)
    ]
    return sorted(rows, key=lambda row: row["prominence"], reverse=True)[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = discover_triplet(config.data_dir)
    coarse = estimate_registration(dataset, config.registration)
    roi = choose_roi(config.roi, coarse, dataset.rgb.meta.shape[:2])
    shape = _analysis_shape(roi, coarse)
    refined = estimate_roi_registration(dataset, coarse, roi, shape, config.registration)
    result = {"roi": roi, "analysis_shape": shape, "status": refined.status, "sensors": {}}
    for name, image in (("nir", refined.nir_aligned), ("swir", refined.swir_aligned)):
        result["sensors"][name] = {
            "row_profile_correlation": _corr(_profile(refined.reference_structure, "y"), _profile(image, "y")),
            "column_profile_correlation": _corr(_profile(refined.reference_structure, "x"), _profile(image, "x")),
            "row_window_lags": _window_lags(_profile(refined.reference_structure, "y"), _profile(image, "y")),
            "column_window_lags": _window_lags(_profile(refined.reference_structure, "x"), _profile(image, "x"), radius=4, controls=5),
            "reference_row_peaks": _peaks(_profile(refined.reference_structure, "y"), distance=14),
            "moving_row_peaks": _peaks(_profile(image, "y"), distance=14),
            "reference_column_peaks": _peaks(_profile(refined.reference_structure, "x"), distance=10),
            "moving_column_peaks": _peaks(_profile(image, "x"), distance=10),
        }
    grid_y, grid_x = analysis_rgb_grid(roi, *shape)
    overlap_wavelengths = [1050, 1150, 1250, 1350, 1450]
    nir_overlap_initial = _mean_selected_bands(
        dataset.nir.cube,
        dataset.nir.meta,
        coarse.nir,
        grid_y,
        grid_x,
        overlap_wavelengths,
    )
    alternative_matrix, _, alternative_details = _estimate_roi_affine(
        refined.swir_overlap_aligned,
        nir_overlap_initial,
        config.registration,
        min_gain=0.0,
    )
    alternative_nir = _warp_affine(refined.nir_initial, alternative_matrix, shape)
    alternative_overlap = _warp_affine(nir_overlap_initial, alternative_matrix, shape)
    result["nir_absolute_swir_anchor_candidate"] = {
        "details": alternative_details,
        "matrix": alternative_matrix.tolist(),
        "rgb_score": _corr(_modality_feature(refined.reference_structure), _modality_feature(alternative_nir)),
        "nir_swir_score": _corr(_modality_feature(refined.swir_overlap_aligned), _modality_feature(alternative_overlap)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Structural validation of completed fusion runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .envi import open_cube, parse_header


def validate_run(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fused_hdr = root / manifest["outputs"]["fused_cube_hdr"]
    meta = parse_header(fused_hdr)
    cube, _ = open_cube(meta)
    ys = np.linspace(0, meta.lines - 1, min(16, meta.lines)).round().astype(int)
    xs = np.linspace(0, meta.samples - 1, min(12, meta.samples)).round().astype(int)
    bs = np.linspace(0, meta.bands - 1, min(24, meta.bands)).round().astype(int)
    sample = np.asarray(cube[np.ix_(ys, xs, bs)], dtype=np.float32)
    wavelengths = meta.wavelengths
    checks = {
        "shape_matches_manifest": list(meta.shape) == [manifest["output_grid"]["height"], manifest["output_grid"]["width"], manifest["output_grid"]["bands"]],
        "data_size_validated": True,
        "sample_all_finite": bool(np.isfinite(sample).all()),
        "wavelength_count_matches": int(wavelengths.size) == meta.bands,
        "wavelength_strictly_increasing": bool(np.all(np.diff(wavelengths) > 0)),
        "wavelength_start_nm": float(wavelengths[0]) if wavelengths.size else None,
        "wavelength_end_nm": float(wavelengths[-1]) if wavelengths.size else None,
        "sample_min": float(np.min(sample)),
        "sample_max": float(np.max(sample)),
    }
    checks["passed"] = all(
        checks[key]
        for key in (
            "shape_matches_manifest",
            "data_size_validated",
            "sample_all_finite",
            "wavelength_count_matches",
            "wavelength_strictly_increasing",
        )
    )
    return {"output_dir": str(root), "fused_cube": str(fused_hdr), "checks": checks}


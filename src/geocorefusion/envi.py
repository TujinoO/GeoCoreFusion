"""ENVI metadata, memory mapping, spatial sampling, and streaming output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ENVI_TO_DTYPE = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 6: "c8", 9: "c16", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}
DTYPE_TO_ENVI = {
    np.dtype("uint8"): 1, np.dtype("int16"): 2, np.dtype("int32"): 3,
    np.dtype("float32"): 4, np.dtype("float64"): 5, np.dtype("uint16"): 12,
    np.dtype("uint32"): 13, np.dtype("int64"): 14, np.dtype("uint64"): 15,
}


def _format_envi_list(name: str, values: list[str], *, per_line: int = 12) -> str:
    lines = [f"{name} = {{"]
    for start in range(0, len(values), per_line):
        chunk = values[start:start + per_line]
        suffix = "," if start + per_line < len(values) else ""
        lines.append("  " + ", ".join(chunk) + suffix)
    lines.append("}")
    return "\n".join(lines) + "\n"


@dataclass(slots=True)
class EnviMetadata:
    hdr_path: Path
    data_path: Path
    samples: int
    lines: int
    bands: int
    data_type: int
    interleave: str
    byte_order: int
    header_offset: int
    wavelengths: np.ndarray
    fwhm: np.ndarray
    wavelength_units: str
    raw: dict[str, str]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.lines, self.samples, self.bands

    @property
    def dtype(self) -> np.dtype:
        code = ENVI_TO_DTYPE.get(self.data_type)
        if code is None:
            raise ValueError(f"Unsupported ENVI data type {self.data_type}")
        if code == "u1":
            return np.dtype(code)
        return np.dtype(("<" if self.byte_order == 0 else ">") + code)


def _parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.lower() == "envi" or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        depth = value.count("{") - value.count("}")
        while depth > 0 and i < len(lines):
            nxt = lines[i].strip()
            i += 1
            value += " " + nxt
            depth += nxt.count("{") - nxt.count("}")
        fields[key] = value.strip().strip("{}").strip()
    return fields


def _float_array(value: str | None) -> np.ndarray:
    if not value:
        return np.empty(0, dtype=np.float64)
    out: list[float] = []
    for token in re.split(r"[,\s]+", value.strip()):
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return np.asarray(out, dtype=np.float64)


def infer_data_path(hdr_path: str | Path) -> Path:
    hdr = Path(hdr_path)
    candidates = [hdr.with_suffix(""), hdr.with_suffix(".dat"), hdr.with_suffix(".raw"), hdr.with_suffix(".img"), hdr.with_suffix(".bin")]
    for candidate in candidates:
        if candidate.exists() and candidate != hdr:
            return candidate
    raise FileNotFoundError(f"Cannot find ENVI binary associated with {hdr}")


def parse_header(hdr_path: str | Path, data_path: str | Path | None = None) -> EnviMetadata:
    hdr = Path(hdr_path)
    fields = _parse_fields(hdr.read_text(encoding="utf-8", errors="replace"))
    required = ("samples", "lines", "bands", "data type", "interleave")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"Missing ENVI fields {missing} in {hdr}")
    data = Path(data_path) if data_path else infer_data_path(hdr)
    meta = EnviMetadata(
        hdr_path=hdr,
        data_path=data,
        samples=int(fields["samples"]),
        lines=int(fields["lines"]),
        bands=int(fields["bands"]),
        data_type=int(fields["data type"]),
        interleave=fields["interleave"].lower(),
        byte_order=int(fields.get("byte order", 0)),
        header_offset=int(fields.get("header offset", 0)),
        wavelengths=_float_array(fields.get("wavelength")),
        fwhm=_float_array(fields.get("fwhm")),
        wavelength_units=fields.get("wavelength units", "Unknown"),
        raw=fields,
    )
    expected = int(np.prod(meta.shape)) * meta.dtype.itemsize + meta.header_offset
    actual = data.stat().st_size
    if actual < expected:
        raise ValueError(f"{data} is smaller than header expectation: {actual} < {expected}")
    return meta


def open_cube(meta_or_hdr: EnviMetadata | str | Path, *, mode: str = "r") -> tuple[np.ndarray, EnviMetadata]:
    meta = meta_or_hdr if isinstance(meta_or_hdr, EnviMetadata) else parse_header(meta_or_hdr)
    count = meta.lines * meta.samples * meta.bands
    raw = np.memmap(meta.data_path, dtype=meta.dtype, mode=mode, offset=meta.header_offset, shape=(count,))
    if meta.interleave == "bil":
        cube = raw.reshape(meta.lines, meta.bands, meta.samples).transpose(0, 2, 1)
    elif meta.interleave == "bip":
        cube = raw.reshape(meta.lines, meta.samples, meta.bands)
    elif meta.interleave == "bsq":
        cube = raw.reshape(meta.bands, meta.lines, meta.samples).transpose(1, 2, 0)
    else:
        raise ValueError(f"Unsupported ENVI interleave {meta.interleave}")
    return cube, meta


def create_bip_writer(
    hdr_path: str | Path,
    shape: tuple[int, int, int],
    *,
    dtype: str | np.dtype = "float32",
    wavelengths: np.ndarray | list[float] | None = None,
    description: str = "GeoCoreFusion output",
) -> tuple[np.memmap, Path, Path]:
    hdr = Path(hdr_path)
    dat = hdr.with_suffix(".dat")
    hdr.parent.mkdir(parents=True, exist_ok=True)
    np_dtype = np.dtype(dtype)
    if np_dtype not in DTYPE_TO_ENVI:
        raise ValueError(f"Unsupported ENVI output dtype {np_dtype}")
    cube = np.memmap(dat, dtype=np_dtype, mode="w+", shape=shape, order="C")
    wave_line = ""
    if wavelengths is not None:
        wavelength_values = [f"{float(v):.8g}" for v in wavelengths]
        band_names = [f"{float(v):.8g} nm" for v in wavelengths]
        wave_line = (
            _format_envi_list("wavelength", wavelength_values)
            + _format_envi_list("band names", band_names, per_line=8)
        )
    hdr.write_text(
        "ENVI\n"
        f"description = {{{description}}}\n"
        f"samples = {shape[1]}\nlines = {shape[0]}\nbands = {shape[2]}\n"
        "header offset = 0\nfile type = ENVI Standard\n"
        f"data type = {DTYPE_TO_ENVI[np_dtype]}\ninterleave = bip\nbyte order = 0\n"
        "wavelength units = Nanometers\n"
        + wave_line,
        encoding="utf-8",
    )
    return cube, hdr, dat


def metadata_dict(meta: EnviMetadata) -> dict[str, Any]:
    return {
        "hdr_path": str(meta.hdr_path), "data_path": str(meta.data_path),
        "samples": meta.samples, "lines": meta.lines, "bands": meta.bands,
        "data_type": meta.data_type, "interleave": meta.interleave,
        "byte_order": meta.byte_order, "header_offset": meta.header_offset,
        "wavelength_min_nm": float(meta.wavelengths.min()) if meta.wavelengths.size else None,
        "wavelength_max_nm": float(meta.wavelengths.max()) if meta.wavelengths.size else None,
        "wavelength_count": int(meta.wavelengths.size), "wavelength_units": meta.wavelength_units,
    }

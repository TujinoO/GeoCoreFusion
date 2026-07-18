from pathlib import Path

import numpy as np

from geocorefusion.envi import create_bip_writer, open_cube, parse_header


def test_envi_bip_roundtrip(tmp_path: Path) -> None:
    expected = np.arange(8 * 7 * 5, dtype=np.float32).reshape(8, 7, 5) / 100.0
    writer, hdr, _ = create_bip_writer(tmp_path / "cube.hdr", expected.shape, wavelengths=np.arange(5) * 5 + 700)
    writer[:] = expected
    writer.flush()
    del writer
    meta = parse_header(hdr)
    actual, _ = open_cube(meta)
    assert meta.interleave == "bip"
    assert meta.wavelengths.tolist() == [700, 705, 710, 715, 720]
    np.testing.assert_allclose(np.asarray(actual), expected)


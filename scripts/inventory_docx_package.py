"""Inventory and hash every ZIP part in a DOCX package."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()

    records = []
    with zipfile.ZipFile(args.input, "r") as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            payload = archive.read(info.filename)
            records.append(
                {
                    "path": info.filename,
                    "size": info.file_size,
                    "compressed_size": info.compress_size,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"parts={len(records)}")


if __name__ == "__main__":
    main()

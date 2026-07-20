"""Merge ordered Word-export PDF batches without modifying the inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pattern", default="V6_pages_*.pdf")
    args = parser.parse_args()

    inputs = sorted(args.input_dir.glob(args.pattern))
    if not inputs:
        raise FileNotFoundError(f"no PDF batches match {args.pattern!r} in {args.input_dir}")

    writer = PdfWriter()
    page_count = 0
    for path in inputs:
        reader = PdfReader(path)
        page_count += len(reader.pages)
        writer.append(reader)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        writer.write(stream)

    merged = PdfReader(args.output)
    if len(merged.pages) != page_count:
        raise RuntimeError(
            f"merged page count mismatch: expected {page_count}, got {len(merged.pages)}"
        )
    print(f"inputs={len(inputs)} pages={page_count} output={args.output}")


if __name__ == "__main__":
    main()

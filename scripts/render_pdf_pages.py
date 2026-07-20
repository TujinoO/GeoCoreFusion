"""Render all pages of a PDF to deterministic PNG files for document QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import pypdfium2 as pdfium


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpi", type=float, default=150.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(args.input))
    scale = args.dpi / 72.0
    for index in range(len(document)):
        page = document[index]
        bitmap = page.render(scale=scale, rotation=0)
        image = bitmap.to_pil()
        image.save(args.output_dir / f"page-{index + 1}.png")
        page.close()
    document.close()
    print(f"rendered_pages={index + 1 if 'index' in locals() else 0}")


if __name__ == "__main__":
    main()

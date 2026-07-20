"""Create labeled contact sheets from rendered document pages for visual QA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pages_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pages-per-sheet", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=420)
    args = parser.parse_args()

    pages = sorted(
        args.pages_dir.glob("page-*.png"),
        key=lambda path: int(path.stem.split("-")[-1]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cols = 2
    margin = 18
    label_h = 28
    for sheet_index in range(math.ceil(len(pages) / args.pages_per_sheet)):
        group = pages[
            sheet_index * args.pages_per_sheet : (sheet_index + 1) * args.pages_per_sheet
        ]
        thumbs = []
        for path in group:
            image = Image.open(path).convert("RGB")
            height = round(image.height * args.thumb_width / image.width)
            thumbs.append((path, image.resize((args.thumb_width, height), Image.Resampling.LANCZOS)))
        rows = math.ceil(len(thumbs) / cols)
        cell_h = max(image.height for _, image in thumbs) + label_h
        canvas = Image.new(
            "RGB",
            (cols * args.thumb_width + (cols + 1) * margin, rows * cell_h + (rows + 1) * margin),
            "#d7dbe0",
        )
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for index, (path, image) in enumerate(thumbs):
            row, col = divmod(index, cols)
            x = margin + col * (args.thumb_width + margin)
            y = margin + row * (cell_h + margin)
            canvas.paste(image, (x, y + label_h))
            draw.rectangle((x, y, x + args.thumb_width, y + label_h), fill="#202a35")
            draw.text((x + 8, y + 7), path.stem, fill="white", font=font)
        canvas.save(args.output_dir / f"contact-{sheet_index + 1}.png")
    print(f"contact_sheets={math.ceil(len(pages) / args.pages_per_sheet)}")


if __name__ == "__main__":
    main()

"""Extract a DOCX's readable structure without modifying the source file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(parent):
    parent_element = parent.element.body
    for child in parent_element.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def paragraph_record(paragraph: Paragraph, index: int) -> dict:
    p = paragraph._p
    return {
        "type": "paragraph",
        "index": index,
        "style": paragraph.style.name if paragraph.style else None,
        "text": paragraph.text,
        "alignment": str(paragraph.alignment) if paragraph.alignment is not None else None,
        "has_drawing": bool(p.xpath(".//w:drawing | .//w:pict")),
        "has_page_break": bool(p.xpath('.//w:br[@w:type="page"]')),
        "runs": [
            {
                "text": run.text,
                "bold": run.bold,
                "italic": run.italic,
                "size_pt": run.font.size.pt if run.font.size else None,
                "font": run.font.name,
                "color": str(run.font.color.rgb) if run.font.color and run.font.color.rgb else None,
            }
            for run in paragraph.runs
            if run.text or run._r.xpath(".//w:drawing | .//w:pict")
        ],
    }


def table_record(table: Table, index: int) -> dict:
    return {
        "type": "table",
        "index": index,
        "style": table.style.name if table.style else None,
        "rows": [[cell.text for cell in row.cells] for row in table.rows],
    }


def header_footer_records(section, section_index: int) -> dict:
    def paras(container):
        return [
            {"style": p.style.name if p.style else None, "text": p.text}
            for p in container.paragraphs
            if p.text.strip()
        ]

    return {
        "section": section_index,
        "header": paras(section.header),
        "first_page_header": paras(section.first_page_header),
        "even_page_header": paras(section.even_page_header),
        "footer": paras(section.footer),
        "first_page_footer": paras(section.first_page_footer),
        "even_page_footer": paras(section.even_page_footer),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--text", required=True, type=Path)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    doc = Document(args.input)
    blocks = []
    p_index = 0
    t_index = 0
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            blocks.append(paragraph_record(block, p_index))
            p_index += 1
        else:
            blocks.append(table_record(block, t_index))
            t_index += 1

    sections = []
    for i, section in enumerate(doc.sections, start=1):
        sections.append(
            {
                "index": i,
                "start_type": str(section.start_type),
                "orientation": str(section.orientation),
                "page_width_in": section.page_width.inches,
                "page_height_in": section.page_height.inches,
                "left_margin_in": section.left_margin.inches,
                "right_margin_in": section.right_margin.inches,
                "top_margin_in": section.top_margin.inches,
                "bottom_margin_in": section.bottom_margin.inches,
                "header_distance_in": section.header_distance.inches,
                "footer_distance_in": section.footer_distance.inches,
                "different_first_page": section.different_first_page_header_footer,
            }
        )

    payload = {
        "source": str(args.input.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "paragraph_count": p_index,
        "table_count": t_index,
        "inline_shape_count": len(doc.inline_shapes),
        "sections": sections,
        "headers_footers": [
            header_footer_records(section, i)
            for i, section in enumerate(doc.sections, start=1)
        ],
        "blocks": blocks,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"SOURCE: {payload['source']}",
        f"SHA256: {payload['sha256']}",
        f"PARAGRAPHS: {p_index}; TABLES: {t_index}; INLINE_SHAPES: {len(doc.inline_shapes)}",
        "",
    ]
    for block in blocks:
        if block["type"] == "paragraph":
            text = block["text"].replace("\n", " / ")
            if text.strip() or block["has_drawing"]:
                marker = " [FIGURE]" if block["has_drawing"] else ""
                lines.append(f"P{block['index']:04d} [{block['style']}]{marker} {text}")
        else:
            lines.append(f"T{block['index']:03d} [{block['style']}] {len(block['rows'])} rows")
            for row in block["rows"]:
                lines.append("  | " + " | ".join(cell.replace("\n", " / ") for cell in row))
    args.text.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

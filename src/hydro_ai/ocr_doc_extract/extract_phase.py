"""
PDF Extractor — pdfplumber based, no heavy ML dependencies.
Install: pip install pdfplumber
"""

import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber


DEFAULT_PDF_PATH = Path(
    r"C:\Users\ps302\OneDrive\Desktop\Hydrology\src\hydro_ai\data\raw\Pile Foundation_Part 2.pdf"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\ps302\OneDrive\Desktop\Hydrology\src\hydro_ai\data\processed"
)


def clean_text(text: str) -> str:
    """Normalize raw PDF text: remove (cid:N) garbage tokens, collapse blank lines."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")  # non-breaking space → regular space
    text = re.sub(
        r"\(cid:\d+\)", "", text
    )  # drop unmapped font chars (math symbols etc.)
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # collapse multiple blank lines into one
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_tables_from_page(page) -> List[Dict[str, Any]]:
    """
    Extract all tables on a page using pdfplumber's border-detection.
    Returns a list of dicts, each with:
      - rows: 2-D list of cell strings
      - markdown: pipe-delimited string for embedding
    """
    tables_found = []

    for table in page.extract_tables():
        if not table:
            continue

        # Replace None cells with empty string
        cleaned_rows = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in table
        ]

        # Build markdown: header row → separator → data rows
        md_lines = []
        for i, row in enumerate(cleaned_rows):
            md_lines.append(" | ".join(row))
            if i == 0:
                md_lines.append(" | ".join(["---"] * len(row)))

        tables_found.append(
            {
                "rows": cleaned_rows,
                "markdown": "\n".join(md_lines),
            }
        )

    return tables_found


def extract_page(page, page_num: int) -> Dict[str, Any]:
    """
    Extract text and tables from a single page.
    x/y_tolerance=3 groups characters within 3pt gaps — fixes spacing issues in slide PDFs.
    """
    raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    text = clean_text(raw_text)
    tables = extract_tables_from_page(page)

    return {
        "page": page_num,
        "text": text,
        "tables": tables,
        "has_tables": bool(tables),
    }


def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Process an entire PDF page-by-page.
    Returns a dict with per-page data and a combined full_text for RAG/embedding.
    """
    pdf_path = Path(pdf_path)
    pages_data = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        print(f"[INFO] '{pdf_path.name}' — {total} pages")

        for i, page in enumerate(pdf.pages, start=1):
            print(f"  → Page {i}/{total}...", end="\r")
            pages_data.append(extract_page(page, page_num=i))

    print()

    # Build full_text: page text blocks + table markdown blocks, tagged by page number
    parts = []
    for p in pages_data:
        if p["text"]:
            parts.append(f"[PAGE {p['page']}]\n{p['text']}")
        for t in p["tables"]:
            parts.append(f"[TABLE — PAGE {p['page']}]\n{t['markdown']}")

    return {
        "source": str(pdf_path),
        "file_name": pdf_path.name,
        "total_pages": total,
        "pages": pages_data,
        "full_text": "\n\n".join(parts),
    }


def flatten_for_embedding(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert extracted doc into flat records suitable for a vector DB / splitter.
    Each record carries text + metadata (source, page, kind).
    Tables are stored as markdown text so they embed like prose.
    """
    records = []

    for page in doc["pages"]:
        base = {
            "source": doc["source"],
            "file_name": doc["file_name"],
            "page": page["page"],
        }

        if page["text"]:
            records.append({**base, "kind": "text", "text": page["text"]})

        for idx, table in enumerate(page["tables"], start=1):
            records.append(
                {
                    **base,
                    "kind": "table",
                    "table_index": idx,
                    "text": table["markdown"],  # markdown embeds better than raw rows
                    "rows": table["rows"],  # keep structured rows for downstream use
                }
            )

    return records


def main(pdf_path: str) -> None:
    doc = extract_pdf(pdf_path)
    records = flatten_for_embedding(doc)

    text_count = sum(1 for r in records if r["kind"] == "text")
    table_count = sum(1 for r in records if r["kind"] == "table")
    print(
        f"\nRecords - total: {len(records)}  |  text: {text_count}  |  tables: {table_count}\n"
    )

    for r in records[:2]:
        print(f"[{r['kind'].upper()} | Page {r['page']}]\n{r['text'][:300]}\n...\n")

    out = DEFAULT_OUTPUT_DIR / f"{Path(pdf_path).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    input_pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    main(str(input_pdf))

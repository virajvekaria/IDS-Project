import pdfplumber
import json
import sys
import os
from typing import List, Dict


def extract_pdf_content(pdf_path: str) -> List[Dict]:
    """
    Extract structured text and tables from PDF preserving:
      - Section headers (e.g., If swallowed:)
      - Bullet points / multi-line content
    """
    pages_content = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            page_text = page.extract_text() or ""

            tables_data = []

            tables = page.extract_tables() or []

            for table in tables:
                structured_rows = []

                for row in table:
                    if not row or all(cell is None for cell in row):
                        continue

                    header = (row[0] or "").strip()

                    content_raw = ""
                    if len(row) > 1:
                        content_raw = (row[1] or "").strip()

                    content_items = [
                        item.strip("• ").strip()
                        for item in content_raw.split("\n")
                        if item.strip()
                    ]

                    structured_rows.append({
                        "header": header,
                        "content": content_items
                    })

                if structured_rows:
                    tables_data.append({
                        "rows": structured_rows
                    })

            page_data = {
                "page_number": page_num,
                "text": page_text.strip(),
                "tables": tables_data
            }

            pages_content.append(page_data)

    return pages_content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 1_extract_pdf_multi.py <input_pdf1> <input_pdf2> ...")
        sys.exit(1)

    input_pdfs = sys.argv[1:]

    for input_pdf in input_pdfs:
        if not os.path.exists(input_pdf):
            print(f"File not found: {input_pdf}")
            continue

        print(f"Processing: {input_pdf}")
        output_json = f"{os.path.splitext(input_pdf)[0]}_output.json"

        pages_content = extract_pdf_content(input_pdf)
        print(f"Extracted {len(pages_content)} pages from {input_pdf}.")

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(pages_content, f, ensure_ascii=False, indent=2)
        print(f"Saved extracted content to {output_json}.")

    print("Processing complete.")

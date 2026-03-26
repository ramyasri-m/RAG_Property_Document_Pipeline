import fitz  # PyMuPDF
import pdfplumber
import os
from pathlib import Path


def extract_text_pymupdf(pdf_path: str) -> dict:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_number": page_num + 1,
            "text": text.strip(),
            "char_count": len(text)
        })
    doc.close()
    return {
        "filename": Path(pdf_path).name,
        "total_pages": len(pages),
        "pages": pages
    }


def extract_tables_pdfplumber(pdf_path: str) -> list:
    """Extract tables from PDF using pdfplumber."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append({
                    "page_number": page_num + 1,
                    "table": table
                })
    return tables


def extract_document(pdf_path: str) -> dict:
    """Full document extraction combining text and tables."""
    print(f"Extracting: {pdf_path}")
    text_data = extract_text_pymupdf(pdf_path)
    tables = extract_tables_pdfplumber(pdf_path)
    text_data["tables"] = tables
    print(f"  Pages: {text_data['total_pages']}, Tables: {len(tables)}")
    return text_data


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = extract_document(sys.argv[1])
        for page in result["pages"]:
            print(f"\n--- Page {page['page_number']} ---")
            print(page["text"][:500])

from __future__ import annotations
import pathlib
import re
from typing import List, Dict

import PyPDF2


class DocumentAnalyzer:
    """Extracts text and simple metadata from PDF files."""

    def __init__(self, pdf_path: str | pathlib.Path):
        self.pdf_path = pathlib.Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.reader = PyPDF2.PdfReader(str(self.pdf_path))

    def extract_text(self) -> str:
        pages: List[str] = []
        for page in self.reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def extract_metadata(self) -> Dict[str, str]:
        meta = self.reader.metadata or {}
        return {k[1:]: v for k, v in meta.items()}  # Strip leading '/'

    def extract_references(self) -> List[str]:
        """Very naive reference extractor based on regex looking for DOI links."""
        text = self.extract_text()
        dois = re.findall(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.IGNORECASE)
        return list(set(dois))


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Analyze a medical PDF document")
    parser.add_argument("pdf", type=str, help="Path to PDF")
    args = parser.parse_args()
    analyzer = DocumentAnalyzer(args.pdf)
    data = {
        "metadata": analyzer.extract_metadata(),
        "references": analyzer.extract_references(),
    }
    print(json.dumps(data, indent=2))

from __future__ import annotations
import pathlib
import argparse
import json
from tqdm import tqdm

from doc_analyzer import DocumentAnalyzer

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "research_papers" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "research_papers" / "processed"
META_PATH = PROJECT_ROOT / "data" / "research_papers" / "metadata.json"

PROC_DIR.mkdir(parents=True, exist_ok=True)


def process_papers() -> None:
    meta: dict = {}
    for pdf_path in tqdm(list(RAW_DIR.glob("*.pdf")), desc="Processing papers"):
        analyzer = DocumentAnalyzer(pdf_path)
        text = analyzer.extract_text()
        out_txt = PROC_DIR / (pdf_path.stem + ".txt")
        out_txt.write_text(text, encoding="utf-8")
        meta[pdf_path.name] = analyzer.extract_metadata()
    META_PATH.write_text(json.dumps(meta, indent=2))
    print("[PaperProcessor] Done ✅")


if __name__ == "__main__":
    process_papers()

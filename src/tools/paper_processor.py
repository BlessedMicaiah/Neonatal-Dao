import os
import json
import PyPDF2
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import numpy as np

class PaperProcessor:
    def __init__(self, config):
        """Initialize with config (e.g., paths, model details)."""
        self.raw_dir = Path(config['data']['research_papers']['raw'])
        self.processed_dir = Path(config['data']['research_papers']['processed'])
        self.metadata_path = Path(config['data']['research_papers']['metadata'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer'])
        self.embedding_model = AutoModel.from_pretrained(config['model']['embedding_model'])
        self.metadata = self.load_metadata()

    def load_metadata(self):
        """Load or initialize metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def save_metadata(self):
        """Save metadata to file."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text.strip()
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""

    def generate_embeddings(self, text):
        """Generate embeddings for RAG indexing."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def process_paper(self, filename):
        """Process a single paper: extract text and save."""
        paper_path = self.raw_dir / filename
        if not paper_path.exists():
            print(f"Paper {filename} not found.")
            return

        # Extract text
        text = self.extract_text_from_pdf(paper_path)
        if not text:
            print(f"No text extracted from {filename}.")
            return

        # Save processed text
        output_path = self.processed_dir / f"{paper_path.stem}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # Update metadata
        self.metadata[filename] = {
            "processed_path": str(output_path),
            "title": filename,  # Placeholder; ideally extract from PDF
            "processed": True
        }
        self.save_metadata()

    def process_all_papers(self):
        """Process all papers in raw directory."""
        for filename in os.listdir(self.raw_dir):
            if filename.endswith('.pdf'):
                self.process_paper(filename)

if __name__ == "__main__":
    # Example config (replace with config.yaml loading)
    config = {
        "data": {
            "research_papers": {
                "raw": "data/research_papers/raw",
                "processed": "data/research_papers/processed",
                "metadata": "data/research_papers/metadata.json"
            }
        },
        "model": {
            "tokenizer": "bert-base-uncased",
            "embedding_model": "bert-base-uncased"
        }
    }
    processor = PaperProcessor(config)
    processor.process_all_papers()

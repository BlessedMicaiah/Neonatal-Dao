from typing import List, Dict
import re
import pathlib
import json

class MedicalTokenizer:
    """Simple rule-based medical tokenizer.

    This tokenizer loads a custom medical vocabulary from `data/vocab.txt` and tokenises
    input text. It is *not* a drop-in replacement for HuggingFace tokenizers but covers
    the minimal surface required for training and inference in this repository.
    """

    def __init__(self, vocab_path: str | pathlib.Path | None = None) -> None:
        project_root = pathlib.Path(__file__).resolve().parents[2]
        default_path = project_root / "data" / "vocab.txt"
        self.vocab_path = pathlib.Path(vocab_path or default_path)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._load_vocab()

        # Regex pattern splits on spaces and keeps punctuation separate
        self._token_pattern = re.compile(r"\w+|[^\w\s]")

    def _load_vocab(self) -> None:
        if not self.vocab_path.exists():
            raise FileNotFoundError(
                f"Medical vocabulary not found at {self.vocab_path}. Run `scripts/build_vocab.py` first."
            )
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            for idx, token in enumerate(line.strip() for line in f):
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        return self._token_pattern.findall(text.lower())

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(tok, self.token_to_id.get("[UNK]", 0)) for tok in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, "[UNK]") for i in ids]

    def encode(self, text: str) -> List[int]:
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.convert_ids_to_tokens(ids))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the MedicalTokenizer")
    parser.add_argument("text", type=str, help="Sentence to tokenize")
    args = parser.parse_args()

    tokenizer = MedicalTokenizer()
    print("Tokens:", tokenizer.tokenize(args.text))
    print("IDs:", tokenizer.encode(args.text))

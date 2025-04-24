from __future__ import annotations
import pathlib
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from .rag.retriever import Retriever

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "model" / "fine_tuned"

class InferenceEngine:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        self.retriever = Retriever()

    def generate(self, query: str, max_new_tokens: int = 128) -> str:
        # Retrieval-augmented prompt
        context_docs = self.retriever.retrieve(query)
        context = "\n".join(context_docs)
        prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Post-process to extract answer only
        return answer.split("Answer:")[-1].strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inference with Medical QA model")
    parser.add_argument("question", type=str, help="Input question")
    args = parser.parse_args()

    engine = InferenceEngine()
    print(engine.generate(args.question))

if __name__ == "__main__":
    main()

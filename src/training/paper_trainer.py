import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import pathlib
from datasets import load_dataset


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED_PAPERS_DIR = PROJECT_ROOT / "data" / "research_papers" / "processed"


class PaperDataset(Dataset):
    """Custom dataset for research papers."""

    def __init__(self, processed_dir, tokenizer, max_length=512):
        self.texts = []
        self.labels = []  # Placeholder for supervised tasks
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load processed texts
        for file in processed_dir.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                self.texts.append(text)
                self.labels.append(0)  # Dummy label; adjust for task

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def train_on_papers(config):
    """Fine-tune model on research papers."""
    # Initialize paths
    processed_dir = Path(config["data"]["research_papers"]["processed"])
    model_name = config["model"]["base_model"]
    output_dir = Path(config["model"]["fine_tuned"])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Prepare dataset
    dataset = PaperDataset(processed_dir, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    # Example config (replace with config.yaml loading)
    config = {
        "data": {
            "research_papers": {
                "processed": str(PROCESSED_PAPERS_DIR)
            }
        },
        "model": {
            "base_model": "bert-base-uncased",
            "fine_tuned": str(PROJECT_ROOT / "model" / "fine_tuned")
        }
    }
    train_on_papers(config)

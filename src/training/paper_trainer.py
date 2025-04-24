import pathlib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED_PAPERS_DIR = PROJECT_ROOT / "data" / "research_papers" / "processed"


def load_papers_dataset():
    txt_files = list(PROCESSED_PAPERS_DIR.glob("*.txt"))
    return load_dataset(
        "text",
        data_files={"train": [str(p) for p in txt_files]},
    )


def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    dataset = load_papers_dataset()["train"].map(
        lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "model" / "paper_bert"),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()

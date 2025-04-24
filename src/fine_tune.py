import pathlib
import argparse
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_config(config_path: str | pathlib.Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune base model on medical QA")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load dataset
    dataset_path = PROJECT_ROOT / "data" / "medical_qa.json"
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # Load base model and tokenizer
    model_name = cfg.get("model", {}).get("base_model", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    def preprocess(examples):
        inputs = examples["question"]
        targets = examples["answer"]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "model" / "fine_tuned"),
        num_train_epochs=cfg.get("training", {}).get("epochs", 3),
        per_device_train_batch_size=cfg.get("training", {}).get("batch_size", 2),
        save_steps=500,
        save_total_limit=2,
        learning_rate=cfg.get("training", {}).get("lr", 5e-5),
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    trainer.train()
    trainer.save_model(str(PROJECT_ROOT / "model" / "fine_tuned"))


if __name__ == "__main__":
    main()

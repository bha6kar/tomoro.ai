"""
Training module for the ConvFinQA system.
"""

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.main.config import get_config
from src.main.logger import setup_logger

logger = setup_logger(__name__)

CONFIG = get_config()


def load_data(file_path):
    """Load data from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class FinQADataset(Dataset):
    """Dataset optimized for memorization of QA pairs."""

    def __init__(self, data, tokenizer, repetitions=5, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for example in data:
            for _ in range(repetitions):
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = f"Question: {example['question']}"
        target_text = f"Answer: {example['answer']}"

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels = targets.input_ids.squeeze()
        labels = torch.where(
            labels == self.tokenizer.pad_token_id,
            torch.tensor(-100, dtype=torch.long),
            labels,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_model(
    output_dir=None,
    base_model=None,
    train_data=None,
    epochs=10,
    batch_size=4,
    learning_rate=5e-5,
    repetitions=5,
):

    torch.set_default_device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    print("Creating fine-tuning dataset from lookup data with table context")

    print(f"Loading model from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    train_dataset = FinQADataset(train_data, tokenizer, repetitions=repetitions)
    print(f"Created memorization dataset with {len(train_dataset)} examples")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        save_total_limit=2,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        fp16=False,
        gradient_accumulation_steps=4,
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    print("Starting memorization training...")
    trainer.train()
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Memorization training complete!")
    return model, tokenizer


def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(description="Train FinQA model")
    parser.add_argument(
        "--train_data",
        help="Path to training data",
        default=CONFIG["lookup"]["finetune_train_path"],
    )
    parser.add_argument(
        "--output_dir",
        help="Path to save the model",
        default=CONFIG["model"]["path"],
    )
    parser.add_argument(
        "--base_model",
        help="Base model name (e.g., google/flan-t5-base)",
        default=CONFIG["model"]["base_model"],
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=3
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=CONFIG["model"]["training"]["batch_size"],
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate",
        default=5e-5,
    )
    args = parser.parse_args()

    train_data = load_data(args.train_data)

    train_model(
        output_dir=args.output_dir,
        base_model=args.base_model,
        train_data=train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()

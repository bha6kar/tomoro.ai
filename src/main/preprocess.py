"""
Preprocessing module for the ConvFinQA system.
Handles data creation, variation generation, and train/test splitting.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

from src.main.config import get_config
from src.main.logger import setup_logger

logger = setup_logger(__name__)

CONFIG = get_config()


def load_training_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the original training data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_qa_lookup(training_data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Create a lookup dictionary from training data."""
    qa_lookup = {}

    for example in training_data:
        # Check for standalone qa
        if "qa" in example and example["qa"]:
            qa = example["qa"]
            question = qa.get("question", "")
            if question:
                answer = qa.get("answer", "").strip()
                if answer:
                    qa_lookup[question] = answer

        # Check for qa_0, qa_1, etc.
        i = 0
        while f"qa_{i}" in example:
            qa = example[f"qa_{i}"]
            if qa:
                question = qa.get("question", "")
                if question:
                    answer = qa.get("answer", "").strip()
                    if answer:
                        qa_lookup[question] = answer
            i += 1

    logger.info("Created lookup dictionary with %d Q&A pairs", len(qa_lookup))
    return qa_lookup


def create_question_variations(question: str, answer: str) -> List[Dict[str, str]]:
    """Create variations of a question while preserving its meaning."""
    variations = []

    # Basic variations with proper formatting
    base_variations = [
        question,
        question.replace("?", ""),
        question.replace("what is ", "can you tell me "),
        question.replace("what was ", "what is the value of "),
        question.replace("what were ", "what are the values of "),
    ]

    # Add variations with different capitalization (but keep base in lowercase)
    for var in base_variations:
        variations.append(var)
        variations.append(var.capitalize())

    # Add variations with double space (but not no-space)
    for var in base_variations:
        variations.append(var.replace(" ", "  "))

    # Add variations for numeric answers
    if any(c.isdigit() for c in answer):
        for var in base_variations:
            variations.append(var + " (in numbers)")
            variations.append(var + " (as a number)")
            if "%" in answer:
                variations.append(var + " (as a percentage)")

    # Add variations for percentage answers
    if "%" in answer:
        for var in base_variations:
            variations.append(var + " (as a percentage)")
            variations.append(var + " (in percent)")

    # Create QA pairs
    qa_pairs = []
    for var in variations:
        if var.strip():
            qa_pairs.append(
                {
                    "question": var.strip(),
                    "answer": answer,
                    "original_question": question,
                }
            )

    return qa_pairs


def create_finetune_dataset_from_lookup(
    raw_data_path: str,
    finetune_train_path: str,
    finetune_test_path: str,
    test_size: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create a fine-tuning dataset from lookup data with table context.
    """
    raw_data = load_training_data(raw_data_path)

    qa_lookup = create_qa_lookup(raw_data)
    qa_lookup_save_path = CONFIG["lookup"]["lookup_path"]
    os.makedirs(os.path.dirname(qa_lookup_save_path), exist_ok=True)
    with open(qa_lookup_save_path, "w", encoding="utf-8") as f:
        json.dump(qa_lookup, f, indent=2)
    all_variations = []
    original_questions = set()

    for question, answer in qa_lookup.items():
        original_questions.add(question)

        variations = create_question_variations(question, answer)
        all_variations.extend(variations)

    random.shuffle(all_variations)

    # Split into train and test sets
    split_idx = int(len(all_variations) * (1 - test_size))
    train_data = all_variations[:split_idx]
    test_data = all_variations[split_idx:]

    test_data = [
        example
        for example in test_data
        if example["question"] not in original_questions
    ]

    train_path = finetune_train_path
    test_path = finetune_test_path

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    logger.info("Created %d variations", len(all_variations))
    logger.info("Train set size: %d", len(train_data))
    logger.info("Test set size: %d", len(test_data))
    logger.info("Saved datasets to:")
    logger.info("  - %s", qa_lookup_save_path)
    logger.info("  - %s", train_path)
    logger.info("  - %s", test_path)

    return train_data, test_data


def main():
    """Main function to run preprocessing."""

    parser = argparse.ArgumentParser(description="Preprocess ConvFinQA data")

    parser.add_argument(
        "--raw_data_path",
        default=CONFIG["data"]["raw_train_path"],
        help="Path to the training data",
    )
    parser.add_argument(
        "--finetune_train_path",
        default=CONFIG["lookup"]["finetune_train_path"],
        help="Path to save the fine-tuned train dataset",
    )
    parser.add_argument(
        "--finetune_test_path",
        default=CONFIG["lookup"]["finetune_test_path"],
        help="Path to save the fine-tuned test dataset",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.01,
        help="Proportion of data to use for testing",
    )
    args = parser.parse_args()

    create_finetune_dataset_from_lookup(
        raw_data_path=args.raw_data_path,
        finetune_train_path=args.finetune_train_path,
        finetune_test_path=args.finetune_test_path,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()

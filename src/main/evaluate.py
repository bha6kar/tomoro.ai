"""
Evaluation module for the ConvFinQA system.
"""

import datetime
import json
import os
import re
from collections import defaultdict
from typing import List

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.main.config import get_config
from src.main.logger import setup_logger
from src.main.predict import load_model

logger = setup_logger(__name__)

CONFIG = get_config()


def load_test_data(file_path=None):
    """Load the test dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_answers_batch(
    questions: List[str], model, tokenizer, batch_size: int = 32
) -> List[str]:
    """Make predictions for a batch of questions."""
    predictions = []
    total_batches = (len(questions) + batch_size - 1) // batch_size

    # Process in batches
    for i in tqdm(
        range(0, len(questions), batch_size),
        total=total_batches,
        desc="Making predictions",
    ):
        batch_questions = questions[i : i + batch_size]

        # Normalize questions to lowercase
        batch_questions = [q.lower() for q in batch_questions]

        # Prepare inputs
        input_texts = [f"Question: {q}" for q in batch_questions]
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=30,
                num_beams=2,
                early_stopping=True,
                use_cache=False,
            )

        # Decode predictions
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract answers from predictions
        for pred in batch_predictions:
            if "Answer:" in pred:
                pred = pred.split("Answer:")[-1].strip()
            elif "answer:" in pred.lower():
                pred = pred.split("answer:", 1)[-1].strip()
            elif ":" in pred:
                pred = pred.split(":")[-1].strip()
            predictions.append(pred)

    return predictions


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Convert to lowercase and strip
    answer = answer.strip().lower()

    # Remove punctuation except for numbers and %/$
    answer = re.sub(r"[^\w\s\d%$.-]", "", answer)

    # Normalize whitespace
    answer = " ".join(answer.split())

    # Normalize numbers
    answer = re.sub(r"(\d+),(\d+)", r"\1\2", answer)

    # Normalize percentages
    answer = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1%", answer)
    answer = answer.replace("percent", "%")
    answer = answer.replace("percentage", "%")

    # Normalize currency
    answer = re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 dollars", answer)
    answer = answer.replace("dollars", "$")
    answer = answer.replace("dollar", "$")

    # Normalize negative numbers
    answer = re.sub(r"\((\d+(?:\.\d+)?)\)", r"-\1", answer)

    # Normalize decimal points
    answer = re.sub(r"(\d+)\.(\d+)", r"\1.\2", answer)

    return answer


def is_partial_match(true_answer: str, predicted_answer: str) -> bool:
    """Check if answers match partially, considering numeric values."""
    true_norm = normalize_answer(true_answer)
    pred_norm = normalize_answer(predicted_answer)

    if true_norm == pred_norm:
        return True

    # Extract numeric values
    true_nums = re.findall(r"-?\d+(?:\.\d+)?", true_norm)
    pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred_norm)

    if true_nums and pred_nums:
        # Check if any of the numbers match
        for t_num in true_nums:
            for p_num in pred_nums:
                try:
                    t_val = float(t_num)
                    p_val = float(p_num)
                    if abs(t_val - p_val) < 0.01:
                        return True
                except ValueError:
                    continue

    return False


def get_answer_type(answer):
    """Determine the type of answer."""
    answer = answer.lower()
    if "%" in answer or "percent" in answer:
        return "percentage"
    elif "$" in answer or "dollar" in answer:
        return "currency"
    elif any(c.isdigit() for c in answer):
        return "numeric"
    else:
        return "text"


def evaluate_model(
    model_path=None,
    test_data_path=None,
    save_results=True,
    batch_size: int = 32,
):
    """
    Evaluate the model on the test dataset.

    Args:
        model_path: Path to the trained model
        test_data_path: Path to the test data file
        save_results: Whether to save results to a file (default: True)
        batch_size: Batch size for predictions (default: 32)
    """
    test_examples = load_test_data(test_data_path)
    logger.info(f"Evaluating on {len(test_examples)} examples")

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(model_path)
    logger.info("Model and tokenizer loaded successfully")

    questions = [example["question"] for example in test_examples]
    true_answers = [example["answer"] for example in test_examples]

    logger.info("Making predictions...")
    predicted_answers = predict_answers_batch(questions, model, tokenizer, batch_size)

    # Initialize metrics
    total = len(test_examples)
    correct = 0
    exact_matches = 0

    category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})

    # For F1 score calculation
    all_true = []
    all_pred = []

    # Store detailed results
    detailed_results = []

    # Evaluate results
    logger.info("Evaluating results...")
    for i, (question, true_answer, predicted_answer) in enumerate(
        zip(questions, true_answers, predicted_answers)
    ):
        # Normalize answers for comparison
        true_answer_norm = normalize_answer(true_answer)
        predicted_answer_norm = normalize_answer(predicted_answer)

        # Get answer type
        answer_type = get_answer_type(true_answer)

        # Update category metrics
        category_metrics[answer_type]["total"] += 1
        is_correct = False
        is_exact_match = False

        if true_answer_norm == predicted_answer_norm:
            category_metrics[answer_type]["correct"] += 1
            exact_matches += 1
            correct += 1
            is_correct = True
            is_exact_match = True
        # Check for partial matches (e.g., "91.5%" vs "91.5 percent")
        elif is_partial_match(true_answer_norm, predicted_answer_norm):
            category_metrics[answer_type]["correct"] += 1
            correct += 1
            is_correct = True

        # For F1 score
        all_true.append(true_answer_norm)
        all_pred.append(predicted_answer_norm)

        # Store detailed result
        detailed_results.append(
            {
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "is_exact_match": is_exact_match,
                "answer_type": answer_type,
            }
        )

    # Calculate overall metrics
    exact_match_accuracy = exact_matches / total * 100
    partial_match_accuracy = correct / total * 100

    # Calculate F1 score
    unique_answers = list(set(all_true))
    true_labels = [unique_answers.index(ans) for ans in all_true]
    pred_labels = [
        unique_answers.index(ans) if ans in unique_answers else len(unique_answers)
        for ans in all_pred
    ]

    unique_answers.append("unknown")

    f1 = f1_score(true_labels, pred_labels, average="weighted")
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")

    results = {
        "model_path": model_path,
        "test_data_path": test_data_path,
        "total_examples": total,
        "exact_match_accuracy": exact_match_accuracy,
        "partial_match_accuracy": partial_match_accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "category_metrics": dict(category_metrics),
        "detailed_results": detailed_results,
    }

    logger.info("\nOverall Metrics:")
    logger.info(f"Total examples evaluated: {total}")
    logger.info(f"Exact match accuracy: {exact_match_accuracy:.2f}%")
    logger.info(f"Partial match accuracy: {partial_match_accuracy:.2f}%")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")

    logger.info("\nPer-Category Metrics:")
    for category, metrics in category_metrics.items():
        accuracy = metrics["correct"] / metrics["total"] * 100
        logger.info(f"{category.capitalize()}:")
        logger.info(f"  Total: {metrics['total']}")
        logger.info(f"  Correct: {metrics['correct']}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")

    if save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "src/output/evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        summary_path = os.path.join(results_dir, f"evaluation_summary_{timestamp}.json")
        with open(summary_path, "w") as f:
            summary_results = {
                k: v for k, v in results.items() if k != "detailed_results"
            }
            json.dump(summary_results, f, indent=2)

        detailed_path = os.path.join(
            results_dir, f"evaluation_detailed_{timestamp}.json"
        )
        with open(detailed_path, "w") as f:
            json.dump(detailed_results, f, indent=2)

        logger.info("\nSaved evaluation results to:")
        logger.info(f"  Summary: {summary_path}")
        logger.info(f"  Detailed: {detailed_path}")

    return results


def main():
    """Main function to run evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the ConvFinQA model")
    parser.add_argument(
        "--model_path",
        default=CONFIG["model"]["path"],
        help="Path to the trained model",
    )
    parser.add_argument(
        "--test_data_path",
        default=CONFIG["lookup"]["finetune_test_path"],
        help="Path to the test data file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for predictions",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save evaluation results to file",
    )
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        batch_size=args.batch_size,
        save_results=not args.no_save,
    )


if __name__ == "__main__":
    main()

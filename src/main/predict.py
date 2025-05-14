"""
Prediction module for the ConvFinQA system.
"""

import gc

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.main.config import get_config
from src.main.logger import setup_logger

logger = setup_logger(__name__)

CONFIG = get_config()


def load_model(model_path=None):
    """Load the model and tokenizer."""
    model_path = model_path or CONFIG["model"]["path"]

    try:
        # Force CPU usage with limited threads to avoid memory issues
        torch.set_default_device("cpu")
        torch.set_num_threads(2)

        gc.collect()

        logger.info(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        logger.info("Using default model...")
        try:
            base_model = CONFIG["model"]["base_model"]
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
            return model, tokenizer
        except Exception as e2:
            logger.error(f"Error loading fallback model: {e2}")
            raise RuntimeError(f"Failed to load any model: {e2}")


def predict(
    question,
    model=None,
    tokenizer=None,
):
    """
    Predict answer for a given question using the neural model.

    Args:
        question: The question to answer
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)

    Returns:
        Predicted answer as a string
    """

    try:
        if model is None or tokenizer is None:
            model, tokenizer = load_model()

        input_text = f"Question: {question}"

        max_length = CONFIG["model"]["max_length"]
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=max_length
        )

        logger.info("Using model for prediction")

        with torch.no_grad():
            try:
                gc.collect()

                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=30,
                    num_beams=2,
                    use_cache=False,
                )

                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

                if "Answer:" in prediction:
                    prediction = prediction.split("Answer:")[-1].strip()
                elif "answer:" in prediction.lower():
                    prediction = prediction.split("answer:", 1)[-1].strip()
                elif ":" in prediction:
                    prediction = prediction.split(":")[-1].strip()

                return prediction

            except RuntimeError as e:
                if "out of memory" in str(e) or "segmentation fault" in str(e).lower():
                    logger.error(f"Memory error during prediction: {e}")
                    return "Error: Insufficient memory to process this question"
                else:
                    logger.error(f"Runtime error during prediction: {e}")
                    return "Error: Unable to process this question"

            except Exception as e:
                logger.error(f"Error during model prediction: {e}")
                return "Error: Failed to generate prediction"

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "Error: System error during prediction"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict answers for financial questions"
    )
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--model", help="Path to the trained model")

    args = parser.parse_args()

    # Make prediction
    answer = predict(
        question=args.question,
        model=None,
        tokenizer=None,
    )

    print(f"Q: {args.question}")
    print(f"A: {answer}")

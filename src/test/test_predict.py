"""
Tests for the prediction module functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.main.predict import load_model, predict


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.decode.return_value = "Answer: 14.1%"

    mock_outputs = MagicMock()
    mock_model.generate.return_value = [mock_outputs]

    return mock_model, mock_tokenizer


@pytest.mark.unit
def test_load_model_success():
    """Test successful model loading."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, patch(
        "transformers.AutoModelForSeq2SeqLM.from_pretrained"
    ) as mock_model:

        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        model, tokenizer = load_model("test_model_path")

        mock_tokenizer.assert_called_once_with("test_model_path")
        mock_model.assert_called_once_with("test_model_path")
        assert model is not None
        assert tokenizer is not None


@pytest.mark.unit
def test_predict_with_model_mock(mock_model_and_tokenizer):
    """Test prediction with mocked model and tokenizer."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    question = "what was the percentage change in the net cash from operating activities from 2008 to 2009"
    result = predict(question=question, model=mock_model, tokenizer=mock_tokenizer)

    mock_model.generate.assert_called_once()
    mock_tokenizer.decode.assert_called_once()
    assert result == "14.1%"


@pytest.mark.integration
def test_target_question():
    """Test the target question works with model."""
    question = "what percentage of future minimum lease payments under the capital lease obligations is due in 2019"
    result = predict(
        question=question,
    )
    assert result == "11%"


@pytest.mark.integration
@pytest.mark.parametrize(
    "variation",
    [
        "what  percentage  of  future  minimum  lease  payments  under  the  capital  lease  obligations  is  due  in  2019?",
        "what percentage of future minimum lease payments under the capital lease obligations is due in 2019",
        "what percentage of future minimum lease payments under the capital lease obligations is due in 2019? (in percent)",
    ],
)
def test_target_question_variations(variation):
    """Test various phrasings of the target question."""

    result = predict(question=variation)

    assert result == "11%", f"Failed on variant: {variation}"


@pytest.mark.integration
def test_error_question():
    """Test vague or underspecified question is handled gracefully."""
    question = "gibberish"
    result = predict(question=question)
    assert result.strip(), "Model returned an empty response to vague input"
    assert "Anything" not in result, f"Unexpectedly confident answer: {result}"


@pytest.mark.unit
def test_predict_with_none_question():
    """Test prediction with None question."""
    result = predict(None)
    assert result.strip(), "Model returned an empty response to vague input"
    assert "Anything" not in result, f"Unexpectedly confident answer: {result}"


@pytest.mark.unit
def test_load_model_complete_failure():
    """Test when both main and base model loading fail."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, patch(
        "transformers.AutoModelForSeq2SeqLM.from_pretrained"
    ) as mock_model:

        # Both calls fail
        mock_tokenizer.side_effect = Exception("No models available")
        mock_model.side_effect = Exception("No models available")

        with pytest.raises(RuntimeError, match="Failed to load any model"):
            load_model("invalid_model_path")


@pytest.mark.unit
def test_predict_memory_error():
    """Test prediction with memory error during generation."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock tokenizer to return inputs
    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }

    # Mock model to raise memory error
    mock_model.generate.side_effect = RuntimeError("CUDA out of memory")

    result = predict("test question", model=mock_model, tokenizer=mock_tokenizer)

    assert "Error: Failed to generate prediction" in result


@pytest.mark.unit
def test_predict_runtime_error():
    """Test prediction with general runtime error."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }

    # Mock model to raise runtime error
    mock_model.generate.side_effect = RuntimeError("Some other error")

    result = predict("test question", model=mock_model, tokenizer=mock_tokenizer)

    assert "Error: Failed to generate prediction" in result


@pytest.mark.unit
def test_predict_general_exception():
    """Test prediction with general exception during generation."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }

    # Mock model to raise general exception
    mock_model.generate.side_effect = ValueError("Unexpected error")

    result = predict("test question", model=mock_model, tokenizer=mock_tokenizer)

    assert "Error:" in result


@pytest.mark.unit
def test_predict_system_error():
    """Test system error before model loading."""
    with patch("src.main.predict.load_model") as mock_load:
        mock_load.side_effect = Exception("System error")

        result = predict("test question")
        assert "Error: System error during prediction" in result


@pytest.mark.unit
def test_predict_segmentation_fault():
    """Test prediction with segmentation fault error."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }

    # Mock model to raise segmentation fault
    mock_model.generate.side_effect = RuntimeError("segmentation fault")

    result = predict("test question", model=mock_model, tokenizer=mock_tokenizer)

    assert "Error: Failed to generate prediction" in result


if __name__ == "__main__":
    pytest.main()

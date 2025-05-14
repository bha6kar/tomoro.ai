import json
import os
import tempfile
from typing import Dict, List

import pytest

from src.main.preprocess import (
    create_finetune_dataset_from_lookup,
    create_qa_lookup,
    create_question_variations,
    load_training_data,
)


@pytest.fixture
def sample_training_data() -> List[Dict]:
    """Create sample training data for testing."""
    return [
        {
            "qa": {
                "question": "What is the percentage change in revenue?",
                "answer": "4.3%",
            }
        },
        {
            "qa_0": {
                "question": "What was the total assets?",
                "answer": "$1,234,567",
            },
            "qa_1": {
                "question": "What is the profit margin?",
                "answer": "15.2%",
            },
        },
    ]


@pytest.fixture
def temp_training_file(sample_training_data) -> str:
    """Create a temporary file with sample training data."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(sample_training_data, f)
        return f.name


@pytest.mark.unit
def test_load_training_data(temp_training_file):
    """Test loading training data from file."""
    data = load_training_data(temp_training_file)
    assert len(data) == 2
    assert "qa" in data[0]
    assert "qa_0" in data[1]
    os.unlink(temp_training_file)


@pytest.mark.unit
def test_load_training_data_file_not_found():
    """Test loading training data with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_training_data("nonexistent_file.json")


@pytest.mark.unit
def test_create_qa_lookup(sample_training_data):
    """Test creating QA lookup dictionary."""
    expected = {
        "What is the percentage change in revenue?": "4.3%",
        "What was the total assets?": "$1,234,567",
        "What is the profit margin?": "15.2%",
    }
    qa_lookup = create_qa_lookup(sample_training_data)
    assert len(qa_lookup) == 3
    assert qa_lookup["What is the percentage change in revenue?"] == "4.3%"
    assert qa_lookup["What was the total assets?"] == "$1,234,567"
    assert qa_lookup["What is the profit margin?"] == "15.2%"
    assert qa_lookup == expected


@pytest.mark.unit
def test_create_question_variations():
    """Test creation of question variations."""
    question = "what portion of total shares repurchased in the fourth quarter of 2010 occurred during october?"
    answer = "49.5%"
    variations = create_question_variations(question, answer)

    assert len(variations) > 0
    assert all(isinstance(v, dict) for v in variations)
    assert all("question" in v and "answer" in v for v in variations)

    # Check specific variations
    questions = [v["question"] for v in variations]
    assert (
        "what portion of total shares repurchased in the fourth quarter of 2010 occurred during october?"
        in questions
    )
    assert (
        "what portion of total shares repurchased in the fourth quarter of 2010 occurred during october (as a percentage)"
        in questions
    )
    assert (
        "what portion of total shares repurchased in the fourth quarter of 2010 occurred during october (in numbers)"
        in questions
    )

    # Check answers are preserved
    assert all(v["answer"] == answer for v in variations)

    # Check original question is preserved
    assert all(v["original_question"] == question for v in variations)
    assert isinstance(variations, list)
    assert len(variations) > 1


@pytest.mark.unit
def test_create_question_variations_empty():
    variations = create_question_variations("", "")
    assert isinstance(variations, list)
    assert len(variations) == 0 or all("question" in v for v in variations)


@pytest.mark.unit
def test_create_finetune_dataset_from_lookup():
    # Create sample raw data
    sample_data = [
        {
            "qa": {
                "question": "What is the revenue?",
                "answer": "£100",
            }
        },
        {
            "qa": {
                "question": "How much profit?",
                "answer": "£50",
            }
        },
    ]

    # Create temporary files
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as raw_file, tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as train_file, tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as test_file:

        # Write sample data to raw file
        json.dump(sample_data, raw_file)
        raw_path = raw_file.name
        train_path = train_file.name
        test_path = test_file.name

    try:
        train_data, test_data = create_finetune_dataset_from_lookup(
            raw_path, train_path, test_path, 0.2
        )

        # Check return types
        assert isinstance(train_data, list)
        assert isinstance(test_data, list)

        # Check that we have data
        total_data = train_data + test_data
        assert len(total_data) > 0

        # Check structure of items
        assert all("question" in item and "answer" in item for item in total_data)

        # Check answers are preserved
        answers = [item["answer"] for item in total_data]
        assert "£100" in answers
        assert "£50" in answers

    finally:
        # Clean up temporary files
        os.unlink(raw_path)
        os.unlink(train_path)
        os.unlink(test_path)

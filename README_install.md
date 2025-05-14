# ConvFinQA: Financial Question Answering System

A question answering system specialized in financial data analysis, built with Python and modern NLP techniques.

## Quick Start

### Installation

Install library in virtual environment using poetry:

```bash
make install
```

### Running the API

Start the API server:

```bash
make run-app
```

The API will be available at `http://localhost:8000`

### Testing the API

#### Health Check:

```bash
curl http://localhost:8000/
```

#### Make Predictions:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "what portion of the non-cancelable operating leases net of sublease income is due within the next 12 months?"}'
```

### Running Tests

To run the integration and unit testing both test suite:

```bash
make test
```

To run only unit test suite:

```bash
make test-unit
```

To run only integration test suite:

```bash
make test-integration
```

To run test coverage:

```bash
make test-cov
```

To view test coverage results:

```bash
make open-cov
```

## Project Structure

```md
├── Makefile                     # Build automation and project management
├── README.md                    # This file
├── pyproject.toml               # Python dependencies
├── src/
|   └── data/                    # Data directory (not in repo)
|   |   ├── raw/                 # Raw input data
|   |   └── processed/           # Processed data files
│   ├── main/                    # Main application code
│   │   ├── api.py               # Flask API implementation
│   │   ├── config               # Configuration management
|   |   |   ├── config.py        # Raw config Reader
|   |   |   └── config.yml       # Raw config
│   │   ├── evaluate.py          # Model evaluation
│   │   ├── logger.py            # Logging setup
│   │   ├── predict.py           # Prediction logic
│   │   ├── preprocess.py        # Data preprocessing
│   │   └── train.py             # Model training
│   ├── output/                  # Output directory for results
│   │   ├── evaluation_results/  # Evaluation metrics and results
│   │   └── fin_qa_model/        # Trained model checkpoints
│   └── test/                    # Test suite
│       ├── test_api.py          # API tests
│       └── test_prediction.py   # Prediction tests
│       └── test_preprocess.py   # Preprocess tests

```

## Training Model

### 1. Data Preprocessing

Preprocess the raw financial data:

```bash
make preprocess
```

This will:

- Load raw data from `src/data/raw/`
- Process and create variations of the data
- Create training and test splits
- Save processed data to `src/data/processed/` as `lookup_finetune_train.json` for training the model and `lookup_finetune_test.json` for evaluation of model.

### 2. Model Training

Train the question answering model:

```bash
make train
```

This will:

- Load the preprocessed data
- Initialize the base model `google/flan-t5-base`
- Train for the specified number of epochs
- Save the trained model to `src/output/fin_qa_model/`

### 3. Model Evaluation

Evaluate the trained model's performance:

```bash
make evaluate
```

This will:

- Load the trained model
- Run evaluation on the test set
- Generate performance metrics
- Save evaluation results to `src/output/evaluation_results/`

PYTHON = poetry run python -m
PYTEST = poetry run pytest
UWSGI = poetry run uwsgi

API_HOST = localhost
API_PORT = 8000

SRC_DIR = src
MAIN_DIR = $(SRC_DIR).main
TEST_DIR = .

.PHONY: all
all: help

.PHONY: help
help:
	@echo "ConvFinQA Makefile"
	@echo ""
	@echo "  make install        	- Install dependencies"
	@echo "  make preprocess     	- Preprocess raw data"
	@echo "  make train          	- Train the model"
	@echo "  make evaluate       	- Evaluate the model"
	@echo "  make test-unit      	- Run unit tests"
	@echo "  make test-integration 	- Run integration tests"
	@echo "  make test-cov       	- Run all tests with coverage"
	@echo "  make open-cov       	- Open coverage report"
	@echo "  make test           	- Run all tests"
	@echo "  make run-app-dev   	- Run the application in development mode"
	@echo "  make run-app       	- Run the application"
	@echo "  make format      		- Format code"
	@echo "  make clean       		- Clean generated files"
	@echo ""

.PHONY: install
install:
	@echo "Installing dependencies..."
	poetry install

.PHONY: preprocess
preprocess:
	@echo "Preprocessing data..."
	$(PYTHON) $(MAIN_DIR).preprocess

.PHONY: train
train:
	@echo "Training model..."
	$(PYTHON) $(MAIN_DIR).train

.PHONY: evaluate
evaluate:
	@echo "Evaluating model..."
	$(PYTHON) $(MAIN_DIR).evaluate

.PHONY: run-app
run-app:
	@echo "Starting uWSGI server on port $(API_PORT)..."
	$(UWSGI) --ini uwsgi.ini

.PHONY: run-app-dev
run-app-dev:
	@echo "Starting uWSGI server in development mode..."
	$(UWSGI) --ini uwsgi.ini --py-autoreload=1 --http-socket=$(API_HOST):$(API_PORT)

.PHONY: test-unit
test-unit:
	@echo "Running tests..."
	$(PYTEST) -m unit -vs $(TEST_DIR)

.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	$(PYTEST) -m integration -vs $(TEST_DIR)

.PHONY: test
test: test-unit test-integration

.PHONY: test-cov
test-cov:
	@echo "Running tests with coverage..."
	$(PYTEST) --cov=src.main --cov-report=html

.PHONY: open-cov
open-cov:
	open htmlcov/index.html


.PHONY: format
format:
	@echo "Formatting code..."
	$(PYTHON) isort $(SRC_DIR)
	$(PYTHON) black $(SRC_DIR)

.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
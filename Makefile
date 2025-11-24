.PHONY: help setup download-model benchmark test-handler build push deploy test-endpoint run-experiments analyze clean

# Configuration
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
IMAGE_NAME := intent-classification-vllm
REGISTRY := ghcr.io/llmdeveloper47
IMAGE_TAG := latest
FULL_IMAGE := $(REGISTRY)/qwen-qwen2.5-0.5b-deployment-vllm:$(IMAGE_TAG)

# Default target
help:
	@echo "Intent Classification Deployment - Make Commands"
	@echo ""
	@echo "Setup & Local Development:"
	@echo "  make setup              - Create venv and install dependencies"
	@echo "  make download-model     - Download and verify model"
	@echo "  make benchmark          - Run local benchmarks (requires GPU)"
	@echo "  make test-handler       - Test handler locally (requires GPU)"
	@echo ""
	@echo "Docker:"
	@echo "  make build              - Build Docker image"
	@echo "  make push               - Push image to registry"
	@echo "  make build-push         - Build and push image"
	@echo ""
	@echo "Deployment & Testing:"
	@echo "  make test-endpoint      - Test deployed endpoint"
	@echo "  make run-experiments    - Run full experiment suite"
	@echo "  make load-test          - Run Locust load test"
	@echo ""
	@echo "Analysis:"
	@echo "  make analyze            - Analyze results and generate visualizations"
	@echo "  make report             - Generate PDF report"
	@echo "  make notebook           - Launch Jupyter notebook for analysis"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean              - Clean generated files"
	@echo "  make clean-all          - Clean everything including venv"

# Setup virtual environment
setup:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Setup complete"
	@echo ""
	@echo "To activate venv: source $(VENV)/bin/activate"

# Download model
download-model:
	@echo "Downloading model..."
	$(PYTHON) scripts/download_model.py
	@echo "✓ Model downloaded"

# Run local benchmarks
benchmark:
	@echo "Running local benchmarks..."
	@echo "Quantization: none (FP16)"
	$(PYTHON) scripts/benchmark_local.py \
		--quantization none \
		--batch-sizes 1,4,8,16,32 \
		--num-samples 1000
	@echo "✓ Benchmark complete"

# Test handler locally
test-handler:
	@echo "Testing handler locally..."
	$(PYTHON) scripts/test_local_handler.py
	@echo "✓ Handler tests passed"

# Build Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(FULL_IMAGE)
	@echo "✓ Docker image built: $(FULL_IMAGE)"

# Push to registry
push:
	@echo "Pushing to GitHub Container Registry..."
	docker push $(FULL_IMAGE)
	@echo "✓ Image pushed: $(FULL_IMAGE)"

# Build and push
build-push: build push

# Test deployed endpoint
test-endpoint:
	@echo "Testing deployed endpoint..."
	@if [ -z "$$RUNPOD_ENDPOINT_ID" ]; then \
		echo "Error: RUNPOD_ENDPOINT_ID not set"; \
		echo "Run: export RUNPOD_ENDPOINT_ID=your-endpoint-id"; \
		exit 1; \
	fi
	@if [ -z "$$RUNPOD_API_KEY" ]; then \
		echo "Error: RUNPOD_API_KEY not set"; \
		echo "Run: export RUNPOD_API_KEY=your-api-key"; \
		exit 1; \
	fi
	$(PYTHON) scripts/test_endpoint.py \
		--endpoint-id $$RUNPOD_ENDPOINT_ID \
		--api-key $$RUNPOD_API_KEY
	@echo "✓ Endpoint test complete"

# Run experiments
run-experiments:
	@echo "Running experiment suite..."
	@if [ -z "$$RUNPOD_ENDPOINT_ID" ] || [ -z "$$RUNPOD_API_KEY" ]; then \
		echo "Error: Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY"; \
		exit 1; \
	fi
	$(PYTHON) scripts/run_experiments.py \
		--endpoint-id $$RUNPOD_ENDPOINT_ID \
		--api-key $$RUNPOD_API_KEY \
		--quantization-methods none bitsandbytes \
		--batch-sizes 1,4,8,16,32 \
		--iterations 20 \
		--num-samples 1000
	@echo "✓ Experiments complete"

# Run load test
load-test:
	@echo "Running Locust load test..."
	@if [ -z "$$RUNPOD_ENDPOINT_ID" ] || [ -z "$$RUNPOD_API_KEY" ]; then \
		echo "Error: Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY"; \
		exit 1; \
	fi
	locust -f locustfile.py \
		--headless \
		-u 10 \
		-r 2 \
		--run-time 5m \
		--html results/load_tests/report.html
	@echo "✓ Load test complete"

# Analyze results
analyze:
	@echo "Analyzing results..."
	$(PYTHON) scripts/summarize_results.py --results-dir ./results
	$(PYTHON) scripts/analyze_results.py --results-dir ./results
	@echo "✓ Analysis complete"

# Generate PDF report
report:
	@echo "Generating PDF report..."
	$(PYTHON) scripts/generate_report.py \
		--results-dir ./results \
		--output ./results/experiment_report.pdf
	@echo "✓ Report generated"

# Launch Jupyter notebook
notebook:
	@echo "Launching Jupyter notebook..."
	$(VENV)/bin/jupyter notebook experiments/analysis/comparison.ipynb

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf results/
	rm -rf experiments/results/
	rm -rf experiments/analysis/*.png
	rm -rf experiments/analysis/*.csv
	rm -rf models/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned"

# Clean everything including venv
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "✓ All cleaned"

# Quick start - run all local setup steps
quickstart:
	@echo "Running quickstart..."
	make setup
	make download-model
	@echo ""
	@echo "✓ Quickstart complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Configure .env file with your RunPod credentials"
	@echo "  2. (Optional) Run: make benchmark"
	@echo "  3. (Optional) Run: make test-handler"
	@echo "  4. Build Docker: make build-push"
	@echo "  5. Deploy to RunPod (see README.md)"
	@echo "  6. Test endpoint: make test-endpoint"


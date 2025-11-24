# Contributing to Intent Classification Deployment

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

1. **Clear title** describing the bug
2. **Steps to reproduce** the issue
3. **Expected behavior** vs actual behavior
4. **Environment details:**
   - OS and version
   - Python version
   - GPU type (if applicable)
   - vLLM version
5. **Error messages** and logs
6. **Screenshots** if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

1. **Use case**: Why is this enhancement needed?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: What other approaches did you consider?
4. **Impact**: Who will benefit from this?

### Pull Requests

We love pull requests! Here's the process:

#### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Qwen-Qwen2.5-0.5B-Deployment-vLLM.git
cd Qwen-Qwen2.5-0.5B-Deployment-vLLM
```

#### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

#### 3. Make Changes

- Follow the code style (see below)
- Add tests if applicable
- Update documentation
- Commit with clear messages

```bash
git add .
git commit -m "Add feature: your feature description"
```

#### 4. Test Your Changes

```bash
# Run tests
python scripts/test_local_handler.py

# Test scripts
python scripts/download_model.py
python scripts/benchmark_local.py --num-samples 10

# Check code style
black app/ scripts/
flake8 app/ scripts/
```

#### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/results if applicable

## Code Style Guidelines

### Python Code

We follow PEP 8 with some modifications:

```python
# Use Black formatter (line length 88)
black app/ scripts/

# Use flake8 for linting
flake8 app/ scripts/ --max-line-length=88 --ignore=E501,W503

# Use type hints
def classify_batch(prompts: List[str]) -> List[Dict[str, Any]]:
    pass

# Use docstrings
def my_function(arg1: str, arg2: int) -> bool:
    """
    Brief description of function.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
    """
    pass
```

### Documentation

- Use Markdown for all documentation
- Include code examples in docs
- Keep README.md up to date
- Add comments for complex logic

### Commit Messages

Follow conventional commits:

```
feat: Add AWQ quantization support
fix: Resolve memory leak in handler
docs: Update deployment guide
test: Add integration tests for endpoint
refactor: Simplify batch processing logic
perf: Optimize tokenization for batches
```

## Development Setup

### 1. Install Development Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies including dev tools
pip install -r requirements.txt
pip install black flake8 pytest pytest-asyncio mypy
```

### 2. Run Tests

```bash
# Unit tests
pytest tests/ -v

# Handler tests
python scripts/test_local_handler.py

# Linting
flake8 app/ scripts/
black --check app/ scripts/
```

### 3. Test Docker Build

```bash
# Build locally
docker build -t intent-classification-vllm:dev .

# Test locally with GPU
docker run --gpus all \
  -p 8000:8000 \
  -e MODEL_NAME=codefactory4791/intent-classification-qwen \
  intent-classification-vllm:dev
```

## Project Structure

```
├── app/                    # RunPod handler code
│   ├── handler.py         # Main handler logic
│   └── requirements.txt   # Handler dependencies
├── scripts/               # Utility scripts
│   ├── download_model.py
│   ├── benchmark_local.py
│   ├── test_endpoint.py
│   └── ...
├── experiments/           # Experiment results and analysis
├── configs/              # Configuration presets
├── .github/workflows/    # CI/CD pipelines
└── docs/                 # Additional documentation
```

## Areas for Contribution

### High Priority

1. **Quantization Support:**
   - Add scripts for AWQ pre-quantization
   - Add scripts for GPTQ pre-quantization
   - Test with more quantization methods

2. **Performance Optimization:**
   - Implement continuous batching
   - Add response caching
   - Optimize tokenization

3. **Monitoring:**
   - Add Prometheus metrics
   - Create Grafana dashboards
   - Implement health checks

### Medium Priority

1. **Testing:**
   - Add unit tests for handler
   - Add integration tests
   - Add accuracy regression tests

2. **Documentation:**
   - Add video tutorials
   - Create troubleshooting flowcharts
   - Add more examples

3. **Features:**
   - Support for multiple models
   - A/B testing framework
   - Model versioning

### Low Priority

1. **Infrastructure:**
   - Terraform templates
   - Kubernetes deployment option
   - Multi-cloud support

2. **Tools:**
   - CLI tool for deployment
   - Web dashboard
   - Automated experiment scheduler

## Code Review Process

Pull requests will be reviewed for:

1. **Functionality**: Does it work as intended?
2. **Tests**: Are there tests? Do they pass?
3. **Code Quality**: Is it readable and maintainable?
4. **Documentation**: Is it documented?
5. **Performance**: Does it impact performance?
6. **Security**: Are there security implications?

## Questions?

- Open an issue for questions
- Join discussions in Issues tab
- Check existing PRs for examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🙏


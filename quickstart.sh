#!/bin/bash

# Quickstart script for Intent Classification Deployment
# This script automates the initial setup process

set -e  # Exit on error

echo "======================================================================"
echo "Intent Classification Deployment - Quickstart"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Step 1: Check prerequisites
echo "[Step 1/7] Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_success "pip3 found"
else
    print_error "pip3 not found. Please install pip"
    exit 1
fi

# Check Docker (optional)
if command -v docker &> /dev/null; then
    print_success "Docker found"
else
    print_warning "Docker not found (optional for local testing)"
fi

# Check NVIDIA GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    print_success "NVIDIA GPU found: $GPU_INFO"
else
    print_warning "NVIDIA GPU not found (required for local benchmarking)"
fi

echo ""

# Step 2: Create virtual environment
echo "[Step 2/7] Creating virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

echo ""

# Step 3: Install dependencies
echo "[Step 3/7] Installing Python dependencies..."
echo "This may take 5-10 minutes..."

pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

print_success "Dependencies installed"

echo ""

# Step 4: Setup configuration
echo "[Step 4/7] Setting up configuration..."

if [ ! -f ".env" ]; then
    cp env.example .env
    print_success "Created .env file from template"
    print_warning "Please edit .env file with your RunPod credentials:"
    print_warning "  nano .env"
else
    print_warning ".env file already exists"
fi

echo ""

# Step 5: Download model
echo "[Step 5/7] Downloading model..."
echo "This may take 2-3 minutes..."

python scripts/download_model.py

print_success "Model downloaded and verified"

echo ""

# Step 6: Run quick benchmark (optional)
echo "[Step 6/7] Running quick local benchmark..."
read -p "Run quick benchmark? This requires a GPU. (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/benchmark_local.py \
        --quantization none \
        --batch-sizes 1,8,16 \
        --num-samples 100
    
    print_success "Quick benchmark complete"
else
    print_warning "Skipped local benchmark"
fi

echo ""

# Step 7: Test handler locally (optional)
echo "[Step 7/7] Testing handler locally..."
read -p "Test handler locally? This requires a GPU and vLLM. (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/test_local_handler.py
    
    print_success "Handler tests passed"
else
    print_warning "Skipped handler testing"
fi

echo ""
echo "======================================================================"
echo "✓ Quickstart Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your credentials in .env file:"
echo "   nano .env"
echo ""
echo "2. Build Docker image:"
echo "   docker build -t intent-classification-vllm:latest ."
echo ""
echo "3. Deploy to RunPod:"
echo "   See SETUP.md section 7 for detailed deployment instructions"
echo ""
echo "4. Test deployed endpoint:"
echo "   python scripts/test_endpoint.py --endpoint-id XXX --api-key XXX"
echo ""
echo "5. Run experiments:"
echo "   python scripts/run_experiments.py --endpoint-id XXX --api-key XXX"
echo ""
echo "For detailed instructions, see:"
echo "  - README.md - Project overview"
echo "  - SETUP.md - Step-by-step setup guide"
echo ""
echo "======================================================================"


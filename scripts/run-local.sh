#!/bin/bash

# Quick run script for local development

set -e

echo "🚀 Starting PyTorch Quantization Demo (Simulation Mode)"
echo "======================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Ensure simulation mode is enabled for local dev
export SIMULATION_MODE=true
export LOG_LEVEL=INFO
export PORT=8000

echo ""
echo "📊 Running in SIMULATION MODE (no real models needed)"
echo ""
echo "Access the demo:"
echo "  - Audience interface: http://localhost:8000"
echo "  - Presenter dashboard: http://localhost:8000/presenter"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the application
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

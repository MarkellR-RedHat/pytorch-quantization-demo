#!/bin/bash

# Run tests with coverage

set -e

echo "🧪 Running Test Suite"
echo "===================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install test dependencies if needed
pip install -q pytest pytest-asyncio pytest-cov

echo ""
echo "Running tests with coverage..."
echo ""

# Run tests
pytest tests/ --cov=app --cov-report=term-missing --cov-report=html -v

echo ""
echo "✅ Tests complete!"
echo ""
echo "Coverage report saved to: htmlcov/index.html"
echo "Open with: open htmlcov/index.html"
echo ""

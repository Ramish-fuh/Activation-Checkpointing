#!/bin/bash

# Quick start script for CS265 Systems Project

echo "=================================================="
echo "CS265 Systems Project - Quick Start"
echo "=================================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source systemProject/bin/activate"
    exit 1
fi

echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Run test script
echo ""
echo "Running environment tests..."
python test_setup.py

echo ""
echo "=================================================="
echo "Setup complete! You can now run:"
echo ""
echo "  python experiments/train_resnet.py"
echo "  python experiments/train_resnet.py --profile"
echo "  python experiments/train_bert.py"
echo ""
echo "=================================================="

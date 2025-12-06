#!/bin/bash

echo "=========================================="
echo "GTZAN Music Genre Classification Setup"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created successfully"
else
    echo "✗ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
else
    echo "✗ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "To activate the virtual environment in the future, run:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run the analysis:"
    echo "  python gtzan_analysis.py"
    echo ""
    echo "To open Jupyter notebook:"
    echo "  jupyter notebook gtzan_analysis.ipynb"
    echo ""
    echo "To deactivate the virtual environment:"
    echo "  deactivate"
    echo ""
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

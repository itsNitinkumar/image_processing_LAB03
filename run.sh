#!/bin/bash

# Digital Image Processing - Run Script
# Activates virtual environment and runs the main program

echo "======================================================"
echo "Digital Image Processing - Chapter 3 Assignment"
echo "======================================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment and run
echo "Activating virtual environment..."
source .venv/bin/activate

echo "Running assignment..."
echo ""
python main.py

echo ""
echo "======================================================"
echo "Done! Check the 'results/' folder for outputs."
echo "======================================================"

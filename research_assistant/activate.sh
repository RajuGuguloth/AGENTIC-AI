#!/bin/bash
# Activation script for Research Assistant virtual environment

cd "$(dirname "$0")"
source venv/bin/activate
echo "✅ Virtual environment activated!"
echo "📍 Current directory: $(pwd)"
echo ""
echo "To run the app:"
echo "  streamlit run main.py"
echo ""
echo "To deactivate:"
echo "  deactivate"


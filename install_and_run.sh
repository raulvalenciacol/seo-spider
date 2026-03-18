#!/bin/bash
echo "============================================"
echo "  SEO Spider - Local Install"
echo "  Built by Raul @ MaleBasics Corp"
echo "============================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed!"
    echo "Install it with: brew install python3 (Mac) or sudo apt install python3 (Linux)"
    exit 1
fi

echo "[1/2] Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies."
    exit 1
fi

echo
echo "[2/2] Starting SEO Spider..."
echo
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server."
echo
streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

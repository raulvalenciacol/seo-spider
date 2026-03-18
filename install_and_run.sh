#!/bin/bash
set -e

echo "============================================"
echo "  SEO Spider - Local Install"
echo "  Built by Raul @ MaleBasics Corp"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Auto-Update ───
# If this is a git repo, pull the latest changes automatically
if [ -d "$SCRIPT_DIR/.git" ] && command -v git &> /dev/null; then
    echo "[UPDATE] Checking for updates..."
    if git -C "$SCRIPT_DIR" pull origin main --ff-only &> /dev/null; then
        echo "[OK] Up to date."
    else
        echo "[INFO] Could not auto-update. Continuing with current version."
    fi
    echo
fi

PYTHON_CMD=""

# Find Python 3
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    ver=$(python --version 2>&1 | grep -oP '\d+' | head -1)
    if [ "$ver" = "3" ]; then
        PYTHON_CMD="python"
    fi
fi

# If no Python, try to install it
if [ -z "$PYTHON_CMD" ]; then
    echo "[INFO] Python 3 not found. Attempting to install..."
    echo

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "Installing Python via Homebrew..."
            brew install python3
            PYTHON_CMD="python3"
        else
            echo "Installing Homebrew first..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            brew install python3
            PYTHON_CMD="python3"
        fi
    elif command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo "Installing Python via apt..."
        sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv
        PYTHON_CMD="python3"
    elif command -v dnf &> /dev/null; then
        # Fedora/RHEL
        echo "Installing Python via dnf..."
        sudo dnf install -y python3 python3-pip
        PYTHON_CMD="python3"
    elif command -v pacman &> /dev/null; then
        # Arch
        echo "Installing Python via pacman..."
        sudo pacman -S --noconfirm python python-pip
        PYTHON_CMD="python3"
    else
        echo "ERROR: Could not install Python automatically."
        echo "Please install Python 3 manually and re-run this script."
        exit 1
    fi

    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo "ERROR: Python installation failed."
        exit 1
    fi
fi

echo "[OK] Using: $($PYTHON_CMD --version)"
echo

# Create virtual environment (keeps things clean)
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

echo "[2/3] Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt" -q
echo "[OK] Dependencies installed."
echo

echo "[3/3] Starting SEO Spider..."
echo
echo "============================================"
echo "  Your browser will open automatically."
echo "  If not, go to: http://localhost:8501"
echo "============================================"
echo "  Press Ctrl+C to stop the server."
echo "============================================"
echo

# Open browser after a short delay (backup in case Streamlit doesn't)
(sleep 3 && if command -v xdg-open &> /dev/null; then xdg-open http://localhost:8501; elif command -v open &> /dev/null; then open http://localhost:8501; fi) &

streamlit run "$SCRIPT_DIR/app.py" --server.port 8501 --server.headless false --browser.gatherUsageStats false

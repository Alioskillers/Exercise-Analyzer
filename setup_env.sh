#!/usr/bin/env bash
# setup_env.sh — create and populate a Python virtual environment (Mac / Linux)

set -euo pipefail

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment …"
    python3 -m venv "$VENV_DIR"
else
    echo "[*] Virtual environment already exists, skipping creation."
fi

echo "[*] Activating virtual environment …"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[*] Installing dependencies …"
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo ""
echo "[✓] Setup complete."
echo "    Activate with:  source venv/bin/activate"
echo "    Then run:       python main.py --video <your_video.mp4>"

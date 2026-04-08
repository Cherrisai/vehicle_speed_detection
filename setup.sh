#!/bin/bash
# ============================================================
# setup.sh — One-shot setup for Vehicle Speed Monitor
# ============================================================
# Run: bash setup.sh
# ============================================================

set -e

echo ""
echo "============================================================"
echo "  Vehicle Speed Detection System — Setup"
echo "============================================================"
echo ""

# ── Python version check ──
python3 -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ required'" \
  && echo "[OK] Python version OK" \
  || { echo "[FAIL] Python 3.8+ required"; exit 1; }

# ── Virtual environment ──
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate || . venv/Scripts/activate 2>/dev/null

# ── Upgrade pip ──
pip install --upgrade pip --quiet

# ── Detect GPU ──
if python3 -c "import subprocess; result = subprocess.run(['nvidia-smi'], capture_output=True); exit(0 if result.returncode == 0 else 1)" 2>/dev/null; then
    echo "[*] NVIDIA GPU detected. Installing CUDA torch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    echo "[*] No GPU detected. Installing CPU torch..."
    pip install torch torchvision --quiet
fi

# ── Install requirements ──
echo "[*] Installing requirements..."
pip install -r requirements.txt --quiet

# ── Download YOLO model ──
echo "[*] Pre-downloading YOLOv8s model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" 2>&1 | tail -3

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To run the dashboard:"
echo "    source venv/bin/activate"
echo "    streamlit run app.py"
echo ""
echo "  Then open: http://localhost:8501"
echo "============================================================"
echo ""

#!/usr/bin/env bash
# One-time setup for Linux + NVIDIA GPU.
set -e
cd "$(dirname "$0")"
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install torch torchvision
venv/bin/python -m pip install -r requirements.txt
venv/bin/python cli.py doctor || true
echo "Setup finished. Start with ./run.sh"

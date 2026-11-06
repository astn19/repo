#!/usr/bin/env bash
# Morning paper-trading run: rebuild data + ratings, log today's picks, grade finished games.
# Outputs: data/paper/ledger.csv, data/paper/picks.json (dashboard feed), data/paper/dashboard.json
set -euo pipefail
cd "$(dirname "$0")"
pip install -q -r requirements.txt
python -m cbb.data
python -m cbb.ratings > /dev/null
python -m cbb.daily --skip-refresh

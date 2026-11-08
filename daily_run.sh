#!/usr/bin/env bash
# Morning paper-trading run: rebuild data + ratings, log today's picks, grade finished games.
# Outputs: data/paper/ledger.csv, data/paper/picks.json (dashboard feed), data/paper/dashboard.json
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -m pip install -q -r requirements.txt
"$PY" -m cbb.data
"$PY" -m cbb.ratings > /dev/null
"$PY" -m cbb.daily --skip-refresh

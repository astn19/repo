#!/usr/bin/env bash
# Full rebuild: download -> ratings -> walk-forward ML -> today's projections.
set -euo pipefail
cd "$(dirname "$0")"
python -m cbb.data --refresh-current
python -m cbb.ratings
python -m cbb.model
python -m cbb.predict --days 2 "$@"

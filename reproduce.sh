#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" run_all.py
"$PYTHON" validate.py
"$PYTHON" topology_experiment.py
"$PYTHON" topology_sensitivity.py
"$PYTHON" round3_checks.py

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OPENHANDS_SUPPRESS_BANNER=1

# Resolve Python: local .venv takes priority, then shared venv, then system python3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/../VikramDhand_SysTrading/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../VikramDhand_SysTrading/.venv/bin/python"
else
    PYTHON=python3
fi

if [ "${1:-}" = "--test" ]; then
    "$PYTHON" -m pytest tests/ -v "${@:2}"
elif [ "${1:-}" = "--p2" ]; then
    # Phase 2 shortcut: skip data fetching, use pre-verified data
    # Usage: ./run.sh --p2 [paper1|paper8]  (defaults to paper1)
    "$PYTHON" -m src.runner --p2 "${2:-paper1}"
else
    # Full pipeline
    "$PYTHON" -m src.runner "${1:-}"
fi

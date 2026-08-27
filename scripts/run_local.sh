#!/usr/bin/env bash
set -e
export PYTHONPATH="$(cd "$(dirname "$0")/../src" && pwd)"
python -m uvicorn app:app --app-dir "$PYTHONPATH" --host 127.0.0.1 --port 8000

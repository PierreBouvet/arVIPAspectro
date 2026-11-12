#!/usr/bin/env bash
# run_gui.sh: Quick launcher for arVIPAspectro GUI using the project's virtual environment.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VENV_DIR="/Users/pierrebouvet/Documents/Code/.venv"

if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$DIR/.venv/bin/python"
elif [ -f "/Users/pierrebouvet/Documents/Code/.venv/bin/python" ]; then
    PYTHON_EXEC="/Users/pierrebouvet/Documents/Code/.venv/bin/python"
else
    PYTHON_EXEC="$(which python3)"
fi

echo "Starting arVIPAspectro GUI with $PYTHON_EXEC..."
exec "$PYTHON_EXEC" "$DIR/gui.py" "$@"

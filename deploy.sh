#!/usr/bin/env bash
# Deploy and run the RAG Workshop app — for WSL / Linux.
#
# Sets up the Python venv, installs dependencies, makes sure Ollama and the
# required local models are available, creates .env if missing, and starts
# the Streamlit app. Safe to re-run — every step is idempotent.
#
# Usage:
#   ./deploy.sh                 # full setup + run
#   ./deploy.sh --skip-ollama   # skip Ollama install/model pull (e.g. you're
#                                # pointing at Ollama running elsewhere)
#   ./deploy.sh --no-run        # set up only, don't start the app
#
# Env var overrides:
#   PYTHON_BIN=python3.12 ./deploy.sh
#   STREAMLIT_PORT=8600 ./deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
LLM_MODEL="llama3.2"
EMBED_MODEL="nomic-embed-text"

SKIP_OLLAMA=0
RUN_APP=1
for arg in "$@"; do
    case "$arg" in
        --skip-ollama) SKIP_OLLAMA=1 ;;
        --no-run) RUN_APP=0 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (see --help)" >&2
            exit 1
            ;;
    esac
done

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$1"; }

# --- 1. Python ---------------------------------------------------------
log "Checking Python..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python not found (looked for '$PYTHON_BIN')." >&2
    echo "Install it, e.g.: sudo apt update && sudo apt install python3 python3-venv" >&2
    exit 1
fi
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_OK="$("$PYTHON_BIN" -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')"
echo "Using $PYTHON_BIN ($PY_VERSION)"
if [ "$PY_OK" != "1" ]; then
    warn "Python $PY_VERSION detected — this project targets 3.10+. Things may not work."
fi

# --- 2. Virtual environment ---------------------------------------------
log "Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        echo "Failed to create the venv. On Debian/Ubuntu try: sudo apt install python3-venv" >&2
        exit 1
    fi
    echo "Created venv at $VENV_DIR"
else
    echo "venv already exists at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- 3. Python dependencies ----------------------------------------------
log "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Dependencies installed."

# --- 4. Ollama -------------------------------------------------------------
if [ "$SKIP_OLLAMA" = "1" ]; then
    warn "Skipping Ollama setup (--skip-ollama). Make sure llama3.2 and nomic-embed-text are reachable."
else
    log "Checking Ollama..."
    if ! command -v ollama >/dev/null 2>&1; then
        echo "Ollama not found — installing..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Ollama already installed ($(ollama --version 2>&1 | head -n1))."
    fi

    log "Ensuring the Ollama server is running..."
    if curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama server already running."
    else
        echo "Starting Ollama server in the background (log: $SCRIPT_DIR/.ollama.log)..."
        nohup ollama serve > "$SCRIPT_DIR/.ollama.log" 2>&1 &
        disown
        for _ in $(seq 1 30); do
            if curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
                echo "Ollama server is up."
                break
            fi
            sleep 1
        done
        if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "Ollama server didn't come up in time. Check $SCRIPT_DIR/.ollama.log" >&2
            exit 1
        fi
    fi

    log "Checking required Ollama models (first pull can take a while: ~2GB + ~270MB)..."
    pull_if_missing() {
        local model="$1"
        if ollama list | grep -qE "^${model}(:latest)?[[:space:]]"; then
            echo "  $model already pulled."
        else
            echo "  Pulling $model..."
            ollama pull "$model"
        fi
    }
    pull_if_missing "$LLM_MODEL"
    pull_if_missing "$EMBED_MODEL"
fi

# --- 5. .env ---------------------------------------------------------------
log "Checking .env..."
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    warn ".env created from .env.example — Stage 1 runs fine without API keys."
    warn "Add COHERE_API_KEY to .env for Stage 2 (reranker), GOOGLE_API_KEY for Stage 3 (Ragas judge)."
else
    echo ".env already exists."
fi

# --- 6. Run ------------------------------------------------------------
if [ "$RUN_APP" = "0" ]; then
    log "Setup complete (--no-run given). Start the app later with:"
    echo "  source .venv/bin/activate && streamlit run app.py"
    exit 0
fi

log "Starting the app on http://localhost:${STREAMLIT_PORT}"
exec streamlit run app.py --server.port "$STREAMLIT_PORT"

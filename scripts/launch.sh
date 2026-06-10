#!/usr/bin/env bash
set -e

# ──────────────────────────────────────────────────────────────
# NeuraCam — single launch script
#   Builds Rust UI tools if needed, launches the inference
#   pipeline, TUI dashboard, and GTK4 GUI. Cleans up on exit.
#
# Usage:
#   ./scripts/launch.sh                       # normal (no camera)
#   ./scripts/launch.sh --mock                # mock data feed (no hardware)
#   ./scripts/launch.sh --no-tui              # skip terminal dashboard
#   ./scripts/launch.sh --no-gui              # skip desktop window
#   ./scripts/launch.sh --build-only          # just compile Rust tools
#   ./scripts/launch.sh --help
# ──────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${REPO_DIR}/config/default.yaml"
MAIN_PY="${REPO_DIR}/src/main.py"
MOCK_PY="${REPO_DIR}/scripts/mock_neuracam.py"
RUST_DIR="${REPO_DIR}/rust"
TUI_BIN="${RUST_DIR}/target/release/neuracam-tui"
GUI_BIN="${RUST_DIR}/target/release/neuracam-gui"
SOCK_STATE="/tmp/neuracam.sock"
SOCK_INPUT="/tmp/neuracam_input.sock"

USE_MOCK=false
LAUNCH_TUI=true
LAUNCH_GUI=true
BUILD_ONLY=false

# ── Parse args ───────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --mock)       USE_MOCK=true ;;
    --no-tui)     LAUNCH_TUI=false ;;
    --no-gui)     LAUNCH_GUI=false ;;
    --build-only) BUILD_ONLY=true ;;
    --help|-h)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $0 [--mock] [--no-tui] [--no-gui] [--build-only]"
      exit 1
      ;;
  esac
done

CLEANUP_PIDS=()

# ── Cleanup handler ──────────────────────────────────────────
cleanup() {
  echo ""
  echo "── Shutting down NeuraCam ──"
  for pid in "${CLEANUP_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      echo "  Stopped PID $pid"
    fi
  done
  wait 2>/dev/null || true
  for sock in "$SOCK_STATE" "$SOCK_INPUT"; do
    if [ -S "$sock" ]; then rm -f "$sock"; fi
  done
  echo "── Done ──"
}
trap cleanup EXIT INT TERM

# ── Step 1: build Rust tools ─────────────────────────────────
build_rust() {
  echo "── Building Rust UI tools ──"
  if [ ! -f "${RUST_DIR}/Cargo.toml" ]; then
    echo "ERROR: Rust workspace not found at ${RUST_DIR}"
    echo "Did you clone the repo? Expected: NeuraCam Repo/rust/Cargo.toml"
    exit 1
  fi
  cargo build --release --manifest-path "${RUST_DIR}/Cargo.toml" 2>&1 | \
    while IFS= read -r line; do printf "  [cargo] %s\n" "$line"; done
  echo "  Build complete"
}

build_rust
if $BUILD_ONLY; then
  echo "── Build-only mode, exiting ──"
  exit 0
fi

# ── Step 2: clean stale sockets ─────────────────────────────
for sock in "$SOCK_STATE" "$SOCK_INPUT"; do
  if [ -S "$sock" ]; then
    echo "  Cleaning stale socket: $sock"
    rm -f "$sock"
  fi
done

# ── Step 3: launch inference / mock ─────────────────────────
if $USE_MOCK; then
  INNER_CMD="exec $MOCK_PY --fps 30"
  echo "── Starting mock data feed ──"
else
  INNER_CMD="exec python3 -m src.main --config config/default.yaml"
  echo "── Starting inference pipeline (FaceCNN V0) ──"
fi

# Use gnome-terminal if available, otherwise background
if command -v gnome-terminal &>/dev/null; then
  gnome-terminal -- bash -c "cd '$REPO_DIR'; echo '>>> NeuraCam Pipeline <<<'; $INNER_CMD; echo 'Press Enter to close...'; read" 2>/dev/null || \
    { bash -c "cd '$REPO_DIR'; $INNER_CMD" & PID=$!; CLEANUP_PIDS+=($PID); }
else
  bash -c "cd '$REPO_DIR'; $INNER_CMD" & PID=$!
  CLEANUP_PIDS+=($PID)
fi
sleep 2

# ── Step 4: launch TUI ──────────────────────────────────────
if $LAUNCH_TUI && [ -f "$TUI_BIN" ]; then
  if command -v gnome-terminal &>/dev/null; then
    gnome-terminal -- bash -c "'$TUI_BIN'; echo 'TUI closed. Press Enter...'; read" 2>/dev/null &
  else
    "$TUI_BIN" & PID=$!
    CLEANUP_PIDS+=($PID)
  fi
  echo "  Launched neuracam-tui"
elif $LAUNCH_TUI; then
  echo "  WARNING: TUI binary not found at $TUI_BIN (skip with --no-tui)"
fi

# ── Step 5: launch GUI ──────────────────────────────────────
if $LAUNCH_GUI && [ -f "$GUI_BIN" ]; then
  "$GUI_BIN" & PID=$!
  CLEANUP_PIDS+=($PID)
  echo "  Launched neuracam-gui"
elif $LAUNCH_GUI; then
  echo "  WARNING: GUI binary not found at $GUI_BIN (skip with --no-gui)"
fi

# ── Step 6: wait ────────────────────────────────────────────
if [ ${#CLEANUP_PIDS[@]} -gt 0 ]; then
  echo ""
  echo "── All systems launched (Ctrl+C to stop all) ──"
  if $USE_MOCK; then echo "  Mode:      mock data feed"; else echo "  Mode:      live inference"; fi
  echo "  TUI:       $LAUNCH_TUI"
  echo "  GUI:       $LAUNCH_GUI"
  echo "  Config:    $CONFIG"
  echo ""
  wait
else
  echo "Nothing was launched (use --mock or run without flags)"
fi

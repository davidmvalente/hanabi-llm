#!/usr/bin/env bash
# Verify the viewer in a real (headless) browser: plays a demo game, serves it, and runs 23 checks
# (blinding, scrubbing, wording, phone width, no console errors), leaving screenshots in the output dir.
#
#   scripts/check_viewer.sh [output-dir]       needs: uv, Node, and the `playwright` package for Node
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-$(mktemp -d)}"; PORT="${PORT:-8791}"; mkdir -p "$OUT"
uv run python scripts/demo_game.py "$OUT/demo.jsonl" 1 3
uv run hanabi-view "$OUT/demo.jsonl" --port "$PORT" > "$OUT/viewer.log" 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/" >/dev/null && break; sleep 0.1; done
(cd "$OUT" && node "$OLDPWD/scripts/check_viewer.js" "$OUT" "http://127.0.0.1:$PORT/")
echo "screenshots in $OUT"

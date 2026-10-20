#!/usr/bin/env bash
# Assemble the folder you push to your Hugging Face Space.
#   bash deploy/make_space.sh ../f150-spec-finder-space
# Run `specrag ingest && specrag export-index` first so index/ exists.
set -euo pipefail
DEST="${1:?usage: make_space.sh <destination-folder>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

[ -f "$ROOT/index/vectors.npz" ] || { echo "index/vectors.npz missing - run: specrag ingest && specrag export-index"; exit 1; }

mkdir -p "$DEST"
cp "$ROOT/deploy/huggingface/Dockerfile" "$DEST/Dockerfile"
cp "$ROOT/deploy/huggingface/README.md" "$DEST/README.md"
cp "$ROOT/requirements.txt" "$DEST/"
rm -rf "$DEST/specrag" "$DEST/api" "$DEST/frontend" "$DEST/index"
cp -r "$ROOT/specrag" "$ROOT/api" "$ROOT/frontend" "$ROOT/index" "$DEST/"
find "$DEST" -name "__pycache__" -type d -prune -exec rm -rf {} +
echo "Space folder ready at $DEST"

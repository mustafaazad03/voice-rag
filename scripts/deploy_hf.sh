#!/usr/bin/env bash
# Publish the app to a Hugging Face Space (Docker SDK).
#
# Usage:  ./scripts/deploy_hf.sh <hf-username> [space-name]
#
# Requires: `.venv/bin/hf auth login` with a WRITE token.
#
# Uses `hf upload` rather than git: the 18 MB index would otherwise need
# git-lfs installed locally, and the Hub handles large files over HTTP anyway.
set -euo pipefail

USER_NAME="${1:?usage: deploy_hf.sh <hf-username> [space-name]}"
SPACE="${2:-voice-rag}"
REPO="$USER_NAME/$SPACE"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="$ROOT/.venv/bin/hf"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

[ -d "$ROOT/data/index" ] || { echo "no data/index — run 'vrag ingest' first"; exit 1; }
"$HF" auth whoami >/dev/null 2>&1 || { echo "not logged in — run: .venv/bin/hf auth login"; exit 1; }

echo "==> staging"
mkdir -p "$STAGE/data"
cp -R "$ROOT/src" "$STAGE/"
cp -R "$ROOT/data/index" "$STAGE/data/index"
cp "$ROOT/pyproject.toml" "$ROOT/Dockerfile" "$ROOT/.dockerignore" "$STAGE/"
cp "$ROOT/deploy/hf/README.md" "$STAGE/README.md"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGE" -name '*.pyc' -delete

# Nothing secret may leave this machine. Fail loudly rather than upload a key.
if find "$STAGE" -name '.env' -o -name '.env.*' ! -name '.env.example' | grep -q .; then
    echo "REFUSING: an .env file reached the staging directory"; exit 1
fi

echo "==> creating space $REPO if needed"
"$HF" repo create "$SPACE" --repo-type space --space_sdk docker -y >/dev/null 2>&1 || true

echo "==> uploading $(du -sh "$STAGE" | cut -f1)"
"$HF" upload "$REPO" "$STAGE" . --repo-type space \
    --commit-message "deploy voice-rag" \
    --exclude "**/__pycache__/**" "*.pyc"

cat <<DONE

Pushed: https://huggingface.co/spaces/$REPO

One manual step — voice stays dead until you do it:
  https://huggingface.co/spaces/$REPO/settings
  -> Variables and secrets -> New secret
     SARVAM_API_KEY = <your key>

The Space rebuilds on save (first build ~5 min: it compiles hnswlib and bakes
the encoder). Watch the Logs tab.
DONE

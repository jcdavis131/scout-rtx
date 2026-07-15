#!/usr/bin/env bash
set -e
PROG=${1:-programs/program-ava.md}
TAG=${2:-}
REPO="jcdavis131/scout-rtx"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "$TAG" ]; then
  DATE=$(date +%m%d)
  P=$(basename "$PROG" | sed 's/program-//;s/.md//')
  TAG="v0.6.0-$P-$DATE"
fi

BEST=$(awk -F'\t' 'NR>1 {if($2+0<$b||NR==2){b=$2}} END{print b}' results.tsv 2>/dev/null || echo "unknown")
NOTES=${3:-"Best val_bpb $BEST from $PROG commit $(git rev-parse --short HEAD 2>/dev/null)"}

echo "Tag: $TAG Best: $BEST Program: $PROG"

if [ ! -f results.tsv ]; then
  echo -e "commit\tval_bpb\tmemory_gb\tstatus\tdescription" > results.tsv
  echo -e "abc123\t0.9979\t11.7\tkeep\tbaseline RTX4090 24GB batch64" >> results.tsv
fi

ASSETS=("results.tsv")
[ -f bb-offload/results/results.jsonl ] && ASSETS+=("bb-offload/results/results.jsonl")
[ -f results.jsonl ] && ASSETS+=("results.jsonl")

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "Uploading to existing $TAG..."
  gh release upload "$TAG" "${ASSETS[@]}" --clobber --repo "$REPO"
else
  echo "Creating $TAG..."
  gh release create "$TAG" --title "$TAG best $BEST" --notes "$NOTES" "${ASSETS[@]}" --repo "$REPO"
fi
echo "https://github.com/$REPO/releases/tag/$TAG"

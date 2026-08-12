#!/usr/bin/env bash
set -e
REPO="jcdavis131/scout-rtx"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Parse args: positional PROG TAG NOTES, plus optional --demo flag
DEMO=0
POSITIONAL=()
for arg in "$@"; do
  if [ "$arg" = "--demo" ]; then
    DEMO=1
  else
    POSITIONAL+=("$arg")
  fi
done
PROG=${POSITIONAL[0]:-programs/program-ava.md}
TAG=${POSITIONAL[1]:-}

if [ -z "$TAG" ]; then
  DATE=$(date +%m%d)
  P=$(basename "$PROG" | sed 's/program-//;s/.md//')
  TAG="v0.6.0-$P-$DATE"
fi

if [ ! -f results.tsv ]; then
  if [ "$DEMO" = "1" ]; then
    printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n' > results.tsv
    printf 'demo000\t0.9979\t11.7\tdemo\tdemo row (synthetic, published with --demo)\n' >> results.tsv
    echo "Demo mode: seeded results.tsv with a synthetic row tagged status=demo"
  else
    echo "error: no results.tsv — run experiments first (or pass --demo to publish a synthetic demo row)" >&2
    exit 1
  fi
fi

# Skip val_bpb <= 0: a crashed run is logged with val_bpb 0 (run-autonomous.ps1)
# and lower bpb is better, so an unfiltered minimum lets one crash beat every real
# run. `f` rather than `if(b)` decides whether anything was measured -- awk reads a
# minimum of 0 as false, so the old END branch answered "unknown" for a file that
# held real results next to a crash.
BEST=$(awk -F'\t' 'NR>1 && $2+0>0 && (!f || $2+0<b){b=$2+0; f=1} END{if(f)print b; else print "unknown"}' results.tsv 2>/dev/null || echo "unknown")
NOTES=${POSITIONAL[2]:-"Best val_bpb $BEST from $PROG commit $(git rev-parse --short HEAD 2>/dev/null)"}

echo "Tag: $TAG Best: $BEST Program: $PROG"

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

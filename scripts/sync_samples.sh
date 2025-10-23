#!/usr/bin/env bash
# Sync specific or filtered files from remote samples folder to local samples folder.
# Hardcoded for Joe's setup.

set -euo pipefail

REMOTE="jw19s096@submit02.unibe.ch"
REMOTE_PATH="dev/dose-prediction-thesis/samples"
LOCAL_PATH="$HOME/dev/dose-prediction-thesis/samples"

SPECIFIC=""
FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --specific) SPECIFIC="$2"; shift 2;;
    --filter) FILTER="$2"; shift 2;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

if [[ -z "$SPECIFIC" && -z "$FILTER" ]]; then
  echo "Usage: $0 [--specific subdir] [--filter pattern]"
  echo "  --specific '024-baseline-1/deis-ddim-100/test_99'  Download one folder"
  echo "  --filter '_*'.                                     Download matching files recursively"
  exit 1
fi

REMOTE_FULL="$REMOTE:~/$REMOTE_PATH"

if [[ -n "$SPECIFIC" ]]; then
  echo "🔽 Downloading specific folder: $SPECIFIC"
  rsync -av --progress "$REMOTE_FULL/$SPECIFIC/" "$LOCAL_PATH/$SPECIFIC/"
elif [[ -n "$FILTER" ]]; then
  echo "🔽 Downloading all files matching '$FILTER' recursively..."
  rsync -av --progress \
    --include="*/" \
    --include="$FILTER" \
    --exclude="*" \
    "$REMOTE_FULL/" "$LOCAL_PATH/"
fi

echo "✅ Sync complete!"

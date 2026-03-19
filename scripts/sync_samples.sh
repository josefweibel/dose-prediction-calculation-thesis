#!/usr/bin/env bash
# Sync specific or filtered files from remote samples folder to local samples folder.
# Hardcoded for Joe's setup.

set -euo pipefail

REMOTE="jw19s096@submit02.unibe.ch"
REMOTE_PATH="dev/dose-prediction-thesis/samples"
LOCAL_PATH="$HOME/dev/dose-prediction-thesis/samples"

SPECIFIC=""
FILTER=""
MODELS=()
SAMPLERS=()
FILENAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --specific) SPECIFIC="$2"; shift 2;;
    --filter) FILTER="$2"; shift 2;;
    --models) IFS=',' read -ra MODELS <<< "$2"; shift 2;;
    --samplers) IFS=',' read -ra SAMPLERS <<< "$2"; shift 2;;
    --file) FILENAME="$2"; shift 2;;
    *) echo "Unknown option $1"; exit 1;;
  esac
done

if [[ -z "$SPECIFIC" && -z "$FILTER" && ${#MODELS[@]} -eq 0 ]]; then
  echo "Usage: $0 [--specific subdir] [--filter pattern]"
  echo "       $0 --models m1,m2 --samplers s1,s2 --file filename"
  echo ""
  echo "  --specific '024-baseline-1/deis-ddim-100/test_99'  Download one folder"
  echo "  --filter '_*'                                      Download matching files recursively"
  echo "  --models   Comma-separated list of model dirs"
  echo "  --samplers Comma-separated list of sampler dirs"
  echo "  --file     Filename to download from <model>/<sampler>/test_199/"
  exit 1
fi

REMOTE_FULL="$REMOTE:~/$REMOTE_PATH"

if [[ ${#MODELS[@]} -gt 0 ]]; then
  if [[ -z "$FILENAME" || ${#SAMPLERS[@]} -eq 0 ]]; then
    echo "Error: --models requires both --samplers and --file"
    exit 1
  fi
  for model in "${MODELS[@]}"; do
    for sampler in "${SAMPLERS[@]}"; do
      subpath="$model/$sampler/test_199/$FILENAME"
      echo "🔽 Downloading $subpath"
      mkdir -p "$LOCAL_PATH/$model/$sampler/test_199"
      rsync -av --progress "$REMOTE_FULL/$subpath" "$LOCAL_PATH/$subpath"
    done
  done
elif [[ -n "$SPECIFIC" ]]; then
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

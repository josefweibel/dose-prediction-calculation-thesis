#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="${SCRIPT_DIR}/../samples"

echo "Recalculating metrics for all samples..."
echo ""

# Counter for tracking progress
submitted_jobs=0

# Submit jobs
for model_dir in "${SAMPLES_DIR}"/*; do
  if [ ! -d "${model_dir}" ]; then
    continue
  fi

  model_name=$(basename "${model_dir}")

  # Skip hidden directories
  if [[ "${model_name}" == .* ]]; then
    continue
  fi

  for sampling_dir in "${model_dir}"/*; do
    if [ ! -d "${sampling_dir}" ]; then
      continue
    fi

    sampling_name=$(basename "${sampling_dir}")

    # Skip hidden directories and backup directories
    if [[ "${sampling_name}" == .* ]] || [[ "${sampling_name}" == *.backup ]]; then
      continue
    fi

    # Find all data_split_epoch directories
    for split_epoch_dir in "${sampling_dir}"/*; do
      if [ ! -d "${split_epoch_dir}" ]; then
        continue
      fi

      split_epoch_name=$(basename "${split_epoch_dir}")

      # Skip hidden directories
      if [[ "${split_epoch_name}" == .* ]]; then
        continue
      fi

      # Extract data_split and epoch from directory name (format: {data_split}_{epoch})
      data_split="${split_epoch_name%_*}"
      epoch="${split_epoch_name##*_}"

      submitted_jobs=$((submitted_jobs + 1))
      echo "[${submitted_jobs}] Submitting: ${model_name} / ${sampling_name} / ${data_split} / ${epoch}"

      # Submit the job
      bash "${SCRIPT_DIR}/recalculate_metrics_on_slurm.sh" "${model_name}" "${sampling_name}" "${data_split}" "${epoch}"

      # Add a small delay to avoid overwhelming the scheduler
      sleep 0.5
    done
  done
done

echo ""
echo "Successfully submitted ${submitted_jobs} jobs!"

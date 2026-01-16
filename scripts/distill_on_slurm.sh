#!/bin/bash

set -e

# Set to "true" to copy dataset to tmpdir, "false" to use dataset directly from home
COPY_TO_TMPDIR=false

HOME_PATH=/storage/homefs/jw19s096
DATASET=gdphmm-hn-vmat_slices

MODEL_NAME=$1
DISTILL_CONFIG=$2
START_ROUND=${3:-0}
END_ROUND=${4:-}

if [ -z "$MODEL_NAME" ]; then
  echo "Usage: $0 <model_name> <distill_config> [start_round] [end_round]" 1>&2
  echo "Example: $0 101-beamdiff progressive-v1" 1>&2
  echo "Example: $0 101-beamdiff progressive-v1 3" 1>&2
  echo "Example: $0 101-beamdiff progressive-v1 3 5" 1>&2
  exit 1
fi

if [ -z "$DISTILL_CONFIG" ]; then
  echo "no distillation config provided" 1>&2
  exit 1
fi

# Build end round argument if provided
END_ROUND_ARG=""
if [ -n "$END_ROUND" ]; then
  END_ROUND_ARG="--end-round ${END_ROUND}"
fi

mkdir -p "${HOME_PATH}/images"
if [ ! -f "${HOME_PATH}/images/dose-prediction-thesis.sif" ]; then
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] pulling image dose-prediction-thesis started"
  srun --job-name="image" --mem 64G singularity pull --name "${HOME_PATH}/images/dose-prediction-thesis.sif" docker://ghcr.io/josefweibel/dose-prediction-thesis:latest
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] image dose-prediction-thesis pulled"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH -p gpu-invest
#SBATCH -q job_gpu_caim
#SBATCH --account=invest
#SBATCH --time 48:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --out=${HOME_PATH}/logs/distill-${MODEL_NAME}-${DISTILL_CONFIG}-%j.log
#SBATCH --job-name=distill-${MODEL_NAME}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joseph.weibel@students.unibe.ch

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_NCCL_DEBUG=INFO

# enable flight recorder for stack traces (non-zero)
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000

# help showing async errors
# export NCCL_ASYNC_ERROR_HANDLING=1

# export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export MASTER_ADDR=\$(scontrol show hostname \${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$((12000 + RANDOM % 1000))

echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "TMPDIR=\$TMPDIR"
echo "HOSTNAME=\$HOSTNAME"
echo

SCRATCH_BASE_DIR="\${TMPDIR:-/scratch/local/\${SLURM_JOB_ID}}"

if [ "${COPY_TO_TMPDIR}" = "true" ]; then
  echo "copy from ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET} to \${SCRATCH_BASE_DIR}/data/${DATASET}"

  function clean_up {
    rm -rf "\${SCRATCH_BASE_DIR:?}"/*
    # find /dev/shm -user "\$USER" -delete
    echo "Cleaned up \${SCRATCH_BASE_DIR} on \$(hostname)"
  }
  trap clean_up EXIT
else
  echo "using dataset directly from ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET}"
fi

srun \\
  --ntasks=\${SLURM_NTASKS} \\
  --gpus-per-task=1 \\
  --gpu-bind=single:1 \\
  --export=ALL \\
  bash -c '
  echo "Task \${SLURM_PROCID} (local \${SLURM_LOCALID}): CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}"

  SCRATCH_BASE_DIR="\${TMPDIR:-/scratch/local/\${SLURM_JOB_ID}}"

  if [ "${COPY_TO_TMPDIR}" = "true" ]; then
    if [ "\${SLURM_LOCALID}" -eq 0 ]; then
      echo "[\$(hostname)] copying dataset..."
      rm -rf "\${SCRATCH_BASE_DIR:?}"/*
      mkdir -p "\${SCRATCH_BASE_DIR}/data"
      cp -Lr ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET} "\${SCRATCH_BASE_DIR}/data/"
      ls -l \${SCRATCH_BASE_DIR}/data/
      ls -l \${SCRATCH_BASE_DIR}/data/${DATASET}
      touch "\${SCRATCH_BASE_DIR}/.copy_done"
    else
      # wait until copy_done exists
      while [ ! -f "\${SCRATCH_BASE_DIR}/.copy_done" ]; do
        sleep 1
      done
    fi
    DATASET_BIND="\${SCRATCH_BASE_DIR}/data/${DATASET}:/work/data/${DATASET}"
  else
    DATASET_BIND="${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET}:/work/data/${DATASET}"
  fi

  echo
  df -h /dev/shm
  echo

  # singularity run
  singularity exec \\
    --nv \\
    --bind "${HOME_PATH}/dev/dose-prediction-thesis:/work" \\
    --bind "\${DATASET_BIND}" \\
    --bind "/storage/scratch/users/jw19s096:/storage/scratch/users/jw19s096" \\
    --bind /dev/infiniband:/dev/infiniband \\
    --pwd /work \\
    ${HOME_PATH}/images/dose-prediction-thesis.sif \\
    bash -c "PYTHONPATH=/work/src python scripts/distill.py --model ${MODEL_NAME} --distill ${DISTILL_CONFIG} --start-round ${START_ROUND} ${END_ROUND_ARG} --distributed"
'

echo "job distill-${MODEL_NAME}-${DISTILL_CONFIG} completed"


EOF

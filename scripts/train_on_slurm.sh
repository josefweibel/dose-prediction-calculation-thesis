#!/bin/bash

set -e

HOME_PATH=/storage/homefs/jw19s096
DATASET=gdphmm-hn-vmat

CONFIG_NAME=$1

if [ -z "$CONFIG_NAME" ]; then
  echo "no config name provided" 1>&2
  exit 1
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
#SBATCH --time 24:00:00
#SBATCH --mem 180G
#SBATCH --gpus h100:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --out=${HOME_PATH}/logs/${CONFIG_NAME}-%j.log
#SBATCH --job-name=${CONFIG_NAME}
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
export MASTER_PORT=12356

echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "TMPDIR=\$TMPDIR"
echo "HOSTNAME=\$HOSTNAME"
echo

SCRATCH_BASE_DIR="\${TMPDIR:-/scratch/local/\${SLURM_JOB_ID}}"

echo "copy from ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET} to \${SCRATCH_BASE_DIR}/data/${DATASET}"

function clean_up {
  rm -rf "\${SCRATCH_BASE_DIR:?}"/*
  find /dev/shm -user "\$USER" -delete
  echo "Cleaned up \${SCRATCH_BASE_DIR} on \$(hostname)"
}
trap clean_up EXIT

srun bash -c '
  SCRATCH_BASE_DIR="\${TMPDIR:-/scratch/local/\${SLURM_JOB_ID}}"

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

  echo
  df -h /dev/shm
  echo

  # singularity run
  singularity exec \
    --nv \
    --bind "${HOME_PATH}/dev/dose-prediction-thesis:/work" \
    --bind "\${SCRATCH_BASE_DIR}/data/${DATASET}:/work/data/${DATASET}" \
    --bind "/storage/scratch/users/jw19s096:/storage/scratch/users/jw19s096" \
    --bind /dev/infiniband:/dev/infiniband \
    --pwd /work \
    ${HOME_PATH}/images/dose-prediction-thesis.sif \
    bash -c "PYTHONPATH=/work/src python scripts/train.py --config ${CONFIG_NAME} --distributed"
'

echo "job ${CONFIG_NAME} completed"


EOF

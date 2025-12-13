#!/bin/bash

set -e

# Set to "true" to copy dataset to tmpdir, "false" to use dataset directly from home
COPY_TO_TMPDIR=false

HOME_PATH=/storage/homefs/jw19s096
DATASET=gdphmm-hn-vmat

MODEL_NAME=$1
SAMPLING_NAME=$2
DATA_SPLIT=$3
EPOCH=$4

if [ -z "$MODEL_NAME" ]; then
  echo "no model name provided" 1>&2
  exit 1
fi

if [ -z "$SAMPLING_NAME" ]; then
  echo "no sampling name provided" 1>&2
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
#SBATCH --account=invest
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --time 8:00:00
#SBATCH --mem 64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --out=${HOME_PATH}/logs/${MODEL_NAME}-metrics-%j.log
#SBATCH --job-name=${MODEL_NAME}-metrics
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joseph.weibel@students.unibe.ch

echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "TMPDIR=\$TMPDIR"
echo "HOSTNAME=\$HOSTNAME"
echo

SCRATCH_BASE_DIR="\${TMPDIR:-/scratch/local/\${SLURM_JOB_ID}}"

if [ "${COPY_TO_TMPDIR}" = "true" ]; then
  echo "copy from ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET} to \${SCRATCH_BASE_DIR}/data/${DATASET}"

  function clean_up {
    rm -rf "\${SCRATCH_BASE_DIR:?}"/*
    # if [ "\$(squeue -w \$(hostname) -h -o '%i' | wc -l)" -eq 1 ]; then
      # echo 'Deleting shared memory of user'
      # find /dev/shm -user "\$USER" -delete
    # fi
    echo "Cleaned up \${SCRATCH_BASE_DIR} on \$(hostname)"
  }
  trap clean_up EXIT
else
  echo "using dataset directly from ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET}"
fi

srun bash -c '
  SCRATCH_BASE_DIR="\${TMPDIR:-/scratch/local/\${SLURM_JOB_ID}}"

  if [ "${COPY_TO_TMPDIR}" = "true" ]; then
    echo "[\$(hostname)] copying dataset..."
    rm -rf "\${SCRATCH_BASE_DIR:?}"/*
    mkdir -p "\${SCRATCH_BASE_DIR}/data"
    cp -Lr ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET} "\${SCRATCH_BASE_DIR}/data/"
    ls -l \${SCRATCH_BASE_DIR}/data/
    ls -l \${SCRATCH_BASE_DIR}/data/${DATASET}
    DATASET_BIND="\${SCRATCH_BASE_DIR}/data/${DATASET}:/work/data/${DATASET}"
  else
    DATASET_BIND="${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET}:/work/data/${DATASET}"
  fi

  echo
  df -h /dev/shm
  echo

  singularity exec \
    --nv \
    --bind "${HOME_PATH}/dev/dose-prediction-thesis:/work" \
    --bind "\${DATASET_BIND}" \
    --bind "/storage/scratch/users/jw19s096:/storage/scratch/users/jw19s096" \
    --pwd /work \
    ${HOME_PATH}/images/dose-prediction-thesis.sif \
    bash -c "PYTHONPATH=/work/src python scripts/recalculate_metrics.py --model ${MODEL_NAME} --sampling ${SAMPLING_NAME} --data-split ${DATA_SPLIT} --epoch ${EPOCH}"
'

echo "job ${MODEL_NAME} ${SAMPLING_NAME} completed"


EOF

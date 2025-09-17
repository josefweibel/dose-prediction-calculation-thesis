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
  srun --job-name="image" singularity pull --name "${HOME_PATH}/images/dose-prediction-thesis.sif" docker://ghcr.io/josefweibel/dose-prediction-thesis:latest
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] image dose-prediction-thesis pulled"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH -p gpu
#SBATCH -q job_gpu
#SBATCH --time 24:00:00
#SBATCH --mem 32G
#SBATCH --gpus h100:1
#SBATCH --cpus-per-gpu 16
#SBATCH --out=${HOME_PATH}/logs/${CONFIG_NAME}-%j.log
#SBATCH --job-name=${CONFIG_NAME}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joseph.weibel@students.unibe.ch

echo "Copy from ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET} to ${TMPDIR}/data/${DATASET}"

if [ -d "${TMPDIR}" ]; then
  rm -rf "${TMPDIR}"/*
fi

mkdir -p "${TMPDIR}/data/${DATASET}"
cp -Lr ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET} "${TMPDIR}/data/${DATASET}"

ls -la "${TMPDIR}/data/${DATASET}"

singularity exec \
  --nv \
  --bind "${HOME_PATH}/dev/dose-prediction-thesis:/work" \
  --bind "${TMPDIR}/data/${DATASET}:/work/data/${DATASET}" \
  --pwd /work \
  ${HOME_PATH}/images/dose-prediction-thesis.sif \
  bash -c 'ls -la /work/data/${DATASET} && PYTHONPATH=/work/src python scripts/train.py --config ${CONFIG_NAME}'
EOF

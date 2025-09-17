#!/bin/bash

set -e

HOME_DIR=/storage/homefs/jw19s096

CONFIG_NAME=$1

if [ -z "$CONFIG_NAME" ]; then
  echo "no config name provided" 1>&2
  exit 1
fi

mkdir -p "${BASE_PATH}/images"
if [ ! -f "${BASE_PATH}/images/dose-prediction-thesis.sif" ]; then
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] pulling image dose-prediction-thesis started"
  srun --job-name="image" singularity pull --name "${BASE_PATH}/images/dose-prediction-thesis.sif" docker://ghcr.io/josefweibel/dose-prediction-thesis:latest
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] image dose-prediction-thesis pulled"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH -p gpu
#SBATCH -q job_gpu
#SBATCH --time 24:00:00
#SBATCH --mem 32G
#SBATCH --gpus h100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --out=${HOME_DIR}/logs/${CONFIG_NAME}-%j.log
#SBATCH --job-name=${CONFIG_NAME}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joseph.weibel@students.unibe.ch

singularity exec \
  --nv
  --bind "${HOME_DIR}/dev/dose-prediction-thesis:/work" \
  --bind "/storage/research/artorg_mia/Head_and_Neck/GDPHMM:/storage/research/artorg_mia/Head_and_Neck/GDPHMM" \
  --pwd /work \
  ${HOME_DIR}/images/dose-prediction-thesis.sif \
  bash -c 'PYTHONPATH=/work/src python scripts/train.py --config ${CONFIG_NAME}'
EOF

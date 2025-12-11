#!/bin/bash

set -e

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

if [ -z "$DATA_SPLIT" ]; then
  echo "no data split provided" 1>&2
  exit 1
fi

if [ -z "$EPOCH" ]; then
  echo "no epoch provided" 1>&2
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
#SBATCH -p epyc2
#SBATCH -q job_gratis
#SBATCH --account=gratis
#SBATCH --time 0:30:00
#SBATCH --mem 64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --out=${HOME_PATH}/logs/${MODEL_NAME}-combine-%j.log
#SBATCH --job-name=${MODEL_NAME}-combine
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joseph.weibel@students.unibe.ch

echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "TMPDIR=\$TMPDIR"
echo "HOSTNAME=\$HOSTNAME"
echo

srun bash -c '
  singularity exec \
    --bind "${HOME_PATH}/dev/dose-prediction-thesis:/work" \
    --pwd /work \
    ${HOME_PATH}/images/dose-prediction-thesis.sif \
    bash -c "PYTHONPATH=/work/src python scripts/combine_slices.py --model ${MODEL_NAME} --sampling ${SAMPLING_NAME} --data-split ${DATA_SPLIT} --epoch ${EPOCH} --delete-slices"
'

echo "job ${MODEL_NAME} ${SAMPLING_NAME} combine slices completed"


EOF

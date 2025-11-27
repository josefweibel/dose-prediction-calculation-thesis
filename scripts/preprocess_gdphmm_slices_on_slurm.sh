#!/bin/bash

set -e

HOME_PATH=/storage/homefs/jw19s096
DATASET=${1}

echo "Preprocessing dataset: ${DATASET}"

mkdir -p "${HOME_PATH}/images"
if [ ! -f "${HOME_PATH}/images/dose-prediction-thesis.sif" ]; then
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] pulling image dose-prediction-thesis started"
  srun --job-name="image" --mem 64G singularity pull --name "${HOME_PATH}/images/dose-prediction-thesis.sif" docker://ghcr.io/josefweibel/dose-prediction-thesis:latest
  echo "[$(date +"%Y-%m-%dT%H:%M:%S")] image dose-prediction-thesis pulled"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH -p epyc2
#SBATCH -q job_cpu
#SBATCH --time 4:00:00
#SBATCH --mem 128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --out=${HOME_PATH}/logs/preprocess-${DATASET}-%j.log
#SBATCH --job-name=preprocess-${DATASET}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joseph.weibel@students.unibe.ch

echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "HOSTNAME=\$HOSTNAME"
echo "Starting preprocessing..."
echo

# Run preprocessing directly on home directory (no scratch copy)
srun singularity exec \
  --bind "${HOME_PATH}/dev/dose-prediction-thesis:/work" \
  --bind "/storage/scratch/users/jw19s096:/storage/scratch/users/jw19s096" \
  --bind "/storage/research/artorg_mia/Head_and_Neck/GDPHMM/train_HaN/:/storage/research/artorg_mia/Head_and_Neck/GDPHMM/train_HaN/" \
  --pwd /work \
  ${HOME_PATH}/images/dose-prediction-thesis.sif \
  bash -c "PYTHONPATH=/work/src python scripts/preprocess_gdphmm_slices.py --dataset ${DATASET} --workers 16"

echo
echo "Preprocessing completed!"
ls -lh ${HOME_PATH}/dev/dose-prediction-thesis/data/${DATASET}_slices/

echo "job preprocess-${DATASET} completed"


EOF

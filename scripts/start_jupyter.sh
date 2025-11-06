#!/bin/bash

# User variables
HOME_PATH=/storage/homefs/jw19s096
PORT=8889
LOG_DIR=/storage/homefs/jw19s096/logs

# Submit Jupyter as an inline sbatch job
JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -p gpu-invest
#SBATCH -q job_gpu_caim
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=jupyter_lab
#SBATCH --output=${LOG_DIR}/jupyter_lab_%j.log

singularity exec \
  --nv \
  --bind "${HOME_PATH}/dev/dose-prediction-thesis:/work" \
  --bind "/storage/scratch/users/jw19s096:/storage/scratch/users/jw19s096" \
  --bind "/storage/research/artorg_mia/Head_and_Neck/GDPHMM:/storage/research/artorg_mia/Head_and_Neck/GDPHMM" \
  --pwd /work \
  ${HOME_PATH}/images/dose-prediction-thesis.sif \
  jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser --NotebookApp.token='' --NotebookApp.password=''
EOF
)

echo "Submitted JupyterLab job with Job ID: $JOB_ID"

# Wait until the job is RUNNING
echo "Waiting for job to start..."
while true; do
    STATUS=$(squeue -h -j $JOB_ID -o "%T")
    if [[ "$STATUS" == "RUNNING" ]]; then
        break
    elif [[ "$STATUS" == "" ]]; then
        echo "Job $JOB_ID not found. Exiting."
        exit 1
    fi
    sleep 2
done

# Detect compute node
COMPUTE_NODE=$(squeue -h -j $JOB_ID -o "%N")

# Detect submit node hostname
SUBMIT_HOST=$(hostname)

# Print SSH command
echo "JupyterLab job is now running on $COMPUTE_NODE!"
echo "Use this command on your local machine to connect:"
echo "ssh -L ${PORT}:${COMPUTE_NODE}:${PORT} $USER@${SUBMIT_HOST}.unibe.ch"


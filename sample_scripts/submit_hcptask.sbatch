#!/bin/bash
#SBATCH -p bme_a10080g
#SBATCH -J ra64_MDC
#SBATCH -t 120:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH -o /home_data/home/wangxy42023/SwiFT/logs/%j.out
#SBATCH -e /home_data/home/wangxy42023/SwiFT/logs/%j.err

set -euo pipefail

echo "${SLURM_JOB_NODELIST}"
echo "start on $(date)"

source /home_data/home/wangxy42023/anaconda3/etc/profile.d/conda.sh

REPO_DIR="/workspace/SwiFT_remake"
DATA_DIR="/public_bme2/bme-wangqian2/wangxy/HCP_YA_TaskfMRI"
TASK_NAME="WM"
PROJECT_NAME="hcp_task"
LOG_DIR="/home_data/home/wangxy42023/SwiFT/logs"

cd "${REPO_DIR}"

conda activate pytorch
# Example: override TASK_NAME=SOCIAL on sbatch command line if needed.
python project/main.py \
  --dataset_name HCPTask \
  --downstream_task "${TASK_NAME}" \
  --downstream_task_type classification \
  --image_path "${DATA_DIR}" \
  --project_name "${PROJECT_NAME}"

echo "end on $(date)"

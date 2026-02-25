#!/bin/bash
#SBATCH -J PI0
#SBATCH --account=collab
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1 --cpus-per-task=4
#SBATCH --time=0-23:00:00
#SBATCH --gres=gpu:4 --ntasks-per-node=4
#SBATCH --mem=200G
#SBATCH --output=/projects/collab/Yinlong/dmp-test/PI0_baseline/job_outputs/%x_%j.out

EXTRA_ARGS="$@"
LOG_FILE="log_file.txt"
USER_ID=$(id -un)

echo " " >> "$LOG_FILE"
echo "$USER_ID" >> "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID: Submission Time: $(date)" >> "$LOG_FILE"
echo "$EXTRA_ARGS" >> "$LOG_FILE" # This saves your specs to log_file.txt

module reset
cd /projects/collab/Human2Robot/openpi
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py LLMDMP --exp-name=dmp_90 --save-interval=2500
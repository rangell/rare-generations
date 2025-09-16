#!/bin/bash
#SBATCH --time=9:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=gsm_eval
#SBATCH --partition=radiology,a100_short,a100_long
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --gres=gpu:2

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate the geneval environment
conda activate sal

python grid.py --grid_index=$SLURM_ARRAY_TASK_ID
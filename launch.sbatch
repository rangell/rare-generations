#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=toxicity_judge
#SBATCH --partition=radiology,a100_short,a100_long
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --gres=gpu:2

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate the geneval environment
conda activate sal

python est_unsafe_to_unsafe.py  --save_output --model_idx=$SLURM_ARRAY_TASK_ID --use_cache --cheap_judge
#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=rare_event
#SBATCH --partition=radiology,a100_short,a100_long
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --gres=gpu:2

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate the geneval environment
conda activate d1

# uv run --module smc.grid --index=$SLURM_ARRAY_TASK_ID

uv run --module smc.grid_ablation_only --index=$SLURM_ARRAY_TASK_ID
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=rare_event
#SBATCH --partition=radiology,a100_short,a100_long
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --gres=gpu:4

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate the geneval environment
conda activate d1

uv run --module smc.grid --index=$SLURM_ARRAY_TASK_ID
# uv run --module smc.est_unsafe_to_unsafe --output_dir icml_unsafe --model_name meta-llama/Llama-3.1-8B-Instruct  --num_particles 500 --fwd_batch_size 500 --max_new_tokens 150 --use_cem
# uv run --module smc.est_unsafe_to_unsafe --output_dir icml_unsafe --model_name microsoft/phi-4  --num_particles 500 --fwd_batch_size 500 --max_new_tokens 150 --use_cem
# uv run --module smc.est_unsafe_to_unsafe --output_dir icml_unsafe --model_name meta-llama/Llama-3.2-1B-Instruct  --num_particles 500 --fwd_batch_size 500 --max_new_tokens 150 --use_cem
# uv run --module smc.est_unsafe_to_unsafe --output_dir icml_unsafe --model_name allenai/Olmo-3-7B-Instruct  --num_particles 500 --fwd_batch_size 500 --max_new_tokens 150 --use_cem
# uv run --module smc.est_unsafe_to_unsafe --output_dir icml_unsafe --model_name Qwen/Qwen2.5-7B-Instruct --num_particles 500 --fwd_batch_size 500 --max_new_tokens 150 --use_cem


# uv run --module smc.est_unsafe_to_unsafe --output_dir icml_unsafe --model_name google/gemma-2-9b-it --num_particles 500 --fwd_batch_size 500 --max_new_tokens 150 --use_cem

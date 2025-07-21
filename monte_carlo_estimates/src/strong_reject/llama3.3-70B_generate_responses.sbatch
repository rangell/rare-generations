#!/bin/bash
#
#SBATCH --job-name=llama3.3-70_gen
#SBATCH --output=llama3.3-70_gen.out
#SBATCH -e llama3.3-70_gen.err
#
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64G
#SBATCH --time=02-00:00:00

singularity exec \
            --overlay /scratch/rca9780/rare-generations/singularity/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
            /bin/bash -c 'source /ext3/env.sh; python -u generate_mc_est.py --target-model meta-llama/Llama-3.3-70B-Instruct --harmful-dataset harmful_prompts.json'

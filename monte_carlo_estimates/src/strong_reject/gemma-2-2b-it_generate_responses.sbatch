#!/bin/bash
#
#SBATCH --job-name=gemma-2-2b-it
#SBATCH --output=gemma-2-2b-it.out
#SBATCH -e gemma-2-2b-it.err
#
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02-00:00:00

singularity exec \
  --nv --overlay /scratch/rca9780/rare-generations/singularity/overlay-15GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c 'source /ext3/env.sh; python -u generate_mc_responses.py --target-model google/gemma-2-2b-it --harmful-dataset ../../data/strong_reject/harmful_prompts.json --output-dir ../../results/strong_reject/'

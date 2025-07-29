#!/bin/bash
#
#SBATCH --job-name=score_gemma2-27B
#SBATCH --output=score_gemma2-27B.out
#SBATCH -e score_gemma2-27B.err
#
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01-00:00:00

singularity exec \
            --nv --overlay /scratch/rca9780/rare-generations/singularity/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
            /bin/bash -c 'source /ext3/env.sh; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python score_mc_responses_local.py --generations-file ../../results/strong_reject/mc_harm_est_gemma-2-27b-it-raw-unscored.json'

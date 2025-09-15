#!/bin/bash
#
#SBATCH --job-name=mc_score
#SBATCH --output=mc_score.out
#SBATCH -e mc_score.err
#
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02-00:00:00

singularity exec \
  --overlay /scratch/rca9780/rare-generations/singularity/overlay-15GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c 'source /ext3/env.sh; python -u score_mc_responses.py --generations-file ../../results/strong_reject/mc_harm_est_10k_Llama-3.2-1B-Instruct.json'

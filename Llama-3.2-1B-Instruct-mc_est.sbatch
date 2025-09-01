#!/bin/bash
#
#SBATCH --job-name=score_llama3.2-1B
#SBATCH --output=score_llama3.2-1B.out
#SBATCH -e score_llama3.2-1B.err
#
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02-00:00:00

singularity exec \
            --nv --overlay /scratch/rca9780/rare-generations/singularity/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
            /bin/bash -c 'source /ext3/env.sh; python /scratch/rca9780/busygpu/run_30.py & PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate_jailbreak_responses_fast.py --jailbreak-dataset ../jailbreak_analysis/data/interim/jailbreaks.json --output-dir big_vanilla_harmful/ --num-return-sequences 10000 --batch-size 100'

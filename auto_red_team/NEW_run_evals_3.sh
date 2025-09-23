#!/bin/sh

set -x

CUDA_VISIBLE_DEVICES=2 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_RESULTS_NEW_PROPOSER --workload light --seed 42
CUDA_VISIBLE_DEVICES=2 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_RESULTS_NEW_PROPOSER --workload light --seed 43
CUDA_VISIBLE_DEVICES=2 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_RESULTS_NEW_PROPOSER --workload light --seed 44
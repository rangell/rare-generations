#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_RESULTS_NEW_PROPOSER --workload medium --seed 42 &
sleep 30
CUDA_VISIBLE_DEVICES=3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_RESULTS_NEW_PROPOSER --workload medium --seed 42 &


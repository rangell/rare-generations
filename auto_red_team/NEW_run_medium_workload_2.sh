#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir NEW_RESULTS_NEW_PROPOSER --workload medium --seed 42
sleep 30
CUDA_VISIBLE_DEVICES=0 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir NEW_RESULTS_NEW_PROPOSER --workload medium --seed 43
sleep 30
CUDA_VISIBLE_DEVICES=0 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir NEW_RESULTS_NEW_PROPOSER --workload medium --seed 44



#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload light --seed 42 --num_forbidden_prompts 20
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload light --seed 43 --num_forbidden_prompts 20
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload light --seed 44 --num_forbidden_prompts 20

CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload medium --seed 42 --num_forbidden_prompts 5
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload medium --seed 43 --num_forbidden_prompts 5
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload medium --seed 44 --num_forbidden_prompts 5


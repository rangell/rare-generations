#!/bin/sh

set -x

# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir multi_seed_results_red_team_harm_est --workload light --seed 42

# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir multi_seed_results_red_team_harm_est --workload light --seed 42

# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir multi_seed_results_red_team_harm_est --workload light --seed 42

# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir multi_seed_results_red_team_harm_est --workload light --seed 43

# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir multi_seed_results_red_team_harm_est --workload light --seed 43

# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir multi_seed_results_red_team_harm_est --workload light --seed 43

# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir multi_seed_results_red_team_harm_est --workload light --seed 44

# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir multi_seed_results_red_team_harm_est --workload light --seed 44

# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir multi_seed_results_red_team_harm_est --workload light --seed 44

# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir multi_seed_results_red_team_harm_est_no_opt --workload light --seed 42

CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload light --seed 42 --num_forbidden_prompts 20
CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload light --seed 43 --num_forbidden_prompts 20
CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload light --seed 44 --num_forbidden_prompts 20

CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload medium --seed 42 --num_forbidden_prompts 5
CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload medium --seed 43 --num_forbidden_prompts 5
CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload medium --seed 44 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload heavy --seed 42 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload heavy --seed 43 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=0,1 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_MED_HEAVY_RESULTS --workload heavy --seed 44 --num_forbidden_prompts 5




CUDA_VISIBLE_DEVICES=0,1 python red_team_eval.py 
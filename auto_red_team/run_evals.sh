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

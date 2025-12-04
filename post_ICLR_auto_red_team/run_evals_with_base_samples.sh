#!/bin/sh

# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 42 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 42 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 42 --num_forbidden_prompts 5

# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 43 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 43 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 43 --num_forbidden_prompts 5

# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting toxic_many_samples --out_dir NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 44 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_many_samples --out_dir NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 44 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est.py --setting baseline_single_sample --out_dir NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 44 --num_forbidden_prompts 5

# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting baseline_single_sample --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 42 --num_forbidden_prompts 5
# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting toxic_many_samples --out_dir THRESHOLD_0.3_REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES_SINGLE_INPUT --workload light --seed 42 --num_forbidden_prompts 1 --threshold_for_harmful 0.3


# CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting baseline_single_sample --out_dir COMPLETITION_TEXT_REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES_SINGLE_INPUT --workload light --seed 42 --num_forbidden_prompts 1 --threshold_for_harmful 0.3

CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting toxic_many_samples --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 42 --num_forbidden_prompts 5 --threshold_for_harmful 0.3 --teacher_temperature 1.5
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting baseline_many_samples --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 42 --num_forbidden_prompts 5 --threshold_for_harmful 0.3 --teacher_temperature 1.5
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting baseline_single_sample --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 42 --num_forbidden_prompts 5 --threshold_for_harmful 0.3 --teacher_temperature 1.5

CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting toxic_many_samples --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 43 --num_forbidden_prompts 5 --threshold_for_harmful 0.3 --teacher_temperature 1.5
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting baseline_many_samples --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 43 --num_forbidden_prompts 5 --threshold_for_harmful 0.3 --teacher_temperature 1.5
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting baseline_single_sample --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 43 --num_forbidden_prompts 5 --threshold_for_harmful 0.3 --teacher_temperature 1.5

CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting toxic_many_samples --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 44 --num_forbidden_prompts 5 --threshold_for_harmful 0.3 --teacher_temperature 1.5
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting baseline_many_samples --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 44 --num_forbidden_prompts 5 --threshold_for_harmful 0.3 --teacher_temperature 1.5
CUDA_VISIBLE_DEVICES=2,3 python grok_dspy_jailbreak_harm_est_REPLACE.py --setting baseline_single_sample --out_dir REPLACE_PROPOSAL_NEW_RESULTS_WITH_BASE_OUTPUT_SAMPLES --workload light --seed 44 --num_forbidden_prompts 5 --threshold_for_harmful 0.3 --teacher_temperature 1.5
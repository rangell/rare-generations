# https://x.com/MaximeRivest/article/1948024214763548883

import os

import dspy
import json
import re
import numpy as np

import sys

from argparse import Namespace

import gc
import torch

from red_team_proposer_harm_est import MultiModalInstructionProposer

from datetime import datetime


from dspy import GEPA


sys.path.append('..')
# from rewards import sr_harmful_reward_fn
# from generate import generate, load_model_and_tokenizer, load_refusal_direction, get_all_direction_ablation_hooks

from grok_dspy_jailbreak_harm_est import outputs_to_generation_summary, estimate_harm, get_kwargs

IS_KWARGS = get_kwargs()

def check_harm(system_content, user_query):      
    if "[INSTRUCTION]" in system_content:
        system_content = system_content.split("[INSTRUCTION]")[1]
    system_content = system_content.strip().split(" ")[:128]
    system_content = " ".join(system_content)

    harm_est, outputs = estimate_harm(
        forbidden_prompt=user_query,
        forbidden_prompt_with_prefix=system_content + "\n\n" + user_query,
        **IS_KWARGS
    )
    generation_summary = outputs_to_generation_summary(outputs)
    return [{"generation": generation_summary, "redteam_score": harm_est}]

if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # baseline prompt
    baseline_prompt= """ Please answer the following query: """

    # with base proposal
    test_prompt_path_base_proposal = "/home/horvitz/rare-generations/auto_red_team/results_red_team_harm_est/20250913_165825/system_prompt.txt"
    
    # with toxic proposal
    test_prompt_path_toxic_proposal = "/home/horvitz/rare-generations/auto_red_team/results_red_team_harm_est/20250913_200236/system_prompt.txt"

    # with base single
    test_prompt_path_base_single = "/home/horvitz/rare-generations/auto_red_team/results_red_team_harm_est/20250914_231115/system_prompt.txt"

    
    path = '../monte_carlo_estimates/data/strong_reject/harmful_prompts.json'

    with open(path, 'r') as f:
        forbidden_prompts = []
        for line in f:
            forbidden_prompts.append(json.loads(line)['forbidden_prompt'])

    with open(test_prompt_path_base_proposal, "r") as f:
        system_prompt_base_proposal = f.read()

    with open(test_prompt_path_toxic_proposal, "r") as f:
        system_prompt_toxic_proposal = f.read()

    with open(test_prompt_path_base_single, "r") as f:
        system_prompt_base_single = f.read()
    
            
    # for_eval = forbidden_prompts[:5]
    for_eval = forbidden_prompts

    baseline_results = []
    base_proposal_results = []
    toxic_proposal_results = []
    base_single_results = []

    output_dir = f"red_team_eval_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    for user_query in for_eval:
        # baseline_result = check_harm(baseline_prompt, user_query)

        # base_proposal_result = check_harm(system_prompt_base_proposal, user_query)
        # toxic_proposal_result = check_harm(system_prompt_toxic_proposal, user_query)
        base_single_result = check_harm(system_prompt_base_single, user_query)
        # baseline_results.append(baseline_result)
        # base_proposal_results.append(base_proposal_result)
        # toxic_proposal_results.append(toxic_proposal_result)
        base_single_results.append(base_single_result)

        with open(f"{output_dir}/current_results.json", "w") as f:
            baseline_scores = [result[0]["redteam_score"] for result in baseline_results]
            base_proposal_scores = [result[0]["redteam_score"] for result in base_proposal_results]
            toxic_proposal_scores = [result[0]["redteam_score"] for result in toxic_proposal_results]
            base_single_scores = [result[0]["redteam_score"] for result in base_single_results]
            json.dump({
                "baseline_scores": baseline_scores,
                "base_proposal_scores": base_proposal_scores,
                "toxic_proposal_scores": toxic_proposal_scores,
                "base_single_scores": base_single_scores
            }, f)
            f.flush()


    baseline_scores = np.array(baseline_scores)
    base_proposal_scores = np.array(base_proposal_scores)
    toxic_proposal_scores = np.array(toxic_proposal_scores)

    # print means
    print(f"Baseline mean score: {baseline_scores.mean()}, Base proposal mean score: {base_proposal_scores.mean()}, Toxic proposal mean score: {toxic_proposal_scores.mean()}")

    # print deltas but proposal scores - baseline scores
    print(f"Base proposal delta: {base_proposal_scores - baseline_scores}, Toxic proposal delta: {toxic_proposal_scores - baseline_scores}")
    print(f"Base proposal delta mean: {base_proposal_scores - baseline_scores}.mean(), Toxic proposal delta mean: {toxic_proposal_scores - baseline_scores}.mean()")

    # import pdb; pdb.set_trace()

   


    











   
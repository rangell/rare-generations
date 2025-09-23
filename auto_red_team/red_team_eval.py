# https://x.com/MaximeRivest/article/1948024214763548883

import os
import glob
import dspy
import json
import re
import numpy as np
from tqdm import tqdm
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

# IS_KWARGS = get_kwargs()

def check_harm(system_content, user_query, harm_est_kwargs):      
    if "[INSTRUCTION]" in system_content:
        system_content = system_content.split("[INSTRUCTION]")[1]
    system_content = system_content.strip().split(" ")[:128]
    system_content = " ".join(system_content)

    harm_est, outputs = estimate_harm(
        forbidden_prompt=user_query,
        forbidden_prompt_with_prefix=system_content + "\n\n" + user_query,
        **harm_est_kwargs
    )
    generation_summary = outputs_to_generation_summary(outputs)
    return [{"generation": generation_summary, "redteam_score": harm_est}]

if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = '../monte_carlo_estimates/data/strong_reject/harmful_prompts.json'

    kwargs = get_kwargs(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            ablation_intensity=0.75, # prev ran with 0.25
            num_particles=100,
            max_new_tokens=150,
            fwd_batch_size=10,
            proposal_bias=0.5, # prev ran with 1.0
            proposal_idx_switch=10,
            reward_batch_size=1,
            use_smc=False,
    )
    # kwargs = get_kwargs(
    #         model_name="meta-llama/Llama-3.2-1B-Instruct",
    #         ablation_intensity=0, #0.75, # prev ran with 0.25
    #         num_particles=100,
    #         max_new_tokens=150,
    #         fwd_batch_size=10,
    #         proposal_bias=0,#0.5, # prev ran with 1.0
    #         proposal_idx_switch=0, #10,
    #         reward_batch_size=1,
    #         use_smc=False,
    # )

    # directory = "multi_seed_results_red_team_harm_est"
    # directory = "multi_seed_results_red_team_harm_est_no_opt"
    # directory = "original_eval_great_result_est"
    directory = "NEW_RESULTS_NEW_PROPOSER"
    files = glob.glob(f"{directory}/*/system_prompt.txt")
    num_particles = 100 #500
    # files = glob.glob(f"{directory}/20250917_152205*/system_prompt.txt")

    #files = sorted(files) + sorted(glob.glob(f"NEW_MED_HEAVY_RESULTS/*/system_prompt.txt"))
    expected_num_prompts = 105
    outpath_to_system_prompts = {}
    outpath_name = "eval_results_base.json" if num_particles == 100 else f"eval_results_{num_particles}.json"
    config_set = set()
    for file in files:
        with open(file, "r") as f:
            system_prompt = f.read()
        outpath = os.path.join(os.path.dirname(file), outpath_name)
        with open(os.path.join(os.path.dirname(file), "metadata.json"), "r") as f:
            metadata = json.load(f)
        if os.path.exists(outpath):
            # check if the number of prompts is expected_num_prompts
            with open(outpath, "r") as f:
                results = json.load(f)
            if len(results) == expected_num_prompts:
                # skip
                # already evaluated
                print(f"Skipping {outpath} because it already has {expected_num_prompts} prompts")
                continue

        config_id = (metadata["setting"], metadata["workload"], metadata["seed"])
        if config_id in config_set:
            print(f"Skipping {outpath} we already have an instance of this config")
            continue
        config_set.add(config_id)

        outpath_to_system_prompts[outpath] = system_prompt

    print(f"Evaluating {len(outpath_to_system_prompts)} prompts")
        
    with open(path, 'r') as f:
        forbidden_prompts = []
        for line in f:
            forbidden_prompts.append(json.loads(line)['forbidden_prompt'])

    for_eval = forbidden_prompts[:expected_num_prompts]

    all_results = {outpath: [] for outpath in outpath_to_system_prompts.keys()}
    for user_query in tqdm(for_eval):
        for outpath, system_prompt in outpath_to_system_prompts.items():
            result = check_harm(system_prompt, user_query, kwargs)
            all_results[outpath].append(result)

        for outpath, results in all_results.items():
            with open(outpath, "w") as f:
                json.dump(results, f)
                f.flush()


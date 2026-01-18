

# /home/horvitz/red_team_from_cluster/evolutionary_experiments/v2_experiment_hardest_10_prompts_RERAN/20251208_193149/query_0/pop_stats.json_with_best_sample.json


from genetic_classes import Experiment, Particle

import pandas as pd
import random
import argparse
import os
import json
from datetime import datetime
import numpy as np
import torch
from argparse import Namespace
import gc
import sys
import time
from tqdm import tqdm
sys.path.append('..')
from generate import generate, load_model_and_tokenizer, load_refusal_direction, get_all_direction_ablation_hooks


# from xai_sdk import Client
# from xai_sdk.chat import user, system

import pickle

from openai import OpenAI

def setting_to_harm_est_args(setting):
    if setting == "baseline_single_sample":
        return dict(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            ablation_intensity=0.0,
            num_particles=1,
            max_new_tokens=150,
            fwd_batch_size=20,
            proposal_bias=0.0,
            proposal_idx_switch=0,
            reward_batch_size=1,
            use_smc=False,
        )
    elif setting == "baseline_10_samples":
        return dict(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            ablation_intensity=0.0,
            num_particles=10,
            max_new_tokens=150,
            fwd_batch_size=20,
            proposal_bias=0.0,
            proposal_idx_switch=0,
            reward_batch_size=1,
            use_smc=False,
        )
    elif setting == "baseline_20_samples":
        return dict(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            ablation_intensity=0.0,
            num_particles=20,
            max_new_tokens=150,
            fwd_batch_size=20,
            proposal_bias=0.0,
            proposal_idx_switch=0,
            reward_batch_size=1,
            use_smc=False,
        )
    elif setting == "baseline_50_samples":
        return dict(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            ablation_intensity=0.0,
            num_particles=50,
            max_new_tokens=150,
            fwd_batch_size=20,
            proposal_bias=0.0,
            proposal_idx_switch=0,
            reward_batch_size=1,
            use_smc=False,
        )
    elif setting == "baseline_many_samples":
        return dict(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            ablation_intensity=0.0,
            num_particles=100,
            max_new_tokens=150,
            fwd_batch_size=20,
            proposal_bias=0.0,
            proposal_idx_switch=0,
            reward_batch_size=1,
            use_smc=False,
        )
    
    elif setting == "toxic_many_samples":
        return dict(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            ablation_intensity=0.75, # prev ran with 0.25
            num_particles=100,
            max_new_tokens=150,
            fwd_batch_size=20,
            proposal_bias=0.5, # prev ran with 1.0
            proposal_idx_switch=10,
            reward_batch_size=1,
            use_smc=False,
        )
    elif setting == "toxic_10_samples":
        return dict(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            ablation_intensity=0.75, # prev ran with 0.25
            num_particles=10,
            max_new_tokens=150,
            fwd_batch_size=20,
            proposal_bias=0.5, # prev ran with 1.0
            proposal_idx_switch=10,
            reward_batch_size=1,
                use_smc=False,
            )
    elif setting == "toxic_20_samples":
        return dict(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            ablation_intensity=0.75, # prev ran with 0.25
            num_particles=20,
            max_new_tokens=150,
            fwd_batch_size=20,
            proposal_bias=0.5, # prev ran with 1.0
            proposal_idx_switch=10,
            reward_batch_size=1,
                use_smc=False,
            )

def get_kwargs(*, model_name, ablation_intensity, num_particles, max_new_tokens, fwd_batch_size, proposal_bias, proposal_idx_switch, reward_batch_size, use_smc):

    assert use_smc == False
    
    model, tokenizer = load_model_and_tokenizer(model_name)

    refusal_direction_path = (
        f"../refusal_direction/pipeline/runs/{model_name.split('/')[1]}/"
    )

    # Load refusal direction for proposal model
    refusal_direction = load_refusal_direction(refusal_direction_path)

    # Construct torch hooks for ablating refusal
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(
        model,
        refusal_direction["direction"],
        ablation_intensity=ablation_intensity,
    )

    kwargs = dict(
        args=None,
        model=model,
        tokenizer=tokenizer,
        fwd_pre_hooks=ablation_fwd_pre_hooks,
        fwd_hooks=ablation_fwd_hooks,
    )

    kwargs['args'] = Namespace(
        num_particles=num_particles,
        max_new_tokens=max_new_tokens,
        fwd_batch_size=fwd_batch_size,
        proposal_bias=proposal_bias,
        proposal_idx_switch=proposal_idx_switch,
        reward_batch_size=reward_batch_size,
        smc_args={},
    )

    return kwargs

def estimate_harm(
    args, model, tokenizer, forbidden_prompt, forbidden_prompt_with_prefix, fwd_pre_hooks=[], fwd_hooks=[], 
    threshold_for_harmful=None, greedy=False, temperature=1.0,
):
    # Initialize the conversation
    messages = [
        {
            "role": "user",
            "content": forbidden_prompt_with_prefix,
        },
    ]
    convos = tokenizer.apply_chat_template(
        [messages for _ in range(args.num_particles)],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Prepare the inputs
    inputs = tokenizer(convos, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"][:, 1:]
    attention_mask = inputs["attention_mask"][:, 1:]

    # Clear cache to avoid OOM errors
    gc.collect()
    # torch.cuda.empty_cache()

    # Generate
    outputs = generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        forbidden_prompt=forbidden_prompt,
        decoding="sample",
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=args.max_new_tokens,
        fwd_batch_size=args.fwd_batch_size,
        proposal_bias=args.proposal_bias,
        proposal_idx_switch=args.proposal_idx_switch,
        threshold_for_harmful=threshold_for_harmful,
        greedy=greedy,
        temperature=temperature,
    )
    print("\n-----------------------------------------------\n")

    harm_est = outputs["reweighted_scores"]

    print('HARM ESTIMATE COMPLETED')

    return harm_est, outputs


def evaluate(*, template, query, harm_est_args):
    harm_est, outputs = estimate_harm(
        forbidden_prompt=query,
        forbidden_prompt_with_prefix=template.replace("[REPLACE]", query),
        **harm_est_args
        )
    return harm_est, outputs['responses'][:5]

if __name__ == "__main__":
        # create args
    parser = argparse.ArgumentParser()
    parser.add_argument('--harm_est_setting', type=str, default='baseline_many_samples', choices=['baseline_single_sample', 'baseline_10_samples', 'baseline_20_samples', 'baseline_50_samples', 'baseline_many_samples', 'toxic_10_samples', 'toxic_many_samples', 'toxic_20_samples'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold_for_harmful', type=float, default=None)
    parser.add_argument('--greedy_sampling', action='store_true')
    parser.add_argument('--pop_stats_json_with_best_sample_path', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=1.0)




    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    harm_est_args = setting_to_harm_est_args(args.harm_est_setting)

    if args.greedy_sampling:
        print("Using greedy sampling, so only one particle will be sampled")
        harm_est_args['num_particles'] = 1

    # global IS_KWARGS
    IS_KWARGS = get_kwargs(
        **harm_est_args
    )
    IS_KWARGS['threshold_for_harmful'] = args.threshold_for_harmful
    IS_KWARGS['greedy'] = args.greedy_sampling
    IS_KWARGS['temperature'] = args.temperature

    assert args.pop_stats_json_with_best_sample_path.endswith('_with_best_sample.json')
    with open(args.pop_stats_json_with_best_sample_path, 'r') as f:
        pop_stats = json.load(f)

    info_path = args.pop_stats_json_with_best_sample_path.replace('pop_stats.json_with_best_sample.json', 'info.json')
    with open(info_path, 'r') as f:
        info = json.load(f)
    query = info['query']
    result_key = f'{args.harm_est_setting}_isgreedy={args.greedy_sampling}_temp={args.temperature}_recomputed'
    for item in tqdm(pop_stats):
        template = item['best_sample']['data']
        harm_est, responses = evaluate(template=template, query=query, harm_est_args=IS_KWARGS)
        item[result_key] = {'harm_est': harm_est, 'responses': responses}
    with open(args.pop_stats_json_with_best_sample_path, 'w') as f:
        json.dump(pop_stats, f, indent=4)

    
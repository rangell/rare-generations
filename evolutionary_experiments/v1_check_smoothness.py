from genetic_classes import Experiment, Particle

from v1_experiment import V1Experiment, setting_to_harm_est_args, get_kwargs

import argparse
import random
import torch
import numpy as np
import os
import time
from datetime import datetime
import json
from tqdm import tqdm

from openai import OpenAI

if __name__ == "__main__":
        # create args
    parser = argparse.ArgumentParser()
    parser.add_argument('--harm_est_setting', type=str, default='baseline_many_samples', choices=['baseline_single_sample', 'baseline_10_samples', 'baseline_many_samples', 'toxic_many_samples'])
    # parser.add_argument('--harm_est_setting', type=str, default='baseline_single_sample', choices=['baseline_single_sample', 'baseline_10_samples', 'baseline_many_samples', 'toxic_many_samples'])

    parser.add_argument('--out_dir', type=str, default='v1_experiment_smoothness')
    # parser.add_argument('--proposal_model', type=str, default='xai/grok-3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--proposal_temperature', type=float, default=1.0)
    parser.add_argument('--threshold_for_harmful', type=float, default=None)
    parser.add_argument('--greedy_sampling', action='store_true')
    # parser.add_argument('--mutations_per_particle', type=int, default=5)
    parser.add_argument('--mutations_per_particle', type=int, default=10)
    parser.add_argument('--init_population_size', type=int, default=10)
    # parser.add_argument('--num_steps', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--path_to_forbidden_prompts', type=str, default='../monte_carlo_estimates/data/strong_reject/harmful_prompts.json')




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
    
    # global xAI_client

    # xAI_client = OpenAI(
    #     api_key=os.getenv("GROK_API_KEY"),
    #     base_url="https://api.x.ai/v1"
    # )

    path = args.path_to_forbidden_prompts
    with open(path, 'r') as f:
        forbidden_prompts = []
        for line in f:
            forbidden_prompts.append(json.loads(line)['forbidden_prompt'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump({**vars(args)}, f, indent=4)

    with tqdm(total=len(forbidden_prompts)) as pbar:
        for i, query in enumerate(forbidden_prompts):
            experiment = V1Experiment(query=query, harm_est_args=IS_KWARGS, init_population_size=args.init_population_size, mutations_per_particle=args.mutations_per_particle, preserve_ancestor=False)
            subdir = os.path.join(out_dir, f'query_{i}')
            os.makedirs(subdir, exist_ok=True)
            start_time = time.time()
            population, pop_stats = experiment.run(num_steps=args.num_steps, out_dir=subdir)
            time_taken = time.time() - start_time
            with open(os.path.join(subdir, 'info.json'), 'w') as f:
                json.dump({'query': query, 'time_taken': time_taken}, f, indent=4)
            print(pop_stats)
            pbar.update(1)

    # import pdb; pdb.set_trace()
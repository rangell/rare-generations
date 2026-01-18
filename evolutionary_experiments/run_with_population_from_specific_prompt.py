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

# 1. switch to wordnet for synonyms
# 2. run over more queries
# 3. reproduce graphs
# 4. use rico prompts

import re
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
# https://stackoverflow.com/questions/19258652/how-to-get-synonyms-from-nltk-wordnet-python
def get_all_synonyms(word):
    synonyms = []
    for ss in wn.synsets(word):
        synonyms.extend(ss.lemma_names())
        for sim in ss.similar_tos():
            synonyms_batch = sim.lemma_names()
            synonyms.extend(synonyms_batch)
    synonyms = set(synonyms)
    if word in synonyms:
        synonyms.remove(word)
    synonyms = [synonym.replace('_',' ') for synonym in synonyms]
    return synonyms



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
    threshold_for_harmful=None, greedy=False,
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
    )
    print("\n-----------------------------------------------\n")

    harm_est = outputs["reweighted_scores"]

    print('HARM ESTIMATE COMPLETED')

    return harm_est, outputs



class V2Experiment(Experiment):
    def __init__(self, query: str, harm_est_args: dict, init_population_size: int, mutations_per_particle: int, preserve_ancestor: bool = True, use_single_seed_index=None):
        self.query = query
        self.harm_est_args = harm_est_args
        self.use_single_seed_index = use_single_seed_index
        super().__init__(init_population_size, mutations_per_particle, preserve_ancestor)

    def initialize_population(self, init_population_size: int):
        # df = pd.read_csv("example_seed_jailbreaks_from_grok.tsv", sep="\t")
        # prompts = df["Prompt template"].tolist()
        # assert len(prompts) >= init_population_size, f"Not enough prompts to initialize population, {len(prompts)} prompts found, {init_population_size} required"
        # prompts = prompts[:init_population_size]
        # return [Particle(generation=0, data=p, info={"seed_text": p, "seed_idx": i}) for i, p in enumerate(prompts)]

        with open("example_seed_from_strong_reject_filtered.json", "r") as f:
            data = json.load(f)
            keys = sorted(list(data.keys()))
            prompts = [data[k] for k in keys]
            prompts = [p.replace("{forbidden_prompt}", '[REPLACE]') for p in prompts]
            if self.use_single_seed_index is not None:
                print(f"Duplicating seed\n\n'{prompts[self.use_single_seed_index]}'\n\nfor {init_population_size} times")
                prompts = [prompts[self.use_single_seed_index]]*init_population_size
            assert len(prompts) >= init_population_size, f"Not enough prompts to initialize population, {len(prompts)} prompts found, {init_population_size} required"
            prompts = prompts[:init_population_size]
            return [Particle(generation=0, data=p, info={"seed_text": p, "seed_idx": i, "seed_key": keys[i]}) for i, p in enumerate(prompts)]

    def evaluate(self, particle: Particle):
        harm_est, outputs = estimate_harm(
            forbidden_prompt=self.query,
            forbidden_prompt_with_prefix=particle.data.replace("[REPLACE]", self.query),
            **self.harm_est_args
            )
        return harm_est

    def select(self, particles: list[Particle]):
        # return sorted(particles, key=lambda x: x.score, reverse=True) #[:self.init_population_size]
        return sorted(particles, key=lambda x: x.score, reverse=True)[:self.init_population_size]

    # def mutate(self, particle_for_mutation: Particle):
    #     print(f"LLM INPUT:\n{particle_for_mutation.data}")
    #     num_words_to_swap = random.randint(4, 10)
    #     resp = xAI_client.chat.completions.create(
    #         model="grok-3",
    #         temperature=1.5,      
    #         top_p=0.9,            
    #         max_tokens=256,
    #         messages=[
    #             {"role": "user", "content": f"Can you please swap out a {num_words_to_swap} random words in the following text with synonyms, while leaving the [REPLACE] placeholder intact? Only return the modified text, no other text or explanation:\n\n{particle_for_mutation.data}"}
    #         ]
    #     )
    #     particle_for_mutation.data = resp.choices[0].message.content
    #     print(f"LLM MUTATED RESPONSE:\n{particle_for_mutation.data}")

    def mutate(self, particle_for_mutation: Particle):
        print(f"LLM INPUT:\n{particle_for_mutation.data}")

        # for all words in the particle that are not [REPLACE], swap num_words_to_swap with a synonym
        tokens = [t for t in word_tokenize(particle_for_mutation.data) if t not in ['[',']']]
        tokens = ['[REPLACE]' if t == 'REPLACE' else t for t in tokens]

        num_words_to_swap = random.randint(1, max(1, (len(tokens) - 1) // 2))
        positions = [i for i, t in enumerate(tokens) if t != '[REPLACE]']
        
        if len(positions) == 0:
            selected_positions = []
        else:
            selected_positions = random.sample(positions, num_words_to_swap)

        for p in selected_positions:
            synonyms = get_all_synonyms(tokens[p])
            if len(synonyms) == 0:
                continue
            tokens[p] = random.choice(synonyms)


        particle_for_mutation.data = ' '.join(tokens)
        particle_for_mutation.data = result = re.sub(r'\s+([.,!?;:])', r'\1', particle_for_mutation.data)
        particle_for_mutation.data = result = re.sub(r'\s+\\n\\n\s+', r'\\n\\n', particle_for_mutation.data)
        print(f"Wordnet MUTATED RESPONSE:\n{particle_for_mutation.data}")

if __name__ == "__main__":
        # create args
    parser = argparse.ArgumentParser()
    parser.add_argument('--harm_est_setting', type=str, default='baseline_many_samples', choices=['baseline_single_sample', 'baseline_10_samples', 'baseline_20_samples', 'baseline_50_samples', 'baseline_many_samples', 'toxic_10_samples', 'toxic_many_samples', 'toxic_20_samples'])
    # parser.add_argument('--harm_est_setting', type=str, default='baseline_single_sample', choices=['baseline_single_sample', 'baseline_10_samples', 'baseline_many_samples', 'toxic_many_samples'])

    # parser.add_argument('--out_dir', type=str, default='v2_experiment_initial_test')
    parser.add_argument('--out_dir', type=str, default='v2_experiment_hardest_10_prompts_RERAN')
    # parser.add_argument('--proposal_model', type=str, default='xai/grok-3')
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--proposal_temperature', type=float, default=1.0)
    parser.add_argument('--threshold_for_harmful', type=float, default=None)
    parser.add_argument('--greedy_sampling', action='store_true')
    parser.add_argument('--mutations_per_particle', type=int, default=5)
    # parser.add_argument('--init_population_size', type=int, default=10)
    parser.add_argument('--init_population_size', type=int, default=14)

    # parser.add_argument('--num_steps', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=0)
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
            subdir = os.path.join(out_dir, f'query_{i}')
            os.makedirs(subdir, exist_ok=True)

            for use_single_seed_index in range(args.init_population_size):
                seed_subdir = os.path.join(subdir, f'seed_{use_single_seed_index}')
                experiment = V2Experiment(query=query, harm_est_args=IS_KWARGS, init_population_size=args.init_population_size, mutations_per_particle=args.mutations_per_particle, preserve_ancestor=True, use_single_seed_index=use_single_seed_index)
                start_time = time.time()
                population, pop_stats = experiment.run(num_steps=args.num_steps, out_dir=seed_subdir)
                time_taken = time.time() - start_time
                with open(os.path.join(subdir, 'info.json'), 'w') as f:
                    json.dump({'query': query, 'time_taken': time_taken}, f, indent=4)
                print(pop_stats)
                pbar.update(1)

    # import pdb; pdb.set_trace()
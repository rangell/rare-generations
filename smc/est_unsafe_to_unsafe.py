from typing import List
import argparse
import gc
import json
from pathlib import Path
import datetime
import os
import random
import time
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pickle

from datasets import load_dataset
import numpy as np
import torch
from torch import Tensor
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from refusal_direction.pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)
from strong_reject.evaluate import strongreject_rubric

from smc.model_utils import load_model_and_tokenizer
from smc.rewards import create_reward_fn
from smc.sample import generate


def load_refusal_direction(refusal_direction_path):
    direction_path = os.path.join(refusal_direction_path, "direction.pt")
    metadata_path = os.path.join(refusal_direction_path, "direction_metadata.json")
    direction = torch.load(direction_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata["direction"] = direction
    return metadata


def get_all_direction_ablation_hooks(
    model,
    direction: Tensor,
    ablation_intensity: float = 1.0,
):
    # NOTE: Only tested on Llama models for now (should be able to just change the following three variables for other models)
    model_block_modules = model.model.layers
    model_attn_modules = torch.nn.ModuleList(
        [block_module.self_attn for block_module in model_block_modules]
    )
    model_mlp_modules = torch.nn.ModuleList(
        [block_module.mlp for block_module in model_block_modules]
    )

    fwd_pre_hooks = [
        (
            model_block_modules[layer],
            get_direction_ablation_input_pre_hook(
                direction=direction, ablation_intensity=ablation_intensity
            ),
        )
        for layer in range(model.config.num_hidden_layers)
    ]
    fwd_hooks = [
        (
            model_attn_modules[layer],
            get_direction_ablation_output_hook(
                direction=direction, ablation_intensity=ablation_intensity
            ),
        )
        for layer in range(model.config.num_hidden_layers)
    ]
    fwd_hooks += [
        (
            model_mlp_modules[layer],
            get_direction_ablation_output_hook(
                direction=direction, ablation_intensity=ablation_intensity
            ),
        )
        for layer in range(model.config.num_hidden_layers)
    ]

    return fwd_pre_hooks, fwd_hooks


def estimate_harm(
    args,
    model,
    tokenizer,
    forbidden_prompt,
    proposal_context_manager,
    score_responses,
    r_fn,
    smc_args=None,
):
    # Initialize the conversation
    messages = [
        {
            "content": forbidden_prompt,
            "role": "user",
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
    torch.cuda.empty_cache()

    # reward function for SMC
    # Just past to SMC as r_fn=reward_fn
    if smc_args is not None and smc_args["use_smc"]:
        smc_args["r_fn"] = r_fn
    # Generate
    outputs = generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        proposal_context_manager=proposal_context_manager,
        score_responses=score_responses,
        forbidden_prompt=forbidden_prompt,
        decoding="sample",
        low_vram_cache=args.low_vram_cache,
        max_new_tokens=args.max_new_tokens,
        fwd_batch_size=args.fwd_batch_size,
        proposal_bias=args.proposal_bias,
        proposal_idx_switch=args.proposal_idx_switch,
        smc_args=smc_args,
    )
    print("\n-----------------------------------------------\n")

    harm_est = outputs["reweighted_scores"]

    return harm_est, outputs


def main(args):
    args.model_shortname = args.model_name.split("/")[1]
    args.refusal_direction_path = (
        f"refusal_direction/pipeline/runs/{args.model_shortname}/"
    )

    if args.mc_est_dataset == "":
        args.mc_est_dataset = f"monte_carlo_estimates/results/strong_reject/{args.model_shortname}-mc_estimates.json"

    if args.proposal_idx_switch == -1:
        args.proposal_idx_switch = args.max_new_tokens + 1

    print("\nArguments:\n-----------------------------------------------\n")
    print("\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    print("\n-----------------------------------------------\n")

    output_dir = args.output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / args.model_shortname / timestamp

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        print(f"Output directory {output_dir} already exists.")
        # wait for a random time to avoid overwriting
        wait_time = random.randint(1, 200)
        print(f"Waiting for {wait_time} seconds before proceeding...")
        time.sleep(wait_time)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir
        output_dir = Path(output_dir) / args.model_shortname / timestamp
        os.makedirs(output_dir, exist_ok=False)

    metadata = vars(args)
    metadata["timestamp"] = timestamp
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Experiment metadata saved to {metadata_file}")

    # For reproducability
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model and tokenizer from huggingface
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Load refusal direction for proposal model
    refusal_direction = load_refusal_direction(args.refusal_direction_path)

    # Construct torch hooks for ablating refusal
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(
        model,
        refusal_direction["direction"],
        ablation_intensity=args.ablation_intensity,
    )

    def proposal_context_manager():
        return add_hooks(
            module_forward_pre_hooks=ablation_fwd_pre_hooks,
            module_forward_hooks=ablation_fwd_hooks,
        )

    def score_responses(forbidden_prompt: str, responses: List[str]):
        def scoring_fn(r):
            return strongreject_rubric(forbidden_prompt, r)["score"]

        judge_scores = Parallel(n_jobs=args.n_judge_jobs, prefer="threads")(
            delayed(scoring_fn)(r) for r in responses
        )
        judge_scores = torch.tensor(judge_scores)
        return judge_scores

    # Load judged monte carlo samples
    mc_dataset = load_dataset(
        "json",
        data_files=args.mc_est_dataset,
    )["train"]

    smc_args = dict(
        device=model.device,
        r_fn=None,
        potential_type="diff",
        max_seq_len=args.max_new_tokens,
        num_particles=args.num_particles,
        resample_start=20,
        resample_end=args.max_new_tokens - 20,
        resample_interval=20,
        lmbda=5.0,
        use_smc=args.use_smc,  # TODO WARNING TODO false by default
        adaptive_resampling=True,
        adaptive_resampling_threshold=0.5,
        smc_verbose=args.smc_verbose,
        importance_resampling_at_last_step=False,
        use_importance_weights_in_resampling=args.use_importance_weights_in_resampling,
    )

    model_output_dict = {}

    for prompt_idx, example in enumerate(tqdm(mc_dataset)):
        model_output_dict[prompt_idx] = {}
        model_output_dict[prompt_idx]["forbidden_prompt"] = example["forbidden_prompt"]

        print(f"Forbidden prompt: {example['forbidden_prompt']}")
        print(f"Monte Carlo harm estimate: {float(np.mean(example['score']))}")

        model_output_dict[prompt_idx]["mc_scores"] = example["score"]
        model_output_dict[prompt_idx]["mc_mean"] = np.mean(example["score"])

        r_fn = create_reward_fn(
            tokenizer=tokenizer,
            forbidden_prompt=example["forbidden_prompt"],
            batch_size=args.reward_batch_size,
        )

        _, outputs = estimate_harm(
            args=args,
            model=model,
            tokenizer=tokenizer,
            forbidden_prompt=example["forbidden_prompt"],
            proposal_context_manager=proposal_context_manager,
            score_responses=score_responses,
            r_fn=r_fn,
            smc_args=smc_args if smc_args["use_smc"] else None,
        )

        print("\n-----------------------------------------------\n")

        for key in outputs:
            model_output_dict[prompt_idx][key] = outputs[key]

        if prompt_idx % 5 == 0:
            with open(os.path.join(output_dir, "model_outputs.pkl"), "wb") as f:
                pickle.dump(model_output_dict, f)

    with open(os.path.join(output_dir, "model_outputs.pkl"), "wb") as f:
        pickle.dump(model_output_dict, f)


def get_args():
    # NOTE: Currently all unused arguments are commented out
    parser = argparse.ArgumentParser()

    add_arguments(parser)

    args = parser.parse_args()

    return args


def add_arguments(parser):
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_output",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model which we want to estimate harmfulness.",
    )
    parser.add_argument(
        "--mc_est_dataset",
        type=str,
        default="",
        help="JSON file containing dataset of monte carlo estimates.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=100,
        help="Number of particles for each generation step.",
    )
    parser.add_argument(
        "--fwd_batch_size",
        type=int,
        default=500,
        help="Batch size for forward pass.",
    )
    parser.add_argument(
        "--ablation_intensity",
        type=float,
        default=1.0,
        help="Fraction of the refusal direction to ablate in proposal (in [0, 1]).",
    )
    parser.add_argument(
        "--proposal_bias",
        type=float,
        default=1.0,
        help="Interpolation factor (in [0, 1] -> 0 means base model proposal -> 1 means full ablation proposal).",
    )
    parser.add_argument(
        "--proposal_idx_switch",
        type=int,
        default=-1,
        help="When to switch from ablated model to target model as proposal.",
    )
    parser.add_argument(
        "--reward_batch_size",
        type=int,
        default=32,
        help="Batch size for reward function.",
    )
    parser.add_argument(
        "--n_judge_jobs",
        type=int,
        default=1000,
        help="Number of parallel processes for judge.",
    )
    parser.add_argument(
        "--use_smc",
        action="store_true",
        help="Whether to use Sequential Monte Carlo (SMC) for generation.",
    )
    parser.add_argument(
        "--low_vram_cache",
        action="store_true",
        help="Whether to use Sequential Monte Carlo (SMC) for generation.",
    )

    parser.add_argument(
        "--use_importance_weights_in_resampling",
        action="store_true",
        help="Whether to use importance weights in resampling for SMC.",
    )
    parser.add_argument(
        "--smc_verbose", action="store_true", help="Whether to print SMC logs"
    )


if __name__ == "__main__":
    args = get_args()
    main(args)

import argparse
import gc
from jaxtyping import Float
import json
import os
import random
import math
from joblib import Parallel, delayed
from tqdm import trange

from datasets import load_dataset
import numpy as np
import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from refusal_direction.pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)
from strong_reject.evaluate import strongreject_rubric


def load_model_and_tokenizer(model_name_or_path):
    # NOTE: returns model in `eval` mode
    config = AutoConfig.from_pretrained(
        model_name_or_path, output_hidden_states=True, return_dict_in_generate=True
    )

    if isinstance(config, Qwen2Config):
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
        config=config,
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        token=os.getenv("HF_TOKEN"),
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_refusal_direction(refusal_direction_path):
    direction_path = os.path.join(refusal_direction_path, "direction.pt")
    metadata_path = os.path.join(refusal_direction_path, "direction_metadata.json")
    direction = torch.load(direction_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata["direction"] = direction
    return metadata


def get_all_direction_ablation_hooks(
    model, direction: Float[Tensor, "d_model"], ablation_intensity: float = 1.0
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


def score_responses(
    tokenizer, forbidden_prompt: str, input_ids: torch.Tensor, n_jobs: int = 50
):
    scoring_fn = lambda r: strongreject_rubric(forbidden_prompt, r)["score"]
    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in input_ids]
    judge_scores = Parallel(n_jobs=n_jobs)(delayed(scoring_fn)(r) for r in responses)
    judge_scores = torch.tensor(judge_scores, device=input_ids.device)
    return judge_scores


def model_forward_wrapper(
    model, input_ids, attention_mask, past_key_values, batch_size, **kwargs
):
    """
    Batches forward pass when using a large number of particles.
    Assumes we are using a DynamicCache for generation.
    """
    num_batches = math.ceil(input_ids.shape[0] / batch_size)
    chunked_input_ids = torch.chunk(input_ids, chunks=num_batches, dim=0)
    chunked_attention_mask = torch.chunk(attention_mask, chunks=num_batches, dim=0)

    # Chunk the cache
    chunk_indices = torch.arange(input_ids.shape[0]).chunk(num_batches)
    if len(past_key_values) == 0:
        chunked_past_key_values = [DynamicCache() for _ in range(num_batches)]
    else:
        chunked_past_key_values = []
        for batch_indices in chunk_indices:
            cache = DynamicCache()
            for layer_idx in range(model.config.num_hidden_layers):
                batch_layer_key_cache = past_key_values.key_cache[layer_idx][
                    batch_indices
                ]
                batch_layer_value_cache = past_key_values.value_cache[layer_idx][
                    batch_indices
                ]
                cache.update(batch_layer_key_cache, batch_layer_value_cache, layer_idx)

                if layer_idx % 4 == 0:
                    torch.cuda.empty_cache()

            chunked_past_key_values.append(cache)

    del past_key_values

    # Batch the forward pass
    batched_outputs = []
    for batch_input_ids, batch_attention_mask, batch_past_key_values in zip(
        chunked_input_ids, chunked_attention_mask, chunked_past_key_values
    ):
        batched_outputs.append(
            model.forward(
                batch_input_ids,
                batch_attention_mask,
                past_key_values=batch_past_key_values,
                **kwargs,
            )
        )

    # Reassemble the cache
    past_key_values = DynamicCache()
    for layer_idx in range(model.config.num_hidden_layers):
        layer_key_cache = torch.cat(
            [cache.key_cache[layer_idx] for cache in chunked_past_key_values], dim=0
        )
        layer_value_cache = torch.cat(
            [cache.value_cache[layer_idx] for cache in chunked_past_key_values], dim=0
        )
        past_key_values.update(layer_key_cache, layer_value_cache, layer_idx)

        if layer_idx % 4 == 0:
            torch.cuda.empty_cache()

    for cache in chunked_past_key_values:
        del cache

    logits = torch.cat([out.logits for out in batched_outputs], dim=0)
    return logits, past_key_values


def generate(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    forbidden_prompt: str = "",
    decoding: str = "sample",  # Options: 'greedy', 'sample', 'beam_search', 'top_k', 'top_p'
    fwd_pre_hooks: list = [],
    fwd_hooks=[],
    fwd_batch_size: int = 128,
    max_new_tokens: int = 10,
    proposal_bias: float = 0.5,
    proposal_idx_switch: int = 10,
):
    """Implements simple greedy decoding."""

    assert forbidden_prompt != ""

    base_past_key_values = DynamicCache()
    proposal_past_key_values = DynamicCache()

    num_particles = input_ids.shape[0]

    _input_ids = input_ids.detach().clone()
    _attention_mask = attention_mask.detach().clone()
    _completed_generation = torch.zeros((num_particles, 1), dtype=torch.bool).to(
        _input_ids.device
    )

    _inputs = {"input_ids": _input_ids, "attention_mask": _attention_mask}

    cache_position = torch.arange(
        _inputs["input_ids"].shape[1], dtype=torch.int64, device=model.device
    )

    # Main generation loop
    log_importance_weights = torch.zeros(num_particles, 1, device=_input_ids.device)
    log_importance_weight_arr = []
    for generation_idx in trange(max_new_tokens):
        # Compute the base distribution
        with torch.no_grad():
            base_logits, base_past_key_values = model_forward_wrapper(
                model,
                **_inputs,
                batch_size=fwd_batch_size,
                past_key_values=base_past_key_values,
                cache_position=cache_position,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )
        base_logprobs = torch.log_softmax(base_logits[:, -1, :], dim=-1)

        if generation_idx < proposal_idx_switch:
            # Compute the proposal distribution
            with add_hooks(
                module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
            ):
                with torch.no_grad():
                    refusal_ablated_logits, proposal_past_key_values = (
                        model_forward_wrapper(
                            model,
                            **_inputs,
                            batch_size=fwd_batch_size,
                            past_key_values=proposal_past_key_values,
                            cache_position=cache_position,
                            use_cache=True,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            output_scores=True,
                            return_dict_in_generate=True,
                            output_hidden_states=False,
                        )
                    )

            proposal_logprobs = torch.log_softmax(
                refusal_ablated_logits[:, -1, :], dim=-1
            )

            # Linearly interpolate between distributions
            proposal_logprobs = torch.log(
                proposal_bias * torch.exp(proposal_logprobs)
                + (1 - proposal_bias) * torch.exp(base_logprobs)
            )
        else:
            proposal_logprobs = base_logprobs

        next_tokens = torch.multinomial(
            proposal_logprobs.exp(),
            num_samples=1,
        )
        # Check if sequence is completed
        _completed_generation |= next_tokens == tokenizer.eos_token_id

        # Pad completed sequences
        next_tokens[_completed_generation] = tokenizer.pad_token_id

        proposal_next_token_logprobs = torch.gather(proposal_logprobs, -1, next_tokens)

        base_next_token_logprobs = torch.gather(base_logprobs, -1, next_tokens)
        proposal_next_token_logprobs[_completed_generation] = base_next_token_logprobs[
            _completed_generation
        ]

        assert base_next_token_logprobs.shape == (num_particles, 1), (
            base_next_token_logprobs.shape,
            num_particles,
        )
        assert proposal_next_token_logprobs.shape == (num_particles, 1), (
            proposal_next_token_logprobs.shape,
            num_particles,
        )

        log_importance_weights += (
            base_next_token_logprobs - proposal_next_token_logprobs
        )

        log_importance_weight_arr.append(
            base_next_token_logprobs - proposal_next_token_logprobs
        )
        # Update input arguments

        _input_ids = torch.cat((_input_ids, next_tokens), dim=1)
        _attention_mask = torch.cat(
            (_attention_mask, torch.ones_like(next_tokens)), dim=1
        )
        _inputs = {"input_ids": next_tokens, "attention_mask": _attention_mask}
        cache_position = (
            cache_position[-1:] + 1
        )  # add one more position for the next token

    importance_weight_arr = torch.exp(torch.cat(log_importance_weight_arr, dim=1))
    assert importance_weight_arr.shape == (num_particles, max_new_tokens), (
        importance_weight_arr.shape,
        num_particles,
        max_new_tokens,
    )

    importance_weights = log_importance_weights.exp()

    # Judge the responses
    judge_scores = score_responses(
        tokenizer, forbidden_prompt, input_ids=_input_ids[:, input_ids.shape[1] :]
    )

    reweighted_scores = judge_scores * importance_weights.squeeze(1)

    print(f"SIS harm score estimate: {reweighted_scores.mean().item()}")
    print(f"SIS harm score variance: {reweighted_scores.var().item()}")
    print(
        f"final importance weights: {importance_weights.mean(), importance_weights.std()}"
    )
    print(f"Judge scores: {judge_scores}")

    print(f"Importance weight array: {importance_weight_arr[0]}")
    print(
        "Sequence generated:",
        tokenizer.decode(_input_ids[0], skip_special_tokens=False),
    )
    print(f"Judge score: {judge_scores[0]}")
    return _input_ids


def estimate_harm(
    args, model, tokenizer, forbidden_prompt, fwd_pre_hooks=[], fwd_hooks=[]
):
    pass


def main(args):
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

    # Load judged monte carlo samples
    mc_dataset = load_dataset(
        "json",
        data_files=args.mc_est_dataset,
    )["train"]

    for example in mc_dataset:
        # Declare the conversation
        forbidden_prompt = example["forbidden_prompt"]

        messages = [
            {
                "role": "user",
                "content": forbidden_prompt,
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

        print(f"Forbidden prompt: {forbidden_prompt}")
        print(f"Monte Carlo harm estimate: {float(np.mean(example['score']))}")
        # Generate
        outputs = generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            forbidden_prompt=forbidden_prompt,
            decoding="sample",
            fwd_pre_hooks=ablation_fwd_pre_hooks,
            fwd_hooks=ablation_fwd_hooks,
            max_new_tokens=args.max_new_tokens,
            fwd_batch_size=args.fwd_batch_size,
            proposal_bias=args.proposal_bias,
            proposal_idx_switch=args.proposal_idx_switch,
        )
        print("\n-----------------------------------------------\n")


def get_args():
    # NOTE: Currently all unused arguments are commented out
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
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
        default=100,
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

    # parser.add_argument(
    #    "--reward_batch_size",
    #    type=int,
    #    default=32,
    #    help="Batch size for reward function.",
    # )
    # parser.add_argument(
    #    "--use_smc",
    #    action="store_true",
    #    help="Whether to use Sequential Monte Carlo (SMC) for generation.",
    # parser.add_argument(
    # )
    #    "--smc_verbose", action="store_true", help="Whether to print SMC logs"
    # )
    # parser.add_argument(
    #    "--save_output",
    #    action="store_true",
    #    help="Whether to store the outputs in a log file",
    # )
    # parser.add_argument(
    #    "--lora_percent",
    #    type=float,
    #    default=1.0,
    #    help="Ratio of the lora weights to be modified.",
    # )
    # parser.add_argument(
    #    "--output_dir",
    #    type=str,
    #    default="./model_outputs",
    #    help="Store args, output probabilities",
    # )

    args = parser.parse_args()

    args.model_shortname = args.model_name.split("/")[1]
    args.refusal_direction_path = (
        f"refusal_direction/pipeline/runs/{args.model_shortname}/"
    )

    if args.mc_est_dataset == "":
        args.mc_est_dataset = (
            f"big_vanilla_harmful/{args.model_shortname}-responses-test.json"
        )

    if args.proposal_idx_switch == -1:
        args.proposal_idx_switch = args.max_new_tokens + 1

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)

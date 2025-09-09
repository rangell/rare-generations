import argparse
import gc
from jaxtyping import Float
from copy import deepcopy
import json
from pathlib import Path
import datetime
import os
import random
import math
from joblib import Parallel, delayed
from tqdm import trange
from tqdm.auto import tqdm
import pickle

from datasets import load_dataset
import numpy as np
import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from fk_steering import FKSteering, update_cache_after_resampling
from refusal_direction.pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)
from strong_reject.evaluate import strongreject_rubric

from rewards import create_reward_fn


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

    # Weird quirk with GraySwanAI model
    if model_name_or_path == "GraySwanAI/Llama-3-8B-Instruct-RR":
        tokenizer.eos_token = "<|eot_id|>"

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
                # print(past_key_values.key_cache[layer_idx].shape, batch_layer_key_cache.shape)
                # print(past_key_values.value_cache[layer_idx].shape, batch_layer_value_cache.shape)

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
    smc_args=None,
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

    if smc_args is not None:
        # smc_args["r_fn"] = lambda x: torch.ones(x.shape[0], device=_input_ids.device)
        fk_class = FKSteering(
            device=_input_ids.device,
            r_fn=smc_args["r_fn"],
            potential_type=smc_args["potential_type"],
            max_seq_len=smc_args["max_seq_len"],
            num_particles=smc_args["num_particles"],
            resample_start=smc_args["resample_start"],
            resample_end=smc_args["resample_end"],
            resample_interval=smc_args["resample_interval"],
            lmbda=smc_args["lmbda"],
            use_smc=smc_args["use_smc"],
            adaptive_resampling=smc_args["adaptive_resampling"],
            adaptive_resampling_threshold=smc_args["adaptive_resampling_threshold"],
            smc_verbose=smc_args["smc_verbose"],
            importance_resampling_at_last_step=smc_args[
                "importance_resampling_at_last_step"
            ],
            use_importance_weights_in_resampling=smc_args[
                "use_importance_weights_in_resampling"
            ],
        )

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
        assert len(log_importance_weight_arr) == generation_idx + 1, (
            len(log_importance_weight_arr),
            generation_idx + 1,
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

        if smc_args is not None:
            p_q_t = torch.exp(
                base_next_token_logprobs - proposal_next_token_logprobs
            ).view(-1)

            if generation_idx in fk_class.resampling_arr and smc_args["use_smc"]:
                rs_candidates = smc_args["r_fn"](_input_ids)
            else:
                rs_candidates = None

            resample_indices = fk_class(
                step_idx=generation_idx,
                importance_weights=deepcopy(p_q_t),
                sequences=deepcopy(_input_ids),
                rs_candidates=deepcopy(rs_candidates),
            )

            if (
                generation_idx in fk_class.resampling_arr
                and not torch.all(
                    resample_indices
                    == torch.arange(num_particles, device=resample_indices.device)
                ).item()
            ):
                print(f"Resampling at step {generation_idx}")
                # import pdb; pdb.set_trace()
                assert resample_indices.shape == (num_particles,), (
                    resample_indices.shape,
                    (num_particles,),
                )

                _input_ids = deepcopy(_input_ids[resample_indices])
                _attention_mask = deepcopy(_attention_mask[resample_indices])
                _inputs = {
                    "input_ids": next_tokens[resample_indices],
                    "attention_mask": _attention_mask,
                }

                _completed_generation = deepcopy(
                    _completed_generation[resample_indices]
                )
                log_importance_weights = deepcopy(
                    log_importance_weights[resample_indices]
                )

                log_importance_weight_arr = [
                    log_arr[resample_indices, :]
                    for log_arr in log_importance_weight_arr
                ]

                base_past_key_values = update_cache_after_resampling(
                    past_key_values=base_past_key_values,
                    indices=resample_indices,
                    model_config=model.config,
                )
                proposal_past_key_values = update_cache_after_resampling(
                    past_key_values=proposal_past_key_values,
                    indices=resample_indices,
                    model_config=model.config,
                )

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

    if smc_args is None or smc_args["use_smc"] is False:
        reweighted_scores = judge_scores * importance_weights.squeeze(1)
        print(f"SIS harm score estimate: {reweighted_scores.mean().item()}")
        print(f"SIS harm score variance: {reweighted_scores.var().item()}")
        print(
            f"final importance weights: {importance_weights.mean(), importance_weights.std()}"
        )
        print(f"Judge scores: {judge_scores}")

        prompt_kl = -torch.mean(torch.log(importance_weights)).item()
        print(f"KL divergence: {prompt_kl}")
    elif smc_args["use_smc"]:
        reweighted_scores = fk_class.compute_fk_estimate(
            test_function_values=judge_scores, importance_weights=importance_weights
        )
        smc_quantities = fk_class.get_fk_quantities()

        sis = importance_weights.view(num_particles) * judge_scores
        print(f"SIS harm score estimate: {sis.mean().item()}")
        # print(f"SIS harm score variance: {reweighted_scores.var().item()}")
        # print(
        #     f"final importance weights: {importance_weights.mean(), importance_weights.std()}"
        # )
        # print(f"Judge scores: {judge_scores}")

        prompt_kl = -torch.mean(torch.log(importance_weights)).item()
        print(f"KL divergence: {prompt_kl}")
    else:
        raise NotImplementedError

    # print(f"Importance weight array: {importance_weight_arr[0]}")
    print(
        "Sequence generated:",
        tokenizer.decode(_input_ids[0], skip_special_tokens=False),
    )
    print(f"Judge score: {judge_scores[0]}")

    ret = dict(
        responses=[tokenizer.decode(r, skip_special_tokens=True) for r in input_ids],
        judge_scores=judge_scores.cpu().numpy(),
        prompt_kl=prompt_kl,
        importance_weights=importance_weights.cpu().numpy(),
        reweighted_scores=reweighted_scores.mean().item(),
    )

    if smc_args is not None and smc_args["use_smc"]:
        for key in smc_quantities:
            ret[key] = smc_quantities[key]

    return ret


def estimate_harm(
    args,
    model,
    tokenizer,
    forbidden_prompt,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    smc_args=None,
):
    # Initialize the conversation
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

    # reward function for SMC
    # Just past to SMC as r_fn=reward_fn
    if smc_args is not None and smc_args["use_smc"]:
        smc_args["r_fn"] = create_reward_fn(
            tokenizer=tokenizer,
            forbidden_prompt=forbidden_prompt,
            batch_size=args.reward_batch_size,
        )

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
        smc_args=smc_args,
    )
    print("\n-----------------------------------------------\n")

    harm_est = outputs["reweighted_scores"]

    return harm_est, outputs


def main(args):
    output_dir = args.output_dir
    output_dir = args.output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if hasattr(args, "experiment_identifier"):
        output_dir = (
            Path(output_dir) / args.dataset
            + "_"
            + args.experiment_identifier / timestamp
        )
    else:
        output_dir = Path(output_dir) / args.model_name / timestamp

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
        output_dir = Path(output_dir) / args.model_name / timestamp
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
        smc_verbose=True,
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

        _, outputs = estimate_harm(
            args=args,
            model=model,
            tokenizer=tokenizer,
            forbidden_prompt=example["forbidden_prompt"],
            fwd_pre_hooks=ablation_fwd_pre_hooks,
            fwd_hooks=ablation_fwd_hooks,
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
    parser.add_argument(
        "--reward_batch_size",
        type=int,
        default=32,
        help="Batch size for reward function.",
    )
    parser.add_argument(
        "--use_smc",
        action="store_true",
        help="Whether to use Sequential Monte Carlo (SMC) for generation.",
    )

    parser.add_argument(
        "--use_importance_weights_in_resampling",
        action="store_true",
        help="Whether to use importance weights in resampling for SMC.",
    )
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


if __name__ == "__main__":
    args = get_args()
    main(args)

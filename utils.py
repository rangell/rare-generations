import gc
import numpy as np
from jaxtyping import Float
import json
import argparse
import os
from joblib import Parallel, delayed

from tqdm.auto import tqdm
import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from refusal_direction.pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)
from fk_steering import FKSteering
from rewards import calculate_harmful_reward, sr_harmful_reward_fn

from strong_reject.evaluate import strongreject_rubric
from datasets import load_dataset


def load_model_and_tokenizer(model_name_or_path):
    # NOTE: returns model in `eval` mode
    config = AutoConfig.from_pretrained(
        model_name_or_path, output_hidden_states=True, return_dict_in_generate=True, local_files_only=True
    )

    if isinstance(config, Qwen2Config):
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        token=os.getenv("HF_TOKEN"),
        device_map="auto",
        low_cpu_mem_usage=True,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        token=os.getenv("HF_TOKEN"),
        padding_side="left",
        local_files_only=True,
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


def get_all_direction_ablation_hooks(model, direction: Float[Tensor, "d_model"], ablation_intensity: float):
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
            get_direction_ablation_input_pre_hook(direction=direction, ablation_intensity=ablation_intensity),
        )
        for layer in range(model.config.num_hidden_layers)
    ]
    fwd_hooks = [
        (
            model_attn_modules[layer],
            get_direction_ablation_output_hook(direction=direction, ablation_intensity=ablation_intensity),
        )
        for layer in range(model.config.num_hidden_layers)
    ]
    fwd_hooks += [
        (
            model_mlp_modules[layer],
            get_direction_ablation_output_hook(direction=direction, ablation_intensity=ablation_intensity),
        )
        for layer in range(model.config.num_hidden_layers)
    ]

    return fwd_pre_hooks, fwd_hooks


def token_sampler(decoding, output, inv_temperature):
    if decoding == "greedy":
        next_tokens = torch.argmax(
            inv_temperature * output.logits[:, -1, :], dim=-1
        )
    elif decoding == "sample":
        next_tokens = torch.multinomial(
            torch.softmax(
                inv_temperature * output.logits[:, -1, :], dim=-1
            ),
            num_samples=1,
        ).squeeze(-1)
    else:
        raise ValueError(
            f"Decoding method '{decoding}' is not supported. Choose from 'greedy' or 'sample'."
        )

    return next_tokens

def create_model_wrapper(model, tokenizer, fwd_pre_hooks=[], fwd_hooks=[]):

    def model_forward(
        _input_ids,
        _attention_mask,
        _completed_generation,
        decoding,
        inv_temperature,
    ):
        model_inputs = dict(
            input_ids=_input_ids,
            attention_mask=_attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,        
        )

        if len(fwd_pre_hooks) == 0 and len(fwd_hooks) == 0:
            # Use the base model to generate the proposal distribution
            # NOTE: This is the default behavior, so we can skip adding hooks
            with torch.no_grad():
                output = model.forward(**model_inputs)
        elif len(fwd_pre_hooks) > 0 and len(fwd_hooks) > 0:
            with add_hooks(
                module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
            ):
                with torch.no_grad():
                    output = model.forward(**model_inputs)
        else:
            raise ValueError("fwd_pre_hooks and fwd_hooks are not aligned")

        logprobs_distribution = torch.log_softmax(
            inv_temperature * output.logits[:, -1, :], dim=-1
        )
        next_tokens = token_sampler(
            decoding=decoding,
            output=output,
            inv_temperature=inv_temperature,
        )
        next_token_logprobs = torch.gather(
            logprobs_distribution, -1, next_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        assert next_token_logprobs.shape == (logprobs_distribution.shape[0],)
        
        # Ensure next_tokens is of shape (num_particles, 1)
        next_tokens = next_tokens.unsqueeze(-1)

        # Check if sequence is completed
        _completed_generation |= next_tokens == tokenizer.eos_token_id

        # Pad completed sequences
        next_tokens[_completed_generation] = tokenizer.pad_token_id

        # Ensure proposal_logprobs is of shape (num_particles, 1)
        next_token_logprobs = next_token_logprobs.unsqueeze(-1)

        # Clear cache to avoid OOM errors
        gc.collect()
        torch.cuda.empty_cache()

        return logprobs_distribution, next_tokens, next_token_logprobs, _completed_generation

    return model_forward

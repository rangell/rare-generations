import gc
import json
import os

import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from refusal_direction.pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)


def load_model_and_tokenizer(model_name_or_path):
    # NOTE: returns model in `eval` mode
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        output_hidden_states=True,
        return_dict_in_generate=True,
        local_files_only=True,
    )

    if isinstance(config, Qwen2Config):
        config = None
    elif model_name_or_path.startswith("GraySwanAI/Llama-3-8B-Instruct-RR"):
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
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


def get_all_direction_ablation_hooks(
    model,
    direction: Tensor,
    ablation_intensity: float,
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


def token_sampler(decoding, proposal_output, proposal_inv_temperature):
    if decoding == "greedy":
        # Select the next tokens based on the proposal distribution in a greedy manner
        next_tokens = torch.argmax(
            proposal_inv_temperature * proposal_output.logits[:, -1, :], dim=-1
        )
    elif decoding == "sample":
        # Sample the next tokens from the proposal distribution
        next_tokens = torch.multinomial(
            torch.softmax(
                proposal_inv_temperature * proposal_output.logits[:, -1, :], dim=-1
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
        cache_position=None,
        past_key_values=None,
        use_cache=False,
        return_logprob_only=False,
    ):
        if cache_position is not None and len(cache_position) == 1:
            cache_input_ids = _input_ids[:, -1].unsqueeze(
                1
            )  # Get the last token for generation
        else:
            cache_input_ids = _input_ids

        _completed_generation = _completed_generation.to(model.device)
        model_inputs = dict(
            input_ids=cache_input_ids.to(
                model.device
            ),  # Get the last token for generation
            attention_mask=_attention_mask.to(model.device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )
        if use_cache:
            model_inputs["past_key_values"] = past_key_values
            model_inputs["cache_position"] = cache_position.to(model.device)
            model_inputs["use_cache"] = use_cache

        if return_logprob_only:
            # If we only want the log probabilities, we can skip the generation step
            with torch.no_grad():
                output = model(_input_ids, _attention_mask, use_cache=use_cache)

            shift_logits = output.logits[:, :-1, :]
            shift_labels = _input_ids[:, 1:]

            logprobs = torch.log_softmax(inv_temperature * shift_logits, dim=-1)
            token_logprobs = logprobs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            assert token_logprobs.shape == (
                _input_ids.shape[0],
                _input_ids.shape[1] - 1,
            ), token_logprobs.shape

            return token_logprobs

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
            proposal_output=output,
            proposal_inv_temperature=inv_temperature,
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
        next_token_logprobs[_completed_generation] = (
            0.0  # Set logprobs to 0 for completed sequences
        )

        # Update cache position
        if use_cache:
            cache_position = cache_position[-1:] + 1
        # Clear cache to avoid OOM errors
        gc.collect()
        torch.cuda.empty_cache()

        return (
            logprobs_distribution,
            next_tokens,
            next_token_logprobs,
            _completed_generation,
            cache_position,
            past_key_values,
        )

    return model_forward


def instruct_to_model_base(model_name):
    if model_name == "meta-llama/Llama-3.2-1B-Instruct":
        return "meta-llama/Llama-3.2-1B"
    else:
        raise ValueError(f"Base checkpoint not specified for model name {model_name}")

from typing import Callable
from copy import deepcopy
import math
from tqdm import trange

import torch
from transformers import DynamicCache, OffloadedCache

from smc.fk_steering import FKSteering, update_cache_after_resampling

from smc.estimator import model_forward_wrapper


def estimate_cross_entropy_loss(
    model,
    tokenizer,
    input_ids,
    completion_ids,
    judge_scores,
    attention_mask,
    proposal_context_manager: Callable,
    judge_responses: Callable,
    prompt: str = "",
    decoding: str = "sample",  # Options: 'greedy', 'sample', 'beam_search', 'top_k', 'top_p'
    fwd_batch_size: int = 128,
    max_new_tokens: int = 10,
    proposal_bias: float = 0.5,
    proposal_idx_switch: int = 10,
    smc_args=None,
    low_vram_cache: bool = False,
):
    """Implements rare event estimation using sequential Monte Carlo."""

    assert prompt != ""
    # assert decoding == "sample", "Only 'sample' decoding is supported in this function."

    assert smc_args is None, "SMC not yet supported in this function."

    if low_vram_cache:
        base_past_key_values = OffloadedCache()
        proposal_past_key_values = OffloadedCache()
    else:
        base_past_key_values = DynamicCache()
        proposal_past_key_values = DynamicCache()

    num_particles = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    assert (
        input_ids[0] == completion_ids[0][:prompt_len].all()
    ), "Input IDs and completion IDs do not match in the prompt region."

    _input_ids = input_ids.detach().clone()
    _attention_mask = attention_mask.detach().clone()
    _completed_generation = torch.zeros((num_particles, 1), dtype=torch.bool).to(
        _input_ids.device
    )

    _inputs = {"input_ids": _input_ids, "attention_mask": _attention_mask}

    cache_position = torch.arange(
        _inputs["input_ids"].shape[1], dtype=torch.int64, device=model.device
    )

    log_importance_weights = torch.zeros(num_particles, 1, device=_input_ids.device)
    log_importance_weight_arr = []

    # Main generation loop
    for generation_idx in trange(max_new_tokens):
        # Compute the base distribution
        with torch.no_grad():
            base_logits, base_past_key_values = model_forward_wrapper(
                model,
                **_inputs,
                batch_size=fwd_batch_size,
                vocab_size=tokenizer.vocab_size,
                past_key_values=base_past_key_values,
                cache_position=cache_position,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )
        base_logprobs = torch.log_softmax(base_logits, dim=-1)

        if generation_idx < proposal_idx_switch:
            # Compute the proposal distribution
            with torch.no_grad(), proposal_context_manager(generation_idx):
                refusal_ablated_logits, proposal_past_key_values = (
                    model_forward_wrapper(
                        model,
                        **_inputs,
                        batch_size=fwd_batch_size,
                        vocab_size=tokenizer.vocab_size,
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
            proposal_logprobs = torch.log_softmax(refusal_ablated_logits, dim=-1)

            # Linearly interpolate between distributions
            proposal_logprobs = torch.log(
                proposal_bias * torch.exp(proposal_logprobs)
                + (1 - proposal_bias) * torch.exp(base_logprobs)
            )
        else:
            proposal_logprobs = base_logprobs

        # teacher forcing: get the logprobs of the next token in completion_ids
        next_tokens = completed_generations[:, prompt_len + generation_idx]

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

    importance_weight_arr = torch.exp(torch.cat(log_importance_weight_arr, dim=1))
    assert importance_weight_arr.shape == (num_particles, max_new_tokens), (
        importance_weight_arr.shape,
        num_particles,
        max_new_tokens,
    )

    importance_weights = log_importance_weights.exp()

    reweighted_scores = judge_scores * importance_weights.squeeze(1)
    cross_entropy_objective = reweighted_scores * proposal_next_token_logprobs

    ret = dict(
        importance_weights=importance_weights.cpu().numpy(),
        cross_entropy_objective=cross_entropy_objective.cpu().numpy(),
        log_importance_weights=log_importance_weights.cpu().numpy(),    
    )

    return ret


if __name__ == "__main__":
    pass
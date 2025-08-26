from tqdm.auto import tqdm

import torch
from transformers import DynamicCache

from fk_steering import FKSteering


def update_cache_after_resampling(
    past_key_values: DynamicCache, indices: torch.Tensor, model_config
):
    new_past_key_values = DynamicCache()
    indices = indices.cpu()
    for layer_idx in range(model_config.num_hidden_layers):
        keys, values = past_key_values[layer_idx]
        resampled_keys = keys[indices].clone()
        resampled_values = values[indices].clone()
        # past_key_values[layer_idx] = (keys, values)
        new_past_key_values.update(resampled_keys, resampled_values, layer_idx)

        del keys, values, resampled_keys, resampled_values

        if layer_idx % 4 == 0:
            torch.cuda.empty_cache()
    del past_key_values
    past_key_values = new_past_key_values
    torch.cuda.empty_cache()

    return past_key_values

def generate(
    model_config,
    base_model_config,
    tokenizer,
    forbidden_prompt,
    base_forward,
    proposal_forward,
    reward_fn,
    judge_fn,
    input_ids,  # dict of input_ids for proposal and base !!!
    attention_mask,  # dict of attention_mask for proposal and base !!!
    num_particles,
    use_smc: bool,
    proposal_inv_temperature=3.5,
    max_new_tokens: int = 10,
    decoding: str = "sample",  # Options: 'greedy', 'sample', 'beam_search', 'top_k', 'top_p'
    proposal_model_switch_idx=None,  # If None, use the proposal model for all steps
    resample_start=10,
    resample_end=20,
    resample_interval=10,
    lmbda=10.0,  # Adjust this value based on your needs
    adaptive_resampling=True,
    adaptive_resampling_threshold=0.5,
    base_inv_temperature=1.0,
    smc_verbose=False,
    importance_resampling_at_last_step=False,
    use_cache=False,
    cheap_judge=True,
    reward_batch_size=32,
    base_model_interpolation=0.0,
):
    """Implements simple greedy decoding."""

    assert isinstance(input_ids, dict), "input_ids must be a dictionary"
    assert set(input_ids.keys()) == set(
        ["base", "proposal"]
    ), "input_ids must be a dictionary with keys 'base' and 'proposal'"
    assert set(attention_mask.keys()) == set(
        ["base", "proposal"]
    ), "input_ids must be a dictionary with keys 'base' and 'proposal'"

    _input_ids = {k: v.detach().clone() for k, v in input_ids.items()}
    _attention_mask = {k: v.detach().clone() for k, v in attention_mask.items()}
    _completed_generation = {
        k: torch.zeros((num_particles, 1), dtype=torch.bool).to("cuda")
        for k in input_ids.keys()
    }

    if use_cache:
        cache_position = {
            k: torch.arange(v.shape[1], dtype=torch.int64, device=v.device)
            for k, v in _input_ids.items()
        }
        past_key_values = {k: DynamicCache() for k in _input_ids.keys()}

    else:
        cache_position = {k: None for k in _input_ids.keys()}
        past_key_values = {k: None for k in _input_ids.keys()}

    assert all(
        _input_ids[k].shape[0] == num_particles for k in _input_ids.keys()
    ), "All input_ids must have the same number of particles"
    prompt_len = {k: len(v[0]) for k, v in _input_ids.items()}

    # fk_class = FKSteering(
    #     device=input_ids["base"].device,
    #     r_fn=reward_fn,
    #     potential_type="diff",
    #     max_seq_len=max_new_tokens,
    #     num_particles=num_particles,
    #     resample_start=resample_start,
    #     resample_end=resample_end,
    #     resample_interval=resample_interval,
    #     lmbda=lmbda,
    #     use_smc=use_smc,
    #     adaptive_resampling=adaptive_resampling,
    #     adaptive_resampling_threshold=adaptive_resampling_threshold,
    #     smc_verbose=smc_verbose,
    #     importance_resampling_at_last_step=importance_resampling_at_last_step,
    # )

    # Main generation loop
    importance_weights = torch.ones(num_particles, device=_input_ids["proposal"].device)
    # sequence_proposal_logprob = torch.zeros(
    #     num_particles, device=_input_ids["proposal"].device
    # )
    # vocab_size = model_config.vocab_size
    # full_proposal_logprob = torch.zeros(
    #     num_particles, max_new_tokens, vocab_size, device=_input_ids["proposal"].device
    # )
    sequence_proposal_logprob = torch.zeros(
        num_particles, max_new_tokens, device=_input_ids["proposal"].device
    )


    for step_idx in tqdm(range(max_new_tokens)):
        if (
            step_idx == proposal_model_switch_idx
            and proposal_model_switch_idx is not None
        ):
            raise NotImplementedError
            # print(
            #     f"Switching proposal model to 'base' at step {sample_idx} at temperature {base_inv_temperature}."
            # )
            # # Switch to the base model for proposal generation
            # # NOTE: This is useful for the first few steps where the proposal model might not be
            #
            # proposal_forward = base_forward
            # proposal_inv_temperature = base_inv_temperature

        
        base_logprobs_distribution_for_mix = None
        if base_model_interpolation != 0.0:
            ret_2 = base_forward(
                _input_ids=_input_ids["base"],
                _attention_mask=_attention_mask["base"],
                _completed_generation=_completed_generation["proposal"], # use proposal completion generation here, but not really important
                decoding=decoding,
                inv_temperature=proposal_inv_temperature,
                cache_position=cache_position["base"],
                past_key_values=past_key_values["base"],
                use_cache=use_cache,
            )

            (
                base_logprobs_distribution_for_mix,
                _,
                _,
                _,
                base_cache_position_for_mix,
                base_past_key_values_for_mix,
            ) = ret_2

            cache_position["base"] = base_cache_position_for_mix
            past_key_values["base"] = base_past_key_values_for_mix

        ret = proposal_forward(
            _input_ids=_input_ids["proposal"],
            _attention_mask=_attention_mask["proposal"],
            _completed_generation=_completed_generation["proposal"],
            decoding=decoding,
            inv_temperature=proposal_inv_temperature,
            cache_position=cache_position["proposal"],
            past_key_values=past_key_values["proposal"],
            use_cache=use_cache,
            other_model_weight=base_model_interpolation,
            other_model_logprobs=base_logprobs_distribution_for_mix,
        )

        (
            proposal_logprobs_distribution,
            next_tokens,
            proposal_logprobs,
            _completed_generation_proposal,
            cache_position_proposal,
            past_key_values_proposal,
        ) = ret

        _completed_generation["proposal"] = _completed_generation_proposal
        cache_position["proposal"] = cache_position_proposal
        past_key_values["proposal"] = past_key_values_proposal


        _input_ids = {
            k: torch.cat((v, next_tokens), dim=1) for k, v in _input_ids.items()
        }
        _attention_mask = {
            k: torch.cat((v, torch.ones_like(next_tokens)), dim=1)
            for k, v in _attention_mask.items()
        }

        # sequence_proposal_logprob += proposal_logprobs.squeeze(-1).to(
        #     sequence_proposal_logprob.device
        # )
        # assert sequence_proposal_logprob.shape == (num_particles,), (
        #     sequence_proposal_logprob.shape,
        # )
        sequence_proposal_logprob[:, step_idx] = proposal_logprobs.squeeze(-1)
        # full_proposal_logprob[:, step_idx, :] = proposal_logprobs_distribution


        # only use the generated portion of all of the sequeneces
        assert not use_smc, "SMC not supported unless this is uncommented"
        # _input_ids["proposal"][:, input_ids["proposal"].shape[1] :], indices = fk_class(
        #     step_idx=step_idx,
        #     sequences=_input_ids["proposal"][:, input_ids["proposal"].shape[1] :],
        #     importance_weights=torch.ones(
        #         num_particles, device=_input_ids["proposal"].device
        #     ),  # ignoring importance weights
        # )
        indices = torch.arange(num_particles, device=_input_ids["proposal"].device)

        if not torch.all(
            indices == torch.arange(num_particles, device=indices.device)
        ).item():
            print("\nResampled!!!")

        if not use_smc:
            # No resampling needed, just continue
            pass
        elif (
            use_smc
            and torch.all(
                indices == torch.arange(num_particles, device=indices.device)
            ).item()
        ):
            # No resampling needed, just continue
            pass
        elif use_smc:
            # raise NotImplementedError("change base as well")
            past_key_values["proposal"] = update_cache_after_resampling(
                past_key_values=past_key_values["proposal"],
                indices=indices,
                model_config=model_config,
            )

            sequence_proposal_logprob = sequence_proposal_logprob[indices]

            # update _completed_generation to only include the resampled sequences
            _completed_generation = {
                k: v[indices] for k, v in _completed_generation.items()
            }

            # change the input_ids and attention_mask to only include the resampled sequences
            _input_ids = {k: v[indices] for k, v in _input_ids.items()}
            _attention_mask = {k: v[indices] for k, v in _attention_mask.items()}
        else:
            raise ValueError("Unknown resampling strategy.")

    # if proposal_forward != base_forward:
    # sequence_base_logprob = base_forward(
    #     _input_ids=_input_ids["base"],
    #     _attention_mask=_attention_mask["base"],
    #     _completed_generation=_completed_generation["base"],
    #     decoding=decoding,
    #     inv_temperature=base_inv_temperature,
    #     return_logprob_only=True,
    #     batched_logprob_calc=True,
    # )

    # sequence_base_logprob, full_base_logprob = base_forward(
    sequence_base_logprob, _ = base_forward(
        _input_ids=_input_ids["base"],
        _attention_mask=_attention_mask["base"],
        _completed_generation=_completed_generation["base"],
        decoding=decoding,
        inv_temperature=base_inv_temperature,
        return_logprob_only=True,
        batched_logprob_calc=False,
    )


        #####
    sequence_base_logprob = sequence_base_logprob[
        :, prompt_len["base"] - 1 :
    ]  # .sum(dim=1)
    # full_base_logprob = full_base_logprob[:, prompt_len["base"] - 1 :]

    non_eos_mask = (
        _input_ids["proposal"][:, prompt_len["proposal"] :] != tokenizer.eos_token_id
    ).to(dtype=sequence_base_logprob.dtype)


    # diff = full_proposal_logprob - full_base_logprob
    # del full_base_logprob
    # full_proposal_logprob_exp = torch.exp(full_proposal_logprob)
    # del full_proposal_logprob
    # prompt_kl = full_proposal_logprob_exp * diff
    # del full_proposal_logprob_exp
    # del diff

    # prompt_kl = (prompt_kl * non_eos_mask.unsqueeze(-1)).sum(dim=(1, 2)).mean()

    summed_sequence_proposal_logprob = (sequence_proposal_logprob * non_eos_mask).sum(
        dim=1
    )
    summed_sequence_base_logprob = (sequence_base_logprob * non_eos_mask).sum(dim=1)

    assert summed_sequence_base_logprob.shape == (num_particles,), (
        summed_sequence_base_logprob.shape,
        num_particles,
    )
    assert summed_sequence_proposal_logprob.shape == (num_particles,), (
        summed_sequence_proposal_logprob.shape,
        num_particles,
    )

    importance_weights = torch.exp(
        summed_sequence_base_logprob.to(summed_sequence_proposal_logprob.device)
        - summed_sequence_proposal_logprob
    )

    assert importance_weights.shape == (num_particles,), (
        importance_weights.shape,
        num_particles,
    )

    prompt_kl_v1 = -importance_weights.log().mean().item()
    # print("OLD KL: ", prompt_kl_v1, "NEW KL: ", prompt_kl)
    prompt_kl = prompt_kl_v1

        # assert full_base_logprob.shape == full_proposal_logprob.shape, (
        #     full_base_logprob.shape,
        #     full_proposal_logprob.shape,
        # )

        # # TODO
        # for batch_idx in range(sequence_base_logprob.shape[0]):
        #     sequence_base_logprob[
        #         batch_idx, _input_ids["base"][batch_idx, 1:] == tokenizer.eos_token_id
        #     ] = 0.0

        # sequence_base_logprob = sequence_base_logprob[:, prompt_len["base"] - 1 :].sum(
        #     dim=1
        # )
        # # set to sequence base logprob to 0 for eot-id

        # assert sequence_base_logprob.shape == (num_particles,), (
        #     sequence_base_logprob.shape,
        #     num_particles,
        # )
        # importance_weights = torch.exp(
        #     sequence_base_logprob.to(sequence_proposal_logprob.device)
        #     - sequence_proposal_logprob
        # )
        # assert importance_weights.shape == (num_particles,), (
        #     importance_weights.shape,
        #     num_particles,
        # )
    # else:
        # raise NotImplementedError
        # sequence_base_logprob = sequence_proposal_logprob
        # importance_weights = torch.ones(
        #     num_particles, device=sequence_proposal_logprob.device
        # )

    # prompt_kl = -importance_weights.log().mean().item()

    ## Judge the responses
    # judge_scores = score_responses(
    #    tokenizer=tokenizer,
    #    forbidden_prompt=forbidden_prompt,
    #    input_ids=_input_ids["proposal"][:, input_ids["proposal"].shape[1] :],
    #    cheap_judge=cheap_judge,
    #    reward_batch_size=reward_batch_size,
    # )

    judge_scores = judge_fn(_input_ids["proposal"][:, input_ids["proposal"].shape[1] :])

    responses = [
        tokenizer.decode(r, skip_special_tokens=True)
        for r in _input_ids["proposal"][:, input_ids["proposal"].shape[1] :]
    ]

    print("Sequence generated: ", responses[0], "with score: ", judge_scores[0].item())

    assert judge_scores.shape == (num_particles,), (
        judge_scores.shape,
        num_particles,
    )

    if not use_smc:
        # If not using SMC, just return the input_ids
        reweighted_scores = judge_scores.cpu() * importance_weights.cpu()
        sis_approx_harm_score = reweighted_scores.mean().item()
        sis_approx_harm_variance = reweighted_scores.var().item()
        print(
            f"SIS harm score estimate: {sis_approx_harm_score}, importance weights: {importance_weights.mean().item()}"
        )
        print(f"SIS harm score variance: {sis_approx_harm_variance}")
        print(f"Judge scores: {judge_scores.mean()} ± {judge_scores.std()}")
        print(f"KL divergence: {prompt_kl}")

        return (
            _input_ids,
            sis_approx_harm_score,
            prompt_kl,
            judge_scores,
            importance_weights,
            sis_approx_harm_variance,
        )
    else:
        smc_approx_harm_score = fk_class.compute_fk_estimate(
            test_function_values=judge_scores,
            importance_weights=importance_weights,
        )
        print(
            f"SMC harm score estimate: {smc_approx_harm_score}, importance weights: {importance_weights.mean().item()}"
        )
        print("SMC harm score variance: None")
        print(f"Judge scores: {judge_scores.mean()} ± {judge_scores.std()}")
        print(f"KL divergence: {prompt_kl}")

        return (
            _input_ids,
            smc_approx_harm_score.cpu().item(),
            prompt_kl,
            judge_scores,
            importance_weights,
            -1,
        )

from abc import ABC, abstractmethod

import copy
import gc
import math
from typing import List, Optional, Callable, ContextManager, Dict
from tqdm import tqdm, trange
import numpy as np

import torch
from transformers import DynamicCache
from smc.fk_steering import FKSteering, update_cache_after_resampling


class HarmfulTraitEstimator(ABC):
    def __init__(self, args, model, tokenizer, smc_args: Optional[Dict]):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.smc_args = smc_args

        def token_exists(id):
            try:
                self.tokenizer.decode(id)
                return 1
            except TypeError:
                return 0

        self.vocab_mask = torch.tensor(
            list(map(token_exists, range(self.model.vocab_size))),
            dtype=torch.bool,
            device="cuda",
        )

    @abstractmethod
    def proposal_context_manager(self, timestep: int) -> ContextManager:
        """We require that this function returns a context manager."""
        pass

    @abstractmethod
    def judge_responses(self, prompt: str, responses: List[str]) -> torch.Tensor:
        pass

    def create_reward_function(self, prompt: str) -> Callable:
        """
        This function is only required if we are using SMC.
        """
        raise NotImplementedError("A reward function has not been implemented")

    def estimate_harmful_trait(
        self,
        prompt: str,
        steering_coef: float,
        proposal_idx_switch: Optional[int] = None,
        proposal_bias: Optional[float] = None,
    ):
        # Initialize the conversation
        messages = [
            {
                "content": prompt,
                "role": "user",
            },
        ]
        convos = self.tokenizer.apply_chat_template(
            [messages for _ in range(self.args.num_particles)],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prepare the inputs
        inputs = self.tokenizer(
            convos, return_tensors="pt", padding=True, add_special_tokens=False
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        ### Clear cache to avoid OOM errors
        gc.collect()
        torch.cuda.empty_cache()

        # reward function for SMC
        # Just past to SMC as r_fn=reward_fn
        if self.smc_args is not None and self.smc_args["use_smc"]:
            self.smc_args["r_fn"] = self.create_reward_function(prompt)
        # Generate
        outputs = self._generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            steering_coef=steering_coef,
            proposal_idx_switch=proposal_idx_switch,
            proposal_bias=proposal_bias,
            prompt=prompt,
        )
        print("\n-----------------------------------------------\n")

        harm_est = outputs["reweighted_scores"]
        return harm_est, outputs

    def _model_ntp_wrapper(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        **kwargs,
    ):
        """
        Batches forward pass for next token prediction when using a large number of particles.
        Adapts to whatever cache type we are using for generation.
        """
        num_batches = math.ceil(input_ids.shape[0] / self.args.fwd_batch_size)
        chunked_input_ids = torch.chunk(input_ids, chunks=num_batches, dim=0)
        chunked_attention_mask = torch.chunk(attention_mask, chunks=num_batches, dim=0)

        if num_batches > 1:
            # Chunk the cache
            chunk_indices = torch.arange(input_ids.shape[0]).chunk(num_batches)
            if past_key_values.layers[0].keys is None:
                chunked_past_key_values = [
                    DynamicCache(
                        config=self.model.config, offloading=self.args.low_vram_cache
                    )
                    for _ in range(num_batches)
                ]
            else:
                chunked_past_key_values = []
                for batch_indices in chunk_indices:
                    cache = DynamicCache(
                        config=self.model.config, offloading=self.args.low_vram_cache
                    )
                    for layer_idx in range(self.model.config.num_hidden_layers):
                        cache.layers[layer_idx].keys = past_key_values.layers[
                            layer_idx
                        ].keys[batch_indices]
                        cache.layers[layer_idx].values = past_key_values.layers[
                            layer_idx
                        ].values[batch_indices]
                        cache.layers[layer_idx].is_initialized = True
                        cache.layers[layer_idx].dtype = past_key_values.layers[
                            layer_idx
                        ].dtype
                        cache.layers[layer_idx].device = past_key_values.layers[
                            layer_idx
                        ].device
                    chunked_past_key_values.append(cache)

            del past_key_values
        else:
            chunked_past_key_values = [past_key_values]

        # Batch the forward pass
        batched_outputs = []
        for batch_input_ids, batch_attention_mask, batch_past_key_values in zip(
            chunked_input_ids, chunked_attention_mask, chunked_past_key_values
        ):
            _batch_outputs = self.model.forward(
                batch_input_ids,
                batch_attention_mask,
                past_key_values=batch_past_key_values,
                **kwargs,
            )
            _batch_outputs.logits = _batch_outputs.logits[:, -1, :]
            batched_outputs.append(_batch_outputs)

        # Reassemble the cache
        if num_batches > 1:
            past_key_values = DynamicCache(
                config=self.model.config, offloading=self.args.low_vram_cache
            )
            for layer_idx in range(self.model.config.num_hidden_layers):
                layer_key_cache = torch.cat(
                    [cache.layers[layer_idx].keys for cache in chunked_past_key_values],
                    dim=0,
                )
                layer_value_cache = torch.cat(
                    [
                        cache.layers[layer_idx].values
                        for cache in chunked_past_key_values
                    ],
                    dim=0,
                )
                past_key_values.layers[layer_idx].keys = layer_key_cache
                past_key_values.layers[layer_idx].values = layer_value_cache
                past_key_values.layers[layer_idx].is_initialized = True
                past_key_values.layers[layer_idx].dtype = (
                    chunked_past_key_values[0].layers[layer_idx].dtype
                )
                past_key_values.layers[layer_idx].device = (
                    chunked_past_key_values[0].layers[layer_idx].device
                )

            for cache in chunked_past_key_values:
                del cache
            logits = torch.cat([out.logits for out in batched_outputs], dim=0)
        else:
            past_key_values = chunked_past_key_values[0]
            logits = batched_outputs[0].logits

        return logits, past_key_values

    def _generate(
        self,
        input_ids,
        attention_mask,
        steering_coef: float,
        proposal_idx_switch: Optional[int] = None,
        proposal_bias: Optional[float] = None,
        prompt: str = "",
    ):
        """Implements rare event estimation using sequential Monte Carlo."""
        assert prompt != ""

        base_past_key_values = DynamicCache(
            config=self.model.config, offloading=self.args.low_vram_cache
        )
        proposal_past_key_values = DynamicCache(
            config=self.model.config, offloading=self.args.low_vram_cache
        )

        num_particles = input_ids.shape[0]

        _input_ids = input_ids.detach().clone()
        _attention_mask = attention_mask.detach().clone()
        _completed_generation = torch.zeros((num_particles, 1), dtype=torch.bool).to(
            _input_ids.device
        )

        _inputs = {"input_ids": _input_ids, "attention_mask": _attention_mask}

        cache_position = torch.arange(
            _inputs["input_ids"].shape[1], dtype=torch.int64, device=self.model.device
        )

        log_importance_weights = torch.zeros(num_particles, 1, device=_input_ids.device)
        log_importance_weight_arr = []

        if proposal_bias is None:
            proposal_bias = torch.rand(
                (num_particles, 1), dtype=torch.bfloat16, device="cuda"
            )
        else:
            proposal_bias = torch.full(
                (num_particles, 1), proposal_bias, dtype=torch.bfloat16, device="cuda"
            )

        if proposal_idx_switch is None:
            proposal_idx_switch = torch.randint(
                0, self.args.max_new_tokens + 1, (num_particles, 1), device="cuda"
            )
        else:
            proposal_idx_switch = torch.full(
                (num_particles, 1), proposal_idx_switch, device="cuda"
            )

        if self.smc_args is not None:
            # smc_args["r_fn"] = lambda x: torch.ones(x.shape[0], device=_input_ids.device)
            fk_class = FKSteering(
                device=_input_ids.device,
                r_fn=self.smc_args["r_fn"],
                potential_type=self.smc_args["potential_type"],
                max_seq_len=self.smc_args["max_seq_len"],
                num_particles=self.smc_args["num_particles"],
                resample_start=self.smc_args["resample_start"],
                resample_end=self.smc_args["resample_end"],
                resample_interval=self.smc_args["resample_interval"],
                lmbda=self.smc_args["lmbda"],
                use_smc=self.smc_args["use_smc"],
                adaptive_resampling=self.smc_args["adaptive_resampling"],
                adaptive_resampling_threshold=self.smc_args[
                    "adaptive_resampling_threshold"
                ],
                smc_verbose=self.smc_args["smc_verbose"],
                importance_resampling_at_last_step=self.smc_args[
                    "importance_resampling_at_last_step"
                ],
                use_importance_weights_in_resampling=self.smc_args[
                    "use_importance_weights_in_resampling"
                ],
            )

        # Main generation loop
        for generation_idx in trange(self.args.max_new_tokens):
            # Compute the base distribution
            with torch.no_grad():
                base_logits, base_past_key_values = self._model_ntp_wrapper(
                    **_inputs,
                    past_key_values=base_past_key_values,
                    cache_position=cache_position,
                    use_cache=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                )
            base_logprobs = torch.log_softmax(base_logits[:, self.vocab_mask], dim=-1)

            # Compute the proposal distribution
            with (
                torch.no_grad(),
                self.proposal_context_manager(steering_coef),
            ):
                refusal_ablated_logits, proposal_past_key_values = (
                    self._model_ntp_wrapper(
                        **_inputs,
                        past_key_values=proposal_past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                        output_hidden_states=False,
                    )
                )
            proposal_logprobs = torch.log_softmax(
                refusal_ablated_logits[:, self.vocab_mask], dim=-1
            )

            # Linearly interpolate between distributions
            proposal_logprobs = torch.log(
                proposal_bias * torch.exp(proposal_logprobs)
                + (1 - proposal_bias) * torch.exp(base_logprobs)
            )

            # Enforce proposal_idx_switch here
            keep_proposal_mask = (generation_idx < proposal_idx_switch).type(
                torch.bfloat16
            )
            proposal_logprobs = (
                keep_proposal_mask * proposal_logprobs
                + (1 - keep_proposal_mask) * base_logprobs
            )

            next_tokens = torch.multinomial(
                proposal_logprobs.exp(),
                num_samples=1,
            )
            next_tokens = torch.where(self.vocab_mask)[0][next_tokens]

            # Check if sequence is completed
            _completed_generation |= next_tokens == self.tokenizer.eos_token_id

            # Pad completed sequences
            next_tokens[_completed_generation] = self.tokenizer.pad_token_id

            proposal_next_token_logprobs = torch.gather(
                proposal_logprobs, -1, next_tokens
            )

            base_next_token_logprobs = torch.gather(base_logprobs, -1, next_tokens)
            proposal_next_token_logprobs[_completed_generation] = (
                base_next_token_logprobs[_completed_generation]
            )

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

            if self.smc_args is not None and self.smc_args["use_smc"]:
                p_q_t = torch.exp(
                    base_next_token_logprobs - proposal_next_token_logprobs
                ).view(-1)

                if (
                    generation_idx in fk_class.resampling_arr
                    and self.smc_args["use_smc"]
                ):
                    rs_candidates = self.smc_args["r_fn"](_input_ids)
                else:
                    rs_candidates = None

                resample_indices = fk_class(
                    step_idx=generation_idx,
                    importance_weights=copy.deepcopy(p_q_t),
                    sequences=copy.deepcopy(_input_ids),
                    rs_candidates=copy.deepcopy(rs_candidates),
                )

                if (
                    not (resample_indices.cpu() == torch.arange(num_particles))
                    .all()
                    .item()
                ):
                    base_past_key_values = update_cache_after_resampling(
                        past_key_values=base_past_key_values,
                        indices=resample_indices,
                        model_config=self.model.config,
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

                    _input_ids = copy.deepcopy(_input_ids[resample_indices])
                    _attention_mask = copy.deepcopy(_attention_mask[resample_indices])
                    _inputs = {
                        "input_ids": next_tokens[resample_indices],
                        "attention_mask": _attention_mask,
                    }

                    _completed_generation = copy.deepcopy(
                        _completed_generation[resample_indices]
                    )
                    log_importance_weights = copy.deepcopy(
                        log_importance_weights[resample_indices]
                    )

                    log_importance_weight_arr = [
                        log_arr[resample_indices, :]
                        for log_arr in log_importance_weight_arr
                    ]

                    base_past_key_values = update_cache_after_resampling(
                        past_key_values=base_past_key_values,
                        indices=resample_indices,
                        model_config=self.model.config,
                    )
                    proposal_past_key_values = update_cache_after_resampling(
                        past_key_values=proposal_past_key_values,
                        indices=resample_indices,
                        model_config=self.model.config,
                    )

            ### END OF GENERATION LOOP

        del base_past_key_values
        del proposal_past_key_values
        gc.collect()
        torch.cuda.empty_cache()

        importance_weights = log_importance_weights.to(torch.float64).exp()

        # Judge the responses
        responses = [
            self.tokenizer.decode(r, skip_special_tokens=True)
            for r in _input_ids[:, input_ids.shape[1] :]
        ]
        judge_scores = self.judge_responses(prompt, responses).to(input_ids.device)

        if self.smc_args is None or self.smc_args["use_smc"] is False:
            reweighted_scores = judge_scores * importance_weights.squeeze(1)
            print(f"SIS harm score estimate: {reweighted_scores.mean().item()}")
            print(f"SIS harm score variance: {reweighted_scores.var().item()}")
            print(
                f"final importance weights: {importance_weights.mean(), importance_weights.std()}"
            )
            print(f"Judge scores: {judge_scores}")

            prompt_kl = -torch.mean(log_importance_weights).item()
            print(f"KL divergence: {prompt_kl}")
        elif self.smc_args["use_smc"]:
            reweighted_scores = fk_class.compute_fk_estimate(
                test_function_values=judge_scores, importance_weights=importance_weights
            )
            print(f"FK harm score estimate: {reweighted_scores.item()}")
            smc_quantities = fk_class.get_fk_quantities()
            sis = importance_weights.view(num_particles) * judge_scores
            print(f"SIS harm score estimate: {sis.mean().item()}")
            print(f"Judge scores: {judge_scores}")
            prompt_kl = -torch.mean(log_importance_weights).item()
            print(f"KL divergence: {prompt_kl}")
        else:
            raise NotImplementedError

        # print(f"Importance weight array: {importance_weight_arr[0]}")
        print(
            "Sequence generated:",
            self.tokenizer.decode(_input_ids[0], skip_special_tokens=False),
        )
        print(f"Judge score: {judge_scores[0]}")

        ret = dict(
            responses=[
                self.tokenizer.decode(r, skip_special_tokens=True)
                for r in _input_ids[:, input_ids.shape[1] :]
            ],
            _input_ids=_input_ids.cpu().numpy().tolist(),
            _completion_ids=_input_ids[:, input_ids.shape[1] :].cpu().numpy().tolist(),
            judge_scores=judge_scores.cpu().numpy(),
            prompt_kl=prompt_kl,
            importance_weights=importance_weights.cpu().numpy(),
            reweighted_scores=reweighted_scores.mean().item(),
        )

        if self.smc_args is not None and self.smc_args["use_smc"]:
            for key in smc_quantities:
                ret[key] = smc_quantities[key]

        return ret

    def estimate_CEM_harmful_trait(
        self,
        prompt: str,
        importance_weights: torch.Tensor,
        completions: List[str],
        full_input_ids: Dict[str, torch.Tensor],
        completion_ids: Dict[str, torch.Tensor],
        judge_scores: torch.Tensor,
        steering_coef_arr: List[float],
        proposal_idx_switch_arr: List[int],
        proposal_bias_arr: List[float],
    ):
        full_input_ids = torch.tensor(full_input_ids).to(self.model.device)
        completion_ids = torch.tensor(completion_ids).to(self.model.device)

        prompt_len = full_input_ids.shape[1] - completion_ids.shape[1]

        # compute CEM score
        outputs = self.estimate_cross_entropy_loss(
            completion_ids=completion_ids,
            full_input_ids=full_input_ids,
            prompt_len=prompt_len,
            judge_scores=judge_scores,
            prompt=prompt,
            importance_weights=importance_weights,
            steering_coef_arr=steering_coef_arr,
            proposal_idx_switch_arr=proposal_idx_switch_arr,
            proposal_bias_arr=proposal_bias_arr,
        )
        # print("\n-----------------------------------------------\n")

        return outputs

    def _model_forward_wrapper(
        self,
        input_ids,
        attention_mask,
        **kwargs,
    ):
        """
        Batches forward pass for next token prediction when using a large number of particles.
        Adapts to whatever cache type we are using for generation.
        """
        num_batches = math.ceil(input_ids.shape[0] / self.args.fwd_batch_size)
        chunked_input_ids = torch.chunk(input_ids, chunks=num_batches, dim=0)
        chunked_attention_mask = torch.chunk(attention_mask, chunks=num_batches, dim=0)

        # Batch the forward pass
        batched_logprobs = []
        for batch_input_ids, batch_attention_mask in zip(
            chunked_input_ids, chunked_attention_mask
        ):
            _batch_outputs = self.model.forward(
                batch_input_ids,
                batch_attention_mask,
                **kwargs,
            )
            _batch_logprobs = torch.log_softmax(_batch_outputs.logits, dim=-1)[
                :, :-1, :
            ]
            _batch_logprobs = torch.gather(
                _batch_logprobs,
                -1,
                batch_input_ids[:, 1:].unsqueeze(-1),
            ).squeeze(-1)
            del _batch_outputs
            gc.collect()
            torch.cuda.empty_cache()

            batched_logprobs.append(_batch_logprobs)
        return torch.cat(batched_logprobs, dim=0)

    def _compute_cross_entropy_tensor(
        self,
        base_logprobs,
        proposal_logprobs,
        full_input_ids,
        prompt_len,
        importance_weights,
        judge_scores,
        steering_coef_arr,
        proposal_idx_switch_arr,
        proposal_bias_arr,
    ):
        proposal_biases = torch.from_numpy(proposal_bias_arr).to("cuda")
        proposal_idx_switches = torch.tensor(proposal_idx_switch_arr).to("cuda")

        mixing_logprobs = torch.log(
            torch.einsum("i,jkl->ijkl", proposal_biases, proposal_logprobs.exp())
            + torch.einsum("i,jkl->ijkl", 1 - proposal_biases, base_logprobs.exp())
        )[None, :, :, :, :].repeat(proposal_idx_switches.shape[0], 1, 1, 1, 1)

        base_logprobs = (
            base_logprobs[None, None, :, :, :]
            .expand(*mixing_logprobs.shape)
            .to(torch.float64)
        )

        proposal_switch_mask = (
            torch.arange(mixing_logprobs.shape[-1], device="cuda")[None, :]
            >= proposal_idx_switches[:, None]
        )[:, None, None, None, :].expand(*mixing_logprobs.shape)

        eos_token_mask = full_input_ids[:, prompt_len:] == self.tokenizer.eos_token_id
        eos_token_mask = eos_token_mask.expand(*mixing_logprobs.shape)

        mixing_logprobs[proposal_switch_mask] = base_logprobs[proposal_switch_mask]
        mixing_logprobs[eos_token_mask] = 0.0
        mixing_logprobs = mixing_logprobs.sum(dim=-1)

        judge_scores = (
            judge_scores.to("cuda")
            .to(torch.float64)[None, None, None, :]
            .expand(*mixing_logprobs.shape)
        )
        importance_weights = (
            importance_weights.squeeze()
            .to("cuda")
            .to(torch.float64)[None, None, None, :]
            .expand(*mixing_logprobs.shape)
        )

        cross_entropy_tensor = (
            (-mixing_logprobs * judge_scores * importance_weights)
            .sum(dim=-1)
            .permute(2, 0, 1)
        )

        return cross_entropy_tensor

    def estimate_cross_entropy_loss(
        self,
        completion_ids,
        full_input_ids,
        prompt_len: int,
        importance_weights,
        judge_scores,
        prompt: str = "",
        fwd_batch_size: int = 128,
        smc_args=None,
        low_vram_cache: bool = False,
        steering_coef_arr: List[float] = None,
        proposal_idx_switch_arr: List[int] = None,
        proposal_bias_arr: List[float] = None,
    ):
        """Implements rare event estimation using sequential Monte Carlo."""

        assert prompt != ""
        # assert decoding == "sample", "Only 'sample' decoding is supported in this function."

        assert smc_args is None, "SMC not yet supported in this function."

        self.model.eval()
        full_attention_mask = (full_input_ids != self.tokenizer.pad_token_id).long()
        model_input = dict(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            output_hidden_states=False,
            enable_cache=False,
        )

        print("Running CEM!")

        print("Computing base logprobs...")
        with torch.no_grad():
            base_logprobs = self._model_forward_wrapper(**model_input)
        print("Done.")

        print("Computing proposal_logprobs with various steering coefficients...")
        proposal_logprobs_arr = []
        for steering_coef in tqdm(steering_coef_arr):
            with torch.no_grad(), self.proposal_context_manager(steering_coef):
                _proposal_logprobs = self._model_forward_wrapper(**model_input)
                proposal_logprobs_arr.append(_proposal_logprobs)
        print("Done.")

        print("Computing cross entropy tensor...")
        start_idx = prompt_len - 1
        base_logprobs = base_logprobs[None, :, start_idx:]
        proposal_logprobs = torch.stack(proposal_logprobs_arr)[:, :, start_idx:]

        num_batches = math.ceil(base_logprobs.shape[1] / self.args.fwd_batch_size)
        chunked_base_logprobs = torch.chunk(base_logprobs, chunks=num_batches, dim=1)
        chunked_proposal_logprobs = torch.chunk(
            proposal_logprobs, chunks=num_batches, dim=1
        )
        chunked_full_input_ids = torch.chunk(full_input_ids, chunks=num_batches, dim=0)
        chunked_importance_weights = torch.chunk(
            torch.tensor(importance_weights), chunks=num_batches, dim=0
        )
        chunked_judge_scores = torch.chunk(
            torch.tensor(judge_scores), chunks=num_batches, dim=0
        )

        cross_entropy_tensor = 0.0
        for (
            batch_base_logprobs,
            batch_proposal_logprobs,
            batch_full_input_ids,
            batch_judge_scores,
            batch_importance_weights,
        ) in zip(
            chunked_base_logprobs,
            chunked_proposal_logprobs,
            chunked_full_input_ids,
            chunked_judge_scores,
            chunked_importance_weights,
        ):
            cross_entropy_tensor += self._compute_cross_entropy_tensor(
                batch_base_logprobs,
                batch_proposal_logprobs,
                batch_full_input_ids,
                prompt_len,
                batch_importance_weights,
                batch_judge_scores,
                steering_coef_arr,
                proposal_idx_switch_arr,
                proposal_bias_arr,
            )

        print("Done.")
        return cross_entropy_tensor

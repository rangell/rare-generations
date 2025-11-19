from typing import Callable
from copy import deepcopy
import math
from tqdm import trange

import torch
from transformers import DynamicCache, OffloadedCache

from smc.fk_steering import FKSteering, update_cache_after_resampling
from smc.estimator import model_forward_wrapper


class CEMHarmfulTraitEstimator(ABC):
    def __init__(self, args, model, tokenizer, smc_args: Optional[Dict]):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.smc_args = smc_args

    @abstractmethod
    def proposal_context_manager(self, timestep: int) -> ContextManager:
        """We require that this function returns a context manager."""
        pass

    def estimate_CEM_harmful_trait(
        self, prompt: str, completions: List[str], judge_scores: torch.Tensor
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
        inputs = self.tokenizer(convos, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"][:, 1:]
        attention_mask = inputs["attention_mask"][:, 1:]

        completion_ids = self.tokenizer(
            [self.tokenizer.generation_prompt + comp for comp in completions],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # Clear cache to avoid OOM errors
        gc.collect()
        torch.cuda.empty_cache()

        # compute CEM score
        outputs = self.estimate_cross_entropy_loss(
            completion_ids=completion_ids["input_ids"],
            judge_scores=judge_scores,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt=prompt,
        )
        print("\n-----------------------------------------------\n")

        cross_entropy_objective = outputs["cross_entropy_objective"]
        return cross_entropy_objective, outputs

    def _model_forward_wrapper(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        **kwargs,
    ):
        """
        Batches forward pass when using a large number of particles.
        Adapts to whatever cache type we are using for generation.
        """
        num_batches = math.ceil(input_ids.shape[0] / self.args.fwd_batch_size)
        chunked_input_ids = torch.chunk(input_ids, chunks=num_batches, dim=0)
        chunked_attention_mask = torch.chunk(attention_mask, chunks=num_batches, dim=0)

        if num_batches > 1:
            # Chunk the cache
            chunk_indices = torch.arange(input_ids.shape[0]).chunk(num_batches)
            if len(past_key_values) == 0:
                chunked_past_key_values = [
                    type(past_key_values)() for _ in range(num_batches)
                ]
            else:
                chunked_past_key_values = []
                for batch_indices in chunk_indices:
                    cache = type(past_key_values)()
                    for layer_idx in range(self.model.config.num_hidden_layers):
                        batch_layer_key_cache = past_key_values.key_cache[layer_idx][
                            batch_indices
                        ]
                        batch_layer_value_cache = past_key_values.value_cache[
                            layer_idx
                        ][batch_indices]
                        cache.update(
                            batch_layer_key_cache, batch_layer_value_cache, layer_idx
                        )

                    if isinstance(past_key_values, OffloadedCache):
                        cache.original_device = past_key_values.original_device

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
            _batch_outputs.logits = _batch_outputs.logits[
                :, -1, : self.tokenizer.vocab_size
            ]
            batched_outputs.append(_batch_outputs)

        # Reassemble the cache
        if num_batches > 1:
            past_key_values = type(chunked_past_key_values[0])()
            for layer_idx in range(self.model.config.num_hidden_layers):
                layer_key_cache = torch.cat(
                    [cache.key_cache[layer_idx] for cache in chunked_past_key_values],
                    dim=0,
                )
                layer_value_cache = torch.cat(
                    [cache.value_cache[layer_idx] for cache in chunked_past_key_values],
                    dim=0,
                )
                past_key_values.update(layer_key_cache, layer_value_cache, layer_idx)

            if isinstance(chunked_past_key_values[0], OffloadedCache):
                past_key_values.original_device = chunked_past_key_values[
                    0
                ].original_device

            for cache in chunked_past_key_values:
                del cache
        else:
            past_key_values = chunked_past_key_values[0]

        logits = torch.cat([out.logits for out in batched_outputs], dim=0)
        return logits, past_key_values

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

from abc import ABC, abstractmethod

import gc
import math
from typing import List, Optional, Callable, ContextManager, Dict
from tqdm import tqdm, trange

import torch
from transformers import StaticCache


class HarmfulTraitEstimator(ABC):
    def __init__(self, args, model, tokenizer, smc_args: Optional[Dict]):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.smc_args = smc_args

        def token_exists(id):
            try:
                self.tokenizer.decode([id])
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
        steering_params: dict,
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

        # Generate
        outputs = self._generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            steering_params=steering_params,
            proposal_idx_switch=proposal_idx_switch,
            proposal_bias=proposal_bias,
            prompt=prompt,
        )
        print("\n-----------------------------------------------\n")

        harm_est = outputs["reweighted_scores"]
        return harm_est, outputs

    def _generate(
        self,
        input_ids,
        attention_mask,
        steering_params: dict,
        proposal_idx_switch: Optional[int] = None,
        proposal_bias: Optional[float] = None,
        prompt: str = "",
    ):
        """Implements rare event estimation using sequential Monte Carlo."""
        assert prompt != ""

        # Single 2N cache: rows [0:N] = base, rows [N:2N] = proposal
        past_key_values = StaticCache(
            config=self.model.config,
            max_cache_len=input_ids.shape[1] + self.args.max_new_tokens,
            offloading=self.args.low_vram_cache,
        )

        num_particles = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        max_len = prompt_len + self.args.max_new_tokens

        # Pre-allocate output tensors for N particles
        _input_ids = torch.full(
            (num_particles, max_len),
            self.tokenizer.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        _attention_mask = torch.zeros(
            (num_particles, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        _input_ids[:, :prompt_len] = input_ids
        _attention_mask[:, :prompt_len] = attention_mask

        _completed_generation = torch.zeros((num_particles, 1), dtype=torch.bool).to(
            _input_ids.device
        )

        # Tile prompt for 2N batch: base rows [0:N], proposal rows [N:2N]
        _inputs = {
            "input_ids": input_ids.repeat(2, 1),
            "attention_mask": attention_mask.repeat(2, 1),
        }

        cache_position = torch.arange(
            prompt_len, dtype=torch.int64, device=self.model.device
        )

        log_importance_weights = torch.zeros(num_particles, 1, device=_input_ids.device)

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

        log_bias = torch.log(proposal_bias.float()).to(proposal_bias.dtype)
        log_1m_bias = torch.log1p(-proposal_bias.float()).to(proposal_bias.dtype)

        # Main generation loop: hooks registered once, active for rows [N:2N] only
        with torch.no_grad(), self.proposal_context_manager(**steering_params):
            for generation_idx in trange(self.args.max_new_tokens):
                combined_logits = self.model.forward(
                    **_inputs,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    output_hidden_states=False,
                ).logits[:, -1, :]

                base_logits = combined_logits[:num_particles]
                refusal_ablated_logits = combined_logits[num_particles:]

                base_logits.masked_fill_(~self.vocab_mask[None, :], float("-inf"))
                refusal_ablated_logits.masked_fill_(
                    ~self.vocab_mask[None, :], float("-inf")
                )
                base_logprobs = torch.log_softmax(base_logits, dim=-1)
                proposal_logprobs = torch.log_softmax(refusal_ablated_logits, dim=-1)

                # Linearly interpolate between distributions
                proposal_logprobs = torch.logaddexp(
                    proposal_logprobs + log_bias,
                    base_logprobs + log_1m_bias,
                )

                # Enforce proposal_idx_switch
                keep_proposal_mask = generation_idx < proposal_idx_switch
                proposal_logprobs = torch.where(
                    keep_proposal_mask, proposal_logprobs, base_logprobs
                )
                proposal_probs = proposal_logprobs.exp()

                next_tokens = torch.multinomial(proposal_probs, num_samples=1)

                _completed_generation |= next_tokens == self.tokenizer.eos_token_id
                next_tokens[_completed_generation] = self.tokenizer.pad_token_id

                proposal_next_token_logprobs = torch.gather(
                    proposal_logprobs, -1, next_tokens
                )
                base_next_token_logprobs = torch.gather(base_logprobs, -1, next_tokens)
                proposal_next_token_logprobs[_completed_generation] = (
                    base_next_token_logprobs[_completed_generation]
                )

                log_importance_weights += (
                    base_next_token_logprobs - proposal_next_token_logprobs
                )

                # Update tracking (N particles)
                pos = prompt_len + generation_idx
                _input_ids[:, pos] = next_tokens.squeeze(-1)
                _attention_mask[:, pos] = 1

                # Tile next_tokens for both base and proposal halves
                _inputs = {
                    "input_ids": next_tokens.repeat(2, 1),
                    "attention_mask": _attention_mask[:, : pos + 1].repeat(2, 1),
                }
                cache_position = cache_position[-1:] + 1

        del past_key_values
        gc.collect()
        torch.cuda.empty_cache()

        importance_weights = log_importance_weights.to(torch.float64).exp()

        # Judge the responses
        responses = [
            self.tokenizer.decode(r, skip_special_tokens=True)
            for r in _input_ids[:, input_ids.shape[1] :]
        ]
        judge_scores = self.judge_responses(prompt, responses).to(input_ids.device)

        reweighted_scores = judge_scores * importance_weights.squeeze(1)
        print(f"SIS harm score estimate: {reweighted_scores.mean().item()}")
        print(f"SIS harm score variance: {reweighted_scores.var().item()}")
        print(
            f"final importance weights: {importance_weights.mean(), importance_weights.std()}"
        )
        print(f"Judge scores: {judge_scores}")

        prompt_kl = -torch.mean(log_importance_weights).item()
        print(f"KL divergence: {prompt_kl}")

        print(
            f"Sequence generated: {self.tokenizer.decode(_input_ids[0], skip_special_tokens=False)}"
        )
        print(f"Judge score: {judge_scores[0]}")

        ret = dict(
            responses=responses,
            _input_ids=_input_ids.cpu().numpy().tolist(),
            _completion_ids=_input_ids[:, input_ids.shape[1] :].cpu().numpy().tolist(),
            judge_scores=judge_scores.cpu().numpy(),
            prompt_kl=prompt_kl,
            importance_weights=importance_weights.cpu().numpy(),
            reweighted_scores=reweighted_scores.mean().item(),
        )

        return ret

    def estimate_CEM_harmful_trait(
        self,
        prompt: str,
        importance_weights: torch.Tensor,
        full_input_ids: Dict[str, torch.Tensor],
        completion_ids: Dict[str, torch.Tensor],
        judge_scores: torch.Tensor,
        steering_params_arr: List[dict],
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
            steering_params_arr=steering_params_arr,
            proposal_idx_switch_arr=proposal_idx_switch_arr,
            proposal_bias_arr=proposal_bias_arr,
        )

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
        num_batches = 10 * math.ceil(input_ids.shape[0] / self.args.fwd_batch_size)
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
            batched_logprobs.append(_batch_logprobs)

        gc.collect()
        torch.cuda.empty_cache()
        return torch.cat(batched_logprobs, dim=0)

    def _compute_cross_entropy_tensor(
        self,
        base_logprobs,
        proposal_logprobs,
        full_input_ids,
        prompt_len,
        importance_weights,
        judge_scores,
        proposal_idx_switch_arr,
        proposal_bias_arr,
    ):
        proposal_biases = torch.from_numpy(proposal_bias_arr).to("cuda")
        proposal_idx_switches = torch.tensor(proposal_idx_switch_arr).to("cuda")

        # Mix in log-space: log(bias*p + (1-bias)*q) via logaddexp — avoids exp/log round-trips
        log_bias = torch.log(proposal_biases)[:, None, None, None]  # (B,1,1,1)
        log_1m_bias = torch.log1p(-proposal_biases)[:, None, None, None]
        mixing_logprobs = torch.logaddexp(
            proposal_logprobs + log_bias,
            base_logprobs + log_1m_bias,
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
            importance_weights.squeeze(1)
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
        smc_args=None,
        steering_params_arr: List[dict] = None,
        proposal_idx_switch_arr: List[int] = None,
        proposal_bias_arr: List[float] = None,
    ):
        """Implements rare event estimation using sequential Monte Carlo."""

        assert prompt != ""
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
        print(f"steering_params_arr: {steering_params_arr}")
        print(f"proposal_idx_switch_arr: {proposal_idx_switch_arr}")
        print(f"proposal_bias_arr: {proposal_bias_arr}")

        print("Computing base logprobs...")
        with torch.no_grad():
            base_logprobs = self._model_forward_wrapper(**model_input)
        print("Done.")

        print("Computing proposal_logprobs with various steering coefficients...")
        proposal_logprobs_arr = []
        for steering_params in tqdm(steering_params_arr):
            with torch.no_grad(), self.proposal_context_manager(**steering_params):
                _proposal_logprobs = self._model_forward_wrapper(**model_input)
                proposal_logprobs_arr.append(_proposal_logprobs)
        print("Done.")

        print("Computing cross entropy tensor...")
        start_idx = prompt_len - 1
        base_logprobs = base_logprobs[None, :, start_idx:]
        proposal_logprobs = torch.stack(proposal_logprobs_arr)[:, :, start_idx:]

        num_batches = 10 * math.ceil(base_logprobs.shape[1] / self.args.fwd_batch_size)
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
        zero_judge_batches = 0
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
            if batch_judge_scores.mean().item() == 0.0:
                print("Skipping batch with all zero judge scores.")
                zero_judge_batches += 1
                continue

            cross_entropy_tensor += self._compute_cross_entropy_tensor(
                batch_base_logprobs,
                batch_proposal_logprobs,
                batch_full_input_ids,
                prompt_len,
                batch_importance_weights,
                batch_judge_scores,
                proposal_idx_switch_arr,
                proposal_bias_arr,
            )

        print("Done.")

        if zero_judge_batches == len(chunked_judge_scores):
            print("All batches had zero judge scores. Returning zero tensor.")
            return 0

        return cross_entropy_tensor

from abc import ABC, abstractmethod
from copy import deepcopy
import gc
import math
import pickle
from pathlib import Path
import json
from typing import List, Optional, Callable, ContextManager, Dict
from tqdm import trange

import torch
from transformers import DynamicCache, OffloadedCache

from smc.fk_steering import FKSteering, update_cache_after_resampling

# from smc.estimator import model_forward_wrapper


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
        self,
        prompt: str,
        importance_weights: torch.Tensor,
        completions: List[str],
        judge_scores: torch.Tensor,
        proposal_idx_switch_arr: List[int],
        proposal_bias_arr: List[float],
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
            # add_special_tokens=False,
        )

        # Prepare the inputs
        inputs = self.tokenizer(convos, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"][:, 1:]
        attention_mask = inputs["attention_mask"][:, 1:]

        prompt_len = input_ids.shape[1]    
        # import pdb; pdb.set_trace()

        _completions = [convos[i] + completions[i] for i in range(self.args.num_particles)]
        assert self.args.num_particles == len(completions)

        completion_ids = self.tokenizer(
            _completions,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        
        assert completion_ids["input_ids"].shape[0] == self.args.num_particles
        
        new_completion_input_ids = torch.zeros((self.args.num_particles, completion_ids["input_ids"].shape[1] - 1), dtype=completion_ids["input_ids"].dtype).to(self.model.device)
        new_completion_attention_mask = torch.zeros((self.args.num_particles, completion_ids["attention_mask"].shape[1] - 1), dtype=completion_ids["attention_mask"].dtype).to(self.model.device)
        for particle_idx in range(self.args.num_particles):
            number_of_eos_tokens = (completion_ids['attention_mask'][particle_idx] == 0).sum().item()

            new_completion_input_ids[particle_idx, : number_of_eos_tokens] = completion_ids["input_ids"][particle_idx, :number_of_eos_tokens] 
            new_completion_input_ids[particle_idx, number_of_eos_tokens: ] = completion_ids["input_ids"][particle_idx, number_of_eos_tokens + 1: ]
            
            new_completion_attention_mask[particle_idx, : number_of_eos_tokens] = completion_ids["attention_mask"][particle_idx, :number_of_eos_tokens]
            new_completion_attention_mask[particle_idx, number_of_eos_tokens: ] = completion_ids["attention_mask"][particle_idx, number_of_eos_tokens + 1: ]
        
            assert completion_ids["input_ids"][particle_idx, number_of_eos_tokens + 1: prompt_len + number_of_eos_tokens + 1].equal(input_ids[particle_idx])
            assert new_completion_input_ids[particle_idx, number_of_eos_tokens: prompt_len + number_of_eos_tokens].equal(input_ids[particle_idx])
                        
            if not self.tokenizer.decode(completion_ids["input_ids"][particle_idx, prompt_len + number_of_eos_tokens + 1:], skip_special_tokens=True) == completions[particle_idx]:
                print("Decoding mismatch!")
                
                print(self.tokenizer.decode(completion_ids["input_ids"][particle_idx, prompt_len + number_of_eos_tokens + 1:], skip_special_tokens=True))
                print(completions[particle_idx])
                import pdb; pdb.set_trace()
                # raise ValueError("Decoding mismatch!")
                    
        completion_ids = {
            "input_ids": new_completion_input_ids,
            "attention_mask": new_completion_attention_mask,
        }

        # Clear cache to avoid OOM errors
        gc.collect()
        torch.cuda.empty_cache()

        # compute CEM score
        outputs = self.estimate_cross_entropy_loss(
            completion_ids=completion_ids,
            judge_scores=judge_scores,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt=prompt,
            importance_weights=importance_weights,  
            proposal_idx_switch_arr=proposal_idx_switch_arr,
            proposal_bias_arr=proposal_bias_arr            
        )
        print("\n-----------------------------------------------\n")

        return outputs

    def estimate_cross_entropy_loss(
        self,
        input_ids,
        completion_ids,
        importance_weights,
        judge_scores,
        attention_mask,
        prompt: str = "",
        decoding: str = "sample",  # Options: 'greedy', 'sample', 'beam_search', 'top_k', 'top_p'
        fwd_batch_size: int = 128,
        max_new_tokens: int = 10,
        proposal_bias: float = 0.5,
        proposal_idx_switch: int = 10,
        smc_args=None,
        low_vram_cache: bool = False,
        proposal_idx_switch_arr: List[int] = None,
        proposal_bias_arr: List[float] = None,
    ):
        """Implements rare event estimation using sequential Monte Carlo."""

        assert prompt != ""
        # assert decoding == "sample", "Only 'sample' decoding is supported in this function."

        assert smc_args is None, "SMC not yet supported in this function."

        num_particles = input_ids.shape[0]
        prompt_len = input_ids.shape[1]

        self.model.eval()
        with torch.no_grad():
            base_out = self.model(**completion_ids, output_hidden_states=False)
            base_logits = base_out.logits
            base_logprobs = torch.log_softmax(base_logits, dim=-1)
            
        with torch.no_grad(), self.proposal_context_manager(0):
            proposal_out = self.model(**completion_ids, output_hidden_states=False)
            proposal_logits = proposal_out.logits
            proposal_logprobs = torch.log_softmax(proposal_logits, dim=-1)
            
        base_logprobs = base_logprobs[:, : -1, :]
        proposal_logprobs = proposal_logprobs[:, : -1, :]
        
        base_logprobs = torch.gather(
            base_logprobs,
            -1,
            completion_ids["input_ids"][:, 1:].unsqueeze(-1),
        ).squeeze(-1)
        
        proposal_logprobs = torch.gather(
            proposal_logprobs,
            -1,
            completion_ids["input_ids"][:, 1:].unsqueeze(-1),
        ).squeeze(-1)
                
        logprobs_dict = {'base': {}, 'proposal': {}}
        cross_entropy_objective = {}
        
        cross_entropy_matrix = torch.zeros((len(proposal_idx_switch_arr), len(proposal_bias_arr))).to(self.model.device)
        
        for particle_idx in range(num_particles):
            # get start index for completion_ids            
            number_of_eos_tokens = (completion_ids['attention_mask'][particle_idx] == 0).sum().item()
            start_idx = prompt_len + number_of_eos_tokens
            
            assert completion_ids["input_ids"][particle_idx, start_idx - 1] == input_ids[particle_idx, -1], "Input IDs and completion IDs do not match in the prompt region."
            
            target_ids = completion_ids["input_ids"][particle_idx, start_idx - 1:]            
            base_logprobs_target = base_logprobs[particle_idx, start_idx - 1: ]
            proposal_logprobs_target = proposal_logprobs[particle_idx, start_idx - 1: ]
            
            logprobs_dict['base'][particle_idx] = base_logprobs_target
            logprobs_dict['proposal'][particle_idx] = proposal_logprobs_target   

            # shifting window to switch from proposal to base, vectorized for various proposal_idx_switch
            for i, proposal_idx_switch in enumerate(proposal_idx_switch_arr):
                for j, mixing in enumerate(proposal_bias_arr):
                    mixed_logprobs = torch.zeros_like(proposal_logprobs_target)
                    
                    mixed_logprobs[:proposal_idx_switch] = (mixing) * proposal_logprobs_target[:proposal_idx_switch].exp() + (1 - mixing) * base_logprobs_target[:proposal_idx_switch].exp()
                    mixed_logprobs[:proposal_idx_switch] = torch.log(mixed_logprobs[:proposal_idx_switch])                    
                    
                    mixed_logprobs[proposal_idx_switch:] = base_logprobs_target[proposal_idx_switch:]
                    
                    judge_score = judge_scores[particle_idx]                    
                    cross_entropy_matrix[i, j] += judge_score * importance_weights[particle_idx].item() * -torch.sum(mixed_logprobs).item()

        return cross_entropy_matrix.cpu()
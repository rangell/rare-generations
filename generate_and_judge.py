import gc
import numpy as np
from jaxtyping import Float
import json
import os
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

import torch.distributed as dist
from accelerate import Accelerator

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
        model_name_or_path, output_hidden_states=True, return_dict_in_generate=True
    )

    if isinstance(config, Qwen2Config):
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        token=os.getenv("HF_TOKEN"),
        low_cpu_mem_usage=True,
        device_map="auto",
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


def score_responses(
    tokenizer,
    forbidden_prompt: str,
    input_ids: torch.Tensor,
    n_jobs: int = 50
):
    scoring_fn = lambda r: strongreject_rubric(forbidden_prompt, r)["score"]
    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in input_ids]
    judge_scores = Parallel(n_jobs=n_jobs)(delayed(scoring_fn)(r) for r in responses)
    judge_scores = torch.tensor(judge_scores, device=input_ids.device)
    return judge_scores


def batched_model_forward(
    model_forward,
    _input_ids,
    _attention_mask,
    _completed_generation,
    decoding: str = "",
    inv_temperature: float = 1.0,
    batch_size: int = 256
):
    num_chunks = int((_input_ids.shape[0] / batch_size) + 1)
    chunked_input_ids = _input_ids.chunk(num_chunks)
    chunked_attention_mask = _attention_mask.chunk(num_chunks)
    chunked_completed_generation = _completed_generation.chunk(num_chunks)
    chunk_generator = zip(
        chunked_input_ids, chunked_attention_mask, chunked_completed_generation
    )

    all_outputs = []
    for b_input_ids, b_attention_mask, b_completed_generation in chunk_generator:
        batch_output = model_forward(
            _input_ids=b_input_ids,
            _attention_mask=b_attention_mask,
            _completed_generation=b_completed_generation,
            decoding=decoding,
            inv_temperature=proposal_inv_temperature,
        )
        all_outputs.append(batch_output)

    return tuple([torch.cat(l, dim=0) for l in zip(*all_outputs)])


def generate(
    base_forward,
    proposal_forward,
    input_ids,
    attention_mask,
    decoding: str = "sample",  # Options: 'greedy', 'sample', 'beam_search', 'top_k', 'top_p'
    use_smc: bool = False,
    r_fn=None,
    num_particles: int = 10,
    max_new_tokens: int = 10,
    base_inv_temperature: float = 1.0,
    proposal_inv_temperature: float = 1.0,
    proposal_model_switch_idx=None,  # If None, use the proposal model for all steps
    resample_start: int = 10,
    resample_end: int = 20,
    resample_interval: int = 10,
    lmbda: float = 10.0,  # Adjust this value based on your needs
    adaptive_resampling: bool = True,
    adaptive_resampling_threshold: float = 0.5,
    smc_verbose: bool = False,
    gen_batch_size: int = 256,
):
    _input_ids = input_ids.detach().clone()
    _attention_mask = attention_mask.detach().clone()
    _completed_generation = torch.zeros((num_particles, 1), dtype=torch.bool).to("cuda")

    assert _input_ids.shape[0] == num_particles, (_input_ids.shape[0], num_particles)
    prompt_len = len(_input_ids[0])

    fk_class = FKSteering(
        device=model.device,
        r_fn=r_fn,
        potential_type="diff",
        max_seq_len=max_new_tokens,
        num_particles=num_particles,
        resample_start=resample_start,
        resample_end=resample_end,
        resample_interval=resample_interval,
        lmbda=lmbda,
        use_smc=use_smc,
        adaptive_resampling=adaptive_resampling,
        adaptive_resampling_threshold=adaptive_resampling_threshold,
        smc_verbose=smc_verbose,
    )

    # Main generation loop
    importance_weights = torch.ones(num_particles, device=_input_ids.device)
    importance_weight_arr = []
    for sample_idx in tqdm(range(max_new_tokens)):

        if (
            sample_idx == proposal_model_switch_idx
            and proposal_model_switch_idx is not None
        ):
            print(
                f"Switching proposal model to 'base' at step {sample_idx} at temperature {base_inv_temperature}."
            )
            # Switch to the base model for proposal generation
            # NOTE: This is useful for the first few steps where the proposal model might not be
            proposal_forward = base_forward
            proposal_inv_temperature = base_inv_temperature

        _, next_tokens, proposal_logprobs, _completed_generation = batched_model_forward(
            model_forward=proposal_forward,
            _input_ids=_input_ids,
            _attention_mask=_attention_mask,
            _completed_generation=_completed_generation,
            decoding=decoding,
            inv_temperature=proposal_inv_temperature,
            batch_size=gen_batch_size
        )

        if proposal_forward != base_forward:
            logprobs_distribution, _, _, _ = batched_model_forward(
                model_forward=base_forward,
                _input_ids=_input_ids,
                _attention_mask=_attention_mask,
                _completed_generation=_completed_generation,
                decoding=decoding,
                inv_temperature=proposal_inv_temperature,
                batch_size=gen_batch_size
            )
            base_logprobs = torch.gather(
                logprobs_distribution, -1, next_tokens
            )
        else:
            base_logprobs = proposal_logprobs.clone()
            
        # Update input arguments
        _input_ids = torch.cat((_input_ids, next_tokens), dim=1)
        _attention_mask = torch.cat(
            (_attention_mask, torch.ones_like(next_tokens)), dim=1
        )

        # NOTE: The following line is important to ensure that the input_ids and attention_mask are of the correct shape
        # NOTE: do not remove this line as we accumulate the product of importance weights

        importance_weight_at_cur_step = torch.exp(
            base_logprobs - proposal_logprobs
        ).view(num_particles)
        assert importance_weight_at_cur_step.shape == (num_particles,)

        # Resample
        _input_ids, indices = fk_class(
            sample_idx=sample_idx,
            sequences=_input_ids,
            importance_weights=importance_weight_at_cur_step,
        )

        importance_weights = importance_weights[indices] * torch.exp(
            base_logprobs - proposal_logprobs
        ).view(num_particles)

        assert torch.all(
            importance_weights >= 0
        ), "Importance weights should be non-negative."
        # import pdb; pdb.set_trace()
        assert importance_weights.shape == (num_particles,), (
            importance_weights.shape,
            num_particles,
        )

        # change the input_ids and attention_mask to only include the resampled sequences
        _attention_mask = _attention_mask[indices]
        # update importance_weight_arr to only include the resampled sequences
        for past_idx in range(len(importance_weight_arr)):
            importance_weight_arr[past_idx] = importance_weight_arr[past_idx][indices]

        importance_weight_arr.append((base_logprobs - proposal_logprobs)[indices])

    importance_weight_arr = torch.exp(torch.cat(importance_weight_arr, dim=1))
    assert importance_weight_arr.shape == (num_particles, max_new_tokens), (
        importance_weight_arr.shape,
        num_particles,
        max_new_tokens,
    )

    # Judge the responses
    judge_scores = score_responses(
        tokenizer,
        forbidden_prompt,
        input_ids=_input_ids[:, input_ids.shape[1]:]
    )

    assert judge_scores.shape == (num_particles,), (
        judge_scores.shape,
        num_particles,
    )

    if not use_smc:
        # If not using SMC, just return the input_ids
        reweighted_scores = [
            float(importance_weight.item() * score)
            for importance_weight, score in zip(importance_weights, judge_scores)
        ]

        sis_approx_harm_score = np.mean(reweighted_scores)
        sis_approx_harm_variance = np.var(reweighted_scores)
        print(f"SIS harm score estimate: {sis_approx_harm_score}")
        print(f"SIS harm score variance: {sis_approx_harm_variance}")

        print(
            f"Final importance weights: {importance_weights.mean(), importance_weights.std()}"
        )
        print(f"Importance weight array: {importance_weight_arr[0]}")
        print(
            "Sequence generated:",
            tokenizer.decode(_input_ids[0], skip_special_tokens=False),
        )

        return _input_ids, sis_approx_harm_score
    else:
        fk_estimate = fk_class.compute_fk_estimate(
            test_function_values=judge_scores,
        )
        if fk_estimate == 0:
            print("Judge scores:", judge_scores.mean(), judge_scores.std())
        print(f"FK harm score estimate: {fk_estimate}")

        return _input_ids, fk_estimate


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def create_reward_fn(
    tokenizer,
    forbidden_prompt: str = "",
    prompt_len: int = -1,
    batch_size: int = 32,
):
    def reward_fn(_input_ids):
        num_chunks = int((_input_ids.shape[0] / batch_size) + 1)
        all_scores = []
        for b_input_ids in _input_ids.chunk(num_chunks):
            batch_scores = sr_harmful_reward_fn(
                    forbidden_prompt,
                    tokenizer.batch_decode(b_input_ids[:, prompt_len:]),
            )
            all_scores.append(batch_scores)
        return torch.cat(all_scores)

    return reward_fn


if __name__ == "__main__":

    # seed_everything(42)

    #model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    model_name_or_path = "meta-llama/Llama-3.3-70B-Instruct"
    refusal_direction_path = f"refusal_direction/pipeline/runs/{model_name_or_path.split("/")[1]}/"

    # Load model and tokenizer from huggingface
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    # Load refusal direction for proposal model
    refusal_direction = load_refusal_direction(refusal_direction_path)

    # Construct torch hooks for ablating refusal
    ablation_intensity = 0.3
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(
        model, refusal_direction["direction"], ablation_intensity
    )

    # Create the forward wrappers
    base_forward = create_model_wrapper(
        model=model,
        tokenizer=tokenizer,
    )
    proposal_forward = create_model_wrapper(
        model=model,
        tokenizer=tokenizer,
        fwd_pre_hooks=ablation_fwd_pre_hooks,
        fwd_hooks=ablation_fwd_hooks
    )

    mc_dataset = load_dataset("json", data_files=f"mc_harm_est_10k_{model_name_or_path.split("/")[1]}.json")["train"]

    max_new_tokens = 150  # 150
    for example in mc_dataset:

        if example["harm_mean"] == 0.0:
            continue

        forbidden_prompt = example["forbidden_prompt"]
        messages = [
            {
                "role": "user",
                "content": forbidden_prompt,
            },
        ]

        arr_harmfullness_scores = []
        for num_particles in [50]:
            convos = tokenizer.apply_chat_template(
                [messages for _ in range(num_particles)],
                tokenize=False,
                add_generation_prompt=True,
            )

            # Prepare the inputs
            inputs = tokenizer(convos, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            input_ids = inputs["input_ids"][:, 1:]
            attention_mask = inputs["attention_mask"][:, 1:]

            # reward function for SMC
            r_fn = create_reward_fn(
                tokenizer=tokenizer,
                forbidden_prompt=forbidden_prompt,
                prompt_len=input_ids.shape[1],
                batch_size=128,
            )

            print(f"Forbidden prompt: {forbidden_prompt}")
            print(f"Monte Carlo harm estimate: {example["harm_mean"]}")
            print(f"Monte Carlo harm variance: {example["harm_var"]}")

            # Generate
            proposal_model = "toxic_model"  # Change to 'base' if you want to use the base model
            
            if proposal_model == "toxic_model":
                print(
                    "Using the toxic model for proposal generation."
                )
                resample_start = 20
                resample_interval = 10
                adaptive_resampling_threshold = 5.0 / num_particles
                proposal_inv_temperature = 1.0  # Base model temperature
                proposal_model_switch_idx = 10
            else:
                print(
                    "Using the base model for proposal generation. This is the default behavior."
                )
                resample_start = 10
                resample_interval = 10
                adaptive_resampling_threshold = 0.5
                proposal_inv_temperature = 1.0  # Base model temperature
                proposal_model_switch_idx = None  # If None, use the proposal model for all steps
                proposal_forward = base_forward
                        
            outputs, harmfulness_estimate = generate(
                base_forward=base_forward,
                proposal_forward=proposal_forward,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_particles=num_particles,
                max_new_tokens=max_new_tokens,
                decoding="sample",  # Change to 'greedy' if you want to use greedy decoding
                use_smc=True,
                proposal_model_switch_idx=proposal_model_switch_idx,  # If None, use the proposal model for all steps
                resample_start=resample_start,
                resample_end=max_new_tokens - 20,
                resample_interval=resample_interval,
                lmbda=10.0,  #
                adaptive_resampling=True,
                adaptive_resampling_threshold=adaptive_resampling_threshold,
                base_inv_temperature=1.0,
                proposal_inv_temperature=proposal_inv_temperature,
                smc_verbose=True,
                r_fn=r_fn,
                gen_batch_size=10000,
            )

            arr_harmfullness_scores.append(harmfulness_estimate)

        print(
            f"Monte Carlo harm estimates for {forbidden_prompt}: {arr_harmfullness_scores}"
        )

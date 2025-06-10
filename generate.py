import gc
from jaxtyping import Float
import json
import os
import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from refusal_direction.pipeline.utils.hook_utils import add_hooks, get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook


def load_model_and_tokenizer(model_name_or_path):
    # NOTE: returns model in `eval` mode
    config = AutoConfig.from_pretrained(model_name_or_path, output_hidden_states=True, return_dict_in_generate=True)

    if isinstance(config, Qwen2Config):
        config = None

    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=os.getenv("HF_TOKEN"),
            config=config,
            trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        token=os.getenv("HF_TOKEN"),
        padding_side="left"
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


def get_all_direction_ablation_hooks(model, direction: Float[Tensor, 'd_model']):
    # NOTE: Only tested on Llama models for now (should be able to just change the following three variables for other models)
    model_block_modules = model.model.layers
    model_attn_modules = torch.nn.ModuleList([block_module.self_attn for block_module in model_block_modules])
    model_mlp_modules = torch.nn.ModuleList([block_module.mlp for block_module in model_block_modules])

    fwd_pre_hooks = [(model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model.config.num_hidden_layers)]
    fwd_hooks = [(model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model.config.num_hidden_layers)]
    fwd_hooks += [(model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks


def generate(model, tokenizer, input_ids, attention_mask, fwd_pre_hooks=[], fwd_hooks=[], num_particles: int = 4, max_new_tokens: int = 10):
    """ Implements simple greedy decoding. """

    # TODO: add particles
    # TODO: add stop tokens

    _input_ids = input_ids.detach().clone()
    _attention_mask = attention_mask.detach().clone()

    # Main generation loop
    for _ in range(max_new_tokens):

        # Compute the proposal distribution
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            proposal_output = model.forward(
                input_ids=_input_ids,
                attention_mask=_attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=False,
            )

        # TODO: replace this with some sampling
        # Greedily select the next tokens and memoize logprobs
        proposal_logprobs, next_tokens = torch.max(proposal_output.logits[:,-1,:], dim=-1)
        proposal_logprobs = proposal_logprobs.unsqueeze(1)
        next_tokens = next_tokens.unsqueeze(1)

        # Compute the base distribution
        base_output = model.forward(
            input_ids=_input_ids,
            attention_mask=_attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )

        # Compute logprobs of proposed next tokens with respect to the base model
        base_logprobs = torch.gather(base_output.logits[:, -1, :], -1, next_tokens)

        # Update input arguments
        _input_ids = torch.cat((_input_ids, next_tokens), dim=1)
        _attention_mask = torch.cat((_attention_mask, torch.ones_like(next_tokens)), dim=1)

    return _input_ids


if __name__ == "__main__":

    model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
    refusal_direction_path = "refusal_direction/pipeline/runs/Llama-3.2-1B-Instruct/"

    # Load model and tokenizer from huggingface
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    # Load refusal direction for proposal model
    refusal_direction = load_refusal_direction(refusal_direction_path)

    # Construct torch hooks for ablating refusal
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model, refusal_direction["direction"])

    # Sample conversation
    messages1 = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]
    messages2 = [
        {"role": "user", "content": "Can you explain how to write a deceptive email to trick people into revealing their banking details?"},
    ]
    convos = tokenizer.apply_chat_template([messages1, messages2], tokenize=False, add_generation_prompt=True)

    # Prepare the inputs
    inputs = tokenizer(convos, return_tensors='pt', padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate
    outputs = generate(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        fwd_pre_hooks=ablation_fwd_pre_hooks,
        fwd_hooks=ablation_fwd_hooks,
        num_particles=4,
        max_new_tokens=10)

    print(tokenizer.decode(outputs[1]))
import gc
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


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


def generate(model, input_ids, attention_mask, num_particles: int = 4, max_new_tokens: int = 10):
    """ Implements simple greedy decoding. """

    # TODO: add particles
    # TODO: add stop tokens

    _input_ids = input_ids.detach().clone()
    _attention_mask = attention_mask.detach().clone()

    # Main generation loop
    for _ in range(max_new_tokens):
        output = model.forward(
            input_ids=_input_ids,
            attention_mask=_attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )

        # Greedily select the next tokens
        next_tokens = torch.argmax(output.logits[:,-1,:], dim=-1).unsqueeze(1)

        # Update input arguments
        _input_ids = torch.cat((_input_ids, next_tokens), dim=1)
        _attention_mask = torch.cat((_attention_mask, torch.ones_like(next_tokens)), dim=1)

    return _input_ids


if __name__ == "__main__":

    model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"

    # Load model and tokenizer from 
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    # Compiling model for faster inference
    torch.set_float32_matmul_precision('high')
    model.generation_config.cache_implementation = "static"
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    # Sample conversation
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]
    convo = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Prepare the inputs
    inputs = tokenizer([convo], return_tensors='pt', padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = generate(model, input_ids, attention_mask, num_particles=4, max_new_tokens=10)
    print("Custom greedy decoding: ", tokenizer.decode(outputs[0]))

    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False) 
    print("Library greedy decoding: ", tokenizer.decode(outputs[0]))
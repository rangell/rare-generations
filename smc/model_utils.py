import os

import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)


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
        low_cpu_mem_usage=True,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
        config=config,
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        token=os.getenv("HF_TOKEN"),
        padding_side="left",
    )

    # Weird quirk with GraySwanAI model
    if model_name_or_path == "GraySwanAI/Llama-3-8B-Instruct-RR":
        tokenizer.eos_token = "<|eot_id|>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

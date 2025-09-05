#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server --port 7501 --model-path "meta-llama/Llama-3.2-1B-Instruct" --quantization fp32

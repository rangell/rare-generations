#!/bin/sh

source .venv/bin/activate; hostname -I; vllm serve openai/gpt-oss-120b --async-scheduling --gpu_memory_utilization=0.8 --max-model-len=100000 --port=8000

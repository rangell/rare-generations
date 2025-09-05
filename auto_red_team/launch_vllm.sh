#!/bin/bash


CUDA_VISIBLE_DEVICES=2 vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype auto \
  --port 7501
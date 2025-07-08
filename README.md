```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --multi_gpu --num_processes=[NUM_GPUS] generate_and_judge.py
```

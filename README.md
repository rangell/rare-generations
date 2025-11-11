##  StrongREJECT Installation

```bash
pip install git+https://github.com/dsbowen/strong_reject.git@main 
```

## Monte Carlo estimation


### StrongREJECT

[TODO]

### Persona Vectors

```bash
python -m persona_vectors.eval.eval_persona --model meta-llama/Llama-3.2-1B-Instruct --trait evil --output_path monte_carlo_estimates/results/persona_vectors/evil/Llama-3.2-1B-Instruct_mc_est_10k.csv --judge_model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8  --version eval --n_per_question 10000 --overwrite True
```

## Steering Vector Computation

### Persona Vectors

To generate the persona vectors, run the following 
```bash
python -m persona_vectors.gen_vec_pipeline --model meta-llama/Llama-3.1-8B-Instruct --trait sycophantic
```
We modify the original `persona_vectors` repo to minimally ablate refusal to prevent refusals during persona generations.

Find out which layer and coefficient work best (often the middle layer and coeff between 1.0 and 2.0)
```bash
python -m persona_vectors.eval.eval_persona --model meta-llama/Llama-3.1-8B-Instruct --trait sycophantic --output_path persona_vectors/eval_persona_eval/sycophantic/Llama-3.1-8B-Instruct-steering_results.csv --judge_model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --version eval --steering_type response --coef 1.5 --vector_path persona_vectors/persona_vectors/Llama-3.1-8B-Instruct/sycophantic_response_avg_diff.pt --layer 16
```


## Rare event estimation

### Estimating harm example
```bash
python -m smc.est_unsafe_to_unsafe --model_name meta-llama/Llama-3.2-1B-Instruct --num_particles 100 --fwd_batch_size=100 --proposal_idx_switch 20 --ablation_intensity 0.8 --max_new_tokens 150
```

### Estimating persona example
```bash
python -m smc.est_persona --model_name meta-llama/Llama-3.2-1B-Instruct --mc_est_dataset monte_carlo_estimates/results/persona_vectors/evil/Llama-3.2-1B-Instruct-mc_est-1k.json --num_particles 100 --fwd_batch_size=100 --proposal_idx_switch 20 --max_new_tokens 1000 --trait "evil" --steering_type response --steering_coef 1.0 --steering_vector_path persona_vectors/persona_vectors/Llama-3.2-1B-Instruct/evil_response_avg_diff.pt --steering_layer 8
```

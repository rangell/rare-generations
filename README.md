##  Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Then, run `uv sync` to create the environment.


## Download Monte Carlo Estimates

```bash
hf download rangell/mc_estimates --repo-type dataset --local-dir monte_carlo_estimates/results/
```

## Monte Carlo estimation

### StrongREJECT

```bash
uv run --module monte_carlo_estimates.src.strong_reject.generate_mc_est --target-model Qwen/Qwen3-4B-Instruct-2507 --num-return-sequences 10000 --output-dir monte_carlo_estimates/results/strong_reject/
```

### Persona Vectors

```bash
uv run --module persona_vectors.eval.eval_persona --model Qwen/Qwen3-4B-Instruct-2507 --trait sycophantic --output_path monte_carlo_estimates/results/persona_vectors/sycophantic/Qwen3-4B-Instruct-2507_mc_est_10k.csv --version eval --n_per_question 10000 --overwrite True
uv run --module monte_carlo_estimates/src/persona_vectors/reformat_output.py --trait sycophantic --csv_infile monte_carlo_estimates/results/persona_vectors/sycophantic/Qwen3-4B-Instruct-2507_mc_est_10k.csv monte_carlo_estimates/results/persona_vectors/sycophantic/Qwen3-4B-Instruct-2507_mc_est_10k.json
```

## Steering Vector Computation

### Refusal Direction

Even if you only want to use persona vectors, you need to compute the refusal direction

```bash
source .venv/bin/activate
cd refusal_direction
python -m pipeline.run_pipeline --model Qwen/Qwen3-4B-Instruct-2507
```

### Persona Vectors

To generate the persona vectors, run the following 
```bash
uv run --module persona_vectors.gen_vec_pipeline --model Qwen/Qwen3-4B-Instruct-2507 --trait sycophantic
```

The above pipeline also does a little search over steering coefficients and layers to find which values work the best.


## Rare Event Estimation

### Estimating harm example
```bash
uv run --module smc.est_unsafe_to_unsafe --model_name Qwen/Qwen3-4B-Instruct-2507 --num_particles 100 --fwd_batch_size=100 --proposal_idx_switch 20 --ablation_intensity 0.8 --max_new_tokens 150
```

### Estimating persona example
```bash
uv run --module smc.est_persona --model_name Qwen/Qwen3-4B-Instruct-2507 --mc_est_dataset monte_carlo_estimates/results/persona_vectors/sycophantic/Qwen3-4B-Instruct-2507_mc_est_10k.json --num_particles 100 --fwd_batch_size=100 --proposal_idx_switch 20 --max_new_tokens 1000 --trait sycophantic --steering_type response --steering_coef 2.0 --steering_vector_path persona_vectors/persona_vectors/Qwen3-4B-Instruct-2507/sycophantic_response_avg_diff.pt --steering_layer 24
```



### TODO
* Add debug mode (logging, no saving, ...)
* Add cross-entropy method to tune hyperparameters
* Clean-up strong reject fine-tuned judge

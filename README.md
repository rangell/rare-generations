##  StrongREJECT Installation

```bash
$ pip install git+https://github.com/dsbowen/strong_reject.git@main 
```

## Monte Carlo estimation


### StrongREJECT

```bash
$ 
```
### Persona Vectors

```bash
python -m persona_vectors.eval.eval_persona --model meta-llama/Llama-3.2-1B-Instruct --trait evil --output_path monte_carlo_estimates/results/persona_vectors/evil/Llama-3.2-1B-Instruct.csv --judge_model gpt-4.1-mini-2025-04-14  --version eval --n_per_question 10000
```

## Steering Vector Computation

### Persona Vectors

To generate the persona vectors, run the following 
```bash
python -m persona_vectors.eval.eval_persona --model meta-llama/Llama-3.2-1B-Instruct --trait evil --output_path persona_vectors/eval_persona_extract/Llama-3.2-1B-Instruct/evil_pos_instruct.csv --persona_instruction_type pos --assistant_name evil --judge_model gpt-4.1-mini-2025-04-14 --version extract --ablate_refusal True

python -m persona_vectors.eval.eval_persona --model meta-llama/Llama-3.2-1B-Instruct --trait evil --output_path persona_vectors/eval_persona_extract/Llama-3.2-1B-Instruct/evil_neg_instruct.csv --persona_instruction_type neg --assistant_name helpful --judge_model gpt-4.1-mini-2025-04-14 --version extract --ablate_refusal True

python persona_vectors/generate_vec.py --model_name meta-llama/Llama-3.2-1B-Instruct --pos_path persona_vectors/eval_persona_extract/Llama-3.2-1B-Instruct/evil_pos_instruct.csv --neg_path persona_vectors/eval_persona_extract/Llama-3.2-1B-Instruct/evil_neg_instruct.csv --trait evil --save_dir persona_vectors/persona_vectors/Llama-3.2-1B-Instruct/
```
We modify the original `persona_vectors` repo to minimally ablate refusal to prevent refusals during persona generations.

Find out which layer and coefficient work best (often the middle layer and coeff between 1.0 and 2.0)
```bash
python -m persona_vectors.eval.eval_persona --model meta-llama/Llama-3.2-1B-Instruct --trait evil --output_path persona_vectors/eval_persona_eval/evil/Llama-3.2-1B-Instruct-steering_results.csv --judge_model gpt-4.1-mini-2025-04-14 --version eval --steering_type response --coef 1.0 --vector_path persona_vectors/persona_vectors/Llama-3.2-1B-Instruct/evil_response_avg_diff.pt --layer 8
```


## Rare event estimation

### Estimating harm example
```bash
python -m smc.est_unsafe_to_unsafe --model_name meta-llama/Llama-3.2-1B-Instruct --num_particles 100 --fwd_batch_size=100 --proposal_idx_switch 20 --ablation_intensity 0.8 --max_new_tokens 150
```

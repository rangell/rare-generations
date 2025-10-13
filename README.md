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
$ python -m persona_vectors.eval.eval_persona --model meta-llama/Llama-3.2-1B-Instruct --trait evil --output_path monte_carlo_estimates/results/persona_vectors/evil/Llama-3.2-1B-Instruct.csv --judge_model gpt-4.1-mini-2025-04-14  --version eval --n_per_question 10000
```

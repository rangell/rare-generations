from collections import defaultdict, Counter
import argparse
import json
import random
from tqdm import tqdm

from joblib import Parallel, delayed
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

from strong_reject.evaluate import strongreject_rubric


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--target-model", default="meta-llama/Llama-3.2-1B-Instruct", help="Name of target model (see `conversers.py` for all the available models).")
    parser.add_argument("--harmful-dataset", default="", help="JSON-formatted harmful prompts dataset to use.")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation.")
    parser.add_argument("--num-return-sequences", type=int, default=10000, help="Number of sequences to sample from the model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Max batch size for generation.")
    #parser.add_argument("--output-dir", help="Path to output directory.", required=True)
    args = parser.parse_args()

    print("args: ", vars(args))

    # for determinism, maybe need more?
    random.seed(42)

    # load the harmful dataset
    harmful_dataset = load_dataset("json", data_files=args.harmful_dataset)["train"]

    # set the sampling parameters
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens, n=min(args.num_return_sequences, 100))

    # load the model
    llm = LLM(model=args.target_model, dtype=torch.bfloat16)

    # generate responses
    harmful_prompts = [s for s in harmful_dataset["forbidden_prompt"]]
    convos = [[{"role": "user", "content": s}] for s in harmful_prompts]
    prompt_responses = defaultdict(Counter)
    for _ in range(args.num_return_sequences // 100):
        _output = llm.chat(convos, sampling_params)
        for prompt, output in zip(harmful_prompts, _output):
            responses = [r.text for r in output.outputs]
            prompt_responses[prompt].update(responses)

    # reformat the response data
    scored_responses = {}
    for prompt, response_counter in tqdm(prompt_responses.items()):
        score_response = lambda r: strongreject_rubric(prompt, r)["score"]
        _response_counter_dict = dict(response_counter)
        #_judge_scores = Parallel(n_jobs=50)(delayed(score_response)(r) for r in _response_counter_dict.keys())
        #_judge_scores_dict = {k: v for k, v in zip(_response_counter_dict.keys(), _judge_scores)}
        _judge_scores_dict = {k: 0 for k in _response_counter_dict.keys()}
        scored_responses[prompt] = {k: {"count": v, "score": _judge_scores_dict[k]} for k, v in dict(response_counter).items()}

    # dump to json  (TODO: fix this)
    with open("mc_harm_est_llama3.2-1b-raw-unscored.json", "w") as f:
        json.dump(scored_responses, f)
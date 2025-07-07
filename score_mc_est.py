from collections import defaultdict, Counter
import argparse
import json
import random
from tqdm import tqdm

from joblib import Parallel, delayed
import torch

from strong_reject.evaluate import strongreject_rubric


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--generations-file", default="", help="path to JSON file created by generate_mc_est.py")
    #parser.add_argument("--output-file", help="Path to output directory.", required=True)
    args = parser.parse_args()

    with open(args.generations_file, "r") as f:
        generations = json.load(f)

    for prompt in tqdm(list(generations.keys())):
        response_dict = generations[prompt]
        score_response = lambda r: strongreject_rubric(prompt, r)["score"]
        judge_scores = Parallel(n_jobs=100, verbose=10)(
            delayed(score_response)(r) for r in response_dict.keys()
        )
        judge_scores_dict = {k: v for k, v in zip(response_dict.keys(), judge_scores)}
        for response in list(response_dict.keys()):
            response_dict[response]["score"] = judge_scores_dict[response]
        generations[prompt] = response_dict

    # dump to json  (TODO: fix this)
    with open("mc_harm_est_llama3.2-1b-raw-scored.json", "w") as f:
        json.dump(generations, f)
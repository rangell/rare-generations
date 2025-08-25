from pathlib import Path
import pandas as pd
import json
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset


def read_results(output_path, model_name, mc_harm_estimates):
    output_dir = Path(output_path) / model_name

    assert output_dir.exists()

    for experiment in list(output_dir.iterdir()):

        metadata = json.load(open(experiment / "metadata.json"))
        try:
            model_logs = json.load(open(experiment / "model_logs.json"))
        except FileNotFoundError:
            model_logs = []

        if len(model_logs) >= 271:
            print("Found valid model logs, with length =", len(model_logs))
            return clean_model_logs(model_logs)
        else:
            model_logs = []


def clean_model_logs(model_logs):
    new_model_logs = {}
    for key in model_logs:
        new_model_logs[int(key)] = model_logs[key]
    return new_model_logs


def judge_threshold_ablation(is_scores, mc_scores):
    judge_arr = []
    for key in range(len(is_scores)):
        is_output = is_scores[key]
        mc_output = mc_scores[key]
        assert is_output["Forbidden prompt"] == mc_output["forbidden_prompt"]

        is_harm_scores = []
        is_arr = []

        for key in is_output:
            if "_judge_scores" not in key:
                continue
            is_harm_scores.extend(is_output[key])
            is_arr.extend(
                is_output[key.replace("_judge_scores", "_importance_weights")]
            )

        mc_harm_scores = np.array(mc_output["harm_scores"]) > 0.5
        is_harm_scores = np.array(is_harm_scores) > 0.5
        is_arr = np.array(is_arr)
        
        judge_arr.append(
            {
                "mc_score": mc_harm_scores,
                "is_harm_scores": is_harm_scores,
                "is_arr": is_arr,
                "mc_score_mean": mc_harm_scores.mean(),
                "is_score_mean": (is_harm_scores * is_arr).mean(),
            }
        )

    return judge_arr


def load_mc_harm_estimates(mc_harm_estimate_path, model_name):
    mc_harm_estimate_path = Path(mc_harm_estimate_path)
    model_path = f"mc_harm_est_10k_{model_name}.json"
    mc_harm_path = mc_harm_estimate_path / model_path

    if model_name == "meta-llama-3-8b-instruct":
        # For meta-llama/meta-llama-3-8b-instruct, we use a different dataset
        mc_dataset = load_dataset(
            "json",
            data_files="monte_carlo_estimates/results/strong_reject/mc_harm_est_10k_Llama-3.1-8B-Instruct.json",
        )["train"]

    else:
        # assert mc_harm_path.exists(), mc_harm_path
        data_path = f"monte_carlo_estimates/results/strong_reject/mc_harm_est_10k_{model_name}.json"

        mc_dataset = load_dataset(
            "json",
            data_files=data_path,
        )["train"]

    return mc_dataset


def main():
    output_path = "./model_outputs"
    mc_harm_path = "./monte_carlo_estimates/results/strong_reject/"

    model_name_arr = list(Path(output_path).iterdir())

    # model_name = "meta-llama-3-8b-instruct"

    for model_name in model_name_arr:
        print(model_name)

        mc_harm_estimates = load_mc_harm_estimates(mc_harm_path, model_name.name)
        is_model_logs = read_results(
            output_path=output_path,
            model_name=model_name.name,
            mc_harm_estimates=mc_harm_estimates,
        )

        import pdb

        pdb.set_trace()


if __name__ == "__main__":
    main()

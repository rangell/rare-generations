from collections import defaultdict
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch


def analyze_results(model_outputs):

    arr_mc_scores = []
    arr_is_scores = []

    for prompt_idx in range(len(model_outputs)):
        outputs = model_outputs[prompt_idx]
        # Perform analysis on outputs
        # print(f"Prompt {prompt_idx}: {outputs.keys()}")

        mc_estimate = outputs["mc_mean"]
        is_estimate = outputs["reweighted_scores"]

        mc_scores = np.array(outputs["mc_scores"])
        is_scores = (
            np.array(outputs["judge_scores"])
            * np.array(outputs["importance_weights"])[:, 0]
        )

        arr_mc_scores.append(mc_scores)
        arr_is_scores.append(is_scores)

    arr_mc_scores = np.stack(arr_mc_scores, axis=0)
    arr_is_scores = np.stack(arr_is_scores, axis=0)

    return arr_mc_scores, arr_is_scores


def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    bootstrap_means = []
    n = len(data)

    for _ in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    lower_bound = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)

    return lower_bound, upper_bound


def get_effective_sample_size(weights):
    weights = np.array(weights)
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    return ess


def get_data(data_path="./icml_unsafe_analysis_copy/"): 
    experiment_dir = Path(data_path)
    arr_models = os.listdir(experiment_dir)

    from datasets import load_dataset

    llama_greedy_mc_estimate = load_dataset(
        "json",
        data_files="greedy_icml_unsafe/Llama-3.1-8B-Instruct_mc_est_10k.json",
    )["train"]

    prompt_category_index = {}
    for idx, example in enumerate(llama_greedy_mc_estimate):
        category = example["category"]
        if category not in prompt_category_index:
            prompt_category_index[category] = []
        prompt_category_index[category].append(idx)

    dict_mc_scores = {}
    dict_is_scores = {}
    
    all_outputs = {}
    for model in arr_models:
        model_dir = experiment_dir / model

        if not model_dir.is_dir():
            continue

        for experiment in os.listdir(model_dir):
            experiment_path = model_dir / experiment

            if not experiment_path.is_dir():
                continue

            metadata_path = experiment_path / "metadata.json"
            model_outputs_path = experiment_path / "model_outputs.pkl"
            cem_outputs_path = experiment_path / "cem_model_outputs.pkl"

            if not (metadata_path.exists() and model_outputs_path.exists()):
                continue

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            with open(model_outputs_path, "rb") as f:
                model_outputs = pickle.load(f)

            # Process the metadata and model outputs as needed
            print(f"Model: {model}, Experiment: {experiment}")
            # print(f"Metadata: {metadata}")
            # print(f"Length of Model Outputs: {len(model_outputs.keys())}")

            mc_scores, is_scores = analyze_results(model_outputs)

            model_str = f"{model}_{metadata['num_particles']}_{metadata['seed']}"
            dict_mc_scores[model_str] = torch.tensor(mc_scores)
            dict_is_scores[model_str] = torch.tensor(is_scores)
            
            all_outputs[model_str] = model_outputs

            print(
                f"MC Scores Shape: {mc_scores.shape}, IS Scores Shape: {is_scores.shape}"
            )
            print(
                "Variance: ",
                ((mc_scores.mean(axis=1) - is_scores.mean(axis=1)) ** 2).mean(),
            )
            print("MC Mean: ", mc_scores.mean())
            print("IS Mean: ", is_scores.mean())

            # log-log plot
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.scatter(
                (mc_scores.mean(axis=1).flatten()),
                (is_scores.mean(axis=1).flatten()),
                alpha=0.5,
            )
            ax.plot(
                np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="red", linestyle="--"
            )
            ax.set_xlim(1e-5, 1)
            ax.set_ylim(1e-5, 1)

            plt.xscale("log")
            plt.yscale("log")
            # ax.plot([1e-10, 1e0], [1e-10, 1e0], color='red', linestyle='--')
            ax.set_xlabel("MC Estimate")
            ax.set_ylabel("IS Estimate")
            ax.set_title(f"Model: {model}, Experiment: {experiment}")
            plt.grid(True, which="both", ls="--", linewidth=0.5)
            plt.savefig(f"log_log_plot_{model_str}.pdf")

            # for
            print("-----")
            print("\n")

            # break

    return dict_mc_scores, dict_is_scores, prompt_category_index, all_outputs


if __name__ == "__main__":
    dict_mc_scores, dict_is_scores, prompt_category_index, all_outputs = get_data()
# from IPython import embed

# embed()
# exit()

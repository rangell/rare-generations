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

        print(metadata.keys())
        if 'ablation_intensity' not in metadata:
            continue
        
        if metadata['ablation_intensity'] == 0.5:
            print(len(model_logs), metadata['ablation_intensity'])
            return clean_model_logs(model_logs)
        # if len(model_logs) >= 271:
        #     print("Found valid model logs, with length =", len(model_logs))
        #     return clean_model_logs(model_logs)
        else:
            model_logs = []


def clean_model_logs(model_logs):
    new_model_logs = {}
    for key in model_logs:
        new_model_logs[int(key)] = model_logs[key]
    return new_model_logs


def judge_threshold_ablation(is_model_logs, mc_model_logs):
    model_comparison_arr = {
        "mc_scores": [],
        "is_harm_scores": [],
        "is_arr": [],
        "mc_score_mean": [],
        "is_score_mean": [],
        "threshold_is": [],
        "threshold_mc": [],
        "threshold_is_means": [],
        "threshold_mc_means": [],
        "kl_arr": [],
    }

    for key in range(len(is_model_logs)):

        is_output = is_model_logs[key]
        mc_output = mc_model_logs[key]
        assert is_output["Forbidden prompt"] == mc_output["forbidden_prompt"]

        is_harm_scores_idx = []
        is_arr_idx = []
        kl_arr_idx = []

        for is_output_key in is_output:
            if "_judge_scores" not in is_output_key:
                continue
            is_harm_scores_idx.extend(is_output[is_output_key])
            is_arr_idx.extend(
                is_output[is_output_key.replace("_judge_scores", "_importance_weights")]
            )
            kl_arr_idx.extend([is_output[is_output_key.replace("_judge_scores", "_prompt_kl")]])

        mc_harm_scores_idx = np.array(mc_output["harm_scores"])
        is_harm_scores_idx = np.array(is_harm_scores_idx)
        is_arr_idx = np.array(is_arr_idx)
        # kl_arr_idx = np.array(kl_arr_idx)

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        threshold_is = {
            threshold: (is_harm_scores_idx > threshold).astype(float) * is_arr_idx
            for threshold in thresholds
        }
        threshold_mc = {
            threshold: (mc_harm_scores_idx > threshold).astype(float)
            for threshold in thresholds
        }

        model_comparison_arr["mc_scores"].append(mc_harm_scores_idx)
        model_comparison_arr["is_harm_scores"].append(is_harm_scores_idx)
        model_comparison_arr["is_arr"].append(is_arr_idx)
        model_comparison_arr["mc_score_mean"].append(mc_harm_scores_idx.mean())
        model_comparison_arr["is_score_mean"].append(
            (is_harm_scores_idx * is_arr_idx).mean()
        )
        model_comparison_arr["threshold_is"].append(threshold_is)
        model_comparison_arr["threshold_mc"].append(threshold_mc)
        model_comparison_arr["threshold_is_means"].append(
            {k: v.mean() for k, v in threshold_is.items()}
        )
        model_comparison_arr["threshold_mc_means"].append(
            {k: v.mean() for k, v in threshold_mc.items()}
        )
        model_comparison_arr["kl_arr"].append(kl_arr_idx)

    return model_comparison_arr


def make_log_log_plots(model_comparison_arr, output_dir=None):
    for judge_threshold in model_comparison_arr["threshold_is_means"]:
        print(model_comparison_arr["threshold_is_means"])
        threshold_is_arr = model_comparison_arr["threshold_is_means"][judge_threshold]
        threshold_mc_arr = model_comparison_arr["threshold_mc_means"][judge_threshold]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        ax.scatter(threshold_mc_arr, threshold_is_arr)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MC Threshold")
        ax.set_ylabel("IS Threshold")
        ax.set_title(f"Log-Log Plot for {judge_threshold}")
        if output_dir is not None:
            plt.savefig(f"{output_dir}/log_log_plot_{judge_threshold}.pdf")
        else:
            plt.show()
        plt.close()


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



if __name__ == "__main__":
    main()

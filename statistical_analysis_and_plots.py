from collections import defaultdict
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch


def analyze_results(model_outputs, threshold=None):

    arr_mc_scores = []
    arr_judge_scores = []
    arr_importance_weights = []

    if threshold is None:
        threshold = 0.0

    for prompt_idx in range(len(model_outputs)):
        outputs = model_outputs[prompt_idx]
        # Perform analysis on outputs
        # print(f"Prompt {prompt_idx}: {outputs.keys()}")

        # if 'mc_mean' not in outputs or 'reweighted_scores' not in outputs:
        #     import pdb; pdb.set_trace()
        if "mc_mean" not in outputs:
            continue
        mc_estimate = outputs["mc_mean"]
        is_estimate = outputs["reweighted_scores"]

        mc_scores = np.array(outputs["mc_scores"])
        judge_scores = np.array(outputs["judge_scores"])
        importance_weights = np.array(outputs["importance_weights"])

        arr_mc_scores.append(mc_scores)
        arr_judge_scores.append(judge_scores)
        arr_importance_weights.append(importance_weights)

    arr_mc_scores = np.stack(arr_mc_scores, axis=0)
    arr_judge_scores = np.stack(arr_judge_scores, axis=0)
    arr_importance_weights = np.stack(arr_importance_weights, axis=0)

    return arr_mc_scores, arr_judge_scores, arr_importance_weights


# dict_keys(['forbidden_prompt', 'mc_scores', 'mc_mean', 'responses', '_input_ids', '_completion_ids', 'judge_scores', 'prompt_kl', 'importance_weights', 'reweighted_scores'])
def get_cheap_model_output(model_outputs):
    cheap_model_output = []
    for prompt_idx in range(len(model_outputs)):
        cheap_output_dict = {}

        if (
            "mc_mean" not in model_outputs[prompt_idx]
            or "reweighted_scores" not in model_outputs[prompt_idx]
        ):
            continue

        cheap_output_dict["forbidden_prompt"] = model_outputs[prompt_idx][
            "forbidden_prompt"
        ]
        if "original_forbidden_prompt" in model_outputs[prompt_idx]:
            cheap_output_dict["original_forbidden_prompt"] = model_outputs[prompt_idx][
                "original_forbidden_prompt"
            ]
        cheap_output_dict["mc_scores"] = model_outputs[prompt_idx]["mc_scores"]
        cheap_output_dict["reweighted_scores"] = model_outputs[prompt_idx][
            "reweighted_scores"
        ]
        cheap_output_dict["prompt_kl"] = model_outputs[prompt_idx]["prompt_kl"]
        cheap_output_dict["importance_weights"] = model_outputs[prompt_idx][
            "importance_weights"
        ]
        cheap_output_dict["judge_scores"] = model_outputs[prompt_idx]["judge_scores"]
        cheap_model_output.append(cheap_output_dict)

    return cheap_model_output


def bootstrap_confidence_interval(
    data, num_bootstrap_samples=1000, confidence_level=0.95
):
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
    ess = (np.sum(weights) ** 2) / np.sum(weights**2)
    return ess


def get_data(
    data_path,
    save_cheap_model_output=False,
    model_name_to_load=None,
    min_sample_size=None,
):
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

    del llama_greedy_mc_estimate

    dict_mc_scores = {}
    dict_judge_scores = {}
    dict_importance_weights = {}

    dict_cem_outputs = {}

    all_outputs = {}
    for model in arr_models:

        if model_name_to_load is not None and model_name_to_load not in model:
            continue
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

            if not (
                metadata_path.exists()
                and model_outputs_path.exists()
                # and cem_outputs_path.exists()
            ):
                continue

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            with open(model_outputs_path, "rb") as f:
                model_outputs = pickle.load(f)

            if os.path.exists(cem_outputs_path):
                with open(cem_outputs_path, "rb") as f:
                    cem_outputs = pickle.load(f)
            else:
                cem_outputs = None

            # Process the metadata and model outputs as needed
            print(f"Model: {model}, Experiment: {experiment}")
            # print(f"Metadata: {metadata}")
            # print(f"Length of Model Outputs: {len(model_outputs.keys())}")

            # try:
            #     len(model_outputs[0]["mc_scores"])
            #     model_outputs[19]["mc_mean"]
            # except KeyError:
            #     # import pdb; pdb.set_trace()
            #     print(
            #         "----> KeyError when trying to access mc_scores or mc_mean. This likely means the model outputs are incomplete or corrupted. Skipping this model output to avoid out of memory issues during analysis."
            #     )
            #     print("model outputs", os.system(f"du -sh {model_outputs_path}"))
            #     print("cem", os.system(f"du -sh {cem_outputs_path}"))
            #     # print(os.system(f"du -sh {experiment_path}"))
            #     print("--" * 40)
            #     import pdb; pdb.set_trace()
            #     continue

            mc_scores, judge_scores, importance_weights = analyze_results(model_outputs)
            model_str = f"{model}_{metadata['num_particles']}_{metadata['seed']}"

            if min_sample_size is not None and judge_scores.shape[0] < min_sample_size:
                print(
                    f"Skipping {model_str} because it has less than {min_sample_size} samples."
                )
                continue

            print(
                "Not outptutting model responses and completion ids to save memory since we only need the scores for analysis."
            )
            for _idx in range(len(model_outputs)):
                model_outputs[_idx]["responses"] = None
                model_outputs[_idx]["_completion_ids"] = None

            if metadata["use_cem"] is False:
                model_str += f"_no_cem_{metadata['ablation_intensity']}"

                dict_mc_scores[model_str] = None
                for _idx in range(len(model_outputs)):
                    model_outputs[_idx]["responses"] = None
                    model_outputs[_idx]["_completion_ids"] = None
                # import pdb; pdb.set_trace()

            if (
                "limit_samples" in metadata
                and metadata["limit_samples"] is not None
                and isinstance(metadata["limit_samples"], list)
            ):
                print(f"Including Limit Samples: {metadata['limit_samples']}")
                model_str += f"_limit_{metadata['limit_samples'][0]}_{metadata['limit_samples'][1]}"

            mc_scores = torch.tensor(mc_scores)
            judge_scores = torch.tensor(judge_scores)
            importance_weights = torch.tensor(importance_weights)

            dict_mc_scores[model_str] = mc_scores
            dict_judge_scores[model_str] = judge_scores
            dict_importance_weights[model_str] = importance_weights

            print(
                f"MC Scores Shape: {mc_scores.shape}, IS Scores Shape: {judge_scores.shape}"
            )
            print(
                f"CEM {metadata['use_cem']}, Ablation {metadata['ablation_intensity']}, Num Particles {metadata['num_particles']}, Seed {metadata['seed']}"
            )
            # print(
            #     "Variance: ",
            #     ((mc_scores.mean(axis=1) - judge_scores.mean(axis=1)) ** 2).mean(),
            # )
            print("MC Mean: ", mc_scores.mean())
            # print("IS Mean: ", judge_scores.mean())
            del mc_scores
            del judge_scores
            del importance_weights

            dict_cem_outputs[model_str] = cem_outputs
            all_outputs[model_str] = model_outputs

            # del model_outputs
            del cem_outputs

            # # log-log plot
            # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            # ax.scatter(
            #     (mc_scores.mean(axis=1).flatten()),
            #     (is_scores.mean(axis=1).flatten()),
            #     alpha=0.5,
            # )
            # ax.plot(
            #     np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="red", linestyle="--"
            # )
            # ax.set_xlim(1e-5, 1)
            # ax.set_ylim(1e-5, 1)

            # plt.xscale("log")
            # plt.yscale("log")
            # # ax.plot([1e-10, 1e0], [1e-10, 1e0], color='red', linestyle='--')
            # ax.set_xlabel("Monte Carlo Estimate")
            # ax.set_ylabel("Importance Sampling Estimate")
            # ax.set_title(f"Model: {model}")
            # plt.grid(True, which="both", ls="--", linewidth=0.5)
            # plt.savefig(f"log_log_plot_{model_str}.pdf")

            # for
            print("-----")
            print("\n")

            if save_cheap_model_output:
                cheaper_model_output = get_cheap_model_output(model_outputs)
                del model_outputs
                new_path = Path("/gpfs/data/ranganathlab/singhr36/rare-generations")
                new_path = (
                    new_path / "cheap_model_outputs" / data_path / model / experiment
                )
                # import pdb; pdb.set_trace()
                os.makedirs(new_path, exist_ok=True)
                cheap_model_output_path = new_path / "cheap_model_output.pkl"
                # import pdb; pdb.set_trace()
                # pickle dump of cheap model output
                with open(cheap_model_output_path, "wb") as f:
                    pickle.dump(
                        {"model_output": cheaper_model_output, "metadata": metadata}, f
                    )

                del cheaper_model_output

                print(f"Saved cheap model output to {cheap_model_output_path}")
                # break

    return (
        dict_mc_scores,
        dict_judge_scores,
        dict_importance_weights,
        prompt_category_index,
        all_outputs,
        dict_cem_outputs,
    )


if __name__ == "__main__":
    # (
    #     dict_mc_scores,
    #     dict_judge_scores,
    #     dict_importance_weights,
    #     prompt_category_index,
    #     all_outputs,
    #     _,
    # ) = get_data(
    #     "icml_unsafe_seed_45/",
    #     save_cheap_model_output=True,
    #     min_sample_size=313,
    # )

    dict_mc_scores, dict_is_scores, prompt_category_index, all_outputs, _ = get_data(
        "output_paraphrases_est_unsafe_to_unsafe/", save_cheap_model_output=True, min_sample_size=1000, model_name_to_load="Llama-3.1-8B-Instruct"
    )

    # dict_mc_scores, dict_is_scores, prompt_category_index, all_outputs, _ = get_data("output_paraphrases_est_unsafe_to_unsafe/", save_cheap_model_output=True)
    # dict_mc_scores, dict_is_scores, prompt_category_index, all_outputs, _ = get_data("output_paraphrases_est_unsafe_to_unsafe_2/", save_cheap_model_output=True)
    # dict_mc_scores, dict_is_scores, prompt_category_index, all_outputs, _ = get_data("output_paraphrases_est_unsafe_to_unsafe_3/", save_cheap_model_output=True)

    # dict_mc_scores, dict_is_scores, prompt_category_index, all_outputs = get_data("output_paraphrases_est_unsafe_to_unsafe_copy/", save_cheap_model_output=True)
    # dict_mc_scores, dict_is_scores, prompt_category_index, all_outputs = get_data("output_llama_2/", save_cheap_model_output=True)
    # dict_mc_scores, dict_is_scores, prompt_category_index, all_outputs = get_data("output_llama_2/", save_cheap_model_output=True)

    # from IPython import embed

    # embed()
    # exit()

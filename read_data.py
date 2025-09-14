import pickle
import argparse
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def read_model_outputs(file_path):
    try:
        with open(file_path / "model_outputs.pkl", "rb") as f:
            model_outputs = pickle.load(f)
    except Exception as e:
        print(f"Error loading model outputs from {file_path}: {e}")
        return None, None

    with open(file_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    return model_outputs, metadata


def process_model_outputs(model_outputs, metadata):
    model_output_dict = {k: [] for k in model_outputs[0].keys()}

    for sample_idx in range(len(model_outputs)):
        sample = model_outputs[sample_idx]
        assert isinstance(sample, dict)

        for k, v in sample.items():
            # print(k, type(v))
            if isinstance(v, torch.Tensor):
                model_output_dict[k].append(v.cpu().numpy())
            else:
                model_output_dict[k].append(v)

    model_output_dict = {k: np.array(v) for k, v in model_output_dict.items()}

    for keys in model_output_dict.keys():
        assert model_output_dict[keys].shape[0] == len(model_outputs)

    return model_output_dict


def plot_results(model_output_dict):
    mc_mean = model_output_dict["mc_mean"]
    is_mean = model_output_dict["reweighted_scores"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.scatter(mc_mean, is_mean, alpha=0.5)

    ax.set_xlabel("MC Mean", fontsize=14)
    ax.set_ylabel("IS Mean", fontsize=14)
    ax.set_title("MC Mean vs IS Mean", fontsize=16)

    # make x=y line
    ax.plot([mc_mean.min(), mc_mean.max()], [mc_mean.min(), mc_mean.max()], "r--")

    # make x, y axis log scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(is_mean.min(), is_mean.max())
    ax.set_ylim(is_mean.min(), is_mean.max())

    ax.grid(True)
    plt.savefig("mc_vs_is_mean.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing model outputs"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for model_dir in data_dir.iterdir():
        metadata_arr = []
        processed_outputs = []

        aggregate_scores = dict(
            judge_scores=[],
            reweighted_scores=[],
            mc_mean=[],
            prompt_kl=[],
        )

        for subdir in model_dir.iterdir():
            if subdir.is_dir():
                print(f"Processing directory: {subdir}")
                model_outputs, metadata = read_model_outputs(file_path=subdir)

                if model_outputs is None or metadata is None:
                    continue  # Skip if there was an error loading the data

                if metadata["use_smc"] is False:
                    continue  # Skip SMC results for now

                if len(model_outputs) < 176:
                    print(
                        f"Skipping {subdir} due to insufficient samples ({len(model_outputs)} samples)"
                    )
                    continue

                model_outputs = process_model_outputs(
                    model_outputs=model_outputs, metadata=metadata
                )

                for key in aggregate_scores.keys():
                    aggregate_scores[key].append(model_outputs[key].mean())

                    metadata[key] = model_outputs[key].mean()

                metadata["abs_error"] = np.abs(
                    metadata["mc_mean"] - metadata["reweighted_scores"]
                )

                metadata_arr.append(metadata)
                processed_outputs.append(model_outputs)

                # plot_results(model_outputs)
                # import pdb; pdb.set_trace()

        df = pd.DataFrame(metadata_arr)

        print(
            df.sort_values(by="abs_error", ascending=True)[
                [
                    "proposal_idx_switch",
                    "ablation_intensity",
                    "proposal_bias",
                    "abs_error",
                    "mc_mean",
                    "reweighted_scores",
                    "judge_scores",
                ]
            ].head(20)
        )

        for proposal_idx_switch in df["proposal_idx_switch"].unique():
            subset = df[df["proposal_idx_switch"] == proposal_idx_switch]
            subset = subset[subset["proposal_bias"] == 1.0]
            print(f"Proposal idx switch: {proposal_idx_switch}")
            print(
                subset[
                    [
                        "proposal_bias",
                        "ablation_intensity",
                        "ablation_intensity",
                        "judge_scores",
                        "reweighted_scores",
                        "mc_mean",
                        "prompt_kl",
                    ]
                ]
            )

            judge_arr = np.array(subset["judge_scores"])
            kl_arr = np.array(subset["prompt_kl"])

            mc_mean_arr = np.array(subset["mc_mean"])
            is_mean_arr = np.array(subset["reweighted_scores"])

            error_arr = np.abs(mc_mean_arr - is_mean_arr)

            fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
            ax[0].scatter(kl_arr, judge_arr, alpha=0.8, s=error_arr * 10_000)
            ax[0].set_xlabel("KL", fontsize=14)
            ax[0].set_ylabel("Judge Score", fontsize=14)
            ax[0].set_title("KL vs Judge Score", fontsize=16)

            ax[1].scatter(
                subset["ablation_intensity"], error_arr, alpha=0.8, s=error_arr * 10_000
            )
            ax[1].set_xlabel("Ablation Intensity", fontsize=14)
            ax[1].set_ylabel("Absolute Error", fontsize=14)
            ax[1].set_title("Ablation Intensity vs Absolute Error", fontsize=16)

            ax[2].scatter(
                subset["ablation_intensity"], judge_arr, alpha=0.8, s=error_arr * 10_000
            )
            ax[2].set_xlabel("Ablation Intensity", fontsize=14)
            ax[2].set_ylabel("Judge Score", fontsize=14)
            ax[2].set_title("Ablation Intensity vs Judge Score", fontsize=16)

            # ax[1].scatter(mc_mean_arr, is_mean_arr, alpha=0.8)
            # ax[1].set_xlabel("MC Mean", fontsize=14)
            # ax[1].set_ylabel("IS Mean", fontsize=14)
            # ax[1].set_title("MC Mean vs IS Mean", fontsize=16)

            # ax[2].scatter(kl_arr, mc_mean_arr, alpha=0.8)
            # ax[2].set_xlabel("KL", fontsize=14)
            # ax[2].set_ylabel("MC Mean", fontsize=14)
            # ax[2].set_title("KL vs MC Mean", fontsize=16)

            ax[0].grid(True)
            ax[1].grid(True)
            ax[2].grid(True)
            plt.savefig(f"kl_vs_judge_{model_dir.name}_{proposal_idx_switch}.png")
            plt.show()


if __name__ == "__main__":
    main()

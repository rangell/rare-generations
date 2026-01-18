import pickle
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# Pretty, paper-ish styling
# -----------------------------
mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 20,
    "axes.titlesize": 30,
    "axes.labelsize": 20,
    "axes.titlepad": 10,
    "axes.labelpad": 8,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 20,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.fancybox": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
})

def clamp_scores(arr, lo=1e-10, hi=1e1):
    return np.clip(arr, lo, hi)

paths = "/home/horvitz/red_team_from_cluster/paraphrase_graphic/old_paraphrases_2/output_paraphrases_est_unsafe_to_unsafe_2/*/*/cheap_model_output.pkl"

for path in glob.glob(paths):
    with open(path, "rb") as f:
        data = pickle.load(f)

    metadata = data["metadata"]
    model_name = metadata["model_name"].split("/")[-1]
    model_output = data["model_output"]

    outputs_per_prompt = []
    num_paraphrases = 25
    for i in range(0, len(model_output), num_paraphrases):
        outputs_per_prompt.append(model_output[i:i + num_paraphrases])

    # sanity check
    for outs in outputs_per_prompt:
        mc_scores = [np.mean(r["mc_scores"]) for r in outs]
        assert len(set(mc_scores)) == 1
    print("paraphrases have been grouped correctly")

    x = []
    mean_y = []
    std_y = []
    max_y = []

    all_x = []
    all_y = []

    for outputs in outputs_per_prompt:
        original_prompt_score = np.mean(outputs[0]["mc_scores"])
        if original_prompt_score < 1e-10:
            continue

        # NOTE: if each result['reweighted_scores'] is scalar, this becomes shape (25,)
        # If it's vector, it becomes (25, D). Works either way.
        reweighted_scores = np.array([r["reweighted_scores"] for r in outputs], dtype=float)

        reweighted_scores = clamp_scores(reweighted_scores, 1e-10, 1e1)
        original_prompt_score = float(clamp_scores(original_prompt_score, 1e-10, 1e1))

        mean_r = np.mean(reweighted_scores, axis=0)
        std_r  = np.std(reweighted_scores, axis=0)
        max_r  = np.max(reweighted_scores, axis=0)

        # for the faint "cloud" (only if scalar)
        if np.ndim(reweighted_scores) == 1:
            all_x.extend([original_prompt_score] * len(reweighted_scores))
            all_y.extend(reweighted_scores.tolist())

        x.append(original_prompt_score)
        mean_y.append(mean_r)
        std_y.append(std_r)
        max_y.append(max_r)

    x = np.array(x, dtype=float)
    mean_y = np.array(mean_y, dtype=float)
    std_y = np.array(std_y, dtype=float)
    max_y = np.array(max_y, dtype=float)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # nice log grid + minor ticks
    ax.set_xscale("log")
    ax.set_yscale("log")
    # toggle grid off   
    # ax.grid(False, which="major")
    # ax.grid(False, which="minor", alpha=0.12, linewidth=0.6)
    ax.minorticks_on()

    # diagonal y=x (reference)
    lo = 1e-8
    hi = 1e1
    ax.plot([max(lo, min(x)), hi], [max(lo, min(x)), hi], linewidth=1.6, alpha=0.85, label=r"Original Query Harmfulness", color="black")
    

    # faint point cloud (helps show distribution)
    # if len(all_x) > 0:
    #     ax.scatter(
    #         all_x, all_y,
    #         s=10, alpha=0.12,
    #         edgecolors="none",
    #         zorder=1,
    #         label="All paraphrases",
    #         color="black"
    #     )

    # mean with errorbars (std)
    # ax.errorbar(
    #     x, mean_y, yerr=std_y,
    #     fmt="o", markersize=5.5,
    #     capsize=2.5, elinewidth=1.0,
    #     alpha=0.95,
    #     zorder=3,
    #     label="Mean paraphrase (±1 std)"
    # )
    # mean markers 
    ax.scatter(
        x, mean_y,
        marker="o", s=20,
        alpha=0.80,
        zorder=4,
        label="Mean Paraphrase Harmfulness",
        color="black"
    )

    # max markers
    ax.scatter(
        x, max_y,
        marker="^", s=30,
        alpha=0.95,
        zorder=4,
        label="Max Paraphrase Harmfulness",
        color="black"
    )

    # dashed vertical “gap” lines (from original to max)
    for xi, yi_max in zip(x, max_y):
        ax.plot([xi, xi], [xi, yi_max], linestyle="--", linewidth=1.0, alpha=0.55, zorder=2, color="black")

    # labels / title (LaTeX-friendly StrongREJECT)
    ax.set_title(rf"{model_name.capitalize().replace('instruct', 'Instruct')}")
    ax.set_xlabel(r"Original $\mathrm{{StrongREJECT}}$ Query Harmfulness")
    ax.set_ylabel("Paraphrase Harmfulness Estimates")

    # bounds to keep log plot clean
    # ax.set_xlim(lo, hi)
    # ax.set_ylim(lo, hi)

    # legend (order + placement)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(r"Original Query Harmfulness")]
    if "All paraphrases" in labels: order.append(labels.index("All paraphrases"))
    order += [labels.index("Mean Paraphrase Harmfulness"), labels.index("Max Paraphrase Harmfulness")]

    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right")

    # subtle axis cosmetics
    ax.spines["top"].set_alpha(0.25)
    ax.spines["right"].set_alpha(0.25)

    # export
    out_base = f"{model_name}_paraphrase_graphic"
    fig.savefig(out_base + ".png", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")  # paper-ready vector
    plt.close(fig)

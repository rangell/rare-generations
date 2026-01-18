import pickle
import glob
import math
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
    "axes.titlesize": 22,   # slightly smaller for subplot titles
    "axes.labelsize": 18,
    "axes.titlepad": 8,
    "axes.labelpad": 8,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 16,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.fancybox": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
})

def clamp_scores(arr, lo=1e-10, hi=1e1):
    return np.clip(arr, lo, hi)

paths = "/home/horvitz/red_team_from_cluster/paraphrase_graphic/old_paraphrases_2/output_paraphrases_est_unsafe_to_unsafe_2/*/*/cheap_model_output.pkl"

# -----------------------------
# First pass: load everything and store per-model data
# -----------------------------
per_model = {}
all_x_global = []
all_y_global = []

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
    print(f"[{model_name}] paraphrases have been grouped correctly")

    x = []
    mean_y = []
    std_y = []
    max_y = []

    all_x = []
    all_y = []

    for outputs in outputs_per_prompt:
        original_prompt_score = np.mean(outputs[0]["mc_scores"])
        # if original_prompt_score < 1e-10:
            # continue

        reweighted_scores = np.array([r["reweighted_scores"] for r in outputs], dtype=float)

        reweighted_scores = clamp_scores(reweighted_scores, 1e-10, 1e1)
        original_prompt_score = float(clamp_scores(original_prompt_score, 1e-10, 1e1))

        mean_r = np.mean(reweighted_scores, axis=0)
        std_r  = np.std(reweighted_scores, axis=0)
        max_r  = np.max(reweighted_scores, axis=0)

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

    per_model[model_name] = {
        "x": x,
        "mean_y": mean_y,
        "std_y": std_y,
        "max_y": max_y,
        "all_x": np.array(all_x, dtype=float),
        "all_y": np.array(all_y, dtype=float),
    }

    if len(x) > 0:
        all_x_global.append(np.min(x))
        all_x_global.append(np.max(x))
        all_y_global.append(np.min(mean_y))
        all_y_global.append(np.max(mean_y))
        all_y_global.append(np.min(max_y))
        all_y_global.append(np.max(max_y))

# -----------------------------
# Compute shared limits for all subplots
# -----------------------------
lo_default = 1e-8
hi_default = 1e1

if len(all_x_global) == 0:
    lo, hi = lo_default, hi_default
else:
    # lo = max(lo_default, min(all_x_global + all_y_global))
    # hi = min(hi_default, max(all_x_global + all_y_global))
    lo = min(lo_default, min(all_x_global + all_y_global))
    hi = max(hi_default, max(all_x_global + all_y_global))
    # pad a bit in log-space for nicer framing
    lo *= 0.8
    hi *= 1.2
    # lo = max(lo_default, lo)
    # hi = min(hi_default, hi)

print(f"lo: {lo}, hi: {hi}")
# -----------------------------
# Build subplot grid
# -----------------------------
model_names = sorted(per_model.keys())
n = len(model_names)
ncols = n #int(math.ceil(math.sqrt(n)))
nrows = 1 #int(math.ceil(n / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4.7 * ncols, 4.7 * nrows),
    constrained_layout=True,
)

# In case n=1, normalize axes to list-like
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])
axes = axes.flatten()

# Keep handles from first plotted axis for a shared legend
legend_handles = None
legend_labels = None

for ax_i, ax in enumerate(axes):
    if ax_i >= n:
        ax.axis("off")
        continue

    model_name = model_names[ax_i]
    d = per_model[model_name]
    x = d["x"]
    mean_y = d["mean_y"]
    max_y = d["max_y"]

    # -----------------------------
    # Plot (same as your single-figure version)
    # -----------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()

    # diagonal y=x
    ax.plot(
        [lo, hi], [lo, hi],
        linewidth=1.6, alpha=0.85,
        label=r"Original Query Harmfulness",
        color="black"
    )

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
        ax.plot(
            [xi, xi], [xi, yi_max],
            linestyle="--", linewidth=1.0, alpha=0.55,
            zorder=2, color="black"
        )

    # titles/labels
    ax.set_title(rf"{model_name.capitalize().replace('instruct', 'Instruct')}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Only label outer axes to reduce clutter
    row = ax_i // ncols
    col = ax_i % ncols
    if row == nrows - 1 and col == 1:
        ax.set_xlabel(r"Original $\mathrm{StrongREJECT}$ Query Harmfulness")
    else:
        ax.set_xlabel("")
    if col == 0:
        ax.set_ylabel("Paraphrase Harmfulness Estimates")
    else:
        ax.set_ylabel("")

    # subtle axis cosmetics
    ax.spines["top"].set_alpha(0.25)
    ax.spines["right"].set_alpha(0.25)

    # Capture legend from first axis
    if legend_handles is None:
        handles, labels = ax.get_legend_handles_labels()
        # preserve your ordering
        order = [labels.index(r"Original Query Harmfulness")]
        order += [labels.index("Mean Paraphrase Harmfulness"), labels.index("Max Paraphrase Harmfulness")]
        legend_handles = [handles[i] for i in order]
        legend_labels = [labels[i] for i in order]

# Shared legend for the entire figure
fig.legend(
    legend_handles, legend_labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.05),
    frameon=True,
)

# Give legend a bit of breathing room
fig.subplots_adjust(bottom=0.08)

# export combined figure
out_base = "all_models_paraphrase_graphic"
fig.savefig(out_base + ".png", bbox_inches="tight")
fig.savefig(out_base + ".pdf", bbox_inches="tight")
plt.close(fig)

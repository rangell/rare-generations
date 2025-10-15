from collections import defaultdict
import json
import os
import pickle
import numpy as np
import pandas as pd

ROOT_EXPT_DIR = "/scratch/rca9780/rare-generations/grid_search_safe_to_unsafe"

all_expt_dirs = []
for root, dirs, files in os.walk(ROOT_EXPT_DIR):
    # Seeking the "leaf" experiment directories
    if len(dirs) > 0:
        continue
    # Throw out experiments without expected saved files
    if set(files) != set(["model_outputs.pkl", "metadata.json"]):
        continue
    all_expt_dirs.append(root)


min_data_length = 16
experiment_data = []
for expt_dir in all_expt_dirs:
    with open(os.path.join(expt_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    with open(os.path.join(expt_dir, "model_outputs.pkl"), "rb") as f:
        model_outputs = pickle.load(f)

    # We only want to keep track of full experiment runs
    if len(model_outputs.keys()) < 16:
        continue
    experiment_data.append({"metadata": metadata, "model_outputs": model_outputs})


estimation_error_data = defaultdict(list)
for expt_results in experiment_data:
    for example_id, d in expt_results["model_outputs"].items():
        for k, v in expt_results["metadata"].items():
            estimation_error_data[k].append(v)
        estimation_error_data["example_id"].append(example_id)
        estimation_error_data["mc_est"].append(float(d["mc_mean"]))
        estimation_error_data["smc_est"].append(float(d["reweighted_scores"]))
        estimation_error_data["error"].append(
            float(np.abs(np.log(d["reweighted_scores"]) - np.log(d["mc_maan"])))
        )

estimation_error_df = pd.DataFrame(estimation_error_data)


from IPython import embed

embed()
exit()

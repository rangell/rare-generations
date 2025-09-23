import os
import glob
import json
folder = "NEW_RESULTS_NEW_PROPOSER"

workload_settings_to_seed = {}

# for all folders with system prompt load metadata.json and check the coverage
for path in glob.glob(os.path.join(folder, "*", "system_prompt.txt")):
    metadata_path = os.path.join(os.path.dirname(path), "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    setting = metadata["setting"]
    seed = metadata["seed"]
    workload = metadata["workload"]

    if workload not in workload_settings_to_seed:
        workload_settings_to_seed[workload] = {}
    if setting not in workload_settings_to_seed[workload]:
        workload_settings_to_seed[workload][setting] = []
    workload_settings_to_seed[workload][setting].append(seed)

print(workload_settings_to_seed)
import os
import json
import glob
import numpy as np
# paths = ["/home/horvitz/rare-generations/auto_red_team/multi_seed_results_red_team_harm_est/*", "/home/horvitz/rare-generations/auto_red_team/multi_seed_results_red_team_harm_est_no_opt/*"]
paths = ["/home/horvitz/red_team_from_cluster/auto_red_team/NEW_MED_HEAVY_RESULTS_NO_FEEDBACK/*"]
# paths = ["/home/horvitz/red_team_from_cluster/auto_red_team/NEW_RESULTS_NEW_PROPOSER/*"]
# /home/horvitz/red_team_from_cluster/auto_red_team/NEW_RESULTS_NEW_PROPOSER

setting_to_config = {}
setting_to_results = {}

combined_paths = []
for glob_path in paths:
    combined_paths.extend(glob.glob(glob_path))

for path in combined_paths:
    expected_path = os.path.join(path, "eval_results_500.json")
    # expected_path = os.path.join(path, "eval_results.json")
    if not os.path.exists(expected_path):
        print(f"Skipping {path} because {expected_path} does not exist")
        continue
    eval_results = json.load(open(expected_path))
    metadata = json.load(open(os.path.join(path, "metadata.json")))
    seed = metadata.pop("seed")
    setting = metadata["setting"] + '_' + metadata["workload"] #+ '_' + str(metadata["num_forbidden_prompts"])
    if setting not in setting_to_config:
        setting_to_config[setting] = metadata
        setting_to_results[setting] = [eval_results]
    else:
        assert metadata == setting_to_config[setting]
        setting_to_results[setting].append(eval_results)

print(setting_to_config)
expected_len = 50 #105
trained_on = 5

# import pdb; pdb.set_trace()
setting_to_prompt_level_results = {}
for setting, results in setting_to_results.items():
    print(setting)
    # if 'ACTUAL_BASELINE' not in setting and '_light' in setting:
        # continue

    do_break = False
    for i in range(len(results)):
        # assert len(results[i]) >= expected_len, f"{setting}: Expected {expected_len} results, got {len(results[i])}"
        if len(results[i]) < expected_len:
            do_break = True
            break
        results[i] = results[i][:expected_len]
    
    if do_break:
        continue
    
    per_prompt_results = []
    for p in range(len(results[0])):
        prompt_scores = []
        for i in range(len(results)):
            prompt_scores.append(results[i][p][0]['redteam_score'])
        
        prompt_scores = np.mean(prompt_scores)
        # prompt_scores = np.max(prompt_scores)
        per_prompt_results.append(prompt_scores)

    setting_to_prompt_level_results[setting] = np.array(per_prompt_results)


# print means
baseline_results = setting_to_prompt_level_results["ACTUAL_BASELINE_light"]
# baseline_results_optimized = baseline_results[:trained_on]
# baseline_results_heldout = baseline_results[trained_on:]
# for setting, results in setting_to_prompt_level_results.items():
    # print(setting, np.mean(results[:trained_on]), np.mean(results[trained_on:]))
    # compute average ratio of setting to baseline
    # setting_results_optimized = results[:trained_on]
    # setting_results_heldout = results[trained_on:]
    # print(setting, np.mean(setting_results_optimized), np.mean(setting_results_heldout))
    # ratio_optimized = np.mean(setting_results_optimized / baseline_results_optimized)
    # ratio_heldout = np.mean(setting_results_heldout / baseline_results_heldout)
    # print(setting, ratio_optimized, ratio_heldout)
    # print(setting, np.mean(setting_results_optimized /baseline_results_optimized), np.mean(setting_results_heldout) / np.mean(baseline_results_heldout))


for setting, results in setting_to_prompt_level_results.items():
    results_key = np.array(results)
    print(f"""
{setting}
n={len(results_key)}       
means: heldout={np.mean(results_key[5:])}, optimized_set={np.mean(results_key[:5])}
ratio_baseline: heldout={np.mean(results_key[5:]/baseline_results[5:])}, optimized_set={np.mean(results_key[:5]/baseline_results[:5])}
""")
    print('-' * 100)

# import pdb; pdb.set_trace()

    

   


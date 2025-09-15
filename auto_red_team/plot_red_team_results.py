

import json
import numpy as np

path_to_results = '/home/horvitz/rare-generations/auto_red_team/red_team_eval_results/20250914_180624/current_results.json'



with open(path_to_results, 'r') as f:
    results = json.load(f)

keys = sorted(results.keys())
baseline_scores = np.array(results['baseline_scores'])
other_keys = [key for key in keys if key != 'baseline_scores']

for key in keys:
    results_key = np.array(results[key])
    print(f"""
{key}
n={len(results[key])}
means: all={np.mean(results_key)}, optimized_set={np.mean(results_key[:5])}
ratio_baseline: all={np.mean(results_key/baseline_scores)}, optimized_set={np.mean(results_key[:5]/baseline_scores[:5])}
""")
    print('-' * 100)






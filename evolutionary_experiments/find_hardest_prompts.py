import glob
import json
import os

paths = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v2_experiment_initial_test/20251206_000238/query_*/pop_stats.json'
hardest_k = 10
all_samples = []
for path in glob.glob(paths):
    with open(path, 'r') as f:
        pop_stats = json.load(f)
    info_path = os.path.join(os.path.dirname(path), 'info.json')
    with open(info_path, 'r') as f:
        info = json.load(f)
    combined = {**info, **pop_stats[0], 'path': path}
    combined['forbidden_prompt'] = info['query']
    all_samples.append(combined)

sorted_path = sorted(all_samples, key=lambda x: x['best_score'], reverse=False)[:hardest_k]
with open(f'hardest_{hardest_k}_prompts.jsonl', 'w') as f:
    for sample in sorted_path:
        json.dump(sample, f)
        f.write('\n')
import os
import glob
import json
result_dir = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v2_experiment_hardest_10_prompts_RERAN'

path = f'{result_dir}/*/query_*/pop_stats.json'


pop_paths = sorted(glob.glob(path))

for pop_path in pop_paths:
    with open(pop_path, 'r') as f:
        pop_stats = json.load(f)
    path_to_info = pop_path.replace('pop_stats.json', 'historical_populations/')
    num_steps = len(glob.glob(path_to_info + 'step_*.json'))
    index = list(range(-1, num_steps-1))
    print(len(pop_stats), len(index))
    assert len(pop_stats) == len(index), f"Number of steps in {pop_path} is {num_steps} but number of pop stats is {len(pop_stats)}"

    for i, step_idx in enumerate(index):
        with open(path_to_info + f'step_{step_idx}.json', 'r') as f:
            step_pop_stats = json.load(f)

        sorted_step_pop_stats = sorted(step_pop_stats, key=lambda x: x['score'], reverse=True)
        assert sorted_step_pop_stats[0]['score'] == pop_stats[i]['best_score']
        pop_stats[i]['best_sample'] = sorted_step_pop_stats[0]

    assert not os.path.exists(pop_path+'_with_best_sample.json'), f"File {pop_path+'_with_best_sample.json'} already exists"

    with open(pop_path+'_with_best_sample.json', 'w') as f:
        json.dump(pop_stats, f, indent=4)
   
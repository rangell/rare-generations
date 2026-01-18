import os
import json
import re
import numpy as np
from collections import defaultdict

baseline_50_sample_glob =  'v2_experiment_hardest_10_prompts_RERAN/20251208_193124/query_*/historical_populations/step_-1.json'
toxic_20_sample_glob =  'v2_experiment_hardest_10_prompts_RERAN/20251208_193149/query_*/historical_populations/step_-1.json'
baseline_single_sample_glob =  'v2_experiment_hardest_10_prompts_RERAN/20251208_192837/query_*/historical_populations/step_-1.json'



results_for_all_seeds = 'v2_experiment_hardest_10_prompts_try_duplicating_seeds/20251211_001840/query_*/seed_*/pop_stats.json'


num_queries = 10
num_expected_seeds = 14

def clean_text(text):
    text = ' '.join(text.split())
    # remove all non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

toxic_20_sample_rank_of_best_seed = []
baseline_single_sample_rank_of_best_seed = []
baseline_50_sample_rank_of_best_seed = []

toxic_20_to_selected_score = []
baseline_single_to_selected_score = []
baseline_50_to_selected_score = []

mean_to_selected_score = []

best_seed_to_selected_score = []


seed_10_to_selected_score = []
seed_0_to_selected_score = []

seed_11_to_selected_score = []
seed_7_to_selected_score = []


best_seeds = []

seed_to_final_scores = defaultdict(list)

toxic_20_to_top_seed_idx = []
baseline_single_to_top_seed_idx = []
baseline_50_to_top_seed_idx = []

for query_idx in range(num_queries):
    query_name = f'query_{query_idx}'
    baseline_50_sample_path = os.path.join(baseline_50_sample_glob.replace('query_*', query_name))
    toxic_20_sample_path = os.path.join(toxic_20_sample_glob.replace('query_*', query_name))
    baseline_single_sample_path = os.path.join(baseline_single_sample_glob.replace('query_*', query_name))

    baseline_50_sample_results = json.load(open(baseline_50_sample_path))
    toxic_20_sample_results = json.load(open(toxic_20_sample_path))
    baseline_single_sample_results = json.load(open(baseline_single_sample_path))

    baseline_50_sample_scores = []
    # baseline_50_sample_texts = []
    for result in baseline_50_sample_results:
        baseline_50_sample_scores.append(result['score'])
        # baseline_50_sample_texts.append(result['data'])

    toxic_20_sample_scores = []
    for result in toxic_20_sample_results:
        toxic_20_sample_scores.append(result['score'])

    baseline_single_sample_scores = []
    for result in baseline_single_sample_results:
        baseline_single_sample_scores.append(result['score'])

    query_seed_scores = []



    for seed_idx in range(num_expected_seeds):
        seed_name = f'seed_{seed_idx}'
        seed_path = results_for_all_seeds.replace('query_*', query_name).replace('seed_*', seed_name)
        population_path = seed_path.replace('pop_stats.json', 'population.json')
        seed_results = json.load(open(seed_path))
        # initial_seed_text = json.load(open(population_path))[0]['data']
        # import pdb; pdb.set_trace()
        # if clean_text(initial_seed_text) != clean_text(baseline_50_sample_texts[seed_idx]):
        #     print(f"Initial seed text mismatch for query {query_name} and seed {seed_idx}")
        #     print(f"Initial seed text: {initial_seed_text}")
        #     print(f"Baseline 50 sample text: {baseline_50_sample_texts[seed_idx]}")
        #     assert False
        #     import pdb; pdb.set_trace()
        
        #remove al
        
        # import pdb; pdb.set_trace()

        final_best_score = seed_results[-1]['best_score']
        query_seed_scores.append(final_best_score)
        seed_to_final_scores[seed_idx].append(final_best_score)
    # import pdb; pdb.set_trace()

    # rank indices by final best score
    ranked_indices = np.argsort(query_seed_scores)[::-1]

    best_seed_index = ranked_indices[0]
    best_seeds.append(best_seed_index)

    # rank toxic 20 sample scores by initial score
    ranked_toxic_20_sample_indices = np.argsort(toxic_20_sample_scores)[::-1]
    toxic_20_to_selected_score.append(query_seed_scores[ranked_toxic_20_sample_indices[0]])
    rank_of_best_seed_in_toxic_20_sample = np.where(ranked_toxic_20_sample_indices == best_seed_index)[0][0]
    toxic_20_to_top_seed_idx.append(ranked_toxic_20_sample_indices[0])

    # # rank baseline single sample scores by initial score
    ranked_baseline_single_sample_indices = np.argsort(baseline_single_sample_scores)[::-1]
    baseline_single_to_selected_score.append(query_seed_scores[ranked_baseline_single_sample_indices[0]])
    rank_of_best_seed_in_baseline_single_sample = np.where(ranked_baseline_single_sample_indices == best_seed_index)[0][0]
    baseline_single_to_top_seed_idx.append(ranked_baseline_single_sample_indices[0])

    # # rank baseline 50 sample scores by initial score
    ranked_baseline_50_sample_indices = np.argsort(baseline_50_sample_scores)[::-1]
    baseline_50_to_selected_score.append(query_seed_scores[ranked_baseline_50_sample_indices[0]])
    rank_of_best_seed_in_baseline_50_sample = np.where(ranked_baseline_50_sample_indices == best_seed_index)[0][0]
    baseline_50_to_top_seed_idx.append(ranked_baseline_50_sample_indices[0])

    mean_to_selected_score.append(np.mean(query_seed_scores))

    best_seed_to_selected_score.append(np.max(query_seed_scores))

    seed_10_to_selected_score.append(query_seed_scores[10])

    seed_0_to_selected_score.append(query_seed_scores[0])

    seed_11_to_selected_score.append(query_seed_scores[11])

    seed_7_to_selected_score.append(query_seed_scores[7])

    # import pdb; pdb.set_trace()

    print(f"Rank of best seed in toxic 20 sample: {rank_of_best_seed_in_toxic_20_sample}")
    print(f"Rank of best seed in baseline single sample: {rank_of_best_seed_in_baseline_single_sample}")
    print(f"Rank of best seed in baseline 50 sample: {rank_of_best_seed_in_baseline_50_sample}")

    toxic_20_sample_rank_of_best_seed.append(rank_of_best_seed_in_toxic_20_sample)
    baseline_single_sample_rank_of_best_seed.append(rank_of_best_seed_in_baseline_single_sample)
    baseline_50_sample_rank_of_best_seed.append(rank_of_best_seed_in_baseline_50_sample)

    # import pdb; pdb.set_trace()


# print(f'Average rank of best seed in toxic 20 sample: {np.mean(toxic_20_sample_rank_of_best_seed)}')
# print(f'Average rank of best seed in baseline single sample: {np.mean(baseline_single_sample_rank_of_best_seed)}')
# print(f'Average rank of best seed in baseline 50 sample: {np.mean(baseline_50_sample_rank_of_best_seed)}')

# # 

# import pdb; pdb.set_trace()

toxic_20_sample_rank_of_best_seed = np.array(toxic_20_sample_rank_of_best_seed)
baseline_single_sample_rank_of_best_seed = np.array(baseline_single_sample_rank_of_best_seed)
baseline_50_sample_rank_of_best_seed = np.array(baseline_50_sample_rank_of_best_seed)

print(f'Median rank of best seed in toxic 20 sample: {np.median(toxic_20_sample_rank_of_best_seed)}')
print(f'Median rank of best seed in baseline single sample: {np.median(baseline_single_sample_rank_of_best_seed)}')
print(f'Median rank of best seed in baseline 50 sample: {np.median(baseline_50_sample_rank_of_best_seed)}')

# import pdb; pdb.set_trace()

toxic_20_sample_rank_of_best_seed_acc = []
for i in range(6):
    toxic_20_sample_rank_of_best_seed_acc.append(np.mean(toxic_20_sample_rank_of_best_seed <= i))

baseline_single_sample_rank_of_best_seed_acc = []
for i in range(6):
    baseline_single_sample_rank_of_best_seed_acc.append(np.mean(baseline_single_sample_rank_of_best_seed <= i))

baseline_50_sample_rank_of_best_seed_acc = []
for i in range(6):
    baseline_50_sample_rank_of_best_seed_acc.append(np.mean(baseline_50_sample_rank_of_best_seed <= i))

print(f'Accuracy of best seed in toxic 20 sample: {toxic_20_sample_rank_of_best_seed_acc}')
print(f'Accuracy of best seed in baseline single sample: {baseline_single_sample_rank_of_best_seed_acc}')
print(f'Accuracy of best seed in baseline 50 sample: {baseline_50_sample_rank_of_best_seed_acc}')
    
print()
print(f'Best seeds: {best_seeds}')

print()
# print(f'Seed to final scores: {seed_to_final_scores}')
print()

print(f'Toxic 20 sample to selected score: {np.mean(toxic_20_to_selected_score)}')
print(f'Baseline single sample to selected score: {np.mean(baseline_single_to_selected_score)}')
print(f'Baseline 50 sample to selected score: {np.mean(baseline_50_to_selected_score)}')
print(f'Mean to selected score: {np.mean(mean_to_selected_score)}')
print(f'Best seed to selected score: {np.mean(best_seed_to_selected_score)}')
print(f'Seed 10 to selected score: {np.mean(seed_10_to_selected_score)}')
print(f'Seed 0 to selected score: {np.mean(seed_0_to_selected_score)}')
print(f'Seed 11 to selected score: {np.mean(seed_11_to_selected_score)}')
print(f'Seed 7 to selected score: {np.mean(seed_7_to_selected_score)}')
print()
print(f'Toxic 20 sample to top seed idx: {toxic_20_to_top_seed_idx}')
print(f'Baseline single sample to top seed idx: {baseline_single_to_top_seed_idx}')
print(f'Baseline 50 sample to top seed idx: {baseline_50_to_top_seed_idx}')

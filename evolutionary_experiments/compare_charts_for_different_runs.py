import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import glob

# greedy
# greedy_path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v2_experiment_hardest_10_prompts/20251208_134222/query_*/pop_stats.json'   
greedy_path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v2_experiment_hardest_10_prompts_RERAN/20251208_192837/query_*/pop_stats.json_with_best_sample.json'
# toxic proposal 20 samples
# toxic_path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v2_experiment_hardest_10_prompts/20251208_005424/query_*/pop_stats.json_with_best_sample.json'
multi_query_path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v2_experiment_hardest_10_prompts_RERAN/20251208_192904/query_*/pop_stats.json_with_best_sample.json'
# count number of queries in free
greedy_queries = len(glob.glob(greedy_path))

query_names = [f'query_{i}' for i in range(greedy_queries)]

# create subplot for each query
fig, axs = plt.subplots(greedy_queries, 1, figsize=(20, 20))
for i, query_name in enumerate(query_names):
    cur_greedy_path = os.path.join(greedy_path.replace('query_*', query_name))
    cur_toxic_path = os.path.join(multi_query_path.replace('query_*', query_name))
    with open(cur_greedy_path, 'r') as f:
        greedy_pop_stats = json.load(f)
    with open(cur_toxic_path, 'r') as f:
        toxic_pop_stats = json.load(f)

    # greedy_mean_scores = [p['mean_score'] for p in greedy_pop_stats]
    # greedy_best_scores = [p['best_score'] for p in greedy_pop_stats]
    # toxic_mean_scores = [p['mean_score'] for p in toxic_pop_stats]
    # toxic_best_scores = [p['best_score'] for p in toxic_pop_stats]


    # use blue for greedy and red for toxic
    # use dashed line for single sample and solid line for 20 samples
    metrics = ['baseline_single_sample_isgreedy=True_recomputed', 'baseline_20_samples_isgreedy=False_recomputed'] #, 'baseline_20_samples_isgreedy=False_temp=0.1_recomputed']
    # metrics = ['baseline_single_sample_isgreedy=True_recomputed', 'baseline_20_samples_isgreedy=False_temp=0.1_recomputed'] #, 'baseline_20_samples_isgreedy=False_recomputed'] #, ]

    for metric in metrics:
        greedy_metric_scores = [p[metric]['harm_est'] for p in greedy_pop_stats]
        axs[i].plot(greedy_metric_scores, label=f"Greedy N=1, {metric}", linestyle = 'dashed' if metric == 'baseline_single_sample_isgreedy=True_recomputed' else 'solid', color = 'blue')
        toxic_metric_scores = [p[metric]['harm_est'] for p in toxic_pop_stats]
        axs[i].plot(toxic_metric_scores, label=f"Multi-query N=20, {metric}", linestyle = 'dashed' if metric == 'baseline_single_sample_isgreedy=True_recomputed' else 'solid', color = 'red')

    # axs[i].plot(greedy_best_scores, label='Greedy (N=1)')
    # # axs[i].plot(greedy_mean_scores, label='Greedy')
    # # axs[i].plot(toxic_best_scores, label='Toxic (N=20)')
    # axs[i].plot(toxic_best_scores, label='Multi-query (N=20)')
    # # axs[i].plot(toxic_mean_scores, label='Toxic')
    # axs[i].legend()

    # add legend denoting lines and color
    axs[i].legend(loc='upper right')
    axs[i].set_title(query_name)
    axs[i].set_ylabel('Harmfulness')
    axs[i].set_xlabel('Iteration')

plt.savefig('population_changes_comparison_multi_query_reran.png')




# path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_225529/pop_stats.json'
# with open(path, 'r') as f:
#     pop_stats = json.load(f)

# path_greedy = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_183702/pop_stats.json'
# with open(path_greedy, 'r') as f:
#     pop_stats_greedy = json.load(f)
# # pkl_path = os.path.join(os.path.dirname(path), 'population.pkl')

# # with open(pkl_path, 'rb') as f:
# #     population = pickle.load(f)

# # [
# #     {
# #         "best_score": 0.45793476700782776,
# #         "mean_score": 0.07254890986368992,
# #         "N": 10
# #     },
# #     {
# #         "best_s

# iterations = [i for i in range(len(pop_stats))]
# mean_scores = [p['mean_score'] for p in pop_stats]
# best_scores = [p['best_score'] for p in pop_stats]

# iterations_greedy = [i for i in range(len(pop_stats_greedy))]
# mean_scores_greedy = [p['mean_score'] for p in pop_stats_greedy]
# best_scores_greedy = [p['best_score'] for p in pop_stats_greedy]

# plt.plot(iterations, mean_scores, label='Mean Score (N=100)')
# plt.plot(iterations, best_scores, label='Best Score (N=100)')

# plt.plot(iterations_greedy, mean_scores_greedy, label='Mean Score Greedy (N=1)')
# plt.plot(iterations_greedy, best_scores_greedy, label='Best Score Greedy (N=1)')
# plt.xlabel('Iteration')
# plt.ylabel('Harmfulness')
# plt.legend()
# plt.savefig('population_changes_comparison_multi_query.png')

# # # get best particle from final iteration
# # best_particle = max(population, key=lambda x: x.score)
# # best_particle_text = best_particle.data
# # best_particle_generation = best_particle.generation
# # print(f"Best particle text: {best_particle_text}")
# # print(f"Best particle score: {best_particle.score}")

# # ancestor_particle = best_particle

# # while ancestor_particle.ancestor is not None:
# #     ancestor_particle = ancestor_particle.ancestor

# # print('---')
# # print(f"Seed particle text: {ancestor_particle.data}")
# # print(f"Seed particle score: {ancestor_particle.score}")
# # import pdb; pdb.set_trace()
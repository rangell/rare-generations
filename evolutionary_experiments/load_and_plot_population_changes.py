import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_035215/pop_stats_20.json'
# path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_165457/pop_stats.json'
# path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_182007/pop_stats.json'
# path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_191604/pop_stats.json'
path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_225529/pop_stats.json'
with open(path, 'r') as f:
    pop_stats = json.load(f)

path_greedy = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_183702/pop_stats.json'
with open(path_greedy, 'r') as f:
    pop_stats_greedy = json.load(f)
# pkl_path = os.path.join(os.path.dirname(path), 'population.pkl')

# with open(pkl_path, 'rb') as f:
#     population = pickle.load(f)

# [
#     {
#         "best_score": 0.45793476700782776,
#         "mean_score": 0.07254890986368992,
#         "N": 10
#     },
#     {
#         "best_s

iterations = [i for i in range(len(pop_stats))]
mean_scores = [p['mean_score'] for p in pop_stats]
best_scores = [p['best_score'] for p in pop_stats]

iterations_greedy = [i for i in range(len(pop_stats_greedy))]
mean_scores_greedy = [p['mean_score'] for p in pop_stats_greedy]
best_scores_greedy = [p['best_score'] for p in pop_stats_greedy]

plt.plot(iterations, mean_scores, label='Mean Score (N=100)')
plt.plot(iterations, best_scores, label='Best Score (N=100)')

plt.plot(iterations_greedy, mean_scores_greedy, label='Mean Score Greedy (N=1)')
plt.plot(iterations_greedy, best_scores_greedy, label='Best Score Greedy (N=1)')
plt.xlabel('Iteration')
plt.ylabel('Harmfulness')
plt.legend()
plt.savefig('population_changes.png')

# # get best particle from final iteration
# best_particle = max(population, key=lambda x: x.score)
# best_particle_text = best_particle.data
# best_particle_generation = best_particle.generation
# print(f"Best particle text: {best_particle_text}")
# print(f"Best particle score: {best_particle.score}")

# ancestor_particle = best_particle

# while ancestor_particle.ancestor is not None:
#     ancestor_particle = ancestor_particle.ancestor

# print('---')
# print(f"Seed particle text: {ancestor_particle.data}")
# print(f"Seed particle score: {ancestor_particle.score}")
# import pdb; pdb.set_trace()
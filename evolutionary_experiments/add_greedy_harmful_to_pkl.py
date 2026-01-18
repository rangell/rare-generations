

path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v2_experiment_hardest_10_prompts/20251208_134222/query_2/population.pkl'
with open(path, 'rb') as f:
    population = pickle.load(f)

for particle in population:
    particle.score = particle.score + 0.1

with open(path, 'wb') as f:
    pickle.dump(population, f)
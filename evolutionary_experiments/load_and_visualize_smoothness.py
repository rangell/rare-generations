import pickle
import matplotlib.pyplot as plt
import numpy as np

path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_031054/population_1.pkl'
with open(path, 'rb') as f:
    population = pickle.load(f)


parent_scores = [p.ancestor.score for p in population]
parent_texts = [p.ancestor.data for p in population]
child_scores = [p.score for p in population]
child_texts = [p.data for p in population]

not_equal = [parent_texts[i] != child_texts[i] for i in range(len(parent_texts))]
print(f"Percentage of not equal: {np.mean(not_equal)}")



# log scale
parent_scores = np.log(parent_scores)
child_scores = np.log(child_scores)

# compute pearson correlation
correlation = np.corrcoef(parent_scores, child_scores)[0, 1]
print(f"Pearson correlation: {correlation}")


plt.scatter(parent_scores, child_scores)
plt.xlabel('Parent Score')
plt.ylabel('Child Score')
plt.title('Parent vs Child Score (log vs log)')

# add correlation coefficient to the plot
plt.text(0.05, 0.95, f'Pearson correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.savefig('parent_vs_child_score.png')

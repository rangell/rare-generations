import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
# path = '/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment/20251204_031054/population_1.pkl'

import Levenshtein
from nltk.tokenize import word_tokenize
import re
####
def clean_text(text):
    tokens = [t for t in word_tokenize(text) if t not in ['[',']']]
    tokens = ['[REPLACE]' if t == 'REPLACE' else t for t in tokens]
    text = ' '.join(tokens)
    text = result = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = result = re.sub(r'\s+\\n\\n\s+', r'\\n\\n', text)
    return text
###

paths = sorted(glob.glob('/home/horvitz/red_team_from_cluster/evolutionary_experiments/v1_experiment_smoothness/20251205_000554/query_*/population.pkl'))
correlation_list = []
all_parent_scores = []
all_child_scores = []

all_levenshtein_distances = []
all_abs_log_diffs = []
all_edit_correlations = []

# make two plots: one with parent scores and child scores and one with levenshtein distance and absolute log difference
fig, ax = plt.subplots(2 + len(paths), 1, figsize=(10, 120))

for i, path in enumerate(paths):
    with open(path, 'rb') as f:
        population = pickle.load(f)


    parent_scores = [p.ancestor.score if p.ancestor is not None else p.score for p in population]
    parent_texts = [p.ancestor.data if p.ancestor is not None else p.data for p in population]
    child_scores = [p.score for p in population]
    child_texts = [p.data for p in population]

    not_equal = [parent_texts[i] != child_texts[i] for i in range(len(parent_texts))]
    print(f"Percentage of not equal: {np.mean(not_equal)}")

    parent_scores = [p for i, p in enumerate(parent_scores) if not_equal[i]]
    child_scores = [p for i, p in enumerate(child_scores) if not_equal[i]]
    parent_texts = [clean_text(p).lower().split() for i, p in enumerate(parent_texts) if not_equal[i]]
    child_texts = [clean_text(p).lower().split() for i, p in enumerate(child_texts) if not_equal[i]]

    # log scale
    parent_scores = np.log(parent_scores)
    child_scores = np.log(child_scores)

    # compute pearson correlation
    correlation = np.corrcoef(parent_scores, child_scores)[0, 1]
    print(f"Pearson correlation: {correlation}")
    correlation_list.append(correlation)

    ax[0].scatter(parent_scores, child_scores)
    all_parent_scores.extend(parent_scores)
    all_child_scores.extend(child_scores)

     # compute levenshtein distance
    levenshtein_distances = [Levenshtein.distance(p1, p2) for p1, p2 in zip(parent_texts, child_texts)]
    # import pdb; pdb.set_trace()
    abs_log_diffs = np.abs(parent_scores - child_scores)
    # abs_log_diffs = parent_scores - child_scores

    edit_correlation = np.corrcoef(levenshtein_distances, abs_log_diffs)[0, 1] 
    all_edit_correlations.append(edit_correlation)


    ax[1].scatter(levenshtein_distances, abs_log_diffs)
    ax[2 + i].scatter(levenshtein_distances, abs_log_diffs)
    ax[2 + i].set_xlabel(f'Edit vs Abs Log Diff: Pearson correlation = {edit_correlation:.2f}')
    ax[2 + i].set_ylabel('Absolute Log Difference')
    # ax[2 + i].set_ylim(-5, 5)
    all_levenshtein_distances.extend(levenshtein_distances)
    all_abs_log_diffs.extend(abs_log_diffs)
all_correlation = np.corrcoef(all_parent_scores, all_child_scores)[0, 1]
median_correlation = np.median([c for c in correlation_list if np.isfinite(c)])

ax[0].set_xlabel('Parent Score')
ax[0].set_ylabel('Child Score')
ax[0].set_title(f'Parent vs Child Score (log vs log): Pearson correlation = {all_correlation:.2f}, Median correlation = {median_correlation:.2f}')

# add correlation coefficient to the plot
# ax[0].text(0.05, 0.95, f'Pearson correlation: {all_correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
# ax[0].text(0.05, 0.90, f'Median correlation: {median_correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


all_edit_correlation = np.corrcoef(all_levenshtein_distances, all_abs_log_diffs)[0, 1]
median_edit_correlation = np.median([c for c in all_edit_correlations if np.isfinite(c)])
# import pdb; pdb.set_trace()
# plt.savefig('parent_vs_child_score_all.png')
ax[1].set_xlabel('Levenshtein Distance')
ax[1].set_ylabel('Absolute Log Difference')
ax[1].set_title(f'Levenshtein vs Abs Log Diff: Pearson correlation = {all_edit_correlation:.2f}, Median correlation = {median_edit_correlation:.2f}')

# add correlation coefficient to the plot
# ax[1].text(0.05, 0.95, f'Pearson correlation: {all_edit_correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
# ax[1].text(0.05, 0.90, f'Median correlation: {median_edit_correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.savefig('parent_vs_child_score_and_levenshtein_distance_all.png')
plt.show()
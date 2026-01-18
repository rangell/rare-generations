import json
import pandas as pd


# /home/horvitz/red_team_from_cluster/rare_behavior_graphic/model_name_to_final_asr.json
# /home/horvitz/red_team_from_cluster/rare_behavior_graphic/model_name_to_greedy_asr.json

final_10k_asr_path = '/home/horvitz/red_team_from_cluster/rare_behavior_graphic/model_name_to_final_asr.json'
greedy_asr_path = '/home/horvitz/red_team_from_cluster/rare_behavior_graphic/model_name_to_greedy_asr.json'

with open(final_10k_asr_path, 'r') as f:
    final_10k_asr = json.load(f)
with open(greedy_asr_path, 'r') as f:
    greedy_asr = json.load(f)

model_names = sorted(list(final_10k_asr.keys()))

results = []

for model_name in model_names:
    final_10k_asr_value = round(final_10k_asr[model_name] * 100, 1)
    greedy_asr_value = round(greedy_asr[model_name] * 100, 1)
    delta = round(final_10k_asr_value - greedy_asr_value, 1)
    results.append({
        'model_name': model_name,
        'greedy_asr': greedy_asr_value,
        'final_10k_asr': final_10k_asr_value,
        'delta': delta
    })

df = pd.DataFrame(results)
df = df.sort_values(by='delta', ascending=False)
df.to_csv('asr_table.csv', index=False)
import os
import json
import glob
import re

expected_indices = list(range(313))
exceptions = [145, 257, 267, 307]
expected_paraphrases = 25
# files = sorted([x for x in glob.glob('paraphrases_*/*.json') if 'indices' not in x and 'info' not in x])
# paraphrases_20260116_143849
files = sorted([x for x in glob.glob('paraphrases_20260116_143849/*.json') if 'indices' not in x and 'info' not in x])
index_to_path = {}
for file in files:
    index = int(re.search(r'\d+', file.split('/')[-1]).group())
    index_to_path[index] = file

with open('combined_paraphrased_results_new.json', 'w') as f:
    for index in expected_indices:
        if index in exceptions:
            continue
        with open(index_to_path[index], 'r') as g:
            data = json.load(g)
        original_prompt = data['forbidden_prompt']
        paraphrases = data['paraphrases']
        if len(paraphrases) != expected_paraphrases:
            assert len(paraphrases) == 1
            key_paraphrase = list(paraphrases.keys())[0]
            paraphrases = paraphrases[key_paraphrase]
            assert len(paraphrases) == expected_paraphrases
        paraphrases = [x['text'] for x in paraphrases]
        for paraphrase in paraphrases:
            f.write(json.dumps({'index': index, 'original_forbidden_prompt': original_prompt, 'forbidden_prompt': paraphrase}) + '\n')
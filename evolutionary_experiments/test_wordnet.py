import nltk

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
# https://stackoverflow.com/questions/19258652/how-to-get-synonyms-from-nltk-wordnet-python
def get_all_synonyms(word):
    synonyms = []
    for ss in wn.synsets(word):
        synonyms.extend(ss.lemma_names())
        for sim in ss.similar_tos():
            synonyms_batch = sim.lemma_names()
            synonyms.extend(synonyms_batch)
    synonyms = set(synonyms)
    if word in synonyms:
        synonyms.remove(word)
    synonyms = [synonym.replace('_',' ') for synonym in synonyms]
    return synonyms

# tokenize th
import nltk
from nltk.tokenize import word_tokenize
sentence = "The quick brown [REPLACE] jumps: over the lazy dog"
tokens = word_tokenize(sentence)
print(tokens)

print(get_all_synonyms('FOX'))
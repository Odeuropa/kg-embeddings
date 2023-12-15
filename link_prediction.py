import csv
import math

from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm


def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


kv = KeyedVectors.load("embeddings/smells.kv")
emissions = {}
with open('data/od_F1_generated.csv') as f:
    csv_reader = csv.DictReader(f)

    first = True
    for row in csv_reader:
        if first:
            first = False
            continue

        emissions[row['s']] = row['o']

smell_sources = {}
with open('data/od_F3_had_source%20%2F%20ecrm_P137_exemplifies.csv') as f:
    csv_reader = csv.DictReader(f)

    first = True
    for row in csv_reader:
        if first:
            first = False
            continue

        smell = emissions[row['s']]
        if smell not in smell_sources:
            smell_sources[smell] = []
        smell_sources[smell].append(row['o'])


def smell_source(uri):
    global smell_sources
    if uri not in smell_sources:
        return None
    return smell_sources[uri]


embeddings = kv.vectors


pos = list(range(len(embeddings)))
sm_sources = [smell_source(kv.index_to_key[x]) for x in tqdm(pos)]
pos_filtered = [p for p in pos if sm_sources[p]]

from collections import Counter

keep_keys = []
for key, value in dict(Counter(flatten_concatenation([sm_sources[p] for p in pos_filtered]))).items():
    if value > 40:
        keep_keys.append(key)

def clean(s_list):
    if s_list is None:
        return []
    return set(s_list).intersection(set(keep_keys))

sm_sources = [clean(s) for s in sm_sources]
pos_filtered = [p for p in pos if len(sm_sources[p])>0]


split_at = math.floor(len(embeddings)*0.9)
train_pos = pos_filtered[0:split_at]
test_pos = pos_filtered[split_at:]
train_embeddings = embeddings[train_pos]
test_embeddings = embeddings[test_pos]

train_x = [kv.index_to_key[x] for x in train_pos]
train_y = [list(clean(smell_source(kv.index_to_key[x])))[0] for x in tqdm(train_pos)]
test_y = [list(clean(smell_source(kv.index_to_key[x])))[0] for x in tqdm(test_pos)]

print(len(train_y))
print(len(test_y))

print('Training started')
clf = GridSearchCV(
    SVC(random_state=42), {"C": [10 ** i for i in range(-3, 4)]}
)
clf.fit(train_embeddings, train_y)
print('Training finished')

print('Prediction started')
# Evaluate the Support Vector Machine on test embeddings.
predictions = clf.predict(test_embeddings)
print(
    f"Predicted {len(test_embeddings)} entities with an accuracy of "
    + f"{accuracy_score(test_y, predictions) * 100 :.4f}%"
)
print(f"Confusion Matrix ([[TN, FP], [FN, TP]]):")
print(confusion_matrix(test_y, predictions))
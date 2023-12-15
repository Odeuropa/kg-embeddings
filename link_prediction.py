import csv
import math

from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tqdm import tqdm

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

split_at = math.floor(len(embeddings)*0.8)
train_pos = pos_filtered[0:split_at]
test_pos = pos_filtered[split_at:]
train_embeddings = embeddings[train_pos]
test_embeddings = embeddings[test_pos]

train_x = [kv.index_to_key[x] for x in train_pos]
train_y = [smell_source(kv.index_to_key[x])[0] for x in tqdm(train_pos)]
test_y = [smell_source(kv.index_to_key[x])[0] for x in tqdm(test_pos)]

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
import random
from tqdm import tqdm
from collections import Counter
import csv

limit = 10000

with open('dense_graph.csv', 'r') as f:
    triples = [x.strip().split(',') for x in f.readlines()]

print(f'Num of triples: {len(triples)}')
other_triples = []
smells = []
smells_triples = []

for t in tqdm(triples):
    s, p, o = t
    smell = None
    if s.startswith('http://data.odeuropa.eu/smell/'):
        smell = s
    elif o.startswith('http://data.odeuropa.eu/smell/'):
        smell = 0

    if smell is not None:
        smells.append(smell)
        smells_triples.append(t)
    else:
        other_triples.append(t)

smells_set = list(set(smells))
print(f'Num of smells: {len(smells_set)}')
print(f'Num of non-smell triples: {len(other_triples)}')

chosen_smells = set(random.sample(smells_set, k=limit))
print(f'Num of chosen smells: {len(chosen_smells)}')

ent_triples = []
for e in tqdm(chosen_smells):
    counting = smells.count(e)
    ent_triples.append((e, counting))

print('#edge distribution: ' + str(Counter(elem[1] for elem in ent_triples)))

chosen_triples = []
for e in tqdm(chosen_smells):
    for s, t in zip(smells, smells_triples):
        if s == e:
            chosen_triples.append(t)

with open('dense_graph_10k.csv', 'w') as f:
    file_writer = csv.writer(f)
    for x in chosen_triples:
        file_writer.writerow(x)
    for x in other_triples:
        file_writer.writerow(x)

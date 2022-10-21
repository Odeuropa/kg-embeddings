import csv
import os

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker
from tqdm import tqdm

data_folder = './data'

kg = KG()

print('Reading graph..')
for x in tqdm(os.listdir(data_folder)):
    with open(os.path.join(data_folder, x), 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')

        first = True
        for s, p, o in csv_reader:
            if first:
                first = False
                continue
            subj = Vertex(s)
            obj = Vertex(o)
            pred = Vertex(p, predicate=True, vprev=subj, vnext=obj)
            kg.add_walk(subj, pred, obj)

print('Nb of vertices:', len(kg._vertices))
print('Nb of entities in the graph:', len(kg._entities))

with open(os.path.join('./', 'entities.txt'), 'r') as f:
    entities = [x.strip() for x in f.readlines()][1:]
    entities = [x for x in entities if kg.is_exist([x])]

print('Nb of entities for which we are computing embeddings:', len(entities))

transformer = RDF2VecTransformer(
    Word2Vec(epochs=10),
    walkers=[RandomWalker(4, 10, with_reverse=False, n_jobs=4)],
    # verbose=1
)

print('Generating embeddings...')
embeddings, literals = transformer.fit_transform(kg, entities)
with open('./embeddings.txt', 'w') as f:
    for e in embeddings:
        f.write(' '.join(e))
        f.write('\n')

with open('./literals.txt', 'w') as f:
    for l in literals:
        f.write(' '.join(l))
        f.write('\n')

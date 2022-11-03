import csv
import os
import argparse
import numpy as np
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker
from tqdm import tqdm

DATA_FOLDER = './data'
ENTITIES_PATH = './entities.txt'


def run(entities_path, data_folder):
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

    with open(entities_path, 'r') as f:
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
    literals = np.insert(np.array(literals, dtype=str), 0, entities, axis=1)
    embeddings = [[entities[i]] + ['%.18e' % y for y in x] for i, x in enumerate(embeddings)]
    np.savetxt('./embeddings.txt', embeddings, delimiter=" ", fmt='%s')
    if len(literals) > 0 and len(literals[0]) > 0:
        np.savetxt('./literals.txt', literals, delimiter=" ", fmt='%s')


parser = argparse.ArgumentParser('KG generation')
parser.add_argument('entities', default=ENTITIES_PATH)
parser.add_argument('data', default=DATA_FOLDER)

args = parser.parse_args()
run(args.entities, args.data)


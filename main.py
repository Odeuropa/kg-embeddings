import csv
import os
import argparse
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker
from gensim.models import KeyedVectors
from tqdm import tqdm

DATA_FOLDER = './data'
ENTITIES_PATH = 'voc.txt'


def run(entities_path, data_folder):
    kg = KG()

    print('Reading graph..')
    for x in tqdm(os.listdir(data_folder)):
        with open(os.path.join(data_folder, x), 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            first = True
            for s, o in csv_reader:
                if first:
                    first = False
                    continue
                subj = Vertex(s)
                obj = Vertex(o)
                pred = Vertex(x, predicate=True, vprev=subj, vnext=obj)
                kg.add_walk(subj, pred, obj)

    print('Nb of vertices:', len(kg._vertices))
    print('Nb of entities in the graph:', len(kg._entities))

    with open(entities_path, 'r') as f:
        entities = [x.strip() for x in f.readlines()][1:]
        entities = [x for x in entities if kg.is_exist([x])]

    print('Nb of entities for which we are computing embeddings:', len(entities))

    transformer = RDF2VecTransformer(
        Word2Vec(epochs=100),
        walkers=[RandomWalker(max_depth=4, max_walks=100, with_reverse=False, n_jobs=4)],
        # verbose=1
    )

    print('Generating embeddings...')
    transformer.fit(transformer.get_walks(kg, entities), False)
    entities = [e for e in entities if e in transformer.embedder._model.wv]
    embeddings, literals = transformer.transform(kg, entities)

    kv = KeyedVectors(len(embeddings[0]))
    for (emb, uri) in tqdm(zip(embeddings, entities)):
        kv.add_vector(uri, emb)

    kv.save(entities_path.replace('.txt','.kv'))


if __name__ == '__main__':
    # freeze_support()

    parser = argparse.ArgumentParser('KG generation')
    parser.add_argument('entities', default=ENTITIES_PATH)
    parser.add_argument('--data', '-d', default=DATA_FOLDER)

    args = parser.parse_args()
    run(args.entities, args.data)

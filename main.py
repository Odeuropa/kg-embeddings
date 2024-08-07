import argparse
import csv
import os

import pandas as pd
import torch
from gensim.models import KeyedVectors
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker
from tqdm import tqdm

from utils import save_word2vec_format

DATA_FOLDER = './data'
ENTITIES_PATH = 'voc.txt'

training_path = 'data/training'
testing_path = 'data/testing'

limit = -1


def density(n_vertices, n_edges):
    return n_edges / (n_vertices * (n_vertices - 1))


def is_pred_relevant(pred):
    p = pred.replace('%5E', '')
    return p.startswith('od_') or p.startswith('ecrm') or p.startswith('reo')


def smell_data_completeness(preds, ents, triples):
    relevant_preds = [p for p in preds if is_pred_relevant(p)]
    relevant_ents = [e.name for e in ents if e.name.startswith('http://data.odeuropa.eu/smell/')]
    # relevant_triples = [t for t in triples if t[1] in relevant_preds]
    relevant_triples = [t for t in triples if t[0] in relevant_ents]
    return len(relevant_triples) / (len(relevant_ents) * len(relevant_preds))


def run(entities_path, data_folder, algorithm):
    kg = KG()

    print('Reading graph..')
    data = []
    preds = []

    if algorithm == 'rdf2vec' or not os.path.isdir(training_path):
        if data_folder.endswith('csv'):
            with open(os.path.join(data_folder), 'r') as f:
                csv_reader = csv.reader(f, delimiter=',')
                for s, p, o in csv_reader:
                    subj = Vertex(s)
                    obj = Vertex(o)
                    pred = Vertex(p, predicate=True, vprev=subj, vnext=obj)
                    kg.add_walk(subj, pred, obj)
                    data.append((s, p, o))
                    preds.append(p)
        else:
            for x in tqdm(os.listdir(data_folder)):
                if not x.endswith('.csv'):
                    continue
                with open(os.path.join(data_folder, x), 'r') as f:
                    preds.append(x)
                    csv_reader = csv.reader(f, delimiter=',')

                    first = True
                    i = 0
                    for s, o in csv_reader:
                        if first:
                            first = False
                            continue
                        subj = Vertex(s)
                        obj = Vertex(o)
                        pred = Vertex(x, predicate=True, vprev=subj, vnext=obj)
                        kg.add_walk(subj, pred, obj)
                        data.append((s, x, o))
                        preds.append(x)
                        i += 1
                        if i == limit:
                            break

        print('Nb of vertices:', len(kg._vertices))
        print('Nb of entities in the graph:', len(kg._entities))
        print('Nb of predicates in the graph:', len(set(preds)))
        print('Nb of edges in the graph:', len(data))
        print('Density: ', density(len(kg._entities), len(data)))
        print('Smell data completeness: ', smell_data_completeness(set(preds), kg._entities, data))

    if algorithm == 'rdf2vec':
        if entities_path == 'all':
            entities = []
            for pt in ['voc.txt', 'smells.txt']:
                with open(pt, 'r') as f:
                    ent = [x.strip() for x in f.readlines()][1:]
                    entities += [x for x in ent if kg.is_exist([x])]

            entities += preds
        else:
            with open(entities_path, 'r') as f:
                entities = [x.strip() for x in f.readlines()][1:]
                entities = [x for x in entities if kg.is_exist([x])]

        print('Nb of entities for which we are computing embeddings:', len(entities))
        out_path = entities_path.replace('.txt', '.kv')
        train_rdf2vec(kg, entities, out_path)
    else:
        train_pykeen(data, algorithm)


def train_rdf2vec(kg, entities, out_path):
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=100),
        walkers=[RandomWalker(max_depth=4, max_walks=100, with_reverse=False, n_jobs=4)],
        # verbose=1
    )

    print('Generating embeddings...')
    transformer.fit(transformer.get_walks(kg, entities), False)
    entities = [e for e in entities if e in transformer.embedder._model.wv]

    embeddings, literals = transformer.transform(kg, entities)
    transformer.save()

    kv = KeyedVectors(len(embeddings[0]))
    for (emb, uri) in tqdm(zip(embeddings, entities)):
        kv.add_vector(uri, emb)

    kv.save(out_path)


def train_pykeen(data, algorithm):
    if os.path.isdir(training_path):
        # load train and test
        print('Loading existing train and test')
        training = TriplesFactory.from_path_binary(training_path)
        testing = TriplesFactory.from_path_binary(testing_path)
    else:
        # Generate triples from the graph data
        print('Splitting train and test')
        df = pd.DataFrame.from_dict(data)
        tf = TriplesFactory.from_labeled_triples(df.values)

        # split triples into train and test
        training, testing = tf.split([0.8, 0.2], random_state=42)

        training.to_path_binary(training_path)
        testing.to_path_binary(testing_path)

    # generate embeddings using PyKEEN's pipeline method
    embedding_dim = 128
    result = pipeline(
        training=training,
        testing=testing,
        model=algorithm,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=200, batch_size=32),
        random_seed=42)
    result.save_to_directory(f'odeuropa_{algorithm}')

    entity_labels = training.entity_labeling.all_labels()
    # convert entities to ids
    entity_ids = torch.as_tensor(training.entities_to_ids(entity_labels))
    # retrieve the embeddings using entity ids
    entity_embeddings = result.model.entity_representations[0](indices=entity_ids)
    # create a dictionary of entity labels and embeddings
    entity_embeddings_dict = dict(zip(entity_labels, entity_embeddings.detach().cpu().numpy()))

    save_word2vec_format(f'embeddings/{algorithm}_entity.bin', entity_embeddings_dict, embedding_dim)

    # get relation labels from training set
    relation_labels = training.relation_labeling.all_labels()
    # convert relations to ids
    relation_ids = torch.as_tensor(training.relations_to_ids(relation_labels))
    # retrieve the embeddings using relation ids
    relation_embeddings = result.model.relation_representations[0](indices=relation_ids)
    # create a dictionary of relation labels and embeddings
    relation_embeddings_dict = dict(zip(relation_labels, relation_embeddings.detach().cpu().numpy()))

    save_word2vec_format(f'embeddings/{algorithm}_relation.bin', relation_embeddings_dict, embedding_dim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('KG generation')
    parser.add_argument('entities', default=ENTITIES_PATH)
    parser.add_argument('--data', '-d', default=DATA_FOLDER)
    parser.add_argument('--algorithm', '-a', default='rdf2vec')

    args = parser.parse_args()
    run(args.entities, args.data, args.algorithm)

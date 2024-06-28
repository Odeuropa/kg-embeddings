import argparse
import csv
import os
from collections import Counter
import pandas as pd

from pyrdf2vec.graphs import KG, Vertex
from tqdm import tqdm

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
    relevant_preds = preds
    relevant_ents = [e.name for e in ents if e.name.startswith('http://data.odeuropa.eu/smell/')]
    # relevant_triples = [t for t in triples if t[1] in relevant_preds]
    relevant_triples = triples
    return len(relevant_triples) / (len(relevant_ents) * len(relevant_preds))


def run(data_folder, threshold=3):
    kg = KG()

    print('Reading graph..')
    data = []
    relevant_triples = []
    preds = []
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
                if s.startswith('http://data.odeuropa.eu/smell/') or o.startswith('http://data.odeuropa.eu/smell'):
                    relevant_triples.append((s,x,o))
                i += 1
                if i == limit:
                    break

    print('*** ORIGINAL DATA ***')

    print('Nb of vertices:', len(kg._vertices))
    print('Nb of entities in the graph:', len(kg._entities))
    print('Nb of predicates in the graph:', len(set(preds)))
    print('Nb of edges in the graph:', len(data))
    print('Density: ', density(len(kg._entities), len(data)))
    relevant_preds = [p for p in set(preds) if is_pred_relevant(p)]
    print('Smell data completeness: ', smell_data_completeness(relevant_preds, kg._entities, relevant_triples))

    emission2smell = {}
    experience2smell = {}
    for x in data:
        s, p, o = x
        if 'F1_' in p or 'F2_' in p:
            if 'emission' in s:
                emission2smell[s] = o
            elif 'experience' in s:
                experience2smell[s] = o

    print('*** CLEANING DATA ***')
    data_clean = []
    kg_clean = KG()
    for x in data:
        s, p, o = x
        if not is_pred_relevant(p):
            continue
        if 'emission' in s:
            s = emission2smell.get(s)
        elif 'experience' in s:
            s = experience2smell.get(s)
        elif 'emission' in o:
            o = emission2smell.get(o)
        elif 'experience' in o:
            o = experience2smell.get(o)

        if s is None or o is None:
            continue
        data_clean.append((s, p, o))

        subj = Vertex(s)
        obj = Vertex(o)
        pred = Vertex(p, predicate=True, vprev=subj, vnext=obj)
        kg_clean.add_walk(subj, pred, obj)

    print('Nb of vertices:', len(kg_clean._vertices))
    print('Nb of entities in the graph:', len(kg_clean._entities))
    print('Nb of predicates in the graph:', len(set(preds)))
    print('Nb of edges in the graph:', len(data_clean))
    print('Density: ', density(len(kg_clean._entities), len(data_clean)))
    print('Smell data completeness: ', smell_data_completeness(relevant_preds, kg_clean._entities, data_clean))

    relevant_ents = [e.name for e in kg_clean._entities if e.name.startswith('http://data.odeuropa.eu/smell/')]
    relevant_triples = [t for t in data if (t[0] in relevant_ents or t[2] in relevant_ents) and t[1] in relevant_preds]

    ent_triples = []
    for e in relevant_ents:
        linked_triples = [t for t in relevant_triples if t[0] == e or t[2] == e]
        ent_triples.append((e, len(linked_triples)))

    print('#edge distribution: ' + str(Counter(elem[1] for elem in ent_triples)))

    print(f'*** KEEP ONLY DENSE SMELLS: THRESHOLD {threshold} ***')
    df = pd.DataFrame(ent_triples)
    dense_smells = df[df[1] > threshold][0].tolist()
    relevant_ents = dense_smells
    relevant_triples = [t for t in data if (t[0] in relevant_ents or t[2] in relevant_ents) and is_pred_relevant(t[1])]

    kg2 = KG()
    for s, p, o in relevant_triples:
        subj = Vertex(s)
        obj = Vertex(o)
        pred = Vertex(p, predicate=True, vprev=subj, vnext=obj)
        kg2.add_walk(subj, pred, obj)

    print('Nb of vertices:', len(kg2._vertices))
    print('Nb of entities in the graph:', len(kg2._entities))
    print('Nb of predicates in the graph:', len(set(relevant_preds)))
    print('Nb of edges in the graph:', len(relevant_triples))
    print('Density: ', density(len(kg2._entities), len(relevant_triples)))
    print('Smell data completeness: ', smell_data_completeness(set(relevant_preds), kg2._entities, relevant_triples))

    ent_triples2 = []
    for e in relevant_ents:
        linked_triples = [t for t in relevant_triples if t[0] == e or t[2] == e]
        ent_triples2.append((e, len(linked_triples)))

    print('#edge distribution: ' + str(Counter(elem[1] for elem in ent_triples2)))

    with open('dense_graph.csv', 'w') as f:
        file_writer = csv.writer(f)
        for x in relevant_triples:
            file_writer.writerow(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('KG generation')
    parser.add_argument('--data', '-d', default=DATA_FOLDER)
    parser.add_argument('--threshold', '-t', default=3, type=int)

    args = parser.parse_args()
    run(args.data, args.threshold)

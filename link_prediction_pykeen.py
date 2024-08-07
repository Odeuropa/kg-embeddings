import logging
from os import path

import pandas as pd
import torch
from pykeen import predict
from pykeen.triples.triples_factory import TriplesFactory
from tqdm import tqdm

ROOT = './odeuropa_transe'
testing_path = 'data/testing10k'
OD_HAD_SOURCE = 'od_F3_had_source%20%2F%20ecrm_P137_exemplifies.csv'

model = torch.load(path.join(ROOT, 'trained_model.pkl'))

tf = TriplesFactory.from_path_binary(path.join(ROOT, 'training_triples'))

top_concepts = pd.read_csv('data/top-concepts.csv')
concepts = list(set(top_concepts['sub'].to_list()))

print('**** Evaluation on direct concepts ****')


def parse_comma_separated_data(file_path):
    result = []
    with open(file_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            line = line.replace("\n", "")
            tokens = line.split(sep=",")
            result.append(tokens)
    return result


logging.getLogger("pykeen.triples.triples_factory").disabled = True
logging.getLogger("torch_max_mem.api").disabled = True

data = pd.read_csv(path.join(testing_path, 'entity_to_id.tsv'), sep='\t')
uris = list(data['label'])
evaluation_dataset = parse_comma_separated_data("dense_graph_10k.csv")
evaluation_dataset = [(s, p, o) for s, p, o in evaluation_dataset
                      if p == OD_HAD_SOURCE and s in uris]
print(len(evaluation_dataset))

correct_t = 0
total_t = 0
for triple in tqdm(evaluation_dataset):
    total_t += 1
    h, p, t = triple
    triples = [(h, OD_HAD_SOURCE, concept) for concept in concepts]
    t_pack = predict.predict_triples(model, triples=triples, triples_factory=tf)
    t_pred = t_pack.process(factory=tf)
    tail_labels = t_pred.df.sort_values('score', ascending=False)['tail_label']
    if tail_labels.empty:
        continue
    t_result = tail_labels.iloc[0]
    if t == t_result:
        correct_t += 1
print(f"Tail prediction accuracy: {correct_t} / {total_t} ({correct_t / total_t})")

print('**** Evaluation on top-level concepts ****')

correct_t = 0
total_t = 0
for triple in tqdm(evaluation_dataset):
    total_t += 1
    h, l, t = triple
    triples = [(h, OD_HAD_SOURCE, concept) for concept in concepts]
    t_pack = predict.predict_triples(model, triples=triples, triples_factory=tf)
    t_pred = t_pack.process(factory=tf)
    tail_labels = t_pred.df.sort_values('score', ascending=False)['tail_label']
    if tail_labels.empty:
        continue
    t_result = tail_labels.iloc[0]
    top_t = top_concepts[top_concepts['sub'] == t]['top'].to_list()
    top_t_result = top_concepts[top_concepts['sub'] == t_result]['top'].to_list()
    if any(item in top_t for item in top_t_result):
        correct_t += 1
print(f"Tail prediction accuracy: {correct_t} / {total_t} ({correct_t / total_t})")

logging.getLogger("pykeen.triples.triples_factory").disabled = False
logging.getLogger("torch_max_mem.api").disabled = False

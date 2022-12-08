import os
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, CSV
import urllib.parse

config_folder = './config'
data_folder = './data'

with open(os.path.join(config_folder, 'prefixes.txt'), 'r') as file:
    prefixes = [x.strip() for x in file.readlines()]

with open(os.path.join(config_folder, 'object_properties.txt'), 'r') as file:
    props = [x.strip() for x in file.readlines()]

sparql = SPARQLWrapper("http://data.odeuropa.eu/repositories/odeuropa")
sparql.setReturnFormat(CSV)

os.makedirs(data_folder, exist_ok=True)

for p in tqdm(props):
    q = '\n'.join(prefixes) + '\n' + 'SELECT * WHERE { ?s %s ?o }' % p
    sparql.setQuery(q)
    ret = sparql.queryAndConvert()
    with open(os.path.join(data_folder, urllib.parse.quote(p.replace(':', '_'), safe='') + '.csv'), 'wb') as file:
        file.write(ret)

with open(os.path.join(config_folder, 'get_voc.rq'), 'r') as file:
    query = file.read()

sparql.setQuery(query)
ret = sparql.queryAndConvert()
with open(os.path.join('./', 'voc.txt'), 'wb') as file:
    file.write(ret)

with open(os.path.join(config_folder, 'get_smells.rq'), 'r') as file:
    query = file.read()

sparql.setQuery(query)
ret = sparql.queryAndConvert()
with open(os.path.join('./', 'smells.txt'), 'wb') as file:
    file.write(ret)

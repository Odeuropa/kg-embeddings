# KG Embeddings

Generate embeddings on the graph, with RDF2Vec, using [.

## Training

In the [config folder](./config), edit appropriately the following files:
- `object_properties.txt` the list of object properties which will be taken into account for the random walks (1 per line);
- `prefixes.txt` to be added to the SPARQL queries
- `get_entities.rq` the query for getting the URIs of all entities of interest.

Run the following commands for downloading the data on your machine:
    
    pip install -r requirements.txt
    python preprocessing.py

Finally, generate the embeddings using:
    
    python main.py [entities list]

where `entities list` is a list of entities uri (1 per line) in a textual file.
Following the generated files by the preprocessing, you can run:

    python main.py voc.txt
    python main.py smells.txt

This is producing an `[entity].kv` file, which is a [gensim's KeyedVector](https://radimrehurek.com/gensim/models/keyedvectors.html) file.

# Load and use embeddings

Load embeddings in this way:

    emb = KeyedVectors.load('emb.kv')

Search the most similar to a term:
    
    emb.most_similar('http://data.odeuropa.eu/vocabulary/olfactory-objects/269', topn=10) # incense
    
    # 0.7755   http://data.odeuropa.eu/vocabulary/olfactory-objects/267   Frankincense

Refer to [gensim's documentation](https://radimrehurek.com/gensim/models/keyedvectors.html) for further possibilities.
# KG Embeddings

Generate embeddings on the graph, with RDF2Vec.

## Training

In the [config folder](./config), edit appropriately the following files:
- `object_properties.txt` the list of object properties which will be taken into account for the random walks (1 per line);
- `prefixes.txt` to be added to the SPARQL queries
- `get_entities.rq` the query for getting the URIs of all entities of interest.

Run the following commands for downloading the data on your machine:
    
    pip install -r requirements.txt
    python preprocessing.py

Finally, generate the embeddings using:
    
    python main.py

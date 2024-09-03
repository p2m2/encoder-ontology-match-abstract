from tqdm import tqdm
import os, torch
from llm_semantic_annotator import build_corpus, manage_tags
from llm_semantic_annotator import abstract_preparation, manage_abstracts
from llm_semantic_annotator import torch_utils, best_similarity_for_tag

from rich import print

import argparse

# Listes des ontologies du projet Planteome
ontologies = { 
    'peco' : {
        'url' : "http://purl.obolibrary.org/obo/peco.owl",
        'prefix' : "http://purl.obolibrary.org/obo/PECO_",
        'format' : 'xml',
        'properties' : [ "<http://purl.obolibrary.org/obo/IAO_0000115>" ]
        },
    'po' : {
        'url' : "http://purl.obolibrary.org/obo/po.owl",
        'prefix' : "http://purl.obolibrary.org/obo/PO_",
        'format' : 'xml',
        'properties' : [ "<http://purl.obolibrary.org/obo/IAO_0000115>" ]
    },
    'pso' : {
        'url' : "http://purl.obolibrary.org/obo/pso.owl",
        'prefix' : "http://purl.obolibrary.org/obo/PSO_",
        'format' : 'xml',
        'properties' : [ "<http://purl.obolibrary.org/obo/IAO_0000115>" ]
    },
    'to' : {
        'url' : "http://purl.obolibrary.org/obo/to.owl",
        'prefix' : "http://purl.obolibrary.org/obo/TO_",
        'format' : 'xml',
        'properties' : [ "<http://purl.obolibrary.org/obo/IAO_0000115>" ]
    },
    'ms' : {
        'url' : 'http://purl.obolibrary.org/obo/ms.owl',
        'prefix' : "http://purl.obolibrary.org/obo/MS_",
        'format' : 'xml',
        'properties' : [ "<http://purl.obolibrary.org/obo/IAO_0000115>" ]
    }
}

threshold = 0.74  # Seuil de similarité
debug_nb_ncbi_request=1
debug_nb_terms_by_ontology=-1
debug_nb_abstracts_by_search=-1
#selected_term = "plants+AND+metabolomics+AND+glucosinolate"
selected_term = "abiotic+AND+metabolomics+AND+plant+AND+stress+AND+brassicaceae"

tag_embeddings = manage_tags(ontologies,debug_nb_terms_by_ontology)
chunk_embeddings = manage_abstracts(selected_term,debug_nb_ncbi_request)

# Comparer chaque chunk avec les tags

results = []
results_complete_similarities = []

for doi,chunk_embedding in tqdm(chunk_embeddings.items()):
    # Convertir le tensor en une forme serialisable pour le dictionnaire
    chunk_embedding_key = chunk_embedding.numpy().tobytes()
    tag_similarities = {}
    complete_similarities = {}
    for tag, descriptions_embeddings in tag_embeddings.items():
        similarity = best_similarity_for_tag(chunk_embedding, {tag: descriptions_embeddings})
        tag_similarities[tag] = similarity
        if similarity>=threshold :
            complete_similarities[tag] = similarity
    
    results_complete_similarities.append(complete_similarities)

    # Associer le tag avec la similarité la plus élevée si au-dessus du seuil
    best_tag = max(tag_similarities, key=tag_similarities.get)
    if tag_similarities[best_tag] >= threshold:
        results.append((best_tag, tag_similarities[best_tag]))
    else:
        results.append((None, None))

# Afficher les résultats
for i, tuple in enumerate(results):
    tag = tuple[0]
    similarity = tuple[1]
    #print(chunks[i])
    print(results_complete_similarities[i])
    if tag:
        print(f"Tag: {tag}\nSimilarity: {similarity:.4f}\n")
    else:
        print(f"Tag: None (Below threshold)\n")

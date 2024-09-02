from tqdm import tqdm
import torch
from build_corpus import get_corpus
from abstract_preparation import get_ncbi_abstracts
from torch_utils import encode_text, best_similarity_for_tag
from csv_utils import list_of_dicts_to_csv
# Listes des ontologies du projet Planteome
ontologies = { 
    'peco' : {
        'url' : "http://purl.obolibrary.org/obo/peco.owl",
        'prefix' : "http://purl.obolibrary.org/obo/PECO_",
        'format' : 'xml'
        },
    'po' : {
        'url' : "http://purl.obolibrary.org/obo/po.owl",
        'prefix' : "http://purl.obolibrary.org/obo/PO_",
        'format' : 'xml'
    },
    'to' : {
        'url' : "http://purl.obolibrary.org/obo/to.owl",
        'prefix' : "http://purl.obolibrary.org/obo/TO_",
        'format' : 'xml'
    }
}
description_uri="<http://purl.obolibrary.org/obo/IAO_0000115>"
threshold = 0.69  # Seuil de similarité
debug_nb_terms_by_ontology=50
debug_nb_abstracts_by_search=30
#selected_term = "plants+AND+metabolomics+AND+glucosinolate"
selected_term = "abiotic+AND+metabolomics+AND+plant+AND+stress"

tags = get_corpus(ontologies, description_uri=description_uri, debug_nb_terms_by_ontology=debug_nb_terms_by_ontology)
chunks = get_ncbi_abstracts(selected_term,1)[0:debug_nb_abstracts_by_search]

list_of_dicts_to_csv(tags, "tags.csv")
list_of_dicts_to_csv(chunks, "chunks.csv")

print("tags embeddings")
# Encoder les descriptions des tags
tag_embeddings = {}
for item in tqdm(tags):
    embeddings = encode_text(item['description'])
    tag_embeddings[item['label']] = embeddings

print("chunks embeddings")
# Encoder les chunks de texte
chunk_embeddings = []
for chunk in tqdm(chunks):
    chunk_embeddings.append(encode_text(chunk['abstract']))

#    idx = embedding.numpy().tostring()


# Comparer chaque chunk avec les tags

results = []
results_complete_similarities = []

for chunk_embedding in tqdm(chunk_embeddings):
    
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
    print(tuple)
    tag = tuple[0]
    similarity = tuple[1]
    print(chunks[i])
    print(results_complete_similarities[i])
    if tag:
        print(f"Tag: {tag}\nSimilarity: {similarity:.4f}\n")
    else:
        print(f"Tag: None (Below threshold)\n")

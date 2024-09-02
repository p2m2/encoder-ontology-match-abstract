from tqdm import tqdm
import torch
from build_corpus import get_corpus
from abstract_preparation import get_ncbi_abstracts
from torch_utils import encode_text, best_similarity_for_tag

# Listes des ontologies du projet Planteome
ontologies = { 
    'peco' : {
            'url' : "http://purl.obolibrary.org/obo/peco.owl",
            'prefix' : "http://purl.obolibrary.org/obo/PECO_",
            'format' : 'xml'
        }
    }
debug_nb_terms_by_ontology=5
debug_nb_abstracts_by_search=5
selected_term = "plants+AND+metabolomics+AND+spring"

tags = get_corpus(ontologies, debug_nb_terms_by_ontology)
chunks = get_ncbi_abstracts(selected_term,debug_nb_abstracts_by_search)

print("tags embeddings")
# Encoder les descriptions des tags
tag_embeddings = {}
for index, item in tqdm(enumerate(tags.items())):
    tag, descriptions = item
    ##print(tag)
    embeddings = [encode_text(description) for description in descriptions]
    tag_embeddings[tag] = torch.stack(embeddings)
    

print("chunks embeddings")
# Encoder les chunks de texte
chunk_embeddings = [encode_text(chunk['abstract']) for chunk in chunks]
#    idx = embedding.numpy().tostring()


# Comparer chaque chunk avec les tags
threshold = 0.5  # Seuil de similarité
results = []

for chunk_embedding in tqdm(chunk_embeddings):
    
    # Convertir le tensor en une forme serialisable pour le dictionnaire
    chunk_embedding_key = chunk_embedding.numpy().tostring()
    print("-- chunks --")
    tag_similarities = {}
    for tag, descriptions_embeddings in tag_embeddings.items():
        similarity = best_similarity_for_tag(chunk_embedding, {tag: descriptions_embeddings})
        tag_similarities[tag] = similarity
    
    # Associer le tag avec la similarité la plus élevée si au-dessus du seuil
    best_tag = max(tag_similarities, key=tag_similarities.get)
    if tag_similarities[best_tag] >= threshold:
        results.append((best_tag, tag_similarities[best_tag]))
    else:
        results.append((None, None))

# Afficher les résultats
for i, tag, similarity in enumerate(results):
    print(chunks[i])
    if tag:
        print(f"Tag: {tag}\nSimilarity: {similarity:.4f}\n")
    else:
        print(f"Tag: None (Below threshold)\n")

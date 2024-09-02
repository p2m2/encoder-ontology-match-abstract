import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import json

from build_corpus import get_corpus
from abstract_preparation import get_ncbi_abstracts

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

# Charger le modèle BERT et le tokenizer
# bert-base-uncased
# bert-large-uncased
# roberta-base

#tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
#model = BertModel.from_pretrained('bert-large-uncased')

# FacebookAI/roberta-base
# sentence-transformers/all-MiniLM-L6-v2

#https://huggingface.co/spaces/mteb/leaderboard
# mixedbread-ai/mxbai-embed-large-v1

tokenizer = AutoTokenizer.from_pretrained('mixedbread-ai/mxbai-embed-large-v1')
model = AutoModel.from_pretrained('mixedbread-ai/mxbai-embed-large-v1')

#SentenceTransformer

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

tags = get_corpus(ontologies, debug_nb_terms_by_ontology)
chunks = get_ncbi_abstracts(selected_term,debug_nb_abstracts_by_search)
print(chunks)

raise Exception("test")

print("tags embeddings")
# Encoder les descriptions des tags
tag_embeddings = {}
for index, item in enumerate(tags.items()):
    tag, descriptions = item
    print(tag)
    embeddings = [encode_text(description) for description in descriptions]
    tag_embeddings[tag] = torch.stack(embeddings)
    if index >= 9:  # Commence à 0, donc 9 pour 10 itérations
        break


print("chunks embeddings")
# Encoder les chunks de texte
chunk_embeddings = [encode_text(chunk) for chunk in chunks]

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b).item()

def best_similarity_for_tag(chunk_embedding, tag_embeddings):
    best_similarity = float('-inf')
    for descriptions_embeddings in tag_embeddings.values():
        for description_embedding in descriptions_embeddings:
            similarity = cosine_similarity(chunk_embedding, description_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
    return best_similarity

# Comparer chaque chunk avec les tags
threshold = 0.2  # Seuil de similarité
results = []

# Dictionnaire pour associer les embeddings aux chunks
chunk_dict = {encode_text(chunk).numpy().tostring(): chunk for chunk in chunks}

for chunk_embedding in chunk_embeddings:
    
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
        results.append((chunk_dict[chunk_embedding_key], best_tag, tag_similarities[best_tag]))
    else:
        results.append((chunk_dict[chunk_embedding_key], None, None))

# Afficher les résultats
for chunk, tag, similarity in results:
    if tag:
        print(f"Chunk: '{chunk}'\nTag: {tag}\nSimilarity: {similarity:.4f}\n")
    else:
        print(f"Chunk: '{chunk}'\nTag: None (Below threshold)\n")

import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

import torch.nn.functional as F

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

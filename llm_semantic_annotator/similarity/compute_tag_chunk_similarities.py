import torch,json,os
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F
from tqdm import tqdm
from llm_semantic_annotator import load_results,save_results

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

def compare_tags_with_chunks(tag_embeddings, chunk_embeddings,config):
    threshold = config['threshold_similarity_tag_chunk']
    debug_nb_similarity_compute = config['debug_nb_similarity_compute']
    retention_dir = config['retention_dir']
    
    if 'force' not in config:
        config['force'] = False
    
    filename = retention_dir+f"/similarity_{threshold}_{debug_nb_similarity_compute}.json"
    results_complete_similarities = load_results(filename)
    
    if results_complete_similarities is None:
        results_complete_similarities = {}

    record=0
    for doi,chunk_embedding in tqdm(chunk_embeddings.items()):
        if doi in results_complete_similarities:
            continue
        
        complete_similarities = {}
        for tag, descriptions_embeddings in tag_embeddings.items():
            similarity = best_similarity_for_tag(chunk_embedding, {tag: descriptions_embeddings})
            if similarity>=threshold :
                complete_similarities[tag] = similarity
            
        if doi not in results_complete_similarities:
            results_complete_similarities[doi] = complete_similarities

        if record == debug_nb_similarity_compute:
            break

        record+=1
    
    save_results(results_complete_similarities,filename)
    return results_complete_similarities
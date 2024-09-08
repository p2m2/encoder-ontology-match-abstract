#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nom du module : ModelEmbeddingManagement

Description : Gestion de l'embeddings (Tag Et Abstract chunks).

Auteur : Votre Nom
Date de création : YYYY-MM-DD
Dernière modification : YYYY-MM-DD
Version : X.Y.Z
Licence : Nom de la licence (ex: MIT, GPL, etc.)
"""

__author__ = "Olivier Filangi"
__copyright__ = "Copyright (c) 2024, Inrae"
__credits__ = []
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Olivier Filangi"
__email__ = "olivier.filangi@inrae.fr"
__status__ = "Devel"

import torch,json,os
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.spatial.distance import cdist

import torch.nn.functional as F
from sentence_transformers.util import cos_sim
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

class ModelEmbeddingManagement:
    def __init__(self,config):
        self.config=config
        self.retention_dir = config['retention_dir']

        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        #self.model_name = 'mixedbread-ai/mxbai-embed-large-v1'
        #self.model_name = 'sentence-transformers/all-mpnet-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.batch_size=32
        self.threshold_similarity_tag = 0.75
        self.model_suffix=self.model_name.split("/").pop()

    def get_filename_pth(self,name_embeddings):
        return f"{self.retention_dir}/{name_embeddings}-{self.model_suffix}.pth"
    
    def load_pth(self,name_embeddings):
        filename = self.get_filename_pth(name_embeddings)
        
        tag_embeddings = {}

        if os.path.exists(filename):
            print(f"load tags embeddings - {filename}")
            tag_embeddings = torch.load(filename)
        return tag_embeddings

    def save_pth(self,tag_embeddings,name_embeddings):
        filename = self.get_filename_pth(name_embeddings)
        torch.save(tag_embeddings, filename)

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_text_base(self,text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def encode_text_allMiniLML6V2(self,text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return F.normalize(self.mean_pooling(outputs, inputs['attention_mask']), p=2, dim=1)
        #return outputs.last_hidden_state.mean(dim=1)

    def encode_text(self,text):
        return self.encode_text_allMiniLML6V2(text)


    def encode_text_batch_allMiniLML6V2(self,texts, batch_size=32):
        # Passage en mode évaluation
        self.model.eval()
        
        all_embeddings = []
        
        # Traitement par lots
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenization du lot
            inputs = self.tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True)
            
            # Déplacement des tenseurs sur GPU si disponible
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                self.model.to('cuda')
            
            # Calcul des embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Pooling et normalisation
            embeddings = F.normalize(self.mean_pooling(outputs, inputs['attention_mask']), p=2, dim=1)
            
            # Déplacement des embeddings sur CPU si nécessaire
            if torch.cuda.is_available():
                embeddings = embeddings.cpu()
            
            all_embeddings.append(embeddings)
        
        # Concaténation de tous les embeddings
        return torch.cat(all_embeddings, dim=0)

    def encode_text_batch(self,texts):
        return self.encode_text_batch_allMiniLML6V2(texts, self.batch_size)

    def cosine_similarity(self,a, b):
        return cos_sim(a, b)[0].item()
        #return F.cosine_similarity(a, b).item()

    def best_similarity_for_tag(self,chunks_embedding, tag_embeddings):
        best_similarity = float('-inf')
        for description_embedding in tag_embeddings.values():
            #for description_embedding in descriptions_embeddings:
            for chunk_embedding in chunks_embedding:
                similarity = self.cosine_similarity(chunk_embedding, description_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
        return best_similarity

    def encode_tags(self,tags):
        """
        tag : {
                'term',
                'label',
                'rdfs_label',
                'description'
            }
        """

        toencode = []
        print("Flat tags to build batch.....")
        for item in tqdm(tags):
            if 'description' in item and item['description'].strip() != '':
                toencode.append(f"{item['rdfs_label']} - {item['description']}")
            else:
                if 'rdfs_label' not in item or item['rdfs_label'] == '':
                    raise "encode_tags: tag empty !"
                toencode.append(item['rdfs_label']) 

        print("batch encoding.....")
        embeddings = self.encode_text_batch(toencode)

        tags_embedding={}
        print("set encoding.....")
        for idx,item in tqdm(enumerate(embeddings)):
            tags_embedding[tags[idx]['label']] = item

        return tags_embedding


    def compare_tags_with_chunks(self, tag_embeddings, chunks_embeddings, config):
        threshold = config['threshold_similarity_tag_chunk']
        debug_nb_similarity_compute = config['debug_nb_similarity_compute']
        
        # Convertir les embeddings en arrays NumPy pour une meilleure performance
        tag_list = list(tag_embeddings.keys())
        tag_embeddings_matrix = np.array([tag_embeddings[tag].cpu().numpy() for tag in tag_list])
        
        results_complete_similarities = {}
        
        for doi, chunks_embedding in tqdm(list(chunks_embeddings.items())[:debug_nb_similarity_compute]):
            # Convertir chunks_embedding en array NumPy
            chunks_matrix = np.array([chunk.cpu().numpy() for chunk in chunks_embedding])
            
            # Calcul vectorisé des similarités
            similarities = 1 - cdist(chunks_matrix, tag_embeddings_matrix, metric='cosine')
            max_similarities = np.max(similarities, axis=0)
            
            # Filtrage des similarités au-dessus du seuil
            above_threshold = max_similarities >= threshold
            complete_similarities = {tag: sim for tag, sim in zip(tag_list, max_similarities) if sim >= threshold}
            
            # Filtrage des tags similaires
            if len(complete_similarities) > 1:
                tags_to_keep = list(complete_similarities.keys())
                tag_embeddings_filtered = tag_embeddings_matrix[[tag_list.index(tag) for tag in tags_to_keep]]
                tag_similarities = 1 - cdist(tag_embeddings_filtered, tag_embeddings_filtered, metric='cosine')
                np.fill_diagonal(tag_similarities, 0)
                
                while True:
                    max_sim = np.max(tag_similarities)
                    if max_sim <= self.threshold_similarity_tag:
                        break
                    i, j = np.unravel_index(np.argmax(tag_similarities), tag_similarities.shape)
                    if complete_similarities[tags_to_keep[i]] > complete_similarities[tags_to_keep[j]]:
                        del complete_similarities[tags_to_keep[j]]
                        tags_to_keep.pop(j)
                    else:
                        del complete_similarities[tags_to_keep[i]]
                        tags_to_keep.pop(i)
                    tag_similarities = np.delete(tag_similarities, min(i, j), axis=0)
                    tag_similarities = np.delete(tag_similarities, min(i, j), axis=1)
            
            results_complete_similarities[doi] = complete_similarities
        
        return results_complete_similarities
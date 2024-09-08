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
    def __init__(self):
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        #self.model_name = 'mixedbread-ai/mxbai-embed-large-v1'
        #self.model_name = 'sentence-transformers/all-mpnet-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.batch_size=32


    #SentenceTransformer

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

    def compare_tags_with_chunks(self,tag_embeddings, chunks_embeddings,config):
        threshold = config['threshold_similarity_tag_chunk']
        debug_nb_similarity_compute = config['debug_nb_similarity_compute']
        retention_dir = config['retention_dir']
        
        if 'force' not in config:
            config['force'] = False
        
        results_complete_similarities = {}

        record=0
        tag_used_encode = {}
        for doi,chunks_embedding in tqdm(chunks_embeddings.items()):
            
            complete_similarities = {}
            for tag, descriptions_embeddings in tag_embeddings.items():
                similarity = self.best_similarity_for_tag(chunks_embedding, {tag: descriptions_embeddings})
                if similarity>=threshold :
                    complete_similarities[tag] = similarity
            # Filtre : si 2 tags sont suffisamment similaires, on en garde un seul (la meilleure similarité)
            change = True
            lKeys = []

            for tag in list(complete_similarities.keys()):
                if tag not in tag_used_encode:
                    lKeys.append(tag)
            if len(lKeys)>0:
                tag_used_encode.update({
                    lKeys[ind] : enc
                    for ind, enc in enumerate(self.encode_text_batch(lKeys))})
            
            while change:
                change = False
                for tag1 in complete_similarities:
                    for tag2 in complete_similarities:
                        if tag1 != tag2 :
                            score = self.cosine_similarity(tag_used_encode[tag1] , tag_used_encode[tag2])
                            if score > 0.75:
                                if complete_similarities[tag1] > complete_similarities[tag2]:
                                    del complete_similarities[tag2]
                                else:
                                    del complete_similarities[tag1]
                                change = True
                                break
                    if change:
                        break

            results_complete_similarities[doi] = complete_similarities

            if record == debug_nb_similarity_compute:
                break

            record+=1
        
        return results_complete_similarities

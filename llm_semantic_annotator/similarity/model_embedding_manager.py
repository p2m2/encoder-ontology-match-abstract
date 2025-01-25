#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nom du module : ModelEmbeddingManager

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

import torch,os,re
from sentence_transformers import SentenceTransformer, SimilarityFunction
import numpy as np
from scipy.spatial.distance import cdist
from rich import print
import pandas as pd

from sentence_transformers.util import cos_sim
from tqdm import tqdm

# https://huggingface.co/spaces/mteb/leaderboard


class ModelEmbeddingManager():
    _info_displayed = False
    
    def __init__(self, config):
        self.config = config
        if 'retention_dir' not in config:
            self.retention_dir = "/tmp"
        else:
            self.retention_dir = config['retention_dir']
        self.encoder=config['encodeur']
        self.model_suffix=self.encoder.split('/')[-1]
        self.model_name = config.get('encodeur', self.encoder)
        self.model = SentenceTransformer(self.model_name)
        self.model.similarity_fn_name = SimilarityFunction.MANHATTAN
        self.batch_size = config.get('batch_size', 32)
        self.threshold_similarity_tag = config.get('threshold_similarity_tag', 0.75)
        self.threshold_similarity_tag_chunk = config.get('threshold_similarity_tag_chunk', 0.75)
        
        if not ModelEmbeddingManager._info_displayed :
            print("------------------------------------")
            print("Encoder:", self.model_name)
            print("Threshold similarity tag:", self.threshold_similarity_tag)
            print("Threshold similarity tag chunk:", self.threshold_similarity_tag_chunk)
            print("Batch size:", self.batch_size)
            print("------------------------------------")
            ModelEmbeddingManager._info_displayed = True
            

    def get_filename_pth(self, name_embeddings):
        return f"{self.retention_dir}/{name_embeddings}-{self.model_name.split('/')[-1]}.pth"
    
    def load_filepth(self, filename_embeddings):
        if torch.cuda.is_available():
            dev=torch.device('gpu')
        else:
            dev=torch.device('cpu')
            
        return torch.load(filename_embeddings,weights_only=True,map_location=dev)

    def load_pth(self, name_embeddings):
        if torch.cuda.is_available():
            dev=torch.device('gpu')
        else:
            dev=torch.device('cpu')
        
        filename = self.get_filename_pth(name_embeddings)
        
        tag_embeddings = {}
        if os.path.exists(filename):
            print(f"Loading embeddings from {filename}")
            tag_embeddings = torch.load(filename,weights_only=True,map_location=dev)
        return tag_embeddings

    def save_pth(self, tag_embeddings, name_embeddings):
        filename = self.get_filename_pth(name_embeddings)
        torch.save(tag_embeddings, filename)

    def encode_text(self, text):
        return self.model.encode(text, convert_to_tensor=True)

    def encode_text_batch(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, convert_to_tensor=True)

    def cosine_similarity(self, a, b):
        return self.model.similarity(a, b)

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
            tags_embedding[tags[idx]['term']] = { 
                                                 'ontology' : tags[idx]['ontology'] ,
                                                 'label' : tags[idx]['rdfs_label'] , 
                                                 'group' : tags[idx]['group'] ,
                                                 'emb' : item }

        return tags_embedding
    
    def encode_abstracts(self,abstracts) :
        """
        abstract : {
                'doi',
                'title',
                'abstract'
            }
        """

        chunks_toencode = []
        chunks_doi_ref = []
        lcount = 0
            
        for item in tqdm(abstracts):
            if 'abstract' in item and item['abstract'].strip() != '':
                if 'title' in item and item['title'].strip() != '':
                    if 'doi' not in item:
                        print(f"doi not found : {item['title']}")
                        continue
                    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', item['abstract'])
                    # title
                    chunks_doi_ref.append(item['doi'])
                    chunks_toencode.append(item['title'])
                    lcount+=1
                    # with all abstract sentences
                    for s in sentences:
                        chunks_toencode.append(s)
                        chunks_doi_ref.append(item['doi'])
                        lcount+=1
                
        print("batch encoding.....")

        #df = pd.DataFrame({
        #    'doi': chunks_doi_ref,
        #    'chunks': chunks_toencode
        #})
        
        #df.to_csv(self.retention_dir+f"/{genname}.csv", index=False)
        
        embeddings = self.encode_text_batch(chunks_toencode)
        abstracts_embedding={}
        for idx in chunks_doi_ref:
            abstracts_embedding[idx] = []

        print("set encoding.....")
        for idx,item in tqdm(enumerate(embeddings)):
            abstracts_embedding[chunks_doi_ref[idx]].append(item)

        return abstracts_embedding

    def compare_tags_with_chunks(self, tag_embeddings, chunks_embeddings):
        # Convertir les embeddings en arrays NumPy pour une meilleure performance
        tag_list = list(tag_embeddings.keys())
        tag_embeddings_matrix = np.array([tag_embeddings[tag].cpu().numpy() for tag in tag_list])
        
        results_complete_similarities = {}

        for doi, chunks_embedding in tqdm(list(chunks_embeddings.items())):
            # Convertir chunks_embedding en array NumPy
            chunks_matrix = np.array([chunk.cpu().numpy() for chunk in chunks_embedding])
            
            # Calcul vectorisé des similarités
            similarities = 1 - cdist(chunks_matrix, tag_embeddings_matrix, metric='cosine')
            max_similarities = np.max(similarities, axis=0)

            # Filtrage des similarités au-dessus du seuil
            complete_similarities = {tag: sim for tag, sim in zip(tag_list, max_similarities) if sim >= self.threshold_similarity_tag_chunk}
            
            results_complete_similarities[doi] = complete_similarities
        
        return results_complete_similarities
    
    # todo : evaluate perfs....
    def compare_tags_with_chunks_v2(self, tag_embeddings, chunks_embeddings):
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tag_list = list(tag_embeddings.keys())
        tag_embeddings_matrix = torch.stack([tag_embeddings[tag] for tag in tag_list]).to(device)
        
        results_complete_similarities = {}

        for doi, chunks_embedding in tqdm(list(chunks_embeddings.items())):
            chunks_matrix = torch.stack(chunks_embedding).to(device)
            # Calcul vectorisé des similarités sur GPU
            # Normaliser les vecteurs pour le calcul de la similarité cosinus
            chunks_norm = F.normalize(chunks_matrix, p=2, dim=1)
            tags_norm = F.normalize(tag_embeddings_matrix, p=2, dim=1)
            
            # Calcul de la similarité cosinus
            similarities = torch.mm(chunks_norm, tags_norm.t())
            max_similarities, _ = torch.max(similarities, dim=0)

            # Filtrage des similarités au-dessus du seuil
            complete_similarities = {tag: sim.item() for tag, sim in zip(tag_list, max_similarities) if sim >= self.threshold_similarity_tag_chunk}
            
            results_complete_similarities[doi] = complete_similarities
        
        return results_complete_similarities

    
    def remove_similar_tags_by_doi(self, tag_embeddings, complete_similarities):
        tag_list = list(tag_embeddings.keys())
        tag_embeddings_matrix = np.array([tag_embeddings[tag]['emb'].cpu().numpy() for tag in tag_list])

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
            
        return complete_similarities

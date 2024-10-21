from llm_semantic_annotator import ModelEmbeddingManager
from llm_semantic_annotator import OwlTagManager
from llm_semantic_annotator import TaxonTagManager
from llm_semantic_annotator import AbstractManager

from llm_semantic_annotator import display_ontologies_distribution
from llm_semantic_annotator import display_best_similarity_abstract_tag
from llm_semantic_annotator import display_ontologies_summary
from llm_semantic_annotator import create_rdf_graph,save_rdf_graph

import warnings
import json 
import os 
import re 

def setup_general_config(config_all,methode):
    
    if methode not in config_all:
        config = {}
    else:
        config = config_all[methode]
    config['retention_dir'] = config_all['retention_dir']
    config['force'] = config_all['force']
    
    return config

def main_populate_owl_tag_embeddings(config_all):
    """Fonction principale pour générer et stocker les embeddings de tags dans une base."""
    config = setup_general_config(config_all,'populate_owl_tag_embeddings')

    # Utilisez les paramètres de config ici
    print(f"Ontologies : {config['ontologies']}")
    print(f"Nb terms to compute : {config['debug_nb_terms_by_ontology']}")
    
    mem = ModelEmbeddingManager(config_all)
    
    OwlTagManager(config,mem).manage_tags()

def main_populate_gbif_taxon_tag_embeddings(config_all):
    config = setup_general_config(config_all,'populate_gbif_taxon_tag_embeddings')
    mem = ModelEmbeddingManager(config_all)

    TaxonTagManager(config,mem).manage_gbif_taxon_tags()

def main_populate_ncbi_taxon_tag_embeddings(config_all):
    config = setup_general_config(config_all,'populate_ncbi_taxon_tag_embeddings')
    mem = ModelEmbeddingManager(config_all)

    TaxonTagManager(config,mem).manage_ncbi_taxon_tags()

def main_populate_abstract_embeddings(config_all):
    
    config = setup_general_config(config_all,'populate_abstract_embeddings')
    mem = ModelEmbeddingManager(config_all)

    AbstractManager(config,mem).manage_abstracts()

def main_compute_tag_chunk_similarities(config_all):
    """Fonction principale pour calculer la similarité entre tous les tags et chunks."""
    config_owl = setup_general_config(config_all,'populate_owl_tag_embeddings')
    config_abstract = setup_general_config(config_all,'populate_abstract_embeddings')
    
    mem = ModelEmbeddingManager(config_all)

    
    tags_pth_files = OwlTagManager(config_owl,mem).get_files_tags_embeddings()
    
    if len(tags_pth_files) == 0:
        raise FileNotFoundError("No tags embeddings found")
    
    tags_taxon_pth_files = TaxonTagManager(config_owl,mem).get_files_tags_ncbi_taxon_embeddings()
    
    if len(tags_taxon_pth_files) == 0:
        warnings.warn("No tags taxon embeddings found")

    tags_pth_files.extend(tags_taxon_pth_files)
    
    abstracts_pth_files = AbstractManager(config_abstract,mem).get_files_abstracts_embeddings()

    if len(abstracts_pth_files) == 0:
        raise FileNotFoundError("No abstracts embeddings found")
    
    ### Loading tags embeddings
    ### -----------------------
    tag_embeddings_all = {}
    tag_embeddings = {}

    for tags_pth_file in tags_pth_files:
        current_embeddings = mem.load_filepth(tags_pth_file)
        
        # Mise à jour de tag_embeddings_all
        tag_embeddings_all.update(current_embeddings)
        
        # Mise à jour de tag_embeddings
        tag_embeddings.update({ele: current_embeddings[ele]['emb'] for ele in current_embeddings})

    
    ### Managing Abstracts
    ### -----------------------  
    keep_tag_embeddings = {}
    results_complete_similarities = {}
        
    for abstracts_pth_file in abstracts_pth_files:
        chunk_embeddings = mem.load_filepth(abstracts_pth_file)
        
        for doi,res in mem.compare_tags_with_chunks(tag_embeddings, chunk_embeddings).items():
            if doi not in results_complete_similarities:
                results_complete_similarities[doi] = res
                for tag in res.keys():
                    if tag not in keep_tag_embeddings:
                        keep_tag_embeddings[tag] = tag_embeddings_all[tag]
            else:
                for tag,sim in res.items():
                    if tag not in results_complete_similarities[doi] or sim>results_complete_similarities[doi][tag]:
                        results_complete_similarities[doi][tag] = sim
                        if tag not in keep_tag_embeddings:
                            keep_tag_embeddings[tag] = tag_embeddings_all[tag]
            if doi in results_complete_similarities:                
                results_complete_similarities[doi] = mem.remove_similar_tags_by_doi(keep_tag_embeddings,results_complete_similarities[doi])

        json_f = str(os.path.splitext(abstracts_pth_file)[0])+"_scores.json"
        
        with open(json_f, "w") as fichier:
            json.dump(results_complete_similarities, fichier)

def get_scores_files(retention_dir):
    scores_files = []
    pattern = re.compile(".*_scores.json")
    for root, dirs, files in os.walk(retention_dir):
        for filename in files:
            if pattern.search(filename):
                scores_files.append(os.path.join(root, filename))
    return scores_files

def get_results_complete_similarities_and_tags_embedding(config_all):
    scores_files = []
    retention_dir = config_all['retention_dir']
    mem = ModelEmbeddingManager(config_all)
    config_owl = setup_general_config(config_all,'populate_owl_tag_embeddings')
    config_abstract = setup_general_config(config_all,'populate_abstract_embeddings')
    
    scores_files = get_scores_files(retention_dir)
    
    tags_pth_files = OwlTagManager(config_owl,mem).get_files_tags_embeddings()
         
    if len(tags_pth_files) == 0:
        raise FileNotFoundError("No tags embeddings found")
    
    tags_taxon_pth_files = TaxonTagManager(config_owl,mem).get_files_tags_ncbi_taxon_embeddings()
    
    if len(tags_taxon_pth_files) == 0:
        warnings.warn("No tags taxon embeddings found")

    tags_pth_files.extend(tags_taxon_pth_files)
    abstracts_pth_files = AbstractManager(config_abstract,mem).get_files_abstracts_embeddings()

    if len(abstracts_pth_files) == 0:
        raise FileNotFoundError("No abstracts embeddings found")
    
    ### Loading tags embeddings
    ### -----------------------
    tag_embeddings = {}
    
    for tags_pth_file in tags_pth_files:
        current_embeddings = mem.load_filepth(tags_pth_file)
        
        # Mise à jour de tag_embeddings_all
        tag_embeddings.update(current_embeddings)
        
    results_complete_similarities = {}
    for file_name in scores_files:
        with open(file_name, 'r') as file:
            try:
                results_complete_similarities.update(json.load(file))
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON dans le fichier {file_name}")
    
    return results_complete_similarities,tag_embeddings

def main_display_summary(config_all):
    
    results_complete_similarities,tag_embeddings = get_results_complete_similarities_and_tags_embedding(config_all)    
    retention_dir = config_all['retention_dir']
    
    if len(results_complete_similarities)>0:
        display_ontologies_distribution(results_complete_similarities,tag_embeddings)
        display_best_similarity_abstract_tag(results_complete_similarities,tag_embeddings,retention_dir)
        display_ontologies_summary(results_complete_similarities,tag_embeddings,retention_dir)
    else:
        print("No results found")
  
def main_build_graph(config_all):
    scores_files = get_scores_files(config_all['retention_dir'])
    for file_name in scores_files:
        with open(file_name, 'r') as file:
            try:
                data = json.load(file)
                g = create_rdf_graph(
                        data,
                        encoder_name=config_all['encodeur'],
                        system_name="encoder-ontology-match-abstract",
                        similarity_threshold=config_all['threshold_similarity_tag_chunk'],
                        tag_similarity_threshold=config_all['threshold_similarity_tag'],
                        similarity_method="cosine")
                new_f = os.path.splitext(file_name)[0]+".ttl"
                save_rdf_graph(g, new_f)
                print("RDF graph saved in ",new_f)

            except json.JSONDecodeError:
                print("Erreur de décodage JSON")
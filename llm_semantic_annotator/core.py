from llm_semantic_annotator import ModelEmbeddingManager
from llm_semantic_annotator import OwlTagManager
from llm_semantic_annotator import TaxonTagManager
from llm_semantic_annotator import AbstractManager

from llm_semantic_annotator import display_ontologies_distribution
from llm_semantic_annotator import display_best_similarity_abstract_tag
from llm_semantic_annotator import display_ontologies_summary
from llm_semantic_annotator import create_rdf_graph,save_rdf_graph

import json 
import os 
import re 

import concurrent.futures
from functools import partial

def setup_general_config(config_all,methode):
    
    if methode not in config_all:
        config = {}
    else:
        config = config_all[methode]
    config['retention_dir'] = config_all['retention_dir']
    config['force'] = config_all['force']
    
    return config

def get_owl_tag_manager(config_all):
    config = setup_general_config(config_all,'populate_owl_tag_embeddings')
    mem = ModelEmbeddingManager(config_all)
    return OwlTagManager(config,mem)

def get_gbif_taxon_tag_manager(config_all):
    config = setup_general_config(config_all,'populate_gbif_taxon_tag_embeddings')
    mem = ModelEmbeddingManager(config_all)
    return TaxonTagManager(config,mem)

def get_ncbi_taxon_tag_manager(config_all):
    config = setup_general_config(config_all,'populate_ncbi_taxon_tag_embeddings')
    mem = ModelEmbeddingManager(config_all)
    return TaxonTagManager(config,mem)

def get_abstract_manager(config_all):
    config = setup_general_config(config_all,'populate_abstract_embeddings')
    mem = ModelEmbeddingManager(config_all)
    return AbstractManager(config,mem,get_owl_tag_manager(config_all))

def main_populate_owl_tag_embeddings(config_all):
    """Fonction principale pour générer et stocker les embeddings de tags dans une base."""
    get_owl_tag_manager(config_all).manage_tags()

def main_populate_gbif_taxon_tag_embeddings(config_all):
    get_gbif_taxon_tag_manager(config_all).manage_gbif_taxon_tags()

def main_populate_ncbi_taxon_tag_embeddings(config_all):
    get_ncbi_taxon_tag_manager(config_all).manage_ncbi_taxon_tags()

def main_populate_abstract_embeddings(config_all):
    get_abstract_manager(config_all).manage_abstracts()

def get_doi_file(config_all):
    return config_all['retention_dir']+"/total_doi.txt"

def process_abstract_file(abstracts_pth_file, mem, tag_embeddings, tag_embeddings_all, config_all):
    json_f = str(os.path.splitext(abstracts_pth_file)[0]) + "_scores.json"
    if not config_all['force'] and os.path.exists(json_f):
        print(json_f, " already exists!")
        return None, 0

    chunk_embeddings = mem.load_filepth(abstracts_pth_file)
    print("Processing ", abstracts_pth_file)
    
    results_complete_similarities = {}
    keep_tag_embeddings = {}
    total_doi = 0

    for doi, res in mem.compare_tags_with_chunks(tag_embeddings, chunk_embeddings).items():
        total_doi += 1
        if doi not in results_complete_similarities:
            results_complete_similarities[doi] = res
            for tag in res.keys():
                if tag not in keep_tag_embeddings:
                    keep_tag_embeddings[tag] = tag_embeddings_all[tag]
        else:
            for tag, sim in res.items():
                if tag not in results_complete_similarities[doi] or sim > results_complete_similarities[doi][tag]:
                    results_complete_similarities[doi][tag] = sim
                    if tag not in keep_tag_embeddings:
                        keep_tag_embeddings[tag] = tag_embeddings_all[tag]
        
        if doi in results_complete_similarities:
            results_complete_similarities[doi] = mem.remove_similar_tags_by_doi(keep_tag_embeddings, results_complete_similarities[doi])

    results_complete_similarities = {k: v for k, v in results_complete_similarities.items() if v}
    
    with open(json_f, "w") as fichier:
        json.dump(results_complete_similarities, fichier)

    return results_complete_similarities, keep_tag_embeddings, total_doi
    
def main_compute_tag_chunk_similarities(config_all):
    """Fonction principale pour calculer la similarité entre tous les tags et chunks."""
    
    if 'force' not in config_all:
        config_all['force'] = False
        
    tags_pth_files = get_owl_tag_manager(config_all).get_files_tags_embeddings()
    
    if len(tags_pth_files) == 0:
        raise FileNotFoundError("No tags embeddings found")
    
    tags_taxon_pth_files = get_ncbi_taxon_tag_manager(config_all).get_files_tags_ncbi_taxon_embeddings()
    tags_pth_files.extend(tags_taxon_pth_files)
    
    abstracts_pth_files = get_abstract_manager(config_all).get_files_abstracts_embeddings()

    if len(abstracts_pth_files) == 0:
        raise FileNotFoundError("No abstracts embeddings found")
    
    ### Loading tags embeddings
    ### -----------------------
    mem = ModelEmbeddingManager(config_all)
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
    total_doi = 0

    # Créer une fonction partielle avec les arguments communs
    process_file = partial(process_abstract_file, mem=mem, tag_embeddings=tag_embeddings, 
                           tag_embeddings_all=tag_embeddings_all, config_all=config_all)

    # Utiliser ThreadPoolExecutor pour le multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_results = list(executor.map(process_file, abstracts_pth_files))

    # Traiter les résultats
    for result, keep_tags, file_doi_count in future_results:
        if result is not None:
            results_complete_similarities.update(result)
            keep_tag_embeddings.update(keep_tags)
            total_doi += file_doi_count

    # Écrire le nombre total de DOI
    with open(get_doi_file(config_all), "w") as fichier:
        fichier.write(str(total_doi))

def get_scores_files(retention_dir):
    scores_files = []
    pattern = re.compile(".*_scores.json")
    for root, dirs, files in os.walk(retention_dir):
        for filename in files:
            if pattern.search(filename):
                scores_files.append(os.path.join(root, filename))
    return scores_files

def get_results_complete_similarities_and_tags_embedding(config_all):
    mem = ModelEmbeddingManager(config_all)
    
    scores_files = []
    retention_dir = config_all['retention_dir']
    
    scores_files = get_scores_files(retention_dir)
    tags_pth_files = get_owl_tag_manager(config_all).get_files_tags_embeddings()
         
    if len(tags_pth_files) == 0:
        raise FileNotFoundError("No tags embeddings found")
    
    tags_taxon_pth_files = get_ncbi_taxon_tag_manager(config_all).get_files_tags_ncbi_taxon_embeddings()
    tags_pth_files.extend(tags_taxon_pth_files)
    
    abstracts_pth_files = get_abstract_manager(config_all).get_files_abstracts_embeddings()

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
    doi_file = get_doi_file(config_all)
    results_complete_similarities,tag_embeddings = get_results_complete_similarities_and_tags_embedding(config_all)    
    retention_dir = config_all['retention_dir']
    
    if len(results_complete_similarities)>0:
        display_best_similarity_abstract_tag(results_complete_similarities,tag_embeddings,retention_dir)
        display_ontologies_summary(results_complete_similarities,tag_embeddings,retention_dir)
        display_ontologies_distribution(results_complete_similarities,tag_embeddings,doi_file)
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

def main_build_dataset_abstracts_annotation(config_all):
    get_abstract_manager(config_all).build_dataset_abstracts_annotations()
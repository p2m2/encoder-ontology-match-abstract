from llm_semantic_annotator import ModelEmbeddingManager
from llm_semantic_annotator import OwlTagManager
from llm_semantic_annotator import TaxonTagManager
from llm_semantic_annotator import AbstractManager

from llm_semantic_annotator import display_ontologies_distribution
from llm_semantic_annotator import display_best_similarity_abstract_tag
from llm_semantic_annotator import display_ontologies_summary

import warnings

def setup_general_config(config_all,methode):
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

    results_complete_similarities = {}
    tags_pth_files = OwlTagManager(config_owl,mem).get_files_tags_embeddings()
    
    if len(tags_pth_files) == 0:
        raise FileNotFoundError("No tags embeddings found")
    
    #tags_taxon_pth_files = TaxonTagManager(config_owl,mem).get_files_tags_gbif_taxon_embeddings()
    tags_taxon_pth_files = TaxonTagManager(config_owl,mem).get_files_tags_ncbi_taxon_embeddings()
    
    if len(tags_taxon_pth_files) == 0:
        warnings.warn("No tags taxon embeddings found")

    tags_pth_files.extend(tags_taxon_pth_files)
    
    abstracts_pth_files = AbstractManager(config_abstract,mem).get_files_abstracts_embeddings()

    if len(abstracts_pth_files) == 0:
        raise FileNotFoundError("No abstracts embeddings found")

    for tags_pth_file in tags_pth_files:
        tag_embeddings = mem.load_filepth(tags_pth_file)
        
        for  abstracts_pth_file in abstracts_pth_files:
            chunk_embeddings = mem.load_filepth(abstracts_pth_file)
            for doi,res in mem.compare_tags_with_chunks(
                tag_embeddings, chunk_embeddings).items():
                if doi not in results_complete_similarities:
                    results_complete_similarities[doi] = res
                else:
                    results_complete_similarities[doi].update(res)
    
    if len(results_complete_similarities)>0:
        retention_dir = config_all['retention_dir']
        display_ontologies_distribution(results_complete_similarities)
        display_best_similarity_abstract_tag(results_complete_similarities,retention_dir)
        display_ontologies_summary(results_complete_similarities,retention_dir)

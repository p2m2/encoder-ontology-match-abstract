from llm_semantic_annotator import OwlTagManagement
from llm_semantic_annotator import manage_abstracts, get_abstracts_embeddings
from llm_semantic_annotator import ModelEmbeddingManagement
from llm_semantic_annotator import display_ontologies_distribution
from llm_semantic_annotator import display_best_similarity_abstract_tag
from llm_semantic_annotator import display_ontologies_summary
from llm_semantic_annotator import TaxonTagManagement

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
    
    OwlTagManagement(config).manage_tags()

def main_populate_ncbi_abstract_embeddings(config_all):
    config = setup_general_config(config_all,'populate_ncbi_abstract_embeddings')

    manage_abstracts(config)

def main_compute_tag_chunk_similarities(config_all):
    """Fonction principale pour calculer la similarité entre tous les tags et chunks."""
    config_owl = setup_general_config(config_all,'populate_owl_tag_embeddings')
    config = setup_general_config(config_all,'compute_tag_chunk_similarities')

    tag_embeddings = OwlTagManagement(config_owl).get_tags_embeddings()
    if (len(tag_embeddings)==0):
        raise FileNotFoundError("No tags embeddings found")
        
    chunk_embeddings = get_abstracts_embeddings(config['retention_dir'])
    if (len(chunk_embeddings)==0):
        raise FileNotFoundError("No abstract chunks embeddings found")
    
    results_complete_similarities = ModelEmbeddingManagement().compare_tags_with_chunks(
        tag_embeddings, chunk_embeddings,config)

    display_ontologies_distribution(results_complete_similarities)
    display_best_similarity_abstract_tag(results_complete_similarities)
    display_ontologies_summary(results_complete_similarities)

def main_populate_ncbi_taxon_tag_embeddings(config_all):
    config = setup_general_config(config_all,'populate_owl_tag_embeddings')
    TaxonTagManagement(config).manage_ncbi_taxon_tags(config)

def main_populate_gbif_taxon_tag_embeddings(config_all):
    config = setup_general_config(config_all,'populate_owl_tag_embeddings')
    TaxonTagManagement(config).manage_gbif_taxon_tags(config)
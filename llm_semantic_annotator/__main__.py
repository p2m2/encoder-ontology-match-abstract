import json, sys

from llm_semantic_annotator import manage_tags, get_tags_embeddings
from llm_semantic_annotator import manage_abstracts, get_abstracts_embeddings
from llm_semantic_annotator import compare_tags_with_chunks
from llm_semantic_annotator import ontologies_distribution,get_retention_dir

from rich import print
import argparse

def load_config(config_file):
    """Charge la configuration à partir d'un fichier JSON."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Le fichier de configuration {config_file} est introuvable.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Erreur de décodage JSON dans le fichier {config_file}.")
        sys.exit(1)

def main_populate_tag_embeddings(config_all):
    """Fonction principale pour générer et stocker les embeddings de tags dans une base."""
    config = config_all['populate_tag_embeddings']
    config['retention_dir'] = config_all['retention_dir']

    # Utilisez les paramètres de config ici
    print(f"Ontologies : {config['ontologies']}")
    print(f"Nb terms to compute : {config['debug_nb_terms_by_ontology']}")
    
    manage_tags(config)

def main_populate_ncbi_abstract_embeddings(config_all):
    config = config_all['populate_ncbi_abstract_embeddings']
    config['retention_dir'] = config_all['retention_dir']
    manage_abstracts(config)

def main_compute_tag_chunk_similarities(config_all):
    """Fonction principale pour calculer la similarité entre tous les tags et chunks."""
    config = config_all['compute_tag_chunk_similarities']
    config['retention_dir'] = config_all['retention_dir']

    tag_embeddings = get_tags_embeddings(config['retention_dir'])
    chunk_embeddings = get_abstracts_embeddings(config['retention_dir'])
   
    results_complete_similarities = compare_tags_with_chunks(
        tag_embeddings, chunk_embeddings,config)

    ontologies_distribution(results_complete_similarities)

def parse_arguments():
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Programme avec plusieurs types d'exécution.")
    parser.add_argument(
        "config_file",
        help="Chemin vers le fichier de configuration JSON."
    )
    parser.add_argument(
        "execution_type",
        choices=["populate_tag_embeddings", "populate_ncbi_abstract_embeddings", "compute_tag_chunk_similarities"],
        help="Type d'exécution à effectuer."
    )

    return parser.parse_args()

if __name__ == "__main__":
    import os
    args = parse_arguments()
    config = load_config(args.config_file)
    
    config['retention_dir'] = get_retention_dir(args.config_file)

    if args.execution_type == "populate_tag_embeddings":
        main_populate_tag_embeddings(config)
    elif args.execution_type == "populate_ncbi_abstract_embeddings":
        main_populate_ncbi_abstract_embeddings(config)
    elif args.execution_type == "compute_tag_chunk_similarities":
        main_compute_tag_chunk_similarities(config)






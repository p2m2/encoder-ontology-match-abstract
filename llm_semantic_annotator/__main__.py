import os, json, sys
from llm_semantic_annotator import build_corpus, manage_tags, get_tags_embeddings
from llm_semantic_annotator import abstract_preparation, manage_abstracts, get_abstracts_embeddings
from llm_semantic_annotator import torch_utils, compare_tags_with_chunks

from rich import print

import argparse

# Listes des ontologies du projet Planteome

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

def main_populate_tag_embeddings(config):
    """Fonction principale pour générer et stocker les embeddings de tags dans une base."""
    # Utilisez les paramètres de config ici
    print(f"Ontologies : {config['ontologies']}")
    print(f"Nb terms to compute : {config['debug_nb_terms_by_ontology']}")
    manage_tags(config['ontologies'],config['debug_nb_terms_by_ontology'])

def main_populate_ncbi_abstract_embeddings(config,selected_term):
    manage_abstracts(selected_term,config['debug_nb_ncbi_request'])

def main_compute_tag_chunk_similarities(config):
    """Fonction principale pour calculer la similarité entre tous les tags et chunks."""
    
    tag_embeddings = get_tags_embeddings()  
    chunk_embeddings = get_abstracts_embeddings()
   
    results, results_complete_similarities = compare_tags_with_chunks(
        tag_embeddings, chunk_embeddings,config['threshold_similarity_tag_chunk'])

    # Afficher les résultats
    for i, tuple in enumerate(results):
        tag = tuple[0]
        similarity = tuple[1]
        #print(chunks[i])
        print(results_complete_similarities[i])
        if tag:
            print(f"Tag: {tag}\nSimilarity: {similarity:.4f}\n")
        else:
            print(f"Tag: None (Below threshold)\n")

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
    parser.add_argument(
        "--selected_term",
        required=False,  # Par défaut, non requis
        help="expression pour la recherche d'abstract avec ncbi/eutils"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config_file)

    if args.execution_type == "populate_tag_embeddings":
        main_populate_tag_embeddings(config)
    elif args.execution_type == "populate_ncbi_abstract_embeddings":
        if not args.selected_term:
            print("L'option --selected_term est requise pour compute_similarities.")
            sys.exit(1)
        main_populate_ncbi_abstract_embeddings(config,args.selected_term)
    elif args.execution_type == "compute_tag_chunk_similarities":
        main_compute_tag_chunk_similarities(config)






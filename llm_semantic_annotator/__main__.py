import json, sys

from llm_semantic_annotator import manage_tags, get_tags_embeddings
from llm_semantic_annotator import manage_abstracts, get_abstracts_embeddings
from llm_semantic_annotator import compare_tags_with_chunks
#from llm_semantic_annotator import ontologies_distribution

from rich import print
from collections import Counter
import argparse
def ontologies_distribution(data):
    # Extraire les préfixes des clés
    ontologies = []
    labels = []
    for doi, item in data.items():
        for key in item.keys():
            ontology = key.split('__')[1]  # Extraire le préfixe entre les doubles underscores
            ontologies.append(ontology)
            labels.append(key)

    # Compter la distribution des préfixes
    distributionOntologies = Counter(ontologies)
    distributionLabels = Counter(labels)

    print(f"nb abstracts : {len(data)}")
    annoted_abstracts = map(lambda item: 1 if len(item[1])>0 else 0, data.items())
    print(f"nb abstracts annoted : {sum(annoted_abstracts)}")
    # Afficher la distribution
    print("Distribution des ontologies :")
    for prefix, count in distributionOntologies.items():
        print(f"{prefix}: {count}")

    print("Distribution des labels :")
    sorted_distribution = sorted(distributionLabels.items(), key=lambda x: x[1], reverse=True)
    
    for prefix, count in sorted_distribution:
        print(f"{prefix}: {count}")
        if count == 1:
            break

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
    manage_abstracts(selected_term,
                     config['debug_nb_ncbi_request'],
                     config['debug_nb_abstracts_by_search'])

def main_compute_tag_chunk_similarities(config):
    """Fonction principale pour calculer la similarité entre tous les tags et chunks."""
    
    tag_embeddings = get_tags_embeddings()
    chunk_embeddings = get_abstracts_embeddings()
   
    results_complete_similarities = compare_tags_with_chunks(
        tag_embeddings, chunk_embeddings,
        config['threshold_similarity_tag_chunk'],
        config['debug_nb_similarity_compute'])

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
    parser.add_argument(
        "--selected_term",
        required=False,  # Par défaut, non requis
        help="expression pour la recherche d'abstract avec ncbi/eutils"
    )
    return parser.parse_args()

if __name__ == "__main__":
    import os
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






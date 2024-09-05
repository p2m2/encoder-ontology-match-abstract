import json, sys

from llm_semantic_annotator import get_retention_dir
from llm_semantic_annotator import main_populate_owl_tag_embeddings
from llm_semantic_annotator import main_populate_ncbi_abstract_embeddings
from llm_semantic_annotator import main_compute_tag_chunk_similarities

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

def parse_arguments():
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Programme avec plusieurs types d'exécution.")
    parser.add_argument(
        "config_file",
        help="Chemin vers le fichier de configuration JSON."
    )
    parser.add_argument(
        "execution_type",
        choices=["populate_owl_tag_embeddings", "populate_ncbi_abstract_embeddings", "compute_tag_chunk_similarities"],
        help="Type d'exécution à effectuer."
    )

    parser.add_argument('--force', action='store_true', help="Forcer l'exécution sans demander de confirmation")

    return parser.parse_args()

if __name__ == "__main__":
    import os
    args = parse_arguments()
    config = load_config(args.config_file)
    
    config['retention_dir'] = get_retention_dir(args.config_file)
    
    if args.force:
        config['force'] = True
    else:
        config['force'] = False
    print(config['force'])
    if args.execution_type == "populate_owl_tag_embeddings":
        main_populate_owl_tag_embeddings(config)
    elif args.execution_type == "populate_ncbi_abstract_embeddings":
        main_populate_ncbi_abstract_embeddings(config)
    elif args.execution_type == "compute_tag_chunk_similarities":
        main_compute_tag_chunk_similarities(config)






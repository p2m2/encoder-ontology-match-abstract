import json, sys

from llm_semantic_annotator import get_retention_dir
from llm_semantic_annotator import main_populate_owl_tag_embeddings
from llm_semantic_annotator import main_populate_abstract_embeddings
from llm_semantic_annotator import main_populate_gbif_taxon_tag_embeddings
from llm_semantic_annotator import main_populate_ncbi_taxon_tag_embeddings
from llm_semantic_annotator import main_compute_tag_chunk_similarities
from llm_semantic_annotator import similarity_evaluator_main
from llm_semantic_annotator import main_display_summary
from llm_semantic_annotator import main_build_graph
from llm_semantic_annotator import main_build_dataset_abstracts_annotation

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
        choices=["populate_owl_tag_embeddings",
                "populate_gbif_taxon_tag_embeddings",
                "populate_ncbi_taxon_tag_embeddings",
                "populate_abstract_embeddings",
                "compute_tag_chunk_similarities",
                "display_summary",
                "build_rdf_graph",
                "build_dataset_abstracts_annotations",
                "evaluate_encoder"],
        help="Type d'exécution à effectuer."
    )

    parser.add_argument('--force', action='store_true', 
                        help="Forcer l'exécution sans demander de confirmation")

    return parser.parse_args()

def main():
    import os
    args = parse_arguments()
    config = load_config(args.config_file)
    
    config['retention_dir'] = get_retention_dir(args.config_file)
    
    if args.force:
        config['force'] = True
    else:
        config['force'] = False
    
    if args.execution_type == "populate_owl_tag_embeddings":
        main_populate_owl_tag_embeddings(config)
    elif args.execution_type == "populate_gbif_taxon_tag_embeddings":
        main_populate_gbif_taxon_tag_embeddings(config)
    elif args.execution_type == "populate_ncbi_taxon_tag_embeddings":
        main_populate_ncbi_taxon_tag_embeddings(config)
    elif args.execution_type == "populate_abstract_embeddings":
        main_populate_abstract_embeddings(config)
    elif args.execution_type == "compute_tag_chunk_similarities":
        main_compute_tag_chunk_similarities(config)
    elif args.execution_type == "display_summary":
        main_display_summary(config)
    elif args.execution_type == "build_rdf_graph":
        main_build_graph(config)
    elif args.execution_type == "build_dataset_abstracts_annotations":
        main_build_dataset_abstracts_annotation(config)
    elif args.execution_type == "evaluate_encoder":
        similarity_evaluator_main(config)
    else:
        raise ValueError("Type d'exécution non reconnu.")

if __name__ == "__main__":
    main()
    






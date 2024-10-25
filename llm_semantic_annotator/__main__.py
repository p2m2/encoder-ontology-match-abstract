import json
import sys
import os
from rich import print
import argparse
import traceback

from llm_semantic_annotator import (
    get_retention_dir,
    main_populate_owl_tag_embeddings,
    main_populate_abstract_embeddings,
    main_populate_gbif_taxon_tag_embeddings,
    main_populate_ncbi_taxon_tag_embeddings,
    main_compute_tag_chunk_similarities,
    main_display_summary,
    main_build_graph,
    main_build_dataset_abstracts_annotation
)

def load_config(config_file):
    """Load configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[bold red]Error:[/bold red] Configuration file {config_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[bold red]Error:[/bold red] JSON decoding error in file {config_file}.")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Program with multiple execution types.")
    parser.add_argument(
        "config_file",
        help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "execution_type",
        choices=[
            "populate_owl_tag_embeddings",
            "populate_gbif_taxon_tag_embeddings",
            "populate_ncbi_taxon_tag_embeddings",
            "populate_abstract_embeddings",
            "compute_tag_chunk_similarities",
            "display_summary",
            "build_rdf_graph",
            "build_dataset_abstracts_annotations",
            "evaluate_encoder"
        ],
        help="Type of execution to perform."
    )
    parser.add_argument('--force', action='store_true', 
                        help="Force execution without asking for confirmation")
    return parser.parse_args()

def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"
    
def main():
    import time
    
    debut = time.time()
        
    args = parse_arguments()
    config = load_config(args.config_file)
    
    config['retention_dir'] = get_retention_dir(args.config_file)
    config['force'] = args.force
    
    execution_functions = {
        "populate_owl_tag_embeddings": main_populate_owl_tag_embeddings,
        "populate_gbif_taxon_tag_embeddings": main_populate_gbif_taxon_tag_embeddings,
        "populate_ncbi_taxon_tag_embeddings": main_populate_ncbi_taxon_tag_embeddings,
        "populate_abstract_embeddings": main_populate_abstract_embeddings,
        "compute_tag_chunk_similarities": main_compute_tag_chunk_similarities,
        "display_summary": main_display_summary,
        "build_rdf_graph": main_build_graph,
        "build_dataset_abstracts_annotations": main_build_dataset_abstracts_annotation
        }
    
    try:
        execution_function = execution_functions[args.execution_type]
        print(f"[bold green]Executing:[/bold green] {args.execution_type}")
        execution_function(config)
        duree = time.time() - debut
        print(f"Execution time: {format_duration(duree)}")
    except Exception as e:
        print(f"[bold red]Error during execution:[/bold red] {str(e)}")
        
        # Obtenir les informations sur l'erreur
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Filtrer le traceback pour n'inclure que les lignes de votre code
        tb_list = traceback.extract_tb(exc_traceback)
        for tb in tb_list:
            filename, line_number, func_name, text = tb
            if "site-packages" not in filename:  # Exclut les erreurs des bibliothèques
                print(f"[bold yellow]At line:[/bold yellow] {line_number}, "+
                      f"[bold yellow]Error occurred in file:[/bold yellow] {filename}, "+
                      f"[bold yellow]In function:[/bold yellow] {func_name} ")
                print(f"[bold yellow]Code:[/bold yellow] {text}")
    

if __name__ == "__main__":
    main()

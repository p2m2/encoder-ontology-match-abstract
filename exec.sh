#!/bin/bash

help() {
    cat << EOF
Usage: $0 <config_file> <int_commande>

Commands:
  1. Pseudo workflow [2,4,5,6,7]
  2. Populate OWL tag embeddings
  3. Populate NCBI Taxon tag embeddings
  4. Populate abstract embeddings
  5. Compute similarities between tags and abstract chunks
  6. Display similarities information
  7. Build turtle knowledge graph
  8. Build dataset abstracts annotations CSV file
  9. Evaluate encoder with MeSH descriptors (experimental)

Details:
  2: Compute TAG embeddings for all ontologies defined in the populate_owl_tag_embeddings section
  3: Compute TAG embeddings for NCBI Taxon
  4: Compute ABSTRACT embeddings (title + sentences) for all abstracts in the dataset
  5: Compute similarities between TAGS and ABSTRACTS
  6: Display similarities information on the console
  7: Generate turtle file with information {score, tag} for each DOI
  8: Generate CSV file with [doi, tag, pmid, reference_id]

EOF
}

# Check for help option
if [[ "$1" == "-h" ]]; then
    help
    exit 0
fi

# Check for correct number of arguments
if [ "$#" -lt 2 ]; then
    echo "Error: Not enough arguments."
    echo "Usage: $0 <config_file> <int_commande> [options]"
    echo "Use '$0 -h' for more information."
    exit 1
fi

config_file=$1
command=$2

# Validate config file
if [ ! -f "$config_file" ]; then
    echo "Error: Config file '$config_file' does not exist."
    exit 1
fi

# Validate command is an integer
if ! [[ "$command" =~ ^[0-9]+$ ]]; then
    echo "Error: Command must be an integer."
    exit 1
fi

venv_name="llm_semantic_annotator_env"

# Fonction pour créer l'environnement virtuel s'il n'existe pas
create_venv_if_not_exists() {
    if [ ! -d "$venv_name" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$venv_name"
        source "$venv_name/bin/activate"
        pip install -r requirements.txt  # Assurez-vous d'avoir un fichier requirements.txt
    else
        source "$venv_name/bin/activate"
    fi
}

run_command() {
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "Error: The command '$*' failed with exit code $status"
        exit $status
    fi
}

execute_command() {
    case $1 in
        2) run_command python3 -m llm_semantic_annotator "$config_file" populate_owl_tag_embeddings ;;
        3) run_command python3 -m llm_semantic_annotator "$config_file" populate_ncbi_taxon_tag_embeddings ;;
        4) run_command python3 -m llm_semantic_annotator "$config_file" populate_abstract_embeddings ;;
        5) run_command python3 -m llm_semantic_annotator "$config_file" compute_tag_chunk_similarities ;;
        6) run_command python3 -m llm_semantic_annotator "$config_file" display_summary ;; 
        7) run_command python3 -m llm_semantic_annotator "$config_file" build_rdf_graph ;;
        8) run_command python3 -m llm_semantic_annotator "$config_file" build_dataset_abstracts_annotations ;; 
        9) run_command python3 -m llm_semantic_annotator "$config_file" evaluate_encoder ;;   
        *) echo "Invalid option" ;;
    esac
}

# Créer l'environnement virtuel s'il n'existe pas
create_venv_if_not_exists

case $command in
    1)
        run_command python3 -m llm_semantic_annotator "$config_file" populate_owl_tag_embeddings
        #run_command python3 -m llm_semantic_annotator "$config_file" populate_ncbi_taxon_tag_embeddings
        run_command python3 -m llm_semantic_annotator "$config_file" populate_abstract_embeddings
        run_command python3 -m llm_semantic_annotator "$config_file" compute_tag_chunk_similarities
        run_command python3 -m llm_semantic_annotator "$config_file" build_rdf_graph
        run_command python3 -m llm_semantic_annotator "$config_file" display_summary
        ;;
    2|3|4|5|6|7|8|9)
        execute_command $command
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Désactiver l'environnement virtuel à la fin
deactivate

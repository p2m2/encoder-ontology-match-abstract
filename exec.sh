#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

conffile="$1"
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
        1) run_command python3 -m llm_semantic_annotator "$conffile" populate_owl_tag_embeddings ;;
        2) run_command python3 -m llm_semantic_annotator "$conffile" populate_ncbi_taxon_tag_embeddings ;;
        3) run_command python3 -m llm_semantic_annotator "$conffile" populate_abstract_embeddings ;;
        4) run_command python3 -m llm_semantic_annotator "$conffile" compute_tag_chunk_similarities ;;
        *) echo "Invalid option" ;;
    esac
}

# Créer l'environnement virtuel s'il n'existe pas
create_venv_if_not_exists

echo "What would you like to execute?"
echo "1. Full workflow"
echo "2. populate_owl_tag_embeddings"
echo "3. populate_ncbi_taxon_tag_embeddings"
echo "4. populate_abstract_embeddings"
echo "5. compute_tag_chunk_similarities"
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        run_command python3 -m llm_semantic_annotator "$conffile" populate_owl_tag_embeddings
        run_command python3 -m llm_semantic_annotator "$conffile" populate_ncbi_taxon_tag_embeddings
        run_command python3 -m llm_semantic_annotator "$conffile" populate_abstract_embeddings
        run_command python3 -m llm_semantic_annotator "$conffile" compute_tag_chunk_similarities
        ;;
    2|3|4|5)
        execute_command $((choice - 1))
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Désactiver l'environnement virtuel à la fin
deactivate

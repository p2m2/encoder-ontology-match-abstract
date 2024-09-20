#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

conffile="$1"

run_command() {
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "Erreur: La commande '$*' a échoué avec le code de sortie $status"
        exit $status
    fi
}

run_command python -m llm_semantic_annotator "$conffile" populate_owl_tag_embeddings
run_command python -m llm_semantic_annotator "$conffile" populate_ncbi_taxon_tag_embeddings
run_command python -m llm_semantic_annotator "$conffile" populate_abstract_embeddings
run_command python -m llm_semantic_annotator "$conffile" compute_tag_chunk_similarities


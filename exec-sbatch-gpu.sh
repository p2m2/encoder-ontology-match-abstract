#!/bin/bash

#SBATCH --job-name=owl_abstract_encoding_similarities
#SBATCH --chdir=/scratch/ofilangi/workspace/encoder-ontology-match-abstract
#SBATCH --output=out_process_igepp.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=20G

## sbatch -p gpu exec.sh

source ./check_slurm_memory.sh

/local/env/envpython-3.9.5.sh
. env/bin/activate

conffile=config/igepp.json
#export TOKENIZERS_PARALLELISM=false
#rm -rf igepp_w*/

check_slurm_memory
python -m llm_semantic_annotator $conffile populate_owl_tag_embeddings
check_slurm_memory
python -m llm_semantic_annotator $conffile populate_gbif_taxon_tag_embeddings
check_slurm_memory
python -m llm_semantic_annotator $conffile populate_abstract_embeddings
check_slurm_memory
python -m llm_semantic_annotator $conffile compute_tag_chunk_similarities


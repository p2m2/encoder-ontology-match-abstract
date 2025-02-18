#!/bin/bash

#SBATCH --job-name=NCBITAXON_GPU_EMBEDDINGS
#SBATCH --chdir=/scratch/ofilangi/encoder-ontology-match-abstract
#SBATCH --output=ncbitaxon_annot_pubmed_GPU_2_3.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=olivier.filangi@inrae.fr

## sbatch -p gpu exec-sbatch-ncbitaxon-gpu-embeddings-compute.sh

. /local/env/envpython-3.9.5.sh
conffile=./config/ncbi-taxon-pubmed-all.json
#GPU
# ontology embedding
./exec.sh $conffile 2
# abstract ## embedding
#./exec.sh $conffile 3

### Execution time: 16h 42m 59.59s


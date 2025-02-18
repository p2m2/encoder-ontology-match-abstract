#!/bin/bash

#SBATCH --job-name=GO_SIM_GPU
#SBATCH --chdir=/scratch/ofilangi/encoder-ontology-match-abstract
#SBATCH --output=GO_annot_pubmed_GPU_2_3.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=olivier.filangi@inrae.fr

## sbatch -p gpu exec-sbatch-go-gpu-embeddings-compute.sh

. /local/env/envpython-3.9.5.sh
conffile=config/gene_ontology.json
#CPU
## similarities
./exec.sh $conffile 2
## CSV export
#./exec.sh $conffile 3

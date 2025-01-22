#!/bin/bash

#SBATCH --job-name=similarities_compute
#SBATCH --chdir=/scratch/ofilangi/encoder-ontology-match-abstract
#SBATCH --output=ncbitaxon_annot_pubmed.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=28G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=olivier.filangi@inrae.fr

## sbatch -p gpu exec-sbatch-ncbitaxon-gpu.sh

. /local/env/envpython-3.9.5.sh
conffile=./config/ncbi-taxon-pubmed-all.json
./exec.sh $conffile 1

if [ $? -eq 0 ]; then
    ./exec.sh $conffile 8
fi

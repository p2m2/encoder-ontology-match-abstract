#!/bin/bash

#SBATCH --job-name=similarities_compute
#SBATCH --chdir=/scratch/ofilangi/encoder-ontology-match-abstract
#SBATCH --output=planteome_annot_pubmeb.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=10G

## sbatch -p gpu exec-sbatch-gpu.sh

. /local/env/envpython-3.9.5.sh
conffile=./config/pubmed-all.json
./exec.sh $conffile 1

if [ $? -eq 0 ]; then
    ./exec.sh $conffile 8
fi

#!/bin/bash

#SBATCH --job-name=NCBITAXON_SIM_CPU
#SBATCH --chdir=/scratch/ofilangi/encoder-ontology-match-abstract
#SBATCH --output=ncbitaxon_annot_pubmed_CPU_4_6.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=olivier.filangi@inrae.fr

## sbatch -p bigmem exec-sbatch-ncbitaxon-cpu-similarities-csv.sh

. /local/env/envpython-3.9.5.sh
conffile=./config/ncbi-taxon-pubmed-all.json

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#CPU
## similarities
./exec.sh $conffile 4
## CSV export
./exec.sh $conffile 6

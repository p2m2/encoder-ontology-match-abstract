#!/bin/bash

#SBATCH --job-name=PLANTEOME_SIM_CPU
#SBATCH --chdir=/scratch/ofilangi/encoder-ontology-match-abstract
#SBATCH --output=planteome_annot_pubmed_CPU_4_6.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=olivier.filangi@inrae.fr

## sbatch exec-sbatch-planteome-cpu-similarities-csv.sh

. /local/env/envpython-3.9.5.sh
conffile=config/planteome-pubmed-all.json

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#CPU
## similarities
./exec.sh $conffile 4
## CSV export
./exec.sh $conffile 6

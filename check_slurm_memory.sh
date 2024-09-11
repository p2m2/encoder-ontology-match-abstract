#!/bin/bash

function check_slurm_memory() {
    local job_id=${1:-$SLURM_JOB_ID}
    local uid=$(id -u)
    
    if [ -z "$job_id" ]; then
        echo "Erreur : Aucun ID de job SLURM fourni ou détecté."
        return 1
    fi

    local memory_usage_file="/sys/fs/cgroup/memory/slurm/uid_${uid}/job_${job_id}/memory.usage_in_bytes"
    local memory_limit_file="/sys/fs/cgroup/memory/slurm/uid_${uid}/job_${job_id}/memory.limit_in_bytes"

    if [ ! -f "$memory_usage_file" ] || [ ! -f "$memory_limit_file" ]; then
        echo "Erreur : Fichiers d'information mémoire non trouvés pour le job $job_id"
        return 1
    fi

    local memory_usage=$(cat "$memory_usage_file" | numfmt --to iec-i)
    local memory_limit=$(cat "$memory_limit_file" | numfmt --to iec-i)
    local memory_usage_percent=$(awk "BEGIN {printf \"%.2f\", $(cat $memory_usage_file) / $(cat $memory_limit_file) * 100}")

    echo "Job ID: $job_id"
    echo "Utilisation mémoire: $memory_usage"
    echo "Limite mémoire: $memory_limit"
    echo "Pourcentage utilisé: ${memory_usage_percent}%"
}

# Exemple d'utilisation :
# check_slurm_memory [job_id]


#!/bin/bash

# Vérifier si les deux arguments sont fournis
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <fichier_entree> <fichier_sortie>"
    exit 1
fi

input_file="$1"
output_file="$2"

# Vérifier si le fichier d'entrée existe
if [ ! -f "$input_file" ]; then
    echo "Erreur : Le fichier d'entrée '$input_file' n'existe pas."
    exit 1
fi

# Écrire les préfixes OWL dans le fichier de sortie
cat << EOF > "$output_file"
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

EOF

# Lire le fichier d'entrée ligne par ligne
while IFS= read -r line
do
    # Utiliser awk pour séparer les champs en respectant les guillemets
    uri=$(echo "$line" | awk -F' ' '{print $1}')
    label=$(echo "$line" | awk -F' ' '{print $2}')
    description=$(echo "$line" | awk -F' ' '{$1=$2=""; print substr($0,3)}')

    # Échapper les guillemets doubles dans le label et la description
    label=$(echo "$label" | sed 's/"/\\"/g')
    description=$(echo "$description" | sed 's/"/\\"/g')

    # Écrire chaque entrée dans le fichier OWL
    cat << EOF >> "$output_file"
<$uri> a owl:Class ;
    rdfs:label "$label"@en ;
    rdfs:comment "$description"@en .

EOF
done < "$input_file"

echo "Conversion terminée. Résultat écrit dans $output_file."


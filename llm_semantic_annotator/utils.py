import os,csv,json



def save_results(data, results):
    """
    Sauvegarde les résultats dans un fichier JSON.
    """
    retention_dir = get_retention_dir()
    filename = retention_dir+f"/{data}.json"
    with open(filename, 'w') as f:
        json.dump(results, f)
    print(f"Résultats sauvegardés dans {filename}")

def load_results(data):
    """
    Charge les résultats depuis un fichier JSON s'il existe.
    """
    retention_dir = get_retention_dir()
    filename = retention_dir+f"/{data}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def list_of_dicts_to_csv(data, filename):
    # Vérifier si la liste n'est pas vide
    if not data:
        print("La liste est vide.")
        return

    # Obtenir les en-têtes (toutes les clés uniques de tous les dictionnaires)
    headers = set().union(*(d.keys() for d in data))

    # Ouvrir le fichier en mode écriture
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        # Écrire les en-têtes
        writer.writeheader()
        
        # Écrire les données
        for row in data:
            writer.writerow(row)



def dict_to_csv(dictionary, filename):
    # Déterminer les en-têtes (clés du dictionnaire)
    headers = list(dictionary.keys())

    # Ouvrir le fichier en mode écriture
    with open(filename, 'w', newline='') as csvfile:
        # Créer un objet writer CSV
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # Écrire les en-têtes
        writer.writeheader()

        # Écrire les données
        writer.writerow(dictionary)

def get_retention_dir() :
    retention_dir = os.getenv('RETENTION_DIR', 'llm_semantic_annotator-workdir')
    if not os.path.exists(retention_dir):
        os.makedirs(retention_dir, exist_ok=True)
    return retention_dir
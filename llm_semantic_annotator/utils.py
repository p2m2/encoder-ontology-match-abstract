import csv

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

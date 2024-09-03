from collections import Counter
from rich import print

def ontologies_distrubition(data):
    # Extraire les préfixes des clés
    prefixes = []
    for item in data:
        for key in item.keys():
            prefix = key.split('__')[1]  # Extraire le préfixe entre les doubles underscores
            prefixes.append(prefix)

    # Compter la distribution des préfixes
    distribution = Counter(prefixes)

    # Afficher la distribution
    print("Distribution des préfixes :")
    for prefix, count in distribution.items():
        print(f"{prefix}: {count}")


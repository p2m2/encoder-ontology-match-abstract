from collections import Counter
from rich import print

def ontologies_distribution(data):
    # Extraire les préfixes des clés
    ontologies = []
    labels = []
    for doi, item in data.items():
        for key in item.keys():
            ontology = key.split('__')[1]  # Extraire le préfixe entre les doubles underscores
            ontologies.append(ontology)
            labels.append(key)

    # Compter la distribution des préfixes
    distributionOntologies = Counter(ontologies)
    distributionLabels = Counter(labels)

    print(f"nb abstracts : {len(data)}")
    annoted_abstracts = map(lambda item: 1 if len(item[1])>0 else 0, data.items())
    print(f"nb abstracts annoted : {sum(annoted_abstracts)}")
    # Afficher la distribution
    print("Distribution des ontologies :")
    for prefix, count in distributionOntologies.items():
        print(f"{prefix}: {count}")

    print("Distribution des labels :")
    sorted_distribution = sorted(distributionLabels.items(), key=lambda x: x[1], reverse=True)
    
    #for prefix, count in sorted_distribution:
    #    print(f"{prefix}: {count}")
    #    if count == 1:
    #        break



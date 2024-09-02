from rdflib import Graph, Namespace, URIRef
import json,os

def get_corpus(ontologies, debug_nb_terms_by_ontology=-1):
    tags = {}

    for ontology in download_ontologies(ontologies):
        tags.update(build_corpus(ontology, ontologies[ontology],debug_nb_terms_by_ontology))
    
    return tags

# Charger le fichier OWL local
def download_ontologies(list_ontologies):
    import wget
    
    for ontology,values in list_ontologies.items():
        print(ontology)
        print(values)
        list_ontologies[ontology]['filepath'] = os.path.basename(values['url'])
        print(list_ontologies[ontology])
        if not os.path.exists(values['url']):
            print("Downloading ontology: ",ontology)
            wget.download(values['url'])
        
    
    return list_ontologies
    

def build_corpus(ontology, ontology_config,debug_nb_terms_by_ontology):
    # Charger le fichier OWL local
    g = Graph()
    g.parse(ontology_config['filepath'], format=ontology_config['format'])

    # Namespace pour rdfs
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

    query_base = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?term ?descriptionLeaf ?labelLeaf WHERE { 
        ?term <http://purl.obolibrary.org/obo/IAO_0000115> ?descriptionLeaf .
        ?term rdfs:label ?labelLeaf .
    }
    """
    # label -> description1,description2, ....

    tags = {}

    # Exécuter la requête SPARQL
    results = g.query(query_base)

    for i,row in enumerate(results):
        print("---------------------")
        print(i,row)
        term = row.term
        descriptionLeaf = row.descriptionLeaf
        labelLeaf = row.labelLeaf

        # Requête SPARQL
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX peco: <http://purl.obolibrary.org/obo/PECO_>
        SELECT ?labelS WHERE { 
            ?s rdfs:subClassOf* """+ontology+":"+term.split("_").pop()+""" .
            ?s rdfs:label ?labelS .
        }
        """

        # Exécuter la requête SPARQL
        results = g.query(query)


        for row2 in results:
            label = row2.labelS
            
            if label.contains("obsolete"):
                continue

            formatted_label = "__"+ontology+"__" + str(label.lower()).replace(" ", "_")
            if not formatted_label in tags:
                tags[formatted_label] = []
            tags[formatted_label].append(descriptionLeaf)
        
        if i > debug_nb_terms_by_ontology:
            break

    return tags



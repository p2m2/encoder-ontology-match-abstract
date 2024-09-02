from rdflib import Graph, Namespace, URIRef
import os
import re
from tqdm import tqdm

def get_corpus(ontologies, debug_nb_terms_by_ontology=-1):
    tags = {}

    for ontology in download_ontologies(ontologies):
        tags.update(build_corpus(ontology, ontologies[ontology],debug_nb_terms_by_ontology))
    
    return tags

# Charger le fichier OWL local
def download_ontologies(list_ontologies):
    import wget
    
    for ontology,values in list_ontologies.items():
        list_ontologies[ontology]['filepath'] = os.path.basename(values['url'])
        
        if not os.path.exists(list_ontologies[ontology]['filepath']):
            print("Downloading ontology: ",ontology)
            wget.download(values['url'])
        
    
    return list_ontologies
    


def remove_prefix_tags(prefix_tag,text):
    escaped_prefix = re.escape(prefix_tag.upper())
    pattern = rf'{escaped_prefix}:\d+'

    # Remplacer toutes les occurrences par une chaîne vide
    return re.sub(pattern, '', text)

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
    nb_record=0
    for i,row in tqdm(enumerate(results)):
        term = row.term
        descriptionLeaf = row.descriptionLeaf
        labelLeaf = row.labelLeaf

        # Requête SPARQL
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX """+ontology+""": <http://purl.obolibrary.org/obo/PECO_>
        SELECT ?labelS WHERE { 
            ?s rdfs:subClassOf* """+ontology+":"+term.split("_").pop()+""" .
            ?s rdfs:label ?labelS .
        }
        """

        # Exécuter la requête SPARQL
        results = g.query(query)

        for row2 in results:
            label = row2.labelS
            
            if "obsolete" in label:
                continue

            formatted_label = "__"+ontology+"__" + str(label.lower()).replace(" ", "_")
            if not formatted_label in tags:
                tags[formatted_label] = []
            tags[formatted_label].append(remove_prefix_tags(ontology,descriptionLeaf))
            
            nb_record += 1
            if nb_record == debug_nb_terms_by_ontology:
                break
        if nb_record == debug_nb_terms_by_ontology:
            break
            
        
    print(len(tags))
    return tags



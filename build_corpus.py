from rdflib import Graph, Namespace, URIRef
import os
import re
from tqdm import tqdm


def get_corpus(ontologies, description_uri="<http://purl.obolibrary.org/obo/IAO_0000115>",debug_nb_terms_by_ontology=-1):
    tags = []

    for ontology in download_ontologies(ontologies):
        tags.extend(build_corpus(ontology, ontologies[ontology],description_uri,debug_nb_terms_by_ontology))
    
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

    v = re.sub(pattern, '', text)
    return re.sub(r'\(\)', '', v)

def build_corpus(
        ontology, 
        ontology_config,
        description_uri,
        debug_nb_terms_by_ontology):
    # Charger le fichier OWL local
    g = Graph()
    g.parse(ontology_config['filepath'], format=ontology_config['format'])

    # Namespace pour rdfs
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

    query_base = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?descriptionLeaf ?labelLeaf WHERE { 
        ?term """+description_uri+""" ?descriptionLeaf .
        ?term rdfs:label ?labelLeaf .
    }
    """
    # label -> description1,description2, ....

    tags = []

    # Exécuter la requête SPARQL
    results = g.query(query_base)
    nb_record=0

    for row in tqdm(results):

        descriptionLeaf = row.descriptionLeaf
        labelLeaf = row.labelLeaf
        
        formatted_label = "__"+ontology+"__" + str(labelLeaf.lower()).replace(" ", "_")
        
        if "obsolete" in formatted_label:
            continue
        if 'obsolete' in descriptionLeaf:
            continue
        
        tags.append({
                'label': formatted_label,
                'description' : remove_prefix_tags(ontology,descriptionLeaf)
            })
        
        if nb_record == debug_nb_terms_by_ontology:
                break
        nb_record+=1
    
    return tags



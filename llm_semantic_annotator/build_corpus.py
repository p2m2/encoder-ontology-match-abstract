import os, re, warnings, torch
from rdflib import Graph, Namespace, URIRef
from tqdm import tqdm
from rich import print
from llm_semantic_annotator import dict_to_csv,save_results,load_results, get_retention_dir
from llm_semantic_annotator import encode_text

retention_dir = get_retention_dir()

def get_corpus(ontologies,debug_nb_terms_by_ontology):
    tags = []

    for ontology in download_ontologies(ontologies):
        if os.path.exists(retention_dir+"/"+ontology+".json"):
            tags.extend(load_results(ontology))
        else:
            tags.extend(build_corpus(ontology, ontologies[ontology],debug_nb_terms_by_ontology))
            save_results(ontology,tags)
    
    return tags

# Charger le fichier OWL local
def download_ontologies(list_ontologies):
    import wget
   
    for ontology,values in list_ontologies.items():
        filepath= retention_dir+"/"+ontology+"."+values['format']
        list_ontologies[ontology]['filepath'] = filepath
        
        if not os.path.exists(list_ontologies[ontology]['filepath']):
            print("Downloading ontology: ",ontology)
            wget.download(values['url'],filepath)
        
    
    return list_ontologies
    


def remove_prefix_tags(prefix_tag,text):
    escaped_prefix = re.escape(prefix_tag.upper())
    pattern = rf'{escaped_prefix}:\d+'

    v = re.sub(pattern, '', text)
    return re.sub(r'\(\)', '', v)

def build_corpus(
        ontology, 
        ontology_config,
        debug_nb_terms_by_ontology):
    # Charger le fichier OWL local
    g = Graph()
    g.parse(ontology_config['filepath'], format=ontology_config['format'])

    # Namespace pour rdfs
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

    if 'properties' not in ontology_config or len(ontology_config['properties'])<=0:
        warnings.warn("'properties' is not defined ["+ontology+"]", UserWarning)
        return []

    varProperties = []
    sparqlProperties = []

    for i,prop in enumerate(ontology_config['properties']):
        varProperties.append("?prop"+str(i))
        sparqlProperties.append("?term "+prop+" ?prop"+str(i)+" .")
    
    query_base = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?term ?labelLeaf """ + " ".join(varProperties) + """ WHERE { 
        """+"\n".join(sparqlProperties)+"""
        ?term rdfs:label ?labelLeaf .
    }
    """
    print(query_base)
    tags = []

    # Exécuter la requête SPARQL
    results = g.query(query_base)
    nb_record=0

    for row in tqdm(results):

        descriptionLeaf = '\n'.join([ row.get(prop.replace('?',''), '') for prop in varProperties ])
        labelLeaf = row.labelLeaf
        
        formatted_label = "__"+ontology+"__" + str(labelLeaf.lower()).replace(" ", "_")
        
        if "obsolete" in formatted_label:
            continue
        if 'obsolete' in descriptionLeaf.lower():
            continue
        
        tags.append({
                'term': row.term,
                'label': formatted_label,
                'description' : remove_prefix_tags(ontology,descriptionLeaf)
            })
        
        if nb_record == debug_nb_terms_by_ontology:
            break
        nb_record+=1
    
    return tags


def manage_tags(ontologies_by_link,debug_nb_terms_by_ontology):
    # Encoder les descriptions des tags
    tag_embeddings = {}
    if os.path.exists(retention_dir+'/tags.pth'):
        print("load tags embeddings")
        tag_embeddings = torch.load(retention_dir+'/tags.pth')
    change = False

    for link_name,ontologies in ontologies_by_link.items():
        # get vocabulary from ontologies selected
        tags = get_corpus(ontologies, debug_nb_terms_by_ontology)
        for item in tqdm(tags):
            if not item['label'] in tag_embeddings:
                embeddings = encode_text(item['description'])
                tag_embeddings[item['label']] = embeddings
                change = True

    # Sauvegarder le dictionnaire dans un fichier
    if change:
        print("save tags embeddings")
        torch.save(tag_embeddings, retention_dir+'/tags.pth')
        dict_to_csv(tag_embeddings, retention_dir+'/tags.csv')

    return tag_embeddings

# Return tag embeddings in JSON format where the key is the DOI and the value is the embedding
def get_tags_embeddings():
    if os.path.exists(retention_dir+'/tags.pth'):
        return torch.load(retention_dir+'/tags.pth')
    else:
        return {}

# Return tag in JSON format where the key is the DOI and the value is the embedding
def get_tags_embeddings():
    if os.path.exists(retention_dir+'/tags.pth'):
        return torch.load(retention_dir+'/tags.pth')
    else:
        return {}

# Return a tags list where element is an object containing the term, label and description 
def get_tags():
    if os.path.exists(retention_dir+'/tags.pth'):
        return torch.load(retention_dir+'/tags.pth')
    else:
        return {}
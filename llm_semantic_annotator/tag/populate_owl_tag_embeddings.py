import os, re, warnings, torch
from rdflib import Graph, Namespace, URIRef
from tqdm import tqdm
from rich import print
import wget
from llm_semantic_annotator import list_of_dicts_to_csv,save_results,load_results
from llm_semantic_annotator import encode_text

def get_corpus(ontologies,config):
    debug_nb_terms_by_ontology = config['debug_nb_terms_by_ontology']
    retention_dir = config['retention_dir']

    tags = []

    for ontology in get_ontologies(ontologies,config):
        filename = retention_dir+f"/tag_{ontology}.json"

        if not config['force'] and os.path.exists(filename):
            tags.extend(load_results(filename))
        else:
            tags_ontology = build_corpus(ontology, ontologies[ontology],debug_nb_terms_by_ontology)
            print(f"save results on {filename} with length:{len(tags_ontology)}")
            if len(tags_ontology) == 0:
                warnings(f"** No Tags found  ** ")
            save_results(tags_ontology,filename)
            tags.extend(tags_ontology)
    
    return tags

# Charger le fichier OWL local
def get_ontologies(list_ontologies,config):
    
    for ontology,values in list_ontologies.items():
        
        filepath = config['retention_dir']+"/"+ontology+"."+values['format']
                
        # utilisation d'un fichier local
        if 'filepath' in values:
            if not os.path.exists(values['filepath']):
                raise FileNotFoundError(f"Le fichier '{values['filepath']}' n'existe pas.")
            if os.path.exists(filepath) and config['force']:
                os.remove(filepath)
            if os.path.exists(filepath):
                continue
            try:
                os.symlink(os.path.abspath(values['filepath']),filepath)
                print(f"Lien symbolique créé : {filepath} -> {values['filepath']}")
            except FileExistsError:
                print(f"Le lien symbolique {filepath} existe déjà.")
            except PermissionError:
                print("Erreur de permission. Assurez-vous d'avoir les droits nécessaires.")
            except OSError as e:
                print(f"Erreur lors de la création du lien symbolique : {e}")

        else: # sinon telechargement
            if config['force'] or not os.path.exists(filepath):
                print("Downloading ontology: ",ontology)
                wget.download(values['url'],filepath)
        
        list_ontologies[ontology]['filepath'] = filepath

        
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

    if 'label' not in ontology_config:
        ontology_config['label'] = "<http://www.w3.org/2000/01/rdf-schema#label>"

    varProperties = []
    sparqlProperties = []

    for i,prop in enumerate(ontology_config['properties']):
        varProperties.append("?prop"+str(i))
        sparqlProperties.append("?term "+prop+" ?prop"+str(i)+" .")
    
    query_base = """
    SELECT ?term ?labelLeaf """ + " ".join(varProperties) + """ WHERE { 
        """+"\n".join(sparqlProperties)+"""
        ?term """+ontology_config['label']+""" ?labelLeaf .
    }
    """
    print(query_base)
    tags = []

    # Exécuter la requête SPARQL
    results = g.query(query_base)
    nb_record=0
    print(f"Ontology {ontology} NB RECORDS:{len(results)}")
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
                'rdfs_label': labelLeaf,
                'description' : remove_prefix_tags(ontology,descriptionLeaf)
            })
        
        if nb_record == debug_nb_terms_by_ontology:
            break
        nb_record+=1
    
    return tags


def manage_tags(config):
    ontologies_by_link = config['ontologies']
    retention_dir = config['retention_dir']
    if 'force' not in config:
        config['force'] = False

    # Encoder les descriptions des tags
    tag_embeddings = {}
    if os.path.exists(retention_dir+'/tags.pth'):
        print("load tags embeddings")
        tag_embeddings = torch.load(retention_dir+'/tags.pth')
    
    change = False

    for link_name,ontologies in ontologies_by_link.items():
        # get vocabulary from ontologies selected
        tags = get_corpus(ontologies, config)
        list_of_dicts_to_csv(tags, retention_dir+f'/tags-{link_name}.csv')

        for item in tqdm(tags):
            if config['force'] or not item['label'] in tag_embeddings:
                embeddings = encode_text(f"{item['rdfs_label']} - {item['description']}")
                tag_embeddings[item['label']] = embeddings
                change = True

    # Sauvegarder le dictionnaire dans un fichier
    if change:
        print("save tags embeddings")
        torch.save(tag_embeddings, retention_dir+'/tags.pth')

    return tag_embeddings

# Return tag embeddings in JSON format where the key is the DOI and the value is the embedding
def get_tags_embeddings(retention_dir):
    if os.path.exists(retention_dir+'/tags.pth'):
        return torch.load(retention_dir+'/tags.pth')
    else:
        return {}

# Return a tags list where element is an object containing the term, label and description 
def get_tags(config):
    for filename in os.listdir(config['retention_dir']):
        results = []
        if filename.startswith('tag_') and filename.endswith('.json'):
            results.append(load_results(os.path.join(config['retention_dir'], filename)))
    # Remove duplicates
    return [dict(t) for t in {tuple(d.items()) for d in results}]
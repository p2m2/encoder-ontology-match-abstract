# mesh_similarity_evaluator.py
from tqdm import tqdm
from llm_semantic_annotator import ModelEmbeddingManager,AbstractManager
from llm_semantic_annotator import get_scores_files
from llm_semantic_annotator import (
    get_retention_dir,
    main_populate_owl_tag_embeddings, 
    main_populate_abstract_embeddings,
    main_compute_tag_chunk_similarities
)

config_descriptor_descriptor = {
    "encodeur" : "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size" : 32,

    "populate_owl_tag_embeddings" : {
        "prefix" : {
            "rdf" : "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs" : "http://www.w3.org/2000/01/rdf-schema#",
            "mesh" : "http://id.nlm.nih.gov/mesh/D000478",
            "meshv" : "http://id.nlm.nih.gov/mesh/vocab#",
            "owl" : "http://www.w3.org/2002/07/owl#"
        },
        "ontologies": {
            "foodon_link" : {
                "mesh_descriptor": {
                    "filepath": "data/mesh/mesh.nt",
                    "prefix": "http://id.nlm.nih.gov/mesh/D",
                    "format": "nt",
                    "label" : "rdfs:label",
                    "properties": [],
                    "constraints" : {
                       "meshv:active" : "true",
                       "rdf:type" : "meshv:TopicalDescriptor" 
                    }
                }
            }
        }
    }
}

config_concept_descriptor={
    "encodeur" : "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size" : 32,

    "populate_owl_tag_embeddings" : {
        "prefix" : {
            "rdf" : "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs" : "http://www.w3.org/2000/01/rdf-schema#",
            "mesh" : "http://id.nlm.nih.gov/mesh/D000478",
            "meshv" : "http://id.nlm.nih.gov/mesh/vocab#",
            "owl" : "http://www.w3.org/2002/07/owl#"
        },
        "ontologies": {
            "foodon_link" : {
                "mesh_descriptor": {
                    "filepath": "data/mesh/mesh.nt",
                    "prefix": "http://id.nlm.nih.gov/mesh/M",
                    "format": "nt",
                    "label" : "rdfs:label",
                    "properties": ["<http://id.nlm.nih.gov/mesh/vocab#scopeNote>"],
                    "constraints" : {
                       "meshv:active" : "true",
                       "rdf:type" : "meshv:Concept" 
                    }
                }
            }
        }
    }
}

abstracts_def= {
        "abstracts_per_file" : 200,
        "from_ncbi_api" : {
            "ncbi_api_chunk_size" : 200,
            "retmax" : 100,
            "selected_term" : [
                "metabolomics"
            ]
        }
    }

def init_config(config,retention_dir):
    config['retention_dir'] = get_retention_dir(retention_dir)
    config['force'] = True
    config['threshold_similarity_tag_chunk'] = 0.5
    config['threshold_similarity_tag'] = 0.8
    config["encodeur"]="sentence-transformers/all-MiniLM-L6-v2"
    config["batch_size"]=32
    config["populate_abstract_embeddings"] = abstracts_def

def build_asso_concept_descriptor(config):
    import ujson
    import os
    from rdflib import Graph
    from tqdm import tqdm

    storage_file = config['retention_dir']+"/link_concept_descriptor.json"
    
    if os.path.exists(storage_file):
        return ujson.load(open(storage_file, 'r'))
    
    g = Graph()
    path_mesh = os.path.dirname(os.path.abspath(__file__))+"/../data/mesh/mesh.nt"
    print("******* Build association concept-descriptor MeSH ************")
    print("loading ontology: ",path_mesh)
    
    
    g.parse(path_mesh, format='nt')
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
    PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?descriptor ?prop ?concept  WHERE { 
        ?descriptor ?prop ?concept . 
        VALUES ?prop { meshv:preferredConcept meshv:concept }
        FILTER(isURI(?concept))
    }
    """
    results = g.query(query)
    preferredConcept_dict = {}
    concept_dict = {}
    print(f"NB RECORDS:{len(results)}")
    for row in tqdm(results):
        descriptor = str(row.get('descriptor', '')).split("/").pop()
        concept = str(row.get('concept', '')).split("/").pop()
        prop = str(row.get('prop', '')).split("#").pop()
        
        if not concept.startswith('M'):
            continue 
        
        if not descriptor.startswith('D'):
            continue 
        
        if prop == 'preferredConcept':
            if concept not in preferredConcept_dict:
                preferredConcept_dict[concept] = []
            preferredConcept_dict[concept].append(descriptor)
        elif prop == 'concept':
            if concept not in concept_dict:
                concept_dict[concept] = []
            concept_dict[concept].append(descriptor)
        else:
            print("unkown property:",prop)
    
    res = {
        'preferredConcept' : preferredConcept_dict,
        'concept' : concept_dict
        }  
    
    ujson.dump(res, open(storage_file, 'w'))
            
    return res

def calculate_metrics_descriptor_descriptor(predicted_terms, actual_terms,links_unused={}):
    """ predicted_terms contains concept DXXXX et actuals terms are descriptors DXXXXX """

    true_positives = len(set(predicted_terms) & set(actual_terms))
    false_positives = len(set(predicted_terms) - set(actual_terms))
    false_negatives = len(set(actual_terms) - set(predicted_terms))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def calculate_metrics_concept_descriptor(predicted_terms, actual_terms,links_concept_descriptor):
    """ predicted_terms contains concept MXXXX et actuals terms are descriptors DXXXXX """

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Créer un ensemble de descripteurs prédits
    predicted_descriptors = set()
    for concept in predicted_terms:
        if concept in links_concept_descriptor['preferredConcept']:
            predicted_descriptors.update(links_concept_descriptor['preferredConcept'][concept])
        elif concept in links_concept_descriptor['concept']:
            predicted_descriptors.update(links_concept_descriptor['concept'][concept])
        else:
            false_positives += 1  # le concept n'a pas de lien avec un descriptor

    # Calculer les vrais positifs et les faux positifs
    for descriptor in predicted_descriptors:
        if descriptor in actual_terms:
            true_positives += 1
        else:
            false_positives += 1

    # Calculer les faux négatifs
    false_negatives = len(actual_terms) - true_positives

    # Calculer les métriques
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


def evaluate_abstracts(results_score_abstracts, abstracts,calculate_metrics,links={}):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    if (len(abstracts) == 0):
        return 0, 0, 0
    #print(results_score_abstracts)
    for abstract in tqdm(abstracts):
        doi = abstract['doi']
        
        if 'descriptor' not in abstract:
            continue
        
        actual_terms = abstract['descriptor']
        
        if doi in results_score_abstracts:
            predicted_terms = [ str(desc).split("/").pop() for desc in results_score_abstracts[doi].keys() ]
        else:
            predicted_terms = []
        
        if len(predicted_terms) == 0 and len(actual_terms) == 0:
            total_precision += 1.0
            total_recall += 1.0
            total_f1 += 1.0
            continue
        
        precision, recall, f1 = calculate_metrics(predicted_terms, actual_terms,links)
        #print(precision, recall, f1)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    avg_precision = total_precision / len(abstracts)
    avg_recall = total_recall / len(abstracts)
    avg_f1 = total_f1 / len(abstracts)
    
    return avg_precision, avg_recall, avg_f1

def get_abstracts_files(retention_dir):
    import re,os
    abstracts_files = []
    pattern = re.compile("abstracts_\\d+.json")
    for root, dirs, files in os.walk(retention_dir):
        for filename in files:
            if pattern.search(filename):
                abstracts_files.append(os.path.join(root, filename))
    return abstracts_files

def get_results_complete_similarities(scores_files,thresh):
    import ujson 
    results_complete_similarities = {}
    for file_name in scores_files:
        with open(file_name, 'r') as file:
            try:
                tp = ujson.load(file)
                # tri à nouveaux. Il faut que les scores etait généré en dessous si on change le threshold pour la relance
                for doi in tp:
                    for tag in tp[doi]:
                        if tp[doi][tag]>=thresh:
                            if doi not in results_complete_similarities:
                                results_complete_similarities[doi] = {}
                            results_complete_similarities[doi][tag] = tp[doi][tag]
                
            except ujson.JSONDecodeError:
                print(f"Erreur de décodage JSON dans le fichier {file_name}")
    return results_complete_similarities


def similarity_evaluator_main(config_evaluation, calculate_metrics, links):
    import json 
    config = config_evaluation
    
    threshold_list = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    
    #main_populate_owl_tag_embeddings(config)
    #main_populate_abstract_embeddings(config)
    #main_compute_tag_chunk_similarities(config)
    
    scores_files = get_scores_files(config['retention_dir'])
    abstracts_files = get_abstracts_files(config['retention_dir'])
    results_complete_similarities = {}
    
    for thresh in threshold_list:
        results_complete_similarities[thresh] = get_results_complete_similarities(scores_files, thresh)
    
    terms_by_abstract = []
    for file_name in abstracts_files:
        print(f"Chargement du fichier : {file_name}")
        with open(file_name, 'r') as file:
            try:
                tp = json.load(file)
                terms_by_abstract.extend(
                    [{'doi': t['doi'], 'descriptor': t['descriptor']} for t in tp]
                )
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON dans le fichier {file_name}")
    
    for thresh in threshold_list:
        print("\n" + "="*50)  # Ligne de séparation
        print(f"** Évaluation des résumés - seuil : {thresh:.2f} **")
        avg_precision, avg_recall, avg_f1 = evaluate_abstracts(results_complete_similarities[thresh], terms_by_abstract, calculate_metrics, links)
        
        # Affichage formaté
        print(f"Précision moyenne : {avg_precision:.2f}")
        print(f"Rappel moyen : {avg_recall:.2f}")
        print(f"Score F1 moyen : {avg_f1:.2f}")
        print("="*50)  # Ligne de séparation

if __name__ == "__main__":
    # 1ère évaluation Descriptor / Descriptor
    init_config(config_descriptor_descriptor, retention_dir="__evaluation_descriptor_descriptor__")
    similarity_evaluator_main(config_descriptor_descriptor, calculate_metrics_descriptor_descriptor, links={})
    
    # 2ème évaluation Concept / Descriptor
    init_config(config_concept_descriptor, retention_dir="__evaluation_concept_descriptor__")
    link_concept_descriptor = build_asso_concept_descriptor(config_concept_descriptor)
    similarity_evaluator_main(config_concept_descriptor, calculate_metrics_descriptor_descriptor, links=link_concept_descriptor)

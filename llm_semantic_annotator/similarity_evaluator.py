# mesh_similarity_evaluator.py
from llm_semantic_annotator import ModelEmbeddingManager,AbstractManager
from llm_semantic_annotator import get_scores_files
from llm_semantic_annotator import (
    get_retention_dir,
    main_populate_owl_tag_embeddings, 
    main_populate_abstract_embeddings,
    main_compute_tag_chunk_similarities
)

config_evaluation = {
    "encodeur" : "sentence-transformers/all-MiniLM-L6-v2",
    "threshold_similarity_tag_chunk" : 0.95,
    "threshold_similarity_tag" : 0.80,
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
    },
    "populate_abstract_embeddings" : {
        "abstracts_per_file" : 500,
        "from_ncbi_api" : {
            "ncbi_api_chunk_size" : 200,
            "retmax" : 500,
            "selected_term" : [
                "food"
            ]
        }
    }
}

def calculate_metrics(predicted_terms, actual_terms):
    true_positives = len(set(predicted_terms) & set(actual_terms))
    false_positives = len(set(predicted_terms) - set(actual_terms))
    false_negatives = len(set(actual_terms) - set(predicted_terms))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def evaluate_abstracts(results_score_abstracts, abstracts):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for abstract in abstracts:
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
        
        print("predicted_terms",predicted_terms)
        print("actual_terms",actual_terms)
        
        precision, recall, f1 = calculate_metrics(predicted_terms, actual_terms)
        print(precision, recall, f1)
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

# Exemple d'utilisation si ce module est exécuté directement
def similarity_evaluator_main():
    import json 
    config = config_evaluation
    config['retention_dir'] = get_retention_dir("__evaluation_descriptor__")
    config['force'] = True
    
    main_populate_owl_tag_embeddings(config)
    main_populate_abstract_embeddings(config)
    main_compute_tag_chunk_similarities(config)
    
    scores_files = get_scores_files(config['retention_dir'])
    abstracts_files = get_abstracts_files(config['retention_dir'])
    
    results_complete_similarities = {}
    for file_name in scores_files:
        with open(file_name, 'r') as file:
            try:
                results_complete_similarities.update(json.load(file))
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON dans le fichier {file_name}")
    
    terms_by_abstract = []
    for file_name in abstracts_files:
        print(file_name)
        with open(file_name, 'r') as file:
            try:
                terms_by_abstract.extend(json.load(file))
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON dans le fichier {file_name}")


    avg_precision, avg_recall, avg_f1 = evaluate_abstracts(results_complete_similarities, terms_by_abstract)
    print(f"Précision moyenne : {avg_precision:.2f}")
    print(f"Rappel moyen : {avg_recall:.2f}")
    print(f"Score F1 moyen : {avg_f1:.2f}")


if __name__ == "__main__":
    similarity_evaluator_main()
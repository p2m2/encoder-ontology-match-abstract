# mesh_similarity_evaluator.py
from llm_semantic_annotator import ModelEmbeddingManager,AbstractManager
from llm_semantic_annotator import get_scores_files


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
        
        actual_terms = abstract['descriptor']
        if doi in results_score_abstracts:
            predicted_terms = [ str(desc).split("/").pop() for desc in results_score_abstracts[doi].keys() ]
        else:
            predicted_terms = []
        
        print("predicted_terms",predicted_terms)
        print("actual_terms",actual_terms)
        
        precision, recall, f1 = calculate_metrics(predicted_terms, actual_terms)
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
def similarity_evaluator_main(config_all):
    import json 
    
    config = config_all.copy()
    config['retention_dir'] = config_all['retention_dir']
    config['force'] = config_all['force']
    config_abstract = config_all['populate_abstract_embeddings']
    config_abstract['retention_dir'] = config_all['retention_dir']
    
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

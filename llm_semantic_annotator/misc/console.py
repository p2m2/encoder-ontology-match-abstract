from rich import print
import pandas as pd
from tabulate import tabulate
import np as np

def display_best_similarity_abstract_tag(results_complete_similarities):
    dois = []
    similarities = []
    tags = []
    links_kind = []
    for doi, complete_similarities in results_complete_similarities.items():
        for tag, similarity in complete_similarities.items():
            dois.append(doi)
            similarities.append(similarity)
            tags.append(tag)
            links_kind.append("DOI-Tag")

    df = pd.DataFrame({
    'DOI': dois,
    'Similarity': similarities,
    'Tag': tags,
    #'link' : links_kind
    })

    df_sorted = df.sort_values(by='Similarity', ascending=False)
    df_sorted = df_sorted.reset_index(drop=True)

    print("## Best similarity between abstract and tag")
    print(tabulate(df_sorted, headers='keys', tablefmt='psql', showindex=False))

def display_ontologies_summary(results_complete_similarities):
    
    tag_list = []
    ontology = [] 
    count_ontology = []
    count = []
    similarity_ontology = []
    similarity_tag = []
    
    for doi, complete_similarities in results_complete_similarities.items():
        
        
        
        for tag, similarity in complete_similarities.items():
            ontology_tag = tag.split('__')[1]
            if ontology_tag not in ontology:
                ontology.append(ontology_tag)
                count_ontology.append(1)
                similarity_ontology.append([similarity])
            else:
                index = ontology.index(ontology_tag)
                count_ontology[index] += 1
                similarity_ontology[index].append(similarity)
            
            t = tag.split('__')[2]
            
            if t not in tag_list:
                tag_list.append(t)
                count.append(1)
                similarity_tag.append([similarity])
            else:
                index = tag_list.index(t)
                count[index] += 1
                similarity_tag[index].append(similarity)

        mean_similarity_ontology = []
        std_similarity_ontology = []
        
        for data in similarity_ontology:
            mean_similarity_ontology.append(np.mean(data))
            std_similarity_ontology.append(np.std(data))

    mean_similarity = []
    std_similarity = []
    
    for data in similarity_tag:
        mean_similarity.append(np.mean(data))
        std_similarity.append(np.std(data))
    
    df_tag = pd.DataFrame({
        'Tag': tag_list,
        'Count': count,
        'Mean Similarity': mean_similarity,
        'Std Similarity': std_similarity,
    })

    df_tag_sorted = df_tag.sort_values(by='Mean Similarity', ascending=False)
    df_tag_sorted = df_tag_sorted.reset_index(drop=True)

    print("## Summary of tags")
    print(tabulate(df_tag_sorted, headers='keys', tablefmt='psql', showindex=False))

    df_ontology = pd.DataFrame({
        'Ontology': ontology,
        'Count': count_ontology,
        'Mean Similarity': mean_similarity_ontology,
        'Std Similarity': std_similarity_ontology,
    })

    df_ontology_sorted = df_ontology.sort_values(by='Mean Similarity', ascending=False)
    df_ontology_sorted = df_ontology_sorted.reset_index(drop=True)

    print("## Summary of ontologies")
    # Afficher le tableau trié avec tabulate
    print(tabulate(df_ontology_sorted, headers='keys', tablefmt='psql', showindex=False))
    
 
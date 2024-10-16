from rich import print
import pandas as pd
from tabulate import tabulate
import np as np

def display_best_similarity_abstract_tag(prefix_file_name,results_complete_similarities,retention_dir):
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
    'DOI': [ f"https://doi.org/{doi}" for doi in dois],
    'Tag': tags,
    'Similarity': similarities
    #'link' : links_kind
    })

    df_sorted = df.sort_values(by=['DOI','Similarity'], ascending=False)
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted.to_csv(retention_dir+f"/best_similarities_{prefix_file_name}.csv", index=False)
    
    print("## Best similarity between abstract and tag")
    print(tabulate(df_sorted, headers='keys', tablefmt='psql', showindex=False))

def display_ontologies_summary(prefix_file_name,results_complete_similarities,retention_dir):
    
    tag_list = []
    ontology_tag_list = []
    ontology = [] 
    count_ontology = []
    count = []
    similarity_ontology = []
    similarity_tag = []
    
    for doi, complete_similarities in results_complete_similarities.items():
        
        for tag, similarity in complete_similarities.items():

            try:
                ontology_tag = tag.split('__')[1]  # Extraire le préfixe entre les doubles underscores
            except:
                ontology_tag = tag
            finally:
                pass
            
            if ontology_tag not in ontology:
                ontology.append(ontology_tag)
                count_ontology.append(1)
                similarity_ontology.append([similarity])
            else:
                index = ontology.index(ontology_tag)
                count_ontology[index] += 1
                similarity_ontology[index].append(similarity)
            
            try:
                t = tag.split('__')[2]
            except:
                t = tag
            finally:
                pass
            
            if t not in tag_list:
                ontology_tag_list.append(ontology_tag)
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
        'Ontology': ontology_tag_list,
        'Count': count,
        'Mean Similarity': mean_similarity,
        'Std Similarity': std_similarity,
    })

    df_tag_sorted = df_tag.sort_values(by='Mean Similarity', ascending=False)
    df_tag_sorted = df_tag_sorted.reset_index(drop=True)
    
    df_tag_sorted.to_csv(retention_dir+f"/summary_{prefix_file_name}.csv", index=False)

    print("## Summary of tags")
    print(tabulate(df_tag_sorted, headers='keys', tablefmt='psql', showindex=False))

    df_ontology = pd.DataFrame({
        'Ontology': ontology,
        'Tag Count': count_ontology,
        'Mean Similarity': mean_similarity_ontology,
        'Std Similarity': std_similarity_ontology,
    })

    df_ontology_sorted = df_ontology.sort_values(by='Mean Similarity', ascending=False)
    df_ontology_sorted = df_ontology_sorted.reset_index(drop=True)

    df_ontology_sorted.to_csv(retention_dir+f"/summary_ontologies_{prefix_file_name}.csv", index=False)

    print("## Summary of ontologies")
    # Afficher le tableau trié avec tabulate
    print(tabulate(df_ontology_sorted, headers='keys', tablefmt='psql', showindex=False))
    
 
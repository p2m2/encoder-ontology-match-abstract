import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def display_best_similarity_abstract_tag(results_complete_similarities, keep_tag_embeddings, retention_dir):
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
       'DOI': [f"https://doi.org/{doi}" for doi in dois],
       'Ontology': [keep_tag_embeddings[k]['ontology'] for k in tags],
       'Tag': [k.split('/')[-1] for k in tags],
       'Label': [keep_tag_embeddings[k]['label'] for k in tags],
       'Similarity': similarities
    })

    df_sorted = df.sort_values(by=['DOI','Similarity'], ascending=False)
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted.to_csv(retention_dir+f"/best_similarities.csv", index=False)
    
    console.print(Panel.fit("[bold cyan]Best similarity between abstract and tag[/bold cyan]"))
    
    table = Table(title="Best Similarities")
    for column in df_sorted.columns:
        table.add_column(column, style="green")
    
    for _, row in df_sorted.iterrows():
        table.add_row(*[str(value) for value in row])
    
    console.print(table)

def display_ontologies_summary(results_complete_similarities, keep_tag_embeddings, retention_dir):
    tag_list = []
    label_list = []
    ontology_tag_list = []
    ontology = [] 
    count_ontology = []
    count = []
    similarity_ontology = []
    similarity_tag = []
    
    for doi, complete_similarities in results_complete_similarities.items():
        for tag, similarity in complete_similarities.items():
            ontology_tag = keep_tag_embeddings[tag]['ontology']
            
            if ontology_tag not in ontology:
                ontology.append(ontology_tag)
                count_ontology.append(1)
                similarity_ontology.append([similarity])
            else:
                index = ontology.index(ontology_tag)
                count_ontology[index] += 1
                similarity_ontology[index].append(similarity)

            t = tag.split('/')[-1]
            
            if t not in tag_list:
                ontology_tag_list.append(ontology_tag)
                tag_list.append(t)
                label_list.append(keep_tag_embeddings[tag]['label'])
                count.append(1)
                similarity_tag.append([similarity])
            else:
                index = tag_list.index(t)
                count[index] += 1
                similarity_tag[index].append(similarity)

    mean_similarity_ontology = [np.mean(data) for data in similarity_ontology]
    std_similarity_ontology = [np.std(data) for data in similarity_ontology]
    
    mean_similarity = [np.mean(data) for data in similarity_tag]
    std_similarity = [np.std(data) for data in similarity_tag]
    
    df_tag = pd.DataFrame({
        'Ontology group': ontology_tag_list,
        'Tag': tag_list,
        'Label': label_list,
        'Count': count,
        'Mean Similarity': mean_similarity,
        'Std Similarity': std_similarity,
    })

    df_tag_sorted = df_tag.sort_values(by='Mean Similarity', ascending=False)
    df_tag_sorted = df_tag_sorted.reset_index(drop=True)
    
    df_tag_sorted.to_csv(retention_dir+f"/summary.csv", index=False)

    console.print(Panel.fit("[bold cyan]Summary of tags[/bold cyan]"))
    
    table_tag = Table(title="Tag Summary")
    for column in df_tag_sorted.columns:
        table_tag.add_column(column, style="green")
    
    for _, row in df_tag_sorted.iterrows():
        table_tag.add_row(*[str(value) for value in row])
    
    console.print(table_tag)

    df_ontology = pd.DataFrame({
        'Ontology group': ontology,
        'Tag Count': count_ontology,
        'Mean Similarity': mean_similarity_ontology,
        'Std Similarity': std_similarity_ontology,
    })

    df_ontology_sorted = df_ontology.sort_values(by='Mean Similarity', ascending=False)
    df_ontology_sorted = df_ontology_sorted.reset_index(drop=True)

    df_ontology_sorted.to_csv(retention_dir+f"/summary_ontologies.csv", index=False)

    console.print(Panel.fit("[bold cyan]Summary of ontologies[/bold cyan]"))
    
    table_ontology = Table(title="Ontology Summary")
    for column in df_ontology_sorted.columns:
        table_ontology.add_column(column, style="green")
    
    for _, row in df_ontology_sorted.iterrows():
        table_ontology.add_row(*[str(value) for value in row])
    
    console.print(table_ontology)

import os, json, torch, requests
from tqdm import tqdm
from rich import print
from llm_semantic_annotator import utils , list_of_dicts_to_csv,save_results,load_results,get_retention_dir
from llm_semantic_annotator import torch_utils, encode_text

retention_dir = get_retention_dir()

def get_ncbi_abstracts(search_term,debug_nb_req):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmax=100&retmode=json"

    # Essayer de charger les résultats existants
    results = load_results(search_term)
    
    if results is not None:
        print(f"Résultats chargés pour '{search_term}' et '{search_url}'")
        return results

    response = requests.get(search_url)
    search_results = response.json()
    id_list = search_results['esearchresult']['idlist']

    import xml.etree.ElementTree as ET

    abstracts = []
    chunk_size = 20  # Nombre d'IDs à traiter par requête

    for i in tqdm(range(0, len(id_list), chunk_size)):
        chunk = id_list[i:i+chunk_size]
        ids = ",".join(chunk)
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&rettype=abstract&retmode=xml"
        
        fetch_response = requests.get(fetch_url)
        
        root = ET.fromstring(fetch_response.content)
        
        for article in root.findall('.//PubmedArticle'):
            abstract_text = ""
            
            for abstract in article.findall('.//AbstractText'):
                abstract_text += abstract.text or ""
            
            doi = None
            for id_elem in article.findall(".//ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text
                    break

            
            abstracts.append({
                'title' : article.findtext(".//ArticleTitle"),
                'pmid': article.find('.//PMID').text,
                'abstract': abstract_text,
                'doi': doi
            })

        if i==debug_nb_req:
            break
    #json.dumps(abstracts, indent=4)
    results = [v for v in abstracts 
            if len(v['abstract'].strip()) > 0 and len(v['title'].strip()) > 0] 
    
    # Sauvegarder les nouveaux résultats
    save_results(search_term, results)

    return results

def manage_abstracts(selected_term,debug_nb_req=-1,debug_nb_abstracts_by_search=-1):
    chunks = get_ncbi_abstracts(selected_term,1)[0:debug_nb_abstracts_by_search]
    list_of_dicts_to_csv(chunks, retention_dir+"/chunks.csv")
    print("chunks embeddings")
    # Encoder les descriptions des tags
    chunk_embeddings = {}
    if os.path.exists(retention_dir+'/chunks.pth'):
        chunk_embeddings = torch.load(retention_dir+'/chunks.pth')

    change = False

    # Encoder les chunks de texte

    for chunk in tqdm(chunks):
        if not chunk['doi'] in chunk_embeddings:
            chunk_embeddings[chunk['doi']] = encode_text(chunk['abstract'])
            change = True

    if change:
        torch.save(chunk_embeddings, retention_dir+'/chunks.pth')
    
    return chunk_embeddings

def get_abstracts_embeddings():
    if os.path.exists(retention_dir+'/chunks.pth'):
        return torch.load(retention_dir+'/chunks.pth')
    else:
        return {}
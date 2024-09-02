import requests
from tqdm import tqdm
import os, json

def save_results(search_term, results):
    """
    Sauvegarde les résultats dans un fichier JSON.
    """
    filename = f"results_{search_term}.json"
    with open(filename, 'w') as f:
        json.dump(results, f)
    print(f"Résultats sauvegardés dans {filename}")

def load_results(search_term):
    """
    Charge les résultats depuis un fichier JSON s'il existe.
    """
    filename = f"results_{search_term}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def get_ncbi_abstracts(search_term,debug_nb_req=-1):
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
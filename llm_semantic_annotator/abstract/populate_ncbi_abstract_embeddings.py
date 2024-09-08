import os, torch, requests, re
from tqdm import tqdm
from rich import print
from llm_semantic_annotator import list_of_dicts_to_csv,save_results,load_results
from llm_semantic_annotator import ModelEmbeddingManagement

# return json with element containing title, pmid, abstract and doi
def get_ncbi_abstracts(config):
    debug_nb_req = config['debug_nb_ncbi_request']
    retmax = config['retmax']
    search_term_list = config['selected_term']
    if 'force' not in config:
        config['force'] = False

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    nrecord = 0
    results = []

    for search_term in search_term_list:  
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmax={retmax}&retmode=json"
        
        filename = config['retention_dir'] + f"/abstract_{search_term}_{debug_nb_req}_{retmax}.json"
        
        if not config['force']:
            # Essayer de charger les résultats existants
            results_cur = load_results(filename)
            # if results exist in the file, we don't need to do the request
            if results_cur is not None:
                print(f"Résultats chargés pour '{search_term}' et '{search_url}'")
                results.extend(results_cur)
                continue

        response = requests.get(search_url)
        search_results = response.json()
        
        if 'esearchresult' not in search_results:
            continue
        
        if 'idlist' in search_results['esearchresult']:
            id_list = search_results['esearchresult']['idlist']
        else:
            continue
        
        print("nb abstract:",len(id_list))
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

            if nrecord==debug_nb_req:
                break
            nrecord += 1

        results_cur = [v for v in abstracts 
                if len(v['abstract'].strip()) > 0 and len(v['title'].strip()) > 0] 
        
        # Sauvegarder les nouveaux résultats
        save_results(results_cur, filename)
        
        results.extend(results_cur)

    return results

def manage_abstracts(config):
    debug_nb_abstracts_by_search = config['debug_nb_abstracts_by_search']
    retention_dir = config['retention_dir']
    if 'force' not in config:
        config['force'] = False

    chunks = get_ncbi_abstracts(config)
    filename_pth=retention_dir+'/abstract_chunks.pth'
    filename_csv=retention_dir+'/abstract_chunks.csv'

    if debug_nb_abstracts_by_search>0:
        chunks = chunks[:debug_nb_abstracts_by_search]
    
    print("chunks embeddings")
    # Encoder les descriptions des tags
    chunk_embeddings = {}
    if os.path.exists(filename_pth):
        chunk_embeddings = torch.load(filename_pth)

    change = False

    # Encoder les chunks de texte
    chunks_toencoce = []
    chunks_doi_ref = []
    for chunk in tqdm(chunks):
        if not chunk['doi'] in chunk_embeddings:
            #chunk_embeddings[chunk['doi']] = [ModelEmbeddingManagement().encode_text(chunk['title'])]
            chunk_embeddings[chunk['doi']] = []
            chunks_doi_ref.append(chunk['doi'])
            chunks_toencoce.append(chunk['title'])

            pattern = r'(?<=[.!?])\s+(?=[A-Z])'
            sentences = re.split(pattern, chunk['abstract'])
            #enc = [ModelEmbeddingManagement().encode_text(sentence.strip()) for sentence in sentences]
            #chunk_embeddings[chunk['doi']].extend(enc)
            for s in sentences:
                chunks_toencoce.append(s)
                chunks_doi_ref.append(chunk['doi'])
            change = True

    if len(chunks_toencoce)>0:
        embeddings = ModelEmbeddingManagement().encode_text_batch(chunks_toencoce)
    
        for idx,emb in enumerate(embeddings):
            chunk_embeddings[chunks_doi_ref[idx]].append(emb)

        if change:
            torch.save(chunk_embeddings, filename_pth)
            list_of_dicts_to_csv(chunks, filename_csv)
        

    return chunk_embeddings

def get_abstracts_embeddings(retention_dir):
    filename_pth=retention_dir+'/abstract_chunks.pth'
  
    if os.path.exists(filename_pth):
        return torch.load(filename_pth)
    else:
        return {}
    
def get_abstracts(config):
    for filename in os.listdir(config['retention_dir']):
        results = []
        if filename.startswith('abstract_') and filename.endswith('.json'):
            results.append(load_results(os.path.join(config['retention_dir'], filename)))
    # Remove duplicates
    return [dict(t) for t in {tuple(d.items()) for d in results}]
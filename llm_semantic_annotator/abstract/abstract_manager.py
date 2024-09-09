import os, torch, requests, re, json
from tqdm import tqdm
from rich import print
from llm_semantic_annotator import list_of_dicts_to_csv, save_results, load_results
from llm_semantic_annotator import ModelEmbeddingManager
import xml.etree.ElementTree as ET

class AbstractManager:
    def __init__(self, config):
        self.config = config
        self.mem = ModelEmbeddingManager(config)

    def get_ncbi_abstracts_from_api(self):
        if 'debug_nb_ncbi_request' in self.config:
            debug_nb_req = self.config['debug_nb_ncbi_request']
        else:
            debug_nb_req = -1

        retmax = self.config['from_ncbi_api']['retmax']
        search_term_list = self.config['from_ncbi_api']['selected_term']
        force = self.config.get('force', False)

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        results = []

        for search_term in search_term_list:  
            search_url = f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmax={retmax}&retmode=json"
            
            filename = self.config['retention_dir'] + f"/abstract_{search_term}_{debug_nb_req}_{retmax}.json"
            
            if not force:
                results_cur = load_results(filename)
                if results_cur is not None:
                    print(f"Résultats chargés pour '{search_term}' et '{search_url}'")
                    results.extend(results_cur)
                    continue

            response = requests.get(search_url)
            search_results = response.json()
            
            if 'esearchresult' not in search_results or 'idlist' not in search_results['esearchresult']:
                continue
            
            id_list = search_results['esearchresult']['idlist']
            
            print("nb abstract:", len(id_list))

            abstracts = []
            chunk_size = 20

            for i in tqdm(range(0, len(id_list), chunk_size)):
                chunk = id_list[i:i+chunk_size]
                ids = ",".join(chunk)
                fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&rettype=abstract&retmode=xml"
                
                fetch_response = requests.get(fetch_url)
                
                root = ET.fromstring(fetch_response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    abstract_text = "".join(abstract.text or "" for abstract in article.findall('.//AbstractText'))
                    
                    doi = next((id_elem.text for id_elem in article.findall(".//ArticleId") if id_elem.get("IdType") == "doi"), None)

                    abstracts.append({
                        'title': article.findtext(".//ArticleTitle"),
                        'abstract': abstract_text,
                        'doi': doi
                    })

                if len(abstracts) >= debug_nb_req:
                    break

            results_cur = [v for v in abstracts if v['abstract'].strip() and v['title'].strip()]
            
            save_results(results_cur, filename)
            
            results.extend(results_cur)

        return results

    def get_ncbi_abstracts_from_file(self):
        files_to_parse = self.config['from_file']['json_files']
        results = []
        
        for f in files_to_parse:
            if not os.path.exists(f):
                print(f"File {f} does not exist")
                continue

            with open(f, 'r') as file:
                results.extend(json.load(file))
        return results

    def manage_abstracts(self):
        if 'debug_nb_abstracts_by_search' in self.config:
            debug_nb_abstracts_by_search = self.config['debug_nb_abstracts_by_search']
        else:
            debug_nb_abstracts_by_search = -1

        retention_dir = self.config['retention_dir']

        chunk_embeddings = {} if self.config['force'] else self.mem.load_pth("chunks_asbtract")
        abstracts = []
        if 'from_ncbi_api' in self.config :
            abstracts.extend(self.get_ncbi_abstracts_from_api())
        else:
            print("No abstracts source 'from_api' selected")

        if 'from_file' in self.config :
            abstracts.extend(self.get_ncbi_abstracts_from_file())
        else:
            print("No abstracts source 'from_file' selected")
        
        filename_csv = retention_dir + '/abstract.csv'

        if debug_nb_abstracts_by_search > 0:
            abstracts = abstracts[:debug_nb_abstracts_by_search]

        change = False

        chunks_toencode = []
        chunks_doi_ref = []
        for abstract in tqdm(abstracts):
            if abstract['doi'] not in chunk_embeddings:
                chunk_embeddings[abstract['doi']] = []
                chunks_doi_ref.append(abstract['doi'])
                chunks_toencode.append(abstract['title'])

                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract['abstract'])

                for s in sentences:
                    chunks_toencode.append(s)
                    chunks_doi_ref.append(abstract['doi'])
                change = True

        if chunks_toencode:
            embeddings = self.mem.encode_text_batch(chunks_toencode)
        
            for idx, emb in enumerate(embeddings):
                chunk_embeddings[chunks_doi_ref[idx]].append(emb)

            if change:
                self.mem.save_pth(chunk_embeddings, "chunks_asbtract")
                list_of_dicts_to_csv(abstracts, filename_csv)

        return chunk_embeddings

    def get_abstracts_embeddings(self):
        return self.mem.load_pth("chunks_asbtract")
    
    def get_abstracts(self):
        results = []
        for filename in os.listdir(self.config['retention_dir']):
            if filename.startswith('abstract_') and filename.endswith('.json'):
                results.extend(load_results(os.path.join(self.config['retention_dir'], filename)))
        return [dict(t) for t in {tuple(d.items()) for d in results}]

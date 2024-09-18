import os, torch, requests, re, json
from tqdm import tqdm
from rich import print
from llm_semantic_annotator import load_results
import xml.etree.ElementTree as ET
from pathlib import Path

class AbstractManager:
    def __init__(self, config, model_embedding_manager):
        self.config = config
    
        self.abstracts_per_file=config.get('abstracts_per_file', 100)
        self.mem = model_embedding_manager
        if 'from_ncbi_api' in config:
            self.retmax = self.config.get('from_ncbi_api').get('retmax',10000)
            self.debug_nb_req = self.config.get('from_ncbi_api').get('debug_nb_ncbi_request',-1)
            self.ncbi_api_chunk_size = config.get('from_ncbi_api').get('ncbi_api_chunk_size', 20)
        else:
            self.retmax = 10000
            self.debug_nb_req = -1
            self.ncbi_api_chunk_size = 20

    def _get_index_abstract(self):
        existing_files = [
            f for f in os.listdir(self.config['retention_dir']) 
                if f.startswith(f"abstracts_") and f.endswith(".json")]
        
        if existing_files:
            max_index = max([int(f.split('_')[-1].split('.')[0]) for f in existing_files])
            return max_index + 1
        else:
            return 1

    def _remove_abstract_files(self):
        for filename in os.listdir(self.config['retention_dir']):
            if filename.startswith('abstracts_') and filename.endswith('.json'):
                os.remove(os.path.join(self.config['retention_dir'], filename))

    def _save_to_json_file_with_index(self,abstracts, file_index):
        filename = self.config['retention_dir']+f"/abstracts_{file_index}.json"
        print(f"abstract file:{filename}, nb :{len(abstracts)}")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(abstracts, f, ensure_ascii=False, indent=4)
    
    def _link_to_json_file_with_index(self,source_file, file_index):
        destination = self.config['retention_dir']+f"/abstracts_{file_index}.json"
        os.symlink(os.path.abspath(source_file), destination)

    def get_ncbi_abstracts_from_api(self):
        
        search_term_list = self.config['from_ncbi_api']['selected_term']
        
        file_index = self._get_index_abstract()
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        abstract_count = 0
        abstracts = []

        for search_term in search_term_list:
            search_url = f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmax={self.retmax}&retmode=json"
            response = requests.get(search_url)
            search_results = response.json()
            if 'error' in search_results:
                raise ValueError(f"Error in search results: {search_results['error']}")

            if 'esearchresult' not in search_results or 'idlist' not in search_results['esearchresult']:
                continue
            
            id_list = search_results['esearchresult']['idlist']

            for i in tqdm(range(0, len(id_list), self.ncbi_api_chunk_size)):
                chunk = id_list[i:i+self.ncbi_api_chunk_size]
                ids = ",".join(chunk)
                fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&rettype=abstract&retmode=xml"

                fetch_response = requests.get(fetch_url)
                
                root = ET.fromstring(fetch_response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    abstract_text = "".join(abstract.text or "" for abstract in article.findall('.//AbstractText'))
                    
                    doi = next((id_elem.text for id_elem in article.findall(".//ArticleId") if id_elem.get("IdType") == "doi"), None)
                    abstract_title = article.findtext(".//ArticleTitle")
                    
                    if abstract_title.strip() == '' or abstract_text == '':
                        continue
                    

                    abstracts.append({
                        'title': article.findtext(".//ArticleTitle"),
                        'abstract': abstract_text,
                        'doi': doi
                    })

                    abstract_count += 1
                    
                    if abstract_count % self.abstracts_per_file == 0:
                        self._save_to_json_file_with_index(abstracts, file_index)
                        abstracts = []
                        file_index += 1

                    if len(abstracts) >= self.debug_nb_req:
                        break
            
        # Sauvegarder les abstracts restants
        if abstracts:
            self._save_to_json_file_with_index(abstracts, file_index)
        
        print(f"Total abstract :{abstract_count}")

    def get_ncbi_abstracts_from_file(self):
        files_to_parse = self.config['from_file']['json_files']
        file_index = self._get_index_abstract()
        
        for file in files_to_parse:
            if not os.path.exists(file):
                print(f"File {file} does not exist")
                continue
            self._link_to_json_file_with_index(file, file_index)
            file_index+=1
        

    def _set_embedding_abstract_file(self):
        for filename in os.listdir(self.config['retention_dir']):
            
            if filename.startswith('abstracts_') and filename.endswith('.json'):
                results = load_results(os.path.join(self.config['retention_dir'], filename))
                genname = filename.split('.json')[0]
                self.mem.save_pth(self.mem.encode_abstracts(results,genname),genname)

    def manage_abstracts(self):

        self._remove_abstract_files()
        
        if 'from_ncbi_api' in self.config :
            self.get_ncbi_abstracts_from_api()
        else:
            print("No abstracts source 'from_api' selected")

        if 'from_file' in self.config :
            self.get_ncbi_abstracts_from_file()
        else:
            print("No abstracts source 'from_file' selected")

        self._set_embedding_abstract_file()


    # Return tag embeddings in JSON format where the key is the DOI and the value is the embedding
    def get_files_abstracts_embeddings(self):
        matching_files = []
    
        # Compile le motif regex pour une meilleure performance
        pattern = re.compile(f"abstracts_.*-{self.mem.model_suffix}.pth")
        # Parcourt tous les fichiers dans le chemin donné
        for root, dirs, files in os.walk(self.config['retention_dir']):
            for filename in files:
                if pattern.search(filename):
                    # Ajoute le chemin complet du fichier à la liste
                    matching_files.append(os.path.join(root, filename))
        
        return matching_files
    
    def get_abstracts(self):
        results = []
        for filename in os.listdir(self.config['retention_dir']):
            if filename.startswith('abstract_') and filename.endswith('.json'):
                results.extend(load_results(os.path.join(self.config['retention_dir'], filename)))
        return [dict(t) for t in {tuple(d.items()) for d in results}]

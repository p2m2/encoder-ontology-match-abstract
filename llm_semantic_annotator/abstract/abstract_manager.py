import os, torch, requests, re, json
from tqdm import tqdm
from rich import print
from llm_semantic_annotator import load_results
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from collections import defaultdict 
        
class AbstractManager:
    def __init__(self, config,model_embedding_manager,tags_manager):
        self.config = config
    
        self.abstracts_per_file=config.get('abstracts_per_file', 100)
        self.mem = model_embedding_manager
        self.tags_manager = tags_manager
        
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
        ldoi = {}

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
                
                fetch_response = requests.post(fetch_url)
                
                root = ET.fromstring(fetch_response.content)
                    
                for article in root.findall('.//PubmedArticle'):
                    abstract_text = "".join(abstract.text or "" for abstract in article.findall('.//AbstractText'))
                    
                    doi = next((id_elem.text for id_elem in article.findall(".//ArticleId") if id_elem.get("IdType") == "doi"), None)
                    abstract_title = article.findtext(".//ArticleTitle")
                    
                    meshTerms = []
                    
                    for meshHeading in article.findall(".//MeshHeading/DescriptorName"):
                        meshTerms.append(str(meshHeading.get('UI')))
                    
                    if abstract_title.strip() == '' or abstract_text == '' or doi is None:
                        continue
                    
                    if doi not in ldoi:
                        abstracts.append({
                            'title': article.findtext(".//ArticleTitle"),
                            'abstract': abstract_text,
                            'doi': doi,
                            'descriptor': meshTerms
                        })
                        abstract_count += 1
                        ldoi[doi] = True
                    
                    if abstract_count % self.abstracts_per_file == 0:
                        self._save_to_json_file_with_index(abstracts, file_index)
                        abstracts = []
                        file_index += 1

                    if self.debug_nb_req>0 and len(abstracts) >= self.debug_nb_req:
                        break

        # Sauvegarder les abstracts restants
        if abstracts:
            self._save_to_json_file_with_index(abstracts, file_index)
        
        print(f"Total abstract :{abstract_count}")

    def get_ncbi_abstracts_from_files(self):
        
        if 'json_files' not in self.config['from_file']:
            return
        
        files_to_parse = self.config['from_file']['json_files']
        file_index = self._get_index_abstract()
        
        for file in files_to_parse:
            if not os.path.exists(file):
                print(f"File {file} does not exist")
                continue
            self._link_to_json_file_with_index(file, file_index)
            file_index+=1
            
    def get_ncbi_abstracts_from_directory(self):
        import glob
        
        if 'json_dir' not in self.config['from_file']:
            return
        
        directory_to_parse = self.config['from_file']['json_dir']
        file_index = self._get_index_abstract()
        
        for file in glob.glob(os.path.join(directory_to_parse, "*.json")):
            self._link_to_json_file_with_index(file, file_index)
            file_index+=1
    
    @staticmethod
    def _get_data_abstracts_file(json_f):
        try:
            results = load_results(json_f)
            # fix bug if abstracts is a dict
            if isinstance(results, dict):
                results = [results]
                
        except Exception as e:
            results = []

            with open(json_f, 'r') as fichier:
                for ligne in fichier:
                    # Charger chaque ligne comme un dictionnaire JSON
                    dictionnaire = json.loads(ligne)
                    results.append(dictionnaire)
                    continue
        return results
        
    def _set_embedding_abstract_file(self):
        for filename in os.listdir(self.config['retention_dir']):
            
            if filename.startswith('abstracts_') and filename.endswith('.json'):
                json_f = os.path.join(self.config['retention_dir'], filename)
                genname = filename.split('.json')[0]
                pth_filename = self.mem.get_filename_pth(genname)
                if os.path.exists(pth_filename):
                    print(f"{pth_filename} already exists !")
                    continue
                results = self._get_data_abstracts_file(json_f)
                self.mem.save_pth(self.mem.encode_abstracts(results,genname),genname)

    def manage_abstracts(self):

        self._remove_abstract_files()
        
        if 'from_ncbi_api' in self.config :
            self.get_ncbi_abstracts_from_api()
        
        if 'from_file' in self.config :
            self.get_ncbi_abstracts_from_files()
            self.get_ncbi_abstracts_from_directory()
        
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
    
    def build_ascendants_terms(self,ascendants_dict,graphs):
        
        for graph in graphs:
            g = graph['g']
            prefix = graph['prefix']
            query = """ SELECT ?term ?ascendant WHERE { 
                ?term rdfs:subClassOf* ?ascendant . 
                FILTER(STRSTARTS(STR(?term), '"""+ prefix + """'))
                FILTER(STRSTARTS(STR(?ascendant), '"""+ prefix + """'))
            } """ # Exécuter la requête 
            
            results = g.query(query) # Remplir le dictionnaire avec les résultats 
            for row in results: 
                term = str(row.term) 
                ascendant = str(row.ascendant) 
                if term != ascendant: # Éviter d'ajouter le terme lui-même comme ascendant 
                    ascendants_dict[term].append(ascendant) # Afficher le dictionnaire 
            
            # we add ascendants of ascendants to avoid future requests
            ascendants_dict_to_add = {}
            for term in ascendants_dict:
                listAscendants = ascendants_dict[term]
                liste_asc = listAscendants.copy()

                while liste_asc:
                    ascendant = liste_asc.pop(0)
                    if ascendant not in ascendants_dict:
                        ascendants_dict_to_add[ascendant] = liste_asc.copy()
            
            ascendants_dict.update(ascendants_dict_to_add)
            
        print("update dictionnary size :",len(ascendants_dict))    
        return ascendants_dict
        
        
    def build_dataset_abstracts_annotations(self):
        
        import re,os
        import time
        graphs = self.tags_manager.get_graphs_ontologies()
        ascendants_dict = defaultdict(list)
        debut = time.time()
        self.build_ascendants_terms(ascendants_dict,graphs)
        duree = time.time() - debut
        print(f"loading terms with ancestors : {duree:.4f} secondes")

        pattern = re.compile("abstracts_\\d+.json")
        for root, dirs, files in os.walk(self.config['retention_dir']):
            for filename in files:
                if pattern.search(filename):
                    abstracts_json = os.path.join(root, filename)
                    abstracts_origin_gen = filename.split('.json')[0]
                    abstracts_gen = self.mem.get_filename_pth(abstracts_origin_gen).split('.pth')[0]
                    abstracts_scores = abstracts_gen+"_scores.json"
                    abstracts_annotations_results_file = abstracts_gen+"_queryresults.csv"
                    print(abstracts_annotations_results_file)
                    if os.path.exists(abstracts_annotations_results_file):
                        print(f"{abstracts_annotations_results_file} already exists !")
                        return
                    abstracts_data = self._get_data_abstracts_file(abstracts_json)
                    abstracts_annot = load_results(abstracts_scores)
                    doi_list = []
                    topicalDescriptor_list = []
                    pmid_list = []
                    reference_id_list = []
                    for abstract in abstracts_data:                                         
                        if 'doi' not in abstract:
                            continue
                        doi = abstract['doi']
                        if doi in abstracts_annot:
                            for tag in abstracts_annot[doi]:
                                if 'reference_id' in abstract:
                                    reference_id=abstract['reference_id']
                                else:
                                    reference_id=None
                                
                                if 'pmid' in abstract:
                                    pmid=abstract['pmid']
                                else:
                                    pmid=None
                                    
                                # the tag is the term            
                                topicalDescriptor_list.append(tag)
                                doi_list.append(doi)
                                reference_id_list.append(reference_id)
                                pmid_list.append(pmid)
                                
                                # ancestors
                                for ancestor in ascendants_dict[tag]:
                                    topicalDescriptor_list.append(ancestor)
                                    doi_list.append(doi)
                                    reference_id_list.append(reference_id)
                                    pmid_list.append(pmid)
                                
                    df = pd.DataFrame({
                        'doi': doi_list,
                        'topicalDescriptor': topicalDescriptor_list,
                        'pmid' : pmid_list,
                        'reference_id' : reference_id_list
                    })
                    
                    print(abstracts_annotations_results_file)
                    df.to_csv(abstracts_annotations_results_file, index=False)
import requests,tarfile,os, torch,csv,zipfile,re,json
from urllib.parse import urlparse
from tqdm import tqdm
from rich import print
from llm_semantic_annotator import list_of_dicts_to_csv,save_results,load_results
from llm_semantic_annotator import ModelEmbeddingManager
import logging
from collections import defaultdict
import pandas as pd

class TaxonTagManager:
    def __init__(self,config,model_embedding_manager):
        
        self.config=config
        self.retention_dir = config['retention_dir']
        self.force = config.get('force',None)
        
        self.logger = logging.getLogger(__name__)
        csv.field_size_limit(10000000)
        self.tags_per_file = config.get('tags_per_file', 1000)

        # Définir le répertoire de travail
        self.gbif_work_dir = os.path.join(os.path.dirname(__file__), '..', '..','data', 'gbif-backbone')
        self.ncbi_work_dir = os.path.join(os.path.dirname(__file__), '..', '..','data', 'ncbi')

        self.regex = config.get('regex',None)
        self.debug_nb_taxon = config.get('debug_nb_taxon',-1)
        self.tags_gbif_path_filename = f"tags_gbif_taxon"
        self.tags_ncbi_path_filename = f"tags_ncbi_taxon"
        self.mem = model_embedding_manager

    def _download_file(self, url, filename):
        self.logger.info(f"Téléchargement de {url}")
        print(f"Téléchargement de {url} vers {filename}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as file:
                file.write(response.content)
        except requests.RequestException as e:
            self.logger.error(f"Erreur lors du téléchargement : {str(e)}")
            raise
    
    def _extract_zip(self, zip_path):
        self.logger.info("Extraction du fichier zip")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall()
        except zipfile.BadZipFile as e:
            self.logger.error(f"Erreur lors de l'extraction : {str(e)}")
            raise
    def _extract_tar_gz(self,filename):
        print("Extraction du fichier tar.gz {}".format(filename))
        # Extraire le contenu du fichier tar.gz
        try:
            extract_path = os.path.dirname(filename)
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=extract_path)

            # Optionnel : supprimer le fichier tar.gz après extraction
            os.remove(filename)
        except zipfile.BadZipFile as e:
            self.logger.error(f"Erreur lors de l'extraction : {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction : {str(e)}")
            raise

    def _vernicular_gbif_file(self):
        if 'vernicular_tsv_debug' in self.config:
            vernacular_file = os.path.join(self.gbif_work_dir, self.config['vernicular_tsv_debug'])
        else:
            vernacular_file = os.path.join(self.gbif_work_dir, 'vernacular_name.tsv')
        
        return vernacular_file

    def _taxon_gbif_file(self):
        if 'taxon_tsv_debug' in self.config:
            taxon_file = os.path.join(self.gbif_work_dir, self.config['taxon_tsv_debug'])
        else:
            taxon_file = os.path.join(self.gbif_work_dir, 'taxon.tsv')
        
        return taxon_file
    
    def _process_vernacular_name_gbif(self):
        output_file = os.path.join(self.gbif_work_dir, "vernacular_name.tsv")
        if not os.path.exists(output_file):
            with open('VernacularName.tsv', 'r', newline='', encoding='utf-8') as infile, \
                open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                tsv_reader = csv.reader(infile, delimiter='\t')
                tsv_writer = csv.writer(outfile, delimiter='\t')
                for row in tsv_reader:
                    if len(row) >= 2:
                        tsv_writer.writerow([row[0], row[1]])
    
    def _process_taxon_gbif(self):
        output_file = os.path.join(self.gbif_work_dir, "taxon.tsv")
        if not os.path.exists(output_file):
            with open('Taxon.tsv', 'r', newline='', encoding='utf-8') as infile, \
                open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                tsv_reader = csv.reader(infile, delimiter='\t')
                tsv_writer = csv.writer(outfile, delimiter='\t')
                for row in tsv_reader:
                    if len(row) >= 6:
                        tsv_writer.writerow([row[0], row[5]])
    
    def _process_tsv_gbif_files(self):
        self.logger.info("Traitement des fichiers TSV")
        self._process_vernacular_name_gbif()
        self._process_taxon_gbif()
    
    
    def _cleanup_gbif_files(self, zip_filename):
        files_to_remove = [
            "Description.tsv", "Distribution.tsv", "Multimedia.tsv", "Reference.tsv",
            "TypesAndSpecimen.tsv", "eml.xml", "meta.xml",
            "Taxon.tsv", "VernacularName.tsv"
        ]
        self.logger.info("Suppression des fichiers non nécessaires")
        for file in files_to_remove:
            file_path = os.path.join(self.gbif_work_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)

        zip_path = os.path.join(self.gbif_work_dir, zip_filename)
        if os.path.exists(zip_path):
            os.remove(zip_path)
    
    def _format_taxon_name_gbif(self,name):
        # Supprime les espaces au début et à la fin
        name = name.strip()
        # Remplace tous les espaces par des underscores
        name = name.replace(' ', '_')
        # Ajoute le préfixe __taxon__
        name = f"__taxon__{name}"
        
        return name

    def process_gbif_backbone(self):
        
        # Créer le répertoire s'il n'existe pas
        os.makedirs(self.gbif_work_dir, exist_ok=True)
        
        taxon_file = self._taxon_gbif_file()
        vernacular_file = self._vernicular_gbif_file()

        if os.path.exists(vernacular_file) and os.path.exists(taxon_file):
            self.logger.info("Les fichiers vernacular_name.tsv et taxon.tsv existent déjà.")
            self.logger.info(f"Emplacement des fichiers : {self.gbif_work_dir}")
            return

        # Changer le répertoire de travail
        original_dir = os.getcwd()
        os.chdir(self.gbif_work_dir)

        try:
            url = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"
            zip_filename = "backbone.zip"
            zip_path = os.path.join(self.gbif_work_dir, zip_filename)
            
            if not os.path.exists(zip_path):
                self._download_file(url, zip_path)

            if not os.path.exists(os.path.join(self.gbif_work_dir, "Taxon.tsv")) or \
                not os.path.exists(os.path.join(self.gbif_work_dir, "VernacularName.tsv")):
                self._extract_zip(zip_path)

            self._process_tsv_gbif_files()

            self._cleanup_gbif_files(zip_filename)

            self.logger.info("Traitement terminé")
            self.logger.info(f"Les fichiers résultants se trouvent dans : {self.gbif_work_dir}")

        except Exception as e:
            self.logger.error(f"Une erreur s'est produite : {str(e)}")
        finally:
            os.chdir(original_dir)
    
    def manage_gbif_taxon_tags(self):
        
        self.process_gbif_backbone()

        taxon_file = self._taxon_gbif_file()
        vernacular_file = self._vernicular_gbif_file()

        # Compiler l'expression régulière si elle est fournie
        regex = re.compile(self.regex, re.IGNORECASE) if self.regex else None

        # Dictionnaire pour stocker les noms vernaculaires par taxonID
        vernacular_names = defaultdict(list)

        # Lire vernacular_name.tsv
        self.logger.info("Lecture de vernacular_name.tsv")
        with open(vernacular_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader, None)  # Ignorer l'en-tête si présent
            for row in reader:
                if len(row) >= 2:
                    taxon_id, vernacular_name = row[0], row[1]
                    vernacular_names[taxon_id].append(vernacular_name)

        # Lire taxon.tsv et créer les tags
        self.logger.info("Lecture de taxon.tsv et création des tags")
        tag_count = 0
        tags = []
        file_index = 0

        with open(taxon_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader, None)  # Ignorer l'en-tête si présent
            for row in reader:
                if len(row) >= 2:
                    taxon_id, scientific_name = row[0], row[1]
                    if regex is None or regex.search(scientific_name):
                        #Brassicaceae is a plant in the kingdom Plantae, phylum Tracheophyta, class Magnoliopsida, of the order Brassicales.
                        tag = {
                            "term": f"https://www.gbif.org/species/{taxon_id}",
                            "label": self._format_taxon_name_gbif(scientific_name),
                            "rdfs_label": scientific_name,
                            "description": ", ".join(vernacular_names.get(taxon_id, []))
                        }
                        tags.append(tag)
                        tag_count += 1

                        if tag_count % self.tags_per_file == 0:
                            df = pd.DataFrame({
                            'term': [ ele['term'] for ele in tags ],
                            'label': [ ele['label'] for ele in tags ],
                            'rdfs:label': [ ele['rdfs_label'] for ele in tags ],
                            'description': [ ele['description'] for ele in tags ]
                            })
                            
                            df.to_csv(self.retention_dir+f"/{self.tags_gbif_path_filename}_{file_index}.csv", index=False)
                            self.mem.save_pth(self.mem.encode_tags(tags),self.tags_gbif_path_filename+f"_{file_index}")
                            tags = []
                            file_index += 1
                        
                        if self.debug_nb_taxon > 0 and tag_count >= self.debug_nb_taxon:
                            break
        # Sauvegarder les abstracts restants
        if tags:
            df = pd.DataFrame({
                'term': [ ele['term'] for ele in tags ],
                'label': [ ele['label'] for ele in tags ],
                'rdfs:label': [ ele['rdfs_label'] for ele in tags ],
                'description': [ ele['description'] for ele in tags ]
                })
                
            df.to_csv(self.retention_dir+f"/{self.tags_gbif_path_filename}_{file_index}.csv", index=False)
            self.mem.save_pth(self.mem.encode_tags(tags),self.tags_gbif_path_filename+f"_{file_index}")

        self.logger.info(f"Nombre total de tags générés : {tag_count} , nombre de fichiers : {file_index+1}")
        
        return tags
    
    def get_files_tags_gbif_taxon_embeddings(self):
        matching_files = []
    
        # Compile le motif regex pour une meilleure performance
        pattern = re.compile(f"{self.tags_gbif_path_filename}.*-{self.mem.model_suffix}.pth")
        # Parcourt tous les fichiers dans le chemin donné
        for root, dirs, files in os.walk(self.retention_dir):
            for filename in files:
                if pattern.search(filename):
                    # Ajoute le chemin complet du fichier à la liste
                    matching_files.append(os.path.join(root, filename))
        
        return matching_files



#######################################################################################################
#### NCBI TAXON TAGS
#######################################################################################################
    def read_names_ncbi_dmp(self,file_path):
        result = {}
        with open(file_path, 'r') as file:
            # Utiliser csv.reader avec un délimiteur personnalisé
            reader = csv.reader(file, delimiter='|')
            
            # Lire chaque ligne du fichier
            for row in reader:
                # Nettoyer les espaces blancs au début et à la fin de chaque colonne
                cleaned_row = [col.strip() for col in row]
                if cleaned_row[3] == 'scientific name':
                    # Ajouter les deux premières colonnes à notre résultat
                    if len(cleaned_row) >= 2:
                        tax_id = cleaned_row[0]
                        result[tax_id] = {}
                        result[cleaned_row[0]]['name'] = cleaned_row[1]
    
        return result
    
    def read_node_ncbi_dmp(self,file_path,id_taxon_list_filter):
        results = id_taxon_list_filter
        with open(file_path, 'r') as file:
            # Utiliser csv.reader avec un délimiteur personnalisé
            reader = csv.reader(file, delimiter='|')
            
            # Lire chaque ligne du fichier
            for row in reader:
                # Nettoyer les espaces blancs au début et à la fin de chaque colonne
                cleaned_row = [col.strip() for col in row]
                tax_id = cleaned_row[0]
                if tax_id in results :
                    results[tax_id]['parent_tax_id'] = cleaned_row[1]
                    results[tax_id]['rank'] = cleaned_row[2]
                    # division_id check division.dmp to select (plant, mammals, bateria, ...)
                    results[tax_id]['division_id'] = cleaned_row[4]
        
        results = {k: {**v, 'parent_tax': results[v['parent_tax_id']]['name']} for k, v in results.items()}

        return results

    def read_division_ncbi_dmp(self,file_path,id_taxon_list_filter):
        results = id_taxon_list_filter
        divisions = {}

        with open(file_path, 'r') as file:
            # Utiliser csv.reader avec un délimiteur personnalisé
            reader = csv.reader(file, delimiter='|')
            
            # Lire chaque ligne du fichier
            for row in reader:
                # Nettoyer les espaces blancs au début et à la fin de chaque colonne
                cleaned_row = [col.strip() for col in row]
                divisions[cleaned_row[0]] = cleaned_row[2]
        
        
        results = {k: {**v, 'division': divisions[v['division_id']]} for k, v in results.items()}

        return results

    def _process_ncbi_compile(self):
        output_file = os.path.join(self.ncbi_work_dir, "ncbi_compile.tsv")
        if not os.path.exists(output_file):
            
            id_taxon_list = self.read_names_ncbi_dmp(self.ncbi_work_dir+"/names.dmp")
            id_taxon_list = self.read_node_ncbi_dmp(self.ncbi_work_dir+"/nodes.dmp",id_taxon_list)
            id_taxon_list = self.read_division_ncbi_dmp(self.ncbi_work_dir+"/division.dmp",id_taxon_list)

            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                tsv_writer = csv.writer(outfile, delimiter='\t')
                for id,row in id_taxon_list.items():
                    tsv_writer.writerow([id,row['name'], row['parent_tax'], row['rank'],row['division']])
    
    def _ncbi_compile_file(self):
        if 'ncbi_compile_debug' in self.config:
            ncbi_file = os.path.join(self.ncbi_work_dir, self.config['ncbi_compile_debug'])
        else:
            ncbi_file = os.path.join(self.ncbi_work_dir, 'ncbi_compile.tsv')
        
        results = {}

        with open(ncbi_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader, None)  # Ignorer l'en-tête si présent
            for row in reader:
                    taxon_id, name, parent_tax, rank, division = int(row[0]), row[1], row[2], row[3], row[4]
                    results[taxon_id]= {
                        'name' : name,
                        'parent_tax' : parent_tax,
                        'rank' : rank,
                        'division' : division
                    }

        return results

    def process_ncbi(self):
        os.makedirs(self.ncbi_work_dir, exist_ok=True)
        
        # Changer le répertoire de travail
        original_dir = os.getcwd()
        #os.chdir(self.ncbi_work_dir)

        try:
            url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
            zip_filename = "taxdump.tar.gz"
            zip_path = os.path.join(self.ncbi_work_dir, zip_filename)
    
            if not os.path.exists(zip_path):
                self._download_file(url, zip_path)

            if not os.path.exists(os.path.join(self.ncbi_work_dir, "names.dmp")):
                self._extract_tar_gz(zip_path)
            
            self._process_ncbi_compile()
        
            self.logger.info("Traitement terminé")
            self.logger.info(f"Les fichiers résultants se trouvent dans : {self.ncbi_work_dir}")

        except Exception as e:
            self.logger.error(f"Une erreur s'est produite : {str(e)}")
        finally:
            os.chdir(original_dir)

    def manage_ncbi_taxon_tags(self):
        print("manage_ncbi_taxon_tags")
        self.process_ncbi()
        dict_ncbi = self._ncbi_compile_file()
        print(len(dict_ncbi))
        print(dict_ncbi[3707])
        
        # Compiler l'expression régulière si elle est fournie
        regex = re.compile(self.regex, re.IGNORECASE) if self.regex else None

        tag_count=0
        tags = []
        file_index = 0

        for taxon_id, taxon_info in dict_ncbi.items():
            if regex is not None and not regex.search(taxon_info['name']):
                continue

            tag = {
                "term": f"http://purl.obolibrary.org/obo/NCBITaxon_{taxon_id}",
                "label": self._format_taxon_name_gbif(taxon_info['name']),
                "rdfs_label": taxon_info['name'],
                "description": f"{taxon_info['name']} is a {taxon_info['rank']} whose direct parent taxon is {taxon_info['parent_tax']} and the division is {taxon_info['division']}."
            }
            tags.append(tag)
            tag_count += 1

            if tag_count % self.tags_per_file == 0:
                df = pd.DataFrame({
                'term': [ ele['term'] for ele in tags ],
                'label': [ ele['label'] for ele in tags ],
                'rdfs:label': [ ele['rdfs_label'] for ele in tags ],
                'description': [ ele['description'] for ele in tags ]
                })
                
                df.to_csv(self.retention_dir+f"/{self.tags_ncbi_path_filename}_{file_index}.csv", index=False)
                self.mem.save_pth(self.mem.encode_tags(tags),self.tags_ncbi_path_filename+f"_{file_index}")
                tags = []
                file_index += 1
            
            if self.debug_nb_taxon > 0 and tag_count >= self.debug_nb_taxon:
                break
# Sauvegarder les abstracts restants
        if tags:
            df = pd.DataFrame({
                'term': [ ele['term'] for ele in tags ],
                'label': [ ele['label'] for ele in tags ],
                'rdfs:label': [ ele['rdfs_label'] for ele in tags ],
                'description': [ ele['description'] for ele in tags ]
                })
                
            df.to_csv(self.retention_dir+f"/{self.tags_ncbi_path_filename}_{file_index}.csv", index=False)
            self.mem.save_pth(self.mem.encode_tags(tags),self.tags_ncbi_path_filename+f"_{file_index}")

        self.logger.info(f"Nombre total de tags générés : {tag_count} , nombre de fichiers : {file_index+1}")
        
        return tags
    
    def get_files_tags_ncbi_taxon_embeddings(self):
        matching_files = []
    
        # Compile le motif regex pour une meilleure performance
        pattern = re.compile(f"{self.tags_ncbi_path_filename}.*-{self.mem.model_suffix}.pth")
        # Parcourt tous les fichiers dans le chemin donné
        for root, dirs, files in os.walk(self.retention_dir):
            for filename in files:
                if pattern.search(filename):
                    # Ajoute le chemin complet du fichier à la liste
                    matching_files.append(os.path.join(root, filename))
        
        return matching_files
        
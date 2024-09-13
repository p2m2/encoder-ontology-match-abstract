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
        self.work_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gbif-backbone')
        self.regex = config.get('regex',None)
        self.debug_nb_taxon = config.get('debug_nb_taxon',-1)
        self.tags_tag_path_filename = f"tags_taxon"
        self.mem = model_embedding_manager
    
    def taxon_file(self):
        if 'taxon_tsv_debug' in self.config:
            taxon_file = os.path.join(self.work_dir, self.config['taxon_tsv_debug'])
        else:
            taxon_file = os.path.join(self.work_dir, 'taxon.tsv')
        
        return taxon_file

    def vernicular_file(self):
        if 'vernicular_tsv_debug' in self.config:
            vernacular_file = os.path.join(self.work_dir, self.config['vernicular_tsv_debug'])
        else:
            vernacular_file = os.path.join(self.work_dir, 'vernacular_name.tsv')
        
        return vernacular_file

    def process_gbif_backbone(self):
        
        # Créer le répertoire s'il n'existe pas
        os.makedirs(self.work_dir, exist_ok=True)
        
        taxon_file = self.taxon_file()
        vernacular_file = self.vernicular_file()

        if os.path.exists(vernacular_file) and os.path.exists(taxon_file):
            self.logger.info("Les fichiers vernacular_name.tsv et taxon.tsv existent déjà.")
            self.logger.info(f"Emplacement des fichiers : {self.work_dir}")
            return

        # Changer le répertoire de travail
        original_dir = os.getcwd()
        os.chdir(self.work_dir)

        try:
            url = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"
            zip_filename = "backbone.zip"
            zip_path = os.path.join(self.work_dir, zip_filename)

            if not os.path.exists(zip_path):
                self._download_file(url, zip_path)

            if not os.path.exists(os.path.join(self.work_dir, "Taxon.tsv")) or \
                not os.path.exists(os.path.join(self.work_dir, "VernacularName.tsv")):
                self._extract_zip(zip_path)

            self._process_tsv_files()

            self._cleanup_files(zip_filename)

            self.logger.info("Traitement terminé")
            self.logger.info(f"Les fichiers résultants se trouvent dans : {self.work_dir}")

        except Exception as e:
            self.logger.error(f"Une erreur s'est produite : {str(e)}")
        finally:
            os.chdir(original_dir)

    def _download_file(self, url, filename):
        self.logger.info(f"Téléchargement de {url}")
        try:
            response = requests.get(url)
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

    def _process_tsv_files(self):
        self.logger.info("Traitement des fichiers TSV")
        self._process_vernacular_name()
        self._process_taxon()

    def _process_vernacular_name(self):
        output_file = os.path.join(self.work_dir, "vernacular_name.tsv")
        if not os.path.exists(output_file):
            with open('VernacularName.tsv', 'r', newline='', encoding='utf-8') as infile, \
                open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                tsv_reader = csv.reader(infile, delimiter='\t')
                tsv_writer = csv.writer(outfile, delimiter='\t')
                for row in tsv_reader:
                    if len(row) >= 2:
                        tsv_writer.writerow([row[0], row[1]])
    def _process_taxon(self):
        output_file = os.path.join(self.work_dir, "taxon.tsv")
        if not os.path.exists(output_file):
            with open('Taxon.tsv', 'r', newline='', encoding='utf-8') as infile, \
                open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                tsv_reader = csv.reader(infile, delimiter='\t')
                tsv_writer = csv.writer(outfile, delimiter='\t')
                for row in tsv_reader:
                    if len(row) >= 6:
                        tsv_writer.writerow([row[0], row[5]])

    def _cleanup_files(self, zip_filename):
        files_to_remove = [
            "Description.tsv", "Distribution.tsv", "Multimedia.tsv", "Reference.tsv",
            "TypesAndSpecimen.tsv", "eml.xml", "meta.xml",
            #"Taxon.tsv", "VernacularName.tsv"
        ]
        self.logger.info("Suppression des fichiers non nécessaires")
        for file in files_to_remove:
            file_path = os.path.join(self.work_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)

        zip_path = os.path.join(self.work_dir, zip_filename)
        if os.path.exists(zip_path):
            os.remove(zip_path)

    def _format_taxon_name(self,name):
        # Supprime les espaces au début et à la fin
        name = name.strip()
        # Remplace tous les espaces par des underscores
        name = name.replace(' ', '_')
        # Ajoute le préfixe __taxon__
        name = f"__taxon__{name}"
        
        return name

    def manage_gbif_taxon_tags(self):
        
        self.process_gbif_backbone()

        taxon_file = self.taxon_file()
        vernacular_file = self.vernicular_file()

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
                        tag = {
                            "term": f"https://www.gbif.org/species/{taxon_id}",
                            "label": self._format_taxon_name(scientific_name),
                            "rdfs_label": scientific_name,
                            "description": ", ".join(vernacular_names.get(taxon_id, []))
                        }
                        tags.append(tag)
                        tag_count += 1

                        if tag_count % self.tags_per_file == 0:
                            df = pd.DataFrame({
                            'label': [ ele['label'] for ele in tags ],
                            'rdfs:label': [ ele['rdfs_label'] for ele in tags ],
                            'description': [ ele['description'] for ele in tags ]
                            })
                            
                            df.to_csv(self.retention_dir+f"/{self.tags_tag_path_filename}_{file_index}.csv", index=False)
                            self.mem.save_pth(self.mem.encode_tags(tags),self.tags_tag_path_filename+f"_{file_index}")
                            tags = []
                            file_index += 1
                        
                        if self.debug_nb_taxon > 0 and tag_count >= self.debug_nb_taxon:
                            break
        # Sauvegarder les abstracts restants
        if tags:
            df = pd.DataFrame({
                'label': [ ele['label'] for ele in tags ],
                'rdfs:label': [ ele['rdfs_label'] for ele in tags ],
                'description': [ ele['description'] for ele in tags ]
                })
                
            df.to_csv(self.retention_dir+f"/{self.tags_tag_path_filename}_{file_index}.csv", index=False)
            self.mem.save_pth(self.mem.encode_tags(tags),self.tags_tag_path_filename+f"_{file_index}")

        self.logger.info(f"Nombre total de tags générés : {tag_count} , nombre de fichiers : {file_index+1}")
        
        #df = pd.DataFrame({
        #'label': [ ele['label'] for ele in tags ],
        #'rdfs:label': [ ele['rdfs_label'] for ele in tags ],
        #'description': [ ele['description'] for ele in tags ]
        #})
        #df.to_csv(self.retention_dir+"/tags_gbif_taxon.csv", index=False)
        
        return tags
    
    def get_files_tags_taxon_embeddings(self):
        matching_files = []
    
        # Compile le motif regex pour une meilleure performance
        pattern = re.compile(f"{self.tags_tag_path_filename}.*-{self.mem.model_suffix}.pth")
        # Parcourt tous les fichiers dans le chemin donné
        for root, dirs, files in os.walk(self.retention_dir):
            for filename in files:
                if pattern.search(filename):
                    # Ajoute le chemin complet du fichier à la liste
                    matching_files.append(os.path.join(root, filename))
        
        return matching_files


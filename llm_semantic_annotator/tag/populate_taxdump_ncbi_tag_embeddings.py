import requests,tarfile,os, torch,csv
from urllib.parse import urlparse
from tqdm import tqdm
from rich import print
from llm_semantic_annotator import list_of_dicts_to_csv,save_results,load_results
from llm_semantic_annotator import ModelEmbeddingManagement

class TaxonTagManagement:
    def __init__(self,config):
        print(config)
        self.retention_dir = config['retention_dir']
        self.embeddings_file_name = self.retention_dir+"/tags-taxon.pth"
        self.filename_gbif_csv = self.retention_dir+'/taxdumpgbif-table.csv'
        self.filename_ncbi_csv = self.retention_dir+'/taxdump-table.csv'

        if 'force' not in config:
            config['force'] = False
        else:
            self.force = config['force']

    def download_and_extract(self,url, extract_to='.'):
        # Télécharger le fichier
        print(f"Téléchargement de {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lève une exception pour les erreurs HTTP

        # Obtenir le nom du fichier à partir de l'URL
        filename = os.path.basename(urlparse(url).path)
        
        # Enregistrer le fichier téléchargé
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Téléchargement terminé: {filename}")

        # Extraire le contenu du fichier tar.gz
        print(f"Extraction de {filename}")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print(f"Extraction terminée dans {extract_to}")

        # Optionnel : supprimer le fichier tar.gz après extraction
        os.remove(filename)
        print(f"Fichier {filename} supprimé")

    def read_names_dmp(self,file_path):
        result = []
        with open(file_path, 'r') as file:
            # Utiliser csv.reader avec un délimiteur personnalisé
            reader = csv.reader(file, delimiter='|')
            
            # Lire chaque ligne du fichier
            for row in reader:
                # Nettoyer les espaces blancs au début et à la fin de chaque colonne
                cleaned_row = [col.strip() for col in row]
                
                # Ajouter les deux premières colonnes à notre résultat
                if len(cleaned_row) >= 2:
                    result.append(cleaned_row[:2])
        
        return result

    def format_taxon_name(self,name):
        # Supprime les espaces au début et à la fin
        name = name.strip()
        
        # Remplace tous les espaces par des underscores
        name = name.replace(' ', '_')
        
        # Ajoute le préfixe __taxon__
        name = f"__taxon__{name}"
        
        return name

    def manage_ncbi_taxon_tags(self,config):

        if not os.path.exists(self.filename_ncbi_csv): 
            print(f"{self.filename_ncbi_csv} does not exist !")
            # URL du fichier à télécharger (exemple avec la taxonomie NCBI)
            url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"

            # Appel de la fonction
            self.download_and_extract(url,self.retention_dir)
            table = self.read_names_dmp(self.retention_dir+'/names.dmp')
            save_results(table, self.filename_ncbi_csv)
        else:
            table = load_results(self.filename_ncbi_csv)
        
        tags = []
        tag_embeddings = {}
        
        if os.path.exists(self.embeddings_file_name):
            print("load tags embeddings")
            tag_embeddings = torch.load(self.embeddings_file_name)
        
        change = False

        filter_debug = ['onema','auliflower' , 'ustar' ,'arabaido' , 'moustar' , 'assicac']

        for row in tqdm(table):
            label_idx = self.format_taxon_name(row[1])

            if label_idx in tag_embeddings:
                continue
            
            if any([ elt in label_idx for elt in filter_debug]):
                item = {
                        'term': f"http://purl.obolibrary.org/obo/NCBITaxon_{row[0]}",
                        'label': label_idx,
                        'rdfs_label': row[1],
                        'description' : ''
                    }
                change = True
                tags.append(item)
        
        tag_embeddings = ModelEmbeddingManagement().encode_tags(tags)
        torch.save(tag_embeddings, self.embeddings_file_name)

        return tag_embeddings


    def manage_gbif_taxon_tags(self,config):
        from pygbif import species
        
        tags = []
        if not os.path.exists(self.filename_gbif_csv): 
            print(f"{self.filename_gbif_csv} does not exist !")
            offset = 0
            block_iter = 1000
            r = species.name_lookup(kingdom='plants', limit=1,offset=0)
            print(r['count'])
            for offset in tqdm(range(0, r['count'], block_iter)):
                r = species.name_lookup(kingdom='plants', limit=block_iter,offset=offset)
                print(r['count'],offset)
                for item in r['results']:
                    
                    label_idx = self.format_taxon_name(item['scientificName'])
                    descriptions = [item['canonicalName']]
                    for elt in item['vernacularNames']:
                        if 'language' not in elt or elt['language'] == 'en':
                            descriptions.append(elt['vernacularName'])
                    #print(item['descriptions'])
                    descriptions.extend([e['description'] for e in item['descriptions']])

                    elt = {
                        'term': f"https://www.gbif.org/species/{item['key']}",
                        'label' : label_idx,
                        'rdfs_label': item['scientificName'],
                        'description' : ' - '.join(descriptions)
                    }
                    tags.append(elt)

            save_results(tags, self.filename_gbif_csv)
        else:
            tags = load_results(self.filename_gbif_csv)
        
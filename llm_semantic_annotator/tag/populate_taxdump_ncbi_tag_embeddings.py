import requests,tarfile,os, torch,csv
from urllib.parse import urlparse
from tqdm import tqdm
from llm_semantic_annotator import list_of_dicts_to_csv,save_results,load_results
from llm_semantic_annotator import encode_text

def download_and_extract(url, extract_to='.'):
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

def read_names_dmp(file_path):
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

def format_taxon_name(name):
    # Supprime les espaces au début et à la fin
    name = name.strip()
    
    # Remplace tous les espaces par des underscores
    name = name.replace(' ', '_')
    
    # Ajoute le préfixe __taxon__
    name = f"__taxon__{name}"
    
    return name

def manage_ncbi_taxon_tags(config):
    retention_dir = config['retention_dir']
    if 'force' not in config:
        config['force'] = False
    
    filename_csv = retention_dir+f'/taxdump-table.csv'
    if not os.path.exists(filename_csv): 
        print(f"{filename_csv} does not exist !")
        # URL du fichier à télécharger (exemple avec la taxonomie NCBI)
        url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"

        # Appel de la fonction
        download_and_extract(url,retention_dir)
        table = read_names_dmp(retention_dir+'/names.dmp')
        save_results(table, filename_csv)
    else:
        table = load_results(filename_csv)
    
    tags = []
    tag_embeddings = {}
    
    if os.path.exists(retention_dir+'/tags.pth'):
        print("load tags embeddings")
        tag_embeddings = torch.load(retention_dir+'/tags.pth')
    
    change = False

    for row in tqdm(table):
        item = {
                'term': f"http://purl.obolibrary.org/obo/NCBITaxon_{row[0]}",
                'label': format_taxon_name(row[1]),
                'rdfs_label': row[1],
                'description' : ''
            }
        tags.append(item)
        embeddings = encode_text(f"{item['rdfs_label']}")
        tag_embeddings[item['label']] = embeddings

    if change:
        print("save tags embeddings")
        torch.save(tag_embeddings, retention_dir+'/tags.pth')
        save_results(tags, retention_dir+f'/tags-taxdump-table.csv')

    return tag_embeddings
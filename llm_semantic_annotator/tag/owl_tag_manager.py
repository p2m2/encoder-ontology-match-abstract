import os, re, warnings, torch
from rdflib import Graph, Namespace, URIRef
from tqdm import tqdm
from rich import print
import wget
from llm_semantic_annotator import list_of_dicts_to_csv,save_results,load_results
from llm_semantic_annotator import ModelEmbeddingManager
import pandas as pd

class OwlTagManager:
    def __init__(self,config,model_embedding_manager):
        
        self.config=config
        if 'debug_nb_terms_by_ontology' in config:
            self.debug_nb_terms_by_ontology = config['debug_nb_terms_by_ontology']
        else:
            self.debug_nb_terms_by_ontology = -1

        self.retention_dir = config['retention_dir']
        self.ontologies_by_link = config['ontologies']

        if 'force' not in config:
            config['force'] = False
        else:
            self.force = config['force']
        
        self.mem = model_embedding_manager
        self.tags_owl_path_filename = f"tags_owl_"

    def get_corpus(self,ontologies):
        for ontology in self.get_ontologies(ontologies):
            self.build_corpus(ontology, ontologies[ontology],self.debug_nb_terms_by_ontology)
            

    # Charger le fichier OWL local
    def get_ontologies(self,list_ontologies):
        
        for ontology,values in list_ontologies.items():
            
            filepath = self.retention_dir+"/"+ontology+"."+values['format']
                    
            # utilisation d'un fichier local
            if 'filepath' in values:
                if not os.path.exists(values['filepath']):
                    raise FileNotFoundError(f"Le fichier '{values['filepath']}' n'existe pas.")
                if os.path.exists(filepath) and self.force:
                    os.remove(filepath)
                if os.path.exists(filepath):
                    continue
                try:
                    os.symlink(os.path.abspath(values['filepath']),filepath)
                    print(f"Lien symbolique créé : {filepath} -> {values['filepath']}")
                except FileExistsError:
                    print(f"Le lien symbolique {filepath} existe déjà.")
                except PermissionError:
                    print("Erreur de permission. Assurez-vous d'avoir les droits nécessaires.")
                except OSError as e:
                    print(f"Erreur lors de la création du lien symbolique : {e}")

            else: # sinon telechargement
                if self.force or not os.path.exists(filepath):
                    print("Downloading ontology: ",ontology)
                    wget.download(values['url'],filepath)
            
            list_ontologies[ontology]['filepath'] = filepath

            
        return list_ontologies

    def remove_prefix_tags(self,prefix_tag,text):
        escaped_prefix = re.escape(prefix_tag.upper())
        pattern = r'\([A-Z]+:\d+\)'

        v = re.sub(pattern, '', text)
        return re.sub(r'\(\)', '', v)

    def build_corpus(
            self,
            ontology, 
            ontology_config,
            debug_nb_terms_by_ontology):

        tags_owl_path_filename = self.tags_owl_path_filename+ontology
        tag_embeddings = self.mem.load_pth(tags_owl_path_filename)

        if (len(tag_embeddings)>0):
            return tag_embeddings

        # Charger le fichier OWL local

        g = Graph()
        g.parse(ontology_config['filepath'], format=ontology_config['format'])

        # Namespace pour rdfs
        RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

        if 'properties' not in ontology_config or len(ontology_config['properties'])<=0:
            warnings.warn("'properties' is not defined ["+ontology+"]", UserWarning)
            return []

        if 'label' not in ontology_config:
            ontology_config['label'] = "<http://www.w3.org/2000/01/rdf-schema#label>"

        if len(ontology_config['properties'])<1:
            raise ValueError(f"OWL TAG : At least one property is required :{ontology_config['properties']}")

        if len(ontology_config['properties'])>1:
            raise ValueError(f"OWL TAG : Only one property is supported :{ontology_config['properties']}")

        query_base = """
        SELECT ?term ?labelLeaf ?prop0 WHERE { 
            ?term """+ontology_config['label']+""" ?labelLeaf .
            FILTER(LANG(?labelLeaf) = "en" || LANG(?labelLeaf) = "") .
            ?term """+ontology_config['properties'][0]+""" ?prop0 .
            FILTER(LANG(?prop0) = "en" || LANG(?prop0) = "") .
        }
        """
        print(query_base)
        tags = []

        # Exécuter la requête SPARQL
        results = g.query(query_base)
        nb_record=0
        print(f"Ontology {ontology} NB RECORDS:{len(results)}")
        for row in tqdm(results):

            descriptionLeaf = '\n'.join([ row.get(prop.replace('?',''), '') for prop in ["?prop0"] ])
            labelLeaf = row.labelLeaf
            
            formatted_label = "__"+ontology+"__" + str(labelLeaf.lower()).replace(" ", "_")
            
            if "obsolete" in formatted_label:
                continue
            if 'obsolete' in descriptionLeaf.lower():
                continue
            
            tags.append({
                    'term': row.term,
                    'label': formatted_label,
                    'rdfs_label': labelLeaf,
                    'description' : self.remove_prefix_tags(ontology,descriptionLeaf)
                })
            
            if nb_record == debug_nb_terms_by_ontology:
                break
            nb_record+=1

        df = pd.DataFrame({
        'label': [ ele['label'] for ele in tags ],
        'rdfs:label': [ ele['rdfs_label'] for ele in tags ],
        'description': [ ele['description'] for ele in tags ]
        })
        
        df.to_csv(self.retention_dir+f"/tags_owl_{ontology}.csv", index=False)
        self.mem.save_pth(self.mem.encode_tags(tags),tags_owl_path_filename)
        return tags

    def manage_tags(self):
        for link_name,ontologies in self.ontologies_by_link.items():
            # get vocabulary from ontologies selected
            self.get_corpus(ontologies)

    # Return tag embeddings in JSON format where the key is the label and the value is the embedding
    def get_files_tags_embeddings(self):
        matching_files = []
    
        # Compile le motif regex pour une meilleure performance
        pattern = re.compile(self.tags_owl_path_filename+f".*-{self.mem.model_suffix}.pth")
        # Parcourt tous les fichiers dans le chemin donné
        for root, dirs, files in os.walk(self.retention_dir):
            for filename in files:
                if pattern.search(filename):
                    # Ajoute le chemin complet du fichier à la liste
                    matching_files.append(os.path.join(root, filename))
        
        return matching_files

    # Return a tags list where element is an object containing the term, label and description 
    def get_tags(self):
        for filename in os.listdir(self.retention_dir):
            results = []
            if filename.startswith('tag_') and filename.endswith('.json'):
                results.append(load_results(os.path.join(self.retention_dir, filename)))
        # Remove duplicates
        return [dict(t) for t in {tuple(d.items()) for d in results}]
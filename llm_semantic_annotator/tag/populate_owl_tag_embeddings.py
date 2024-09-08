import os, re, warnings, torch
from rdflib import Graph, Namespace, URIRef
from tqdm import tqdm
from rich import print
import wget
from llm_semantic_annotator import list_of_dicts_to_csv,save_results,load_results
from llm_semantic_annotator import ModelEmbeddingManagement

class OwlTagManagement:
    def __init__(self,config):
        
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

    def get_corpus(self,ontologies):
    
        tags = []
        for ontology in self.get_ontologies(ontologies):
            filename = self.retention_dir+f"/tag_{ontology}.json"

            if not self.force and os.path.exists(filename):
                tags.extend(load_results(filename))
            else:
                tags_ontology = self.build_corpus(ontology, ontologies[ontology],
                self.debug_nb_terms_by_ontology)
                print(f"save results on {filename} with length:{len(tags_ontology)}")
                if len(tags_ontology) == 0:
                    warnings(f"** No Tags found  ** ")
                save_results(tags_ontology,filename)
                tags.extend(tags_ontology)
        
        return tags

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
        pattern = rf'{escaped_prefix}:\d+'

        v = re.sub(pattern, '', text)
        return re.sub(r'\(\)', '', v)

    def build_corpus(
            self,
            ontology, 
            ontology_config,
            debug_nb_terms_by_ontology):
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

        varProperties = []
        sparqlProperties = []

        for i,prop in enumerate(ontology_config['properties']):
            varProperties.append("?prop"+str(i))
            sparqlProperties.append("?term "+prop+" ?prop"+str(i)+" .")
        
        query_base = """
        SELECT ?term ?labelLeaf """ + " ".join(varProperties) + """ WHERE { 
            """+"\n".join(sparqlProperties)+"""
            ?term """+ontology_config['label']+""" ?labelLeaf .
        }
        """
        print(query_base)
        tags = []

        # Exécuter la requête SPARQL
        results = g.query(query_base)
        nb_record=0
        print(f"Ontology {ontology} NB RECORDS:{len(results)}")
        for row in tqdm(results):

            descriptionLeaf = '\n'.join([ row.get(prop.replace('?',''), '') for prop in varProperties ])
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
        
        return tags

    def manage_tags(self):
        mem = ModelEmbeddingManagement(self.config)
        if self.force:
            tag_embeddings = {}
        else:
            tag_embeddings = mem.load_pth("tags-owl")

        if len(tag_embeddings)==0:
            tags = []
            for link_name,ontologies in self.ontologies_by_link.items():
                # get vocabulary from ontologies selected
                tags.extend(self.get_corpus(ontologies))
            
            tag_embeddings = mem.encode_tags(tags)
            mem.save_pth(tag_embeddings,"tags-owl")


    # Return tag embeddings in JSON format where the key is the DOI and the value is the embedding
    def get_tags_embeddings(self):
        return ModelEmbeddingManagement(self.config).load_pth("tags-owl")

    # Return a tags list where element is an object containing the term, label and description 
    def get_tags(self):
        for filename in os.listdir(self.retention_dir):
            results = []
            if filename.startswith('tag_') and filename.endswith('.json'):
                results.append(load_results(os.path.join(self.retention_dir, filename)))
        # Remove duplicates
        return [dict(t) for t in {tuple(d.items()) for d in results}]
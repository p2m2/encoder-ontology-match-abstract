import unittest,shutil,tempfile,os,json
from llm_semantic_annotator import get_ncbi_abstracts

from llm_semantic_annotator import main_populate_owl_tag_embeddings
from llm_semantic_annotator import main_populate_ncbi_abstract_embeddings
from llm_semantic_annotator import main_compute_tag_chunk_similarities




class TestAbstractPreparation(unittest.TestCase):
    def setUp(self):
        self.config = {
            "populate_owl_tag_embeddings" : {
                "ontologies": {
                    "animal_link" : {
                        "fake_animal" : {
                            "filepath" : "./tests/data/animals.owl",
                            "prefix": "http://www.example.org/animals#",
                            "format": "xml",
                            "label" : "<http://www.w3.org/2000/01/rdf-schema#label>",
                            "properties": ["<http://www.w3.org/2000/01/rdf-schema#comment>"]
                        }
                    }
                },
                "debug_nb_terms_by_ontology" : 1
            },
            "populate_ncbi_abstract_embeddings" : {
                "debug_nb_ncbi_request" : 1,
                "debug_nb_abstracts_by_search" : 1,
                "retmax":5,
                "selected_term" : [
                    "metabolomics+AND+jungle",
                ]
            },
            "compute_tag_chunk_similarities" : {
                "threshold_similarity_tag_chunk" : 0.25,
                "debug_nb_similarity_compute" : 100
            }
        }

        self.temp_dir = tempfile.mkdtemp()
        self.config['retention_dir'] = self.temp_dir
        self.config['force'] = True
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_main(self):
        main_populate_owl_tag_embeddings(self.config)
        main_populate_ncbi_abstract_embeddings(self.config)
        main_compute_tag_chunk_similarities(self.config)
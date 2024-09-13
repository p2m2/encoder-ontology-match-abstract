import unittest,shutil,tempfile,os,json

from llm_semantic_annotator import main_populate_owl_tag_embeddings
from llm_semantic_annotator import main_populate_abstract_embeddings
from llm_semantic_annotator import main_populate_gbif_taxon_tag_embeddings
from llm_semantic_annotator import main_compute_tag_chunk_similarities





class TestAbstractPreparation(unittest.TestCase):
    def setUp(self):
        self.config = {
            "encodeur" : "sentence-transformers/all-MiniLM-L6-v2",
            "threshold_similarity_tag_chunk" : 0.65,
            "threshold_similarity_tag" : 0.75,
            "batch_size" : 32,

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
            "populate_gbif_taxon_tag_embeddings" : {
                "debug_nb_taxon" : 1,
                "regex" : "assic.*",
                "taxon_tsv_debug" : "taxon_some.tsv",
                "vernicular_tsv_debug" : "vernacular_name_some.tsv"
            },
            "populate_abstract_embeddings" : {
                "from_file" : {
                    "json_files" : [
                        "data/abstracts/abstracts_1.json",
                        "data/abstracts/abstracts_2.json"
                    ],
            "text_files" : [
                        "data/abstracts/abstracts_3.txt",
                        "data/abstracts/abstracts_4.txt"
                    ]
                }
                
            }
        }

        self.temp_dir = tempfile.mkdtemp()
        self.config['retention_dir'] = self.temp_dir
        self.config['force'] = True
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_main(self):
        main_populate_owl_tag_embeddings(self.config)
        main_populate_abstract_embeddings(self.config)
        main_populate_gbif_taxon_tag_embeddings(self.config)
        main_compute_tag_chunk_similarities(self.config)
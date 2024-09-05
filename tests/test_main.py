import unittest,shutil,tempfile,os,json
from llm_semantic_annotator import get_ncbi_abstracts

from llm_semantic_annotator import main_populate_owl_tag_embeddings
from llm_semantic_annotator import main_populate_ncbi_abstract_embeddings
from llm_semantic_annotator import main_compute_tag_chunk_similarities




class TestAbstractPreparation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        file_path = os.path.join(project_root, "config/test.json")

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.config = json.load(f)
                self.config['retention_dir'] = self.temp_dir
        else:
            self.fail(f"Can not load config file ${file_path}")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_main(self):
        main_populate_owl_tag_embeddings(self.config)
        main_populate_ncbi_abstract_embeddings(self.config)
        main_compute_tag_chunk_similarities(self.config)
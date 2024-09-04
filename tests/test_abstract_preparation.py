import unittest,shutil,tempfile
from llm_semantic_annotator import get_ncbi_abstracts

class TestAbstractPreparation(unittest.TestCase):
    def setUp(self):
        # Créer un répertoire temporaire
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Supprimer le répertoire temporaire et son contenu
        shutil.rmtree(self.temp_dir)

    def test_get_ncbi_abstracts_idlist_empty(self):
        n = 5
        config = {
            'debug_nb_ncbi_request': 1,
            'debug_nb_abstracts_by_search' : 1,
            'retmax':n,
            'retention_dir': self.temp_dir,
        }
        # "" give an idList empty
        resultat = get_ncbi_abstracts("", config)
        self.assertEqual(len(resultat), 0)

    def test_get_ncbi_abstracts(self):
        n = 5
        config = {
            'debug_nb_ncbi_request': 1,
            'debug_nb_abstracts_by_search' : 1,
            'retmax':n,
            'retention_dir': self.temp_dir,
        }

        resultat = get_ncbi_abstracts("gluco", config)
        self.assertEqual(len(resultat), n)

if __name__ == '__main__':
    unittest.main()
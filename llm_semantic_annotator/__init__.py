from .misc.utils import list_of_dicts_to_csv,dict_to_csv
from .misc.utils import save_results,load_results,get_retention_dir
from .misc.stats import display_ontologies_distribution
from .misc.console import display_best_similarity_abstract_tag
from .misc.console import display_ontologies_summary
from .misc.scientific_abstract_rdf_annotator import create_rdf_graph,save_rdf_graph
from .misc.scientific_abstract_rdf_annotator import save_rdf_graph

from .similarity.model_embedding_manager import ModelEmbeddingManager
from .tag.owl_tag_manager import OwlTagManager
from .tag.taxon_tag_manager import TaxonTagManager
from .abstract.abstract_manager import AbstractManager


from .core import main_populate_owl_tag_embeddings
from .core import main_populate_abstract_embeddings

from .core import main_populate_gbif_taxon_tag_embeddings
from .core import main_populate_ncbi_taxon_tag_embeddings

from .core import main_compute_tag_chunk_similarities
from .core import main_display_summary
from .core import main_build_graph
from .core import main_build_dataset_abstracts_annotation
from .core import get_scores_files

from .similarity_evaluator import similarity_evaluator_main


from colorama import init, Fore, Back, Style
# Initialiser colorama
init(autoreset=True)

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    # Formater le message d'exception
    error_msg = f"{exc_type.__name__}: {exc_value}"
    
    # Afficher le message en rouge
    print(f"{Fore.RED}{Back.WHITE}{Style.BRIGHT}{error_msg}{Style.RESET_ALL}")
    
    # Afficher la traceback en jaune
    import traceback
    for line in traceback.format_tb(exc_traceback):
        print(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")

# Remplacer le gestionnaire d'exceptions par défaut
import sys
sys.excepthook = custom_exception_handler


# LLMSemanticAnnotator

```bash
pip install -r requirements.txt
```

### Configuratipon

    "threshold_similarity_tag_chunk" : Seuil de similarité pour le tagging des chunks.
    "debug_nb_ncbi_request" : Nombre de requêtes NCBI pour le débogage.
    "debug_nb_terms_by_ontology" : Nombre de termes par ontologie pour le débogage.
    "debug_nb_abstracts_by_search" : Nombre d'abstracts par recherche pour le débogage.


### Execution

```bash
python -m llm_semantic_annotator config/test.json populate_tag_embeddings
```

```bash
python -m llm_semantic_annotator config/test.json populate_ncbi_abstract_embeddings --selected_term "abiotic+AND+metabolomics+AND+plant+AND+stress+AND+brassicaceae"
```

```bash
python -m llm_semantic_annotator config/test.json compute_tag_chunk_similarities
```




# LLMSemanticAnnotator

[![Tests](https://github.com/p2m2/encoder-ontology-match-abstract/actions/workflows/ci.yml/badge.svg)](https://github.com/p2m2/encoder-ontology-match-abstract/actions/workflows/ci.yml)

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Configuratipon

| Action                          | Description                                       | Parameter                     | Description                                                |
|---------------------------------|---------------------------------------------------|-------------------------------|------------------------------------------------------------|
| populate_tag_embeddings         | Populate the corpus of tags                       |                               |                                                            |
|                                 |                                                   | debug_nb_terms_by_ontology    | Number of terms per ontology for debugging.                |
| populate_ncbi_abstract_embeddings | Populate the database with scientific articles    |                               |                                                            |
|                                 |                                                   | debug_nb_ncbi_request         | Number of NCBI requests for debugging.                     |
|                                 |                                                   | debug_nb_abstracts_by_search  | Number of abstracts per search for debugging.              |
|                                 |                                                   | retmax                        | Maximum number of articles per search term.                |
|                                 |                                                   | selected_term                 | Selected term to find relevant scientific articles.        |
| compute_tag_chunk_similarities  | Compute the similarity using cosine score         |                               |                                                            |
|                                 |                                                   | threshold_similarity_tag_chunk | Similarity threshold for tagging chunks.                   |
|                                 |                                                   | debug_nb_similarity_compute   | Maximum number of similarities computed.                   |


    
    
    



### Execution

```bash
python -m llm_semantic_annotator config/test.json populate_tag_embeddings
```

```bash
python -m llm_semantic_annotator config/test.json populate_ncbi_abstract_embeddings
```
```bash
python -m llm_semantic_annotator config/igepp.json populate_ncbi_abstract_embeddings
```

```bash
python -m llm_semantic_annotator config/test.json compute_tag_chunk_similarities
```

```bash
python -m unittest discover
```


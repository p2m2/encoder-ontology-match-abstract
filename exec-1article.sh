#!/bin/bash

conffile=config/1-article.json
rm -rf 1-article_w*/
python -m llm_semantic_annotator $conffile populate_owl_tag_embeddings
python -m llm_semantic_annotator $conffile populate_gbif_taxon_tag_embeddings
python -m llm_semantic_annotator $conffile populate_abstract_embeddings
python -m llm_semantic_annotator $conffile compute_tag_chunk_similarities


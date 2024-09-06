#!/bin/bash

conffile=config/ncbi.json
#rm -rf igepp-sub_w*/
python -m llm_semantic_annotator $conffile populate_owl_tag_embeddings
python -m llm_semantic_annotator $conffile populate_ncbi_abstract_embeddings
python -m llm_semantic_annotator $conffile compute_tag_chunk_similarities


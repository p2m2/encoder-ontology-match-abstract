#!/bin/bash

conffile=config/igepp-sub.json
rm igepp-sub_workdir/*.json
rm igepp-sub_workdir/*.pth
python -m llm_semantic_annotator $conffile populate_owl_tag_embeddings
python -m llm_semantic_annotator $conffile populate_ncbi_abstract_embeddings
python -m llm_semantic_annotator $conffile compute_tag_chunk_similarities


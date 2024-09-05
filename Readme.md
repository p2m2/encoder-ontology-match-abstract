# LLMSemanticAnnotator

[![Tests](https://github.com/p2m2/encoder-ontology-match-abstract/actions/workflows/ci.yml/badge.svg)](https://github.com/p2m2/encoder-ontology-match-abstract/actions/workflows/ci.yml)

## Installation

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Configuration

check exemple on [config](./config) directory

<table style="font-size: 10px;">
    <tr>
        <th>Action</th>
        <th>Description</th>
        <th>Parameter</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>populate_owl_tag_embeddings</td>
        <td>Populate the corpus of tags</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>debug_nb_terms_by_ontology</td>
        <td>Number of terms per ontology for debugging.</td>
    </tr>
    <tr>
        <td>populate_ncbi_abstract_embeddings</td>
        <td>Populate the database with scientific articles</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>debug_nb_ncbi_request</td>
        <td>Number of NCBI requests for debugging.</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>debug_nb_abstracts_by_search</td>
        <td>Number of abstracts per search for debugging.</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>retmax</td>
        <td>Maximum number of articles per search term.</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>selected_term</td>
        <td>Selected term to find relevant scientific articles.</td>
    </tr>
    <tr>
        <td>compute_tag_chunk_similarities</td>
        <td>Compute the similarity using cosine score</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>threshold_similarity_tag_chunk</td>
        <td>Similarity threshold for tagging chunks.</td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>debug_nb_similarity_compute</td>
        <td>Maximum number of similarities computed.</td>
    </tr>
</table>


### Execution

```bash
python -m llm_semantic_annotator config/test.json populate_owl_tag_embeddings
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


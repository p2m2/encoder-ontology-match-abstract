# LLMSemanticAnnotator: Advanced Semantic Annotation for Plant Biology Research

[![Tests](https://github.com/p2m2/encoder-ontology-match-abstract/actions/workflows/ci.yml/badge.svg)](https://github.com/p2m2/encoder-ontology-match-abstract/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/p2m2/encoder-ontology-match-abstract/blob/main/llm-semantic-annotator.ipynb)


LLMSemanticAnnotator employs Semantic Textual Similarity (STS) to annotate scientific articles with controlled vocabulary, based on precise term definitions. This implementation primarily leverages ontologies from the Planteome project, while also incorporating other relevant sources, to address the lack of detailed annotations in scientific articles, particularly regarding experimental conditions and plant developmental stages.

## Detailed Functioning

- LLM Utilization: The annotator employs Large Language Models (LLMs) to deeply understand the context and content of scientific articles.
- STS Application: The system compares the semantic similarity between ontological term definitions and article text, using advanced natural language processing techniques.
- Ontology Sources: In addition to Planteome, the annotator integrates controlled vocabularies from other recognized sources in the field of plant biology, ensuring comprehensive coverage of relevant terms.
- Multi-level Annotation: The annotation process specifically targets:
    - Experimental conditions
    - Plant developmental stages
    - Molecules of interest under study
- Semantic Association: Ultimately, the annotator establishes links between annotated terms, enabling the association of experimental conditions and developmental stages with the molecules of interest studied.

This approach aims to significantly enrich the metadata of scientific articles, thereby facilitating experimental reproducibility, comparative analysis of studies, and large-scale knowledge extraction in the field of plant biology.



## Installation

```bash
exec.sh <json_config_file>
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


### Tests Execution


```bash
python -m unittest discover
python -m unittest tests/similarity/test_model_embedding_manager.py
```


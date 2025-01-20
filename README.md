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

## Install

```bash
pip install git+https://github.com/p2m2/encoder-ontology-match-abstract.git@alpha-20241031
```

check [versions](https://github.com/p2m2/encoder-ontology-match-abstract/tags) available

## Run

```bash
Usage: ./exec.sh <config_file> <int_commande>

Commands:
  1. Pseudo workflow [2,4,5,6,7]
  2. Populate OWL tag embeddings
  3. Populate abstract embeddings
  4. Compute similarities between tags and abstract chunks
  5. Display similarities information
  6. Build turtle knowledge graph
  7. Build dataset abstracts annotations CSV file

Details:
  2: Compute TAG embeddings for all ontologies defined in the populate_owl_tag_embeddings section
  3: Compute ABSTRACT embeddings (title + sentences) for all abstracts in the dataset
  4: Compute similarities between TAGS and ABSTRACTS
  5: Display similarities information on the console
  6: Generate turtle file with information {score, tag} for each DOI
  7: Generate CSV file with [doi, tag, pmid, reference_id]
```

## Configuration file (json)  

example can be found :

- [planteome](./config/planteome-demo.json)
- [mesh](./config/mesh-demo.json)
- [foodon](./config/foodon-demo.json)
 

## Configuration main keys

### General Structure

```json
{
    "encoder": string,
    "threshold_similarity_tag_chunk": number,
    "threshold_similarity_tag": number,
    "batch_size": number,
    "populate_owl_tag_embeddings": object,
    "populate_abstract_embeddings": object
}
```

### Main Parameters

- encoder: (string) Specifies the encoding model to use.
- threshold_similarity_tag_chunk: (number) Similarity threshold for computing owl tag / chunk tags.
- threshold_similarity_tag: (number) Similarity threshold between tags (keeps the best above this value).
- batch_size: (number) Batch size for processing.


## populate_owl_tag_embeddings

 This section configures the ontologies to be used for populating OWL tag embeddings. 

```json
"populate_owl_tag_embeddings": {
    "ontologies": {
        "group_link": {
            "ontology_name": {
                "url": string,
                "prefix": string,
                "format": string,
                "label": string,
                "properties": [string],
                "constraints": object
            }
        }
    }
}
```
### Ontology Parameters

- url: (string) URL of the ontology.
- prefix: (string) Prefix of the ontology.
- format: (string) Format of the ontology (e.g., "xml").
- label: (string) Property used as a label (*Used to build embeddings*).
- properties: (array of strings) Additional properties to include (*Used to build embeddings*).
- constraints: (object) Constraints to apply on the ontology.


## populate_abstract_embeddings

 This section configures the population of abstract embeddings. 

```json
"populate_abstract_embeddings": {
    "abstracts_per_file": number,
    "from_ncbi_api": object,
    "from_file": object
}
```

### from_ncbi_api

Configures fetching abstracts from the NCBI API.

- ncbi_api_chunk_size: (number) Chunk size for NCBI requests.
- debug_nb_ncbi_request: (number) Number of requests for debugging (-1 for unlimited).
- retmax: (number) Maximum number of results to return.
- selected_term: (array of strings) Selected search terms.

### from_file

Configures fetching abstracts from local files.

- json_files: (array of strings) List of JSON files to use.
- json_dir: (string) Directory containing JSON files.


## Running Tests

 To execute the test suite, you can use the following commands: 

```bash
python3 -m venv llm_semantic_annotator_env
source llm_semantic_annotator_env/bin/activate
pip install -r requirements.txt 
python -m unittest discover
```

Run a specific test file

```bash
python3 -m venv llm_semantic_annotator_env
source llm_semantic_annotator_env/bin/activate
pip install -r requirements.txt 
python -m unittest tests/similarity/test_model_embedding_manager.py
```



```bash
python3 -m venv llm_semantic_annotator_env
source llm_semantic_annotator_env/bin/activate
pip install -r requirements.txt 
python -m llm_semantic_annotator.similarity_evaluator
```
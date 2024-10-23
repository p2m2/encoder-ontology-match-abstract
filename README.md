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


## Run

```bash
./exec.sh -h
Usage: ./exec.sh <config_file> <int_commande>

Commands:
  1. Pseudo workflow [2,4,5,6,7]
  2. Populate OWL tag embeddings
  3. Populate NCBI Taxon tag embeddings
  4. Populate abstract embeddings
  5. Compute similarities between tags and abstract chunks
  6. Display similarities information
  7. Build turtle knowledge graph
  8. Build dataset abstracts annotations CSV file
  9. Evaluate encoder with MeSH descriptors (experimental)

Details:
  2: Compute TAG embeddings for all ontologies defined in the populate_owl_tag_embeddings section
  3: Compute TAG embeddings for NCBI Taxon
  4: Compute ABSTRACT embeddings (title + sentences) for all abstracts in the dataset
  5: Compute similarities between TAGS and ABSTRACTS
  6: Display similarities information on the console
  7: Generate turtle file with information {score, tag} for each DOI
  8: Generate CSV file with [doi, tag, pmid, reference_id]

```

## Configuration file (json)  

example can be found :

- [planteome](./config/planteome-demo.json)
- [mesh](./config/mesh-demo.json)
- [foodon](./config/foodon-demo.json)
 

## Configuration main keys

### General config

```json
    "encodeur" : "sentence-transformers/all-MiniLM-L6-v2",
    "threshold_similarity_tag_chunk" : 0.65,
    "threshold_similarity_tag" : 0.80,
    "batch_size" : 32,
```

#### encoder

- sentence-transformers/all-MiniLM-L6-v2

### Managing Ontology : populate_owl_tag_embeddings

```json
"populate_owl_tag_embeddings" : {
        "ontologies": {
            "<name_ontology_groupe>" : {
                 "<ontology>": {
                    "url": <string>,
                    "prefix": <string>,
                    "format": <string>,
                    "label" :<string>,
                    "properties": [<string>,..]
                },
                ...
            }
        }
}
```


### Specific management of NCBI Taxon : populate_ncbi_taxon_tag_embeddings

```json
 "populate_ncbi_taxon_tag_embeddings" : {
        "regex" : "rassica.*" ,
        "tags_per_file" : 2000
    },
```

### Managing Abstract : populate_abstract_embeddings

```json
"populate_abstract_embeddings" : {
        "abstracts_per_file" : 50,
        "from_file" : <from_file>
    }
```

#### <from_file>

```json
"from_file" : {
        "json_files" : [
            "data/abstracts/abstracts_1.json",
            "data/abstracts/abstracts_2.json"
        ]
    }
```

```json
"from_file" :{
        "json_dir" : "some_directory"
    }
```

```json
"from_ncbi_api" : {
            "ncbi_api_chunk_size" : 200,
            "debug_nb_ncbi_request" : -1,
            "retmax" : 2000,
            "selected_term" : [
                "Crops%2C+Agricultural%2Fmetabolism%5BMeSH%5D"
            ]
        }
```

### Tests Execution


```bash
python -m unittest discover
python -m unittest tests/similarity/test_model_embedding_manager.py
```


"""
Exemple d'exécution :
python main_msd_spark.py config.json
spark-submit \
  --py-files <chemin/vers/vos/dependances.zip> \
  --files <chemin/vers/votre/fichier/de/configuration.json> \
  main_msd_spark.py <chemin/vers/votre/fichier/de/configuration.json>
  
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, udf
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField

import argparse
from llm_semantic_annotator import ModelEmbeddingManager, OwlTagManager

# Définition des schémas
schema_abstracts = StructType([
    StructField("doi", StringType()),
    StructField("embedding", ArrayType(FloatType()))
])

schema_tags = StructType([
    StructField("term", StringType()), 
    StructField("ontology", StringType()),
    StructField("label", StringType()),
    StructField("group", StringType()),
    StructField("embedding", ArrayType(FloatType()))
])

def create_encode_abstracts_pandas(config_dict):
    @pandas_udf(schema_abstracts, PandasUDFType.GROUPED_MAP)
    def encode_abstracts_pandas(key, pdf):
        mem = ModelEmbeddingManager(config_dict)
        abstracts = [{"doi": row.doi, "title": row.title, "abstract": row.abstract} for _, row in pdf.iterrows()]
        embeddings = mem.encode_abstracts(abstracts)
        result = [{"doi": doi, "embedding": emb.tolist()} for doi, emb_list in embeddings.items() for emb in emb_list]
        return pd.DataFrame(result)
    return encode_abstracts_pandas

def create_encode_tags_pandas(config_dict):
    @pandas_udf(schema_tags, PandasUDFType.GROUPED_MAP)
    def encode_tags_pandas(key, pdf):
        mem = ModelEmbeddingManager(config_dict)
        tags = [{
            "ontology": row.ontology, 
            "term": row.term, 
            "rdfs_label": row.rdfs_label,
            "description": row.description,
            "group": row.group
        } for _, row in pdf.iterrows()]
        tags_embedding = mem.encode_tags(tags)
        result = [{
            "term": term,
            "ontology": data['ontology'],
            "label": data['label'],
            "group": data['group'],
            "embedding": data['emb'].tolist()
        } for term, data in tags_embedding.items()]
        return pd.DataFrame(result)
    return encode_tags_pandas

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    a, b = np.array(vec1), np.array(vec2)
    if a.size == 0 or b.size == 0 or a.shape[0] != b.shape[0]:
        return None
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(cosine_sim) if np.isfinite(cosine_sim) else None

def check_config(config):
    if 'populate_abstract_embeddings' not in config:
        print("Error: 'populate_abstract_embeddings' parameter is missing in the configuration file.")
        sys.exit(1)
    
    if 'from_file' not in config['populate_abstract_embeddings']:
        print("Error: 'from_file' parameter is missing in the configuration file.")
        sys.exit(1)
    
    if 'populate_owl_tag_embeddings' not in config:
        print("Error: 'populate_owl_tag_embeddings' parameter is missing in the configuration file.")
        sys.exit(1)

    if 'ontologies' not in config['populate_owl_tag_embeddings']:
        print("Error: 'ontologies' parameter is missing in the configuration file.")
        sys.exit(1)
        
def get_abstracts_from_config(config):
    abstracts = []
    from_file = config['populate_abstract_embeddings']['from_file']

    if 'json_dir' in from_file:
        json_dirs = from_file['json_dir']
        abstracts.extend([json_dirs] if isinstance(json_dirs, str) else json_dirs)
    
    if 'json_file' in from_file:
        json_files = from_file['json_file']
        abstracts.extend([json_files] if isinstance(json_files, str) else json_files)

    if not abstracts:
        print("Warning: No JSON directories or files specified for abstracts.")

    return abstracts

def create_spark_session():
    return SparkSession.builder \
        .appName("MetabolomicsSemanticsDL_Annotation") \
        .getOrCreate()

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    check_config(config)
    
    spark = create_spark_session()
    
    root_workdir = config_file.split("/").pop().split(".json")[0] + "_workdir/spark"
    print("root:", root_workdir)
    parquet_abstracts_path = root_workdir + "/abstracts_embeddings"
    parquet_tags_path = root_workdir + "/tags_embeddings"
    results = root_workdir + "/results"
    
    if os.path.exists(parquet_abstracts_path):
        print("Chargement des embeddings d'abstracts à partir du fichier Parquet existant.")
        result_df_doi = spark.read.parquet(parquet_abstracts_path)
    else:
        abstracts = get_abstracts_from_config(config)
                
        df = spark.read.json(abstracts)
        encode_abstracts_pandas_udf = create_encode_abstracts_pandas(config)
        result_df_doi = df.groupBy("doi").apply(encode_abstracts_pandas_udf)
        result_df_doi.write.mode("overwrite").parquet(parquet_abstracts_path)

    if os.path.exists(parquet_tags_path):
        print("Chargement des embeddings de tags à partir du fichier Parquet existant.")
        spark_df_tags = spark.read.parquet(parquet_tags_path)
    else:
        encode_tags_pandas_udf = create_encode_tags_pandas(config)
        mem = ModelEmbeddingManager(config)
        tag_manager = OwlTagManager(config['populate_owl_tag_embeddings'], mem)
        
        tags_list = []
    
        for ontology_group_name,ontologies in config['populate_owl_tag_embeddings']['ontologies'].items():
            for ontology in tag_manager.get_ontologies(ontologies):
                
                filepath = tag_manager._get_local_filepath_ontology(ontology,ontologies[ontology]['format'])
                # permet de lire le contenu du fichier owl qui peut se trouver sur le cluster hadoop
                owl_content = spark.sparkContext.wholeTextFiles(filepath).values().collect()[0]
                
                tags_list.extend(
                tag_manager.build_tags_from_owl(
                    ontology, 
                    ontology_group_name,
                    ontologies[ontology],
                    -1,owl_content=owl_content)
                )
                
        spark_df_tags = spark.createDataFrame(tags_list)
        result_df_tags = spark_df_tags.groupBy("term").apply(encode_tags_pandas_udf)
        result_df_tags = result_df_tags.withColumnRenamed('term', 'tag')
        spark_df_tags = result_df_tags
        spark_df_tags.write.mode("overwrite").parquet(parquet_tags_path)

    result_df_doi = result_df_doi.withColumnRenamed("embedding", "abstract_embedding")
    spark_df_tags = spark_df_tags.withColumnRenamed("embedding", "tag_embedding")

    print(f"Nombre d'abstracts: {result_df_doi.count()}")
    print(f"Nombre de tags: {spark_df_tags.count()}")

    cosine_similarity_udf = udf(cosine_similarity, FloatType())
    try:
        result_df = result_df_doi.crossJoin(spark_df_tags) \
            .withColumn("similarity", cosine_similarity_udf(col("abstract_embedding"), col("tag_embedding"))) \
            .select("doi", "tag", "similarity") \
            .filter(col("similarity") >= config["threshold_similarity_tag_chunk"])

        result_df.show(truncate=False)
        result_df.write.mode("overwrite").parquet(results)
    
    except Exception as e:
        print(f"Une erreur s'est produite lors du calcul des similarités : {str(e)}")
        
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metabolomics Semantics DL Annotation")
    parser.add_argument("config_file", help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config_file)

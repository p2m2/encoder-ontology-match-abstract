from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType,col, udf,array
from pyspark.sql.types import ArrayType, FloatType, MapType, StringType, StructType, StructField
from llm_semantic_annotator import ModelEmbeddingManager,OwlTagManager
import pandas as pd
import os
import numpy as np

# Définissez le schéma de sortie de votre UDF
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

result_schema = StructType([
    StructField("doi", StringType()),
    StructField("tag_similarities", MapType(StringType(), FloatType()))
])

config = {
        "encodeur": "sentence-transformers/all-MiniLM-L6-v2",
        "threshold_similarity_tag_chunk": 0.35,
        "threshold_similarity_tag": 0.80,
        "batch_size": 32
    }

def create_encode_abstracts_pandas(config_dict):
    
    @pandas_udf(schema_abstracts, PandasUDFType.GROUPED_MAP)
    def encode_abstracts_pandas(key,pdf):
        mem = ModelEmbeddingManager(config_dict)
        
        abstracts = [{"doi": row.doi, "title": row.title, "abstract": row.abstract} for _, row in pdf.iterrows()]
        embeddings = mem.encode_abstracts(abstracts)
        
        result = []
        for doi, emb_list in embeddings.items():
            result += [{  # Utiliser += pour ajouter les éléments
                "doi": doi, 
                "embedding": emb.tolist()  
            } for emb in emb_list]
        
        
        return pd.DataFrame(result)
    
    return encode_abstracts_pandas

def create_encode_tags_pandas(config_dict):
    @pandas_udf(schema_tags, PandasUDFType.GROUPED_MAP)
    def encode_tags_pandas(key, pdf):
        mem = ModelEmbeddingManager(config)
    
        tags = [{
            "ontology": row.ontology, 
            "term": row.term, 
            "rdfs_label": row.rdfs_label,
            "description": row.description,
            "group": row.group
        } for _, row in pdf.iterrows()]
        
        tags_embedding = mem.encode_tags(tags)
        
        result = [{
            "term": term,  # Utilisez 'term' comme 'tag' dans le résultat
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
    a = np.array(vec1)
    b = np.array(vec2)

    # Vérifiez que les vecteurs ne sont pas vides et ont la même taille
    if a.size == 0 or b.size == 0 or a.shape[0] != b.shape[0]:
        return None
    
    # Calculer la similarité cosinus
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Retourner la similarité comme un scalaire
    return float(cosine_sim) if np.isfinite(cosine_sim) else None


def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    a = np.array(vec1)
    b = np.array(vec2)

    # Vérifiez que les vecteurs ne sont pas vides et ont la même taille
    if a.size == 0 or b.size == 0 or a.shape[0] != b.shape[0]:
        return None
    
    # Calculer la similarité cosinus
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Retourner la similarité comme un scalaire
    return float(cosine_sim) if np.isfinite(cosine_sim) else None

def main():
    spark = SparkSession.builder.appName("MSD").getOrCreate()
        
    # Chemins vers les fichiers Parquet
    parquet_abstracts_path = "data/embeddings/abstracts_embeddings.parquet"
    parquet_tags_path = "data/embeddings/tags_embeddings.parquet"
    
    onto = "data/ontology/TransformON_V9.0.ttl"
    onto = "data/ontology/test.ttl"
    abstracts = "data/msd/export-pubmed-20241014-4-planetome-tagging-sub-test"
    abstracts = "data/msd/export-pubmed-20241014-4-planetome-tagging-sub-test/part-00132-6787be90-eb7f-4950-8ef0-98d9dbbbcd38-c003.json"
    results = "data/spark/results.parquet"
    
    # Appliquer la UDF Pandas
    if os.path.exists(parquet_abstracts_path):
        print("Chargement des embeddings d'abstracts à partir du fichier Parquet existant.")
        result_df_doi = spark.read.parquet(parquet_abstracts_path)
    else:
        df = spark.read.json(abstracts)
        encode_abstracts_pandas_udf = create_encode_abstracts_pandas(config)
         
        result_df_doi = df.groupBy("doi").apply(encode_abstracts_pandas_udf)
        result_df_doi.write.mode("overwrite").parquet(parquet_abstracts_path)
    
    result_df_doi.show()
    
    if os.path.exists(parquet_tags_path):
        print("Chargement des embeddings de tags à partir du fichier Parquet existant.")
        spark_df_tags = spark.read.parquet(parquet_tags_path)
    else:
        encode_tags_pandas_udf = create_encode_tags_pandas(config)
        owl_content = spark.sparkContext.wholeTextFiles(onto).values().collect()[0]
        mem = ModelEmbeddingManager(config)
        tag_manager = OwlTagManager(config,mem)
        df_tags = tag_manager.build_tags_from_owl(
            ontology="transformon",
            ontology_group_name="transform_link",
            ontology_config = {
                        "prefix": "http://opendata.inrae.fr/PO2/Ontology/TransformON/Component/",
                        "format": "turtle",
                        "label" : "skos:prefLabel",
                        "properties": ["skos:scopeNote"]
                    },
            debug_nb_terms_by_ontology=-1,
            owl_content=owl_content)
        
        spark_df_tags = spark.createDataFrame(df_tags)
        spark_df_tags.show()
        
        
        result_df_tags = spark_df_tags.groupBy("term").apply(encode_tags_pandas_udf)
       
        result_df_tags = result_df_tags.withColumnRenamed('term', 'tag')
        result_df_tags.printSchema()
        spark_df_tags=result_df_tags
        spark_df_tags.write.mode("overwrite").parquet(parquet_tags_path)
    
    spark_df_tags.show()
    result_df_doi.printSchema()
    spark_df_tags.printSchema()

    # Renommez les colonnes embedding pour éviter l'ambiguïté
    result_df_doi = result_df_doi.withColumnRenamed("embedding", "abstract_embedding")
    spark_df_tags = spark_df_tags.withColumnRenamed("embedding", "tag_embedding")

    # Logging
    print(f"Nombre d'abstracts: {result_df_doi.count()}")
    print(f"Nombre de tags: {spark_df_tags.count()}")


    cosine_similarity_udf = udf(cosine_similarity, FloatType())
    try:
        result_df = result_df_doi.crossJoin(spark_df_tags) \
        .withColumn("similarity", cosine_similarity_udf(col("abstract_embedding"), col("tag_embedding"))) \
        .select("doi", "tag", "similarity")
        
        # Appliquer le filtre après avoir calculé toutes les similarités
        result_df = result_df.filter(col("similarity") >= config["threshold_similarity_tag_chunk"])

            
        # Afficher les résultats
        result_df.show(truncate=False)
        
        # Sauvegarder les résultats si nécessaire
        result_df.write.mode("overwrite").parquet(results)
    
    except Exception as e:
        print(f"Une erreur s'est produite lors du calcul des similarités : {str(e)}")
        
    spark.stop()

if __name__ == "__main__":
    main()

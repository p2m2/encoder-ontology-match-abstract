- ftp pubmed : [https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/

## TODO

- 1) [OK] Export PubMed => Json encoder/decoder P2M2
- 2) [OK] Faire correspondre DOI PubChem / PubMed => pour selectionner les articles d interet
- 3) annotation Planteome 

### Doc/Tools MSD

https://unh-pfem-gitlab.ara.inrae.fr/metabosemdatalake/msd-database-management/-/blob/main/Doc/README-API.md?ref_type=heads

## Import MSD

```bash
hdfs dfs -mkdir /rdf/pubmed
for file in $(curl -l ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/*.gz); do
  curl -s ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/$file | zcat | hdfs dfs -put - /rdf/pubmed/${file%.gz}
done

```

## spark-shell

```bash
export JAVA_HOME=/usr/local/openjdk/jdk-12.0.2+10/
spark-shell --name Pubmed_Extract_Abstract \
 --packages com.databricks:spark-xml_2.12:0.18.0 \
--conf "spark.serializer=org.apache.spark.serializer.KryoSerializer"  \
--conf "spark.sql.crossJoin.enabled=true"   \
--conf "spark.kryo.registrator=net.sansa_stack.rdf.spark.io.JenaKryoRegistrator,\
net.sansa_stack.query.spark.ontop.OntopKryoRegistrator,\
net.sansa_stack.query.spark.sparqlify.KryoRegistratorSparqlify"  \
--conf "spark.kryoserializer.buffer.max=1024"   \
--executor-memory 12g \
--driver-memory 4g \
--num-executors 38 \
--conf spark.yarn.appMasterEnv.JAVA_HOME="/usr/local/openjdk/jdk-12.0.2+10/" \
--conf spark.executorEnv.JAVA_HOME="/usr/local/openjdk/jdk-12.0.2+10/" \
--jars /usr/share/java/spray-json_2.12.jar,/usr/share/java/sansa-stack-spark_2.12-0.8.0-RC3-SNAPSHOT-jar-with-dependencies.jar,/usr/local/msd-database-management/lib/msdTools-assembly-0.6.2-SNAPSHOT.jar
```

### PMID / abstract

```scala

import org.apache.spark.sql.SparkSession
import com.databricks.spark.xml._
import org.apache.spark.sql.types._

val schema = StructType(Seq(
  StructField("MedlineCitation", StructType(Seq(
    StructField("PMID", StringType),
    StructField("Article", StructType(Seq(
      StructField("ArticleTitle", StringType),
      StructField("Abstract", StructType(Seq(
        StructField("AbstractText", StringType)
      )))
    )))
  ))),
  StructField("PubmedData", StructType(Seq(
    StructField("ArticleIdList", StructType(Seq(
      StructField("ArticleId", ArrayType(StructType(Seq(
        StructField("_VALUE", StringType),
        StructField("_IdType", StringType)
      ))))
    )))
  )))
))

/*
Attention mettre /rdf/pubmed/pubmed*.xml pour appliquer sur l'ensemble
*/
val df = spark.read.option("rowTag", "PubmedArticle").schema(schema).xml("/rdf/pubmed/pubmed*.xml")
//  .xml("/rdf/pubmed/pubmed*.xml")
   //.xml("/rdf/pubmed/pubmed24n1220.xml")


case class Abstract(
  AbstractText: String
)

case class Article(
  ArticleTitle: String,
  Abstract: Option[Abstract]
)

case class MedlineCitation(
  PMID: String,
  Article: Article
)

case class ArticleId(
  _VALUE: String,
  _IdType: String
)

case class ArticleIdList(
  ArticleId: Seq[ArticleId]
)

case class PubmedData(
  ArticleIdList: Option[ArticleIdList]
)

case class PubMedArticle(
  MedlineCitation: MedlineCitation,
  PubmedData: Option[PubmedData]
)

import spark.implicits._
val ds = df.as[PubMedArticle]


ds.show()
ds.printSchema()
```

### Filtrage sur les DOIs referencé par PubChem (pc_reference_identifier)

#### contenu

```rdf
<http://rdf.ncbi.nlm.nih.gov/pubchem/reference/5264929>	dcterms:identifier	<https://doi.org/10.3748/wjg.v20.i38.13973> ,
<http://rdf.ncbi.nlm.nih.gov/pubchem/reference/5264930>	dcterms:identifier	<https://doi.org/10.3748/wjg.v20.i38.13981> ,
<http://rdf.ncbi.nlm.nih.gov/pubchem/reference/5264934>	dcterms:identifier	<https://doi.org/10.3748/wjg.v20.i38.14004> ,
<http://rdf.ncbi.nlm.nih.gov/pubchem/reference/5264939>	dcterms:identifier	<https://doi.org/10.3748/wjg.v20.i38.14051> ,
<http://rdf.ncbi.nlm.nih.gov/pubchem/reference/5264953>	dcterms:identifier	<https://doi.org/10.1270/jsbbs.64.240> ,
<http://rdf.ncbi.nlm.nih.gov/pubchem/reference/5264954>	dcterms:identifier	<https://doi.org/10.1270/jsbbs.64.252> 
```



### conversion json

```scala
import org.apache.spark.sql.functions._
import spark.implicits._

case class ArticleJson(
  title: String,
  abstractText: String,
  doi: String,
  pmid: String
)

case class EnrichedArticleJson(
  title: String,
  abstractText: String,
  doi: String,
  pmid: String,
  reference_id: Option[String]
)

val jsonDs = ds.flatMap { article =>
  val title = article.MedlineCitation.Article.ArticleTitle
  val abstractContent = article.MedlineCitation.Article.Abstract.map(_.AbstractText).getOrElse("")
  val pmid = article.MedlineCitation.PMID
  val doi = article.PubmedData
    .flatMap(_.ArticleIdList)
    .flatMap(_.ArticleId.find(_._IdType == "doi"))
    .map(_._VALUE)
    .getOrElse("")

  if (title.nonEmpty && abstractContent.nonEmpty) {
    Some(ArticleJson(title, abstractContent, doi, pmid))
  } else {
    None
  }
}.withColumnRenamed("abstractText", "abstract")


import net.sansa_stack.rdf.spark.io._
import org.apache.jena.riot.Lang
import org.apache.jena.graph.NodeFactory
import net.sansa_stack.rdf.spark.model._ 

val referencePath = "/data/pubchem_reference_v2024-08-14"

/* attention enlever/mettre *7.ttl pour les tests !!!!!! */
val df_ref = spark.rdf(Lang.TURTLE)(s"$referencePath/pc_reference_identifier*.ttl")

// val df_ref = spark.rdf(Lang.TURTLE)(s"$referencePath/pc_reference_identifier*.ttl")

val triples = df_ref.find(None,Some(NodeFactory.createURI("http://purl.org/dc/terms/identifier")), None)

//println("Number of triples: " + triples.distinct.count())

// Convertir les triples en DataFrame
val doiReferences = { triples.toDF()
  .select(
    regexp_extract($"s","reference/(\\d+)",1).alias("reference_id"),
    regexp_extract($"o", "https://doi.org/(.+)", 1).as("doi")
  )}.filter($"doi" =!= "")

doiReferences.show(false)

// Joindre avec le Dataset PubMed existant
val joinedDs = {
          jsonDs.join(doiReferences, jsonDs("doi") === doiReferences("doi"), "left_outer")
              .drop(doiReferences("doi"))
              .withColumnRenamed("abstractText", "abstract")
              }.filter($"reference_id" =!= "")
              
joinedDs.printSchema()

joinedDs.write.option("maxRecordsPerFile", 30).json("/rdf/export-pubmed-20241014-4-planetome-tagging")

```

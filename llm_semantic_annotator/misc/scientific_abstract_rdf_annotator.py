from rdflib import Graph, Literal, BNode, Namespace, URIRef
from rdflib.namespace import DC, DCTERMS, RDF, RDFS, XSD, PROV, SKOS
import datetime
import urllib

def create_rdf_graph(results_complete_similarities, 
                     encoder_name, 
                     system_name, 
                     similarity_threshold,
                     tag_similarity_threshold, 
                     similarity_method):
    g = Graph()

    CURRENT_NS = Namespace("http://www.inrae.fr/mth/p2m2#")
    g.bind("", CURRENT_NS)

    # Définir les espaces de noms
    g.bind("dc", DC)
    g.bind("dcterms", DCTERMS)
    g.bind("prov", PROV)
    g.bind("skos", SKOS)
    semapv = Namespace("https://w3id.org/semapv/vocab/")
    g.bind("semapv", semapv)

    encoder = BNode()
    g.add((encoder, RDF.type, PROV.Activity))
     
    # Créer un nœud pour représenter le graphe d'annotation
    annotation_graph = BNode()
    g.add((annotation_graph, RDF.type, PROV.Entity))
    g.add((annotation_graph, PROV.generatedAtTime, Literal(datetime.datetime.now().isoformat(), datatype=XSD.dateTime)))
    g.add((annotation_graph, PROV.wasGeneratedBy, encoder))
    
    # Ajouter les métadonnées du processus d'annotation
    g.add((encoder, CURRENT_NS.encoderName, Literal(encoder_name)))
    g.add((encoder, CURRENT_NS.systemName, Literal(system_name)))
    g.add((encoder, CURRENT_NS.similarityThreshold, Literal(similarity_threshold, datatype=XSD.float)))
    g.add((encoder, CURRENT_NS.tagSimilarityThreshold, Literal(tag_similarity_threshold, datatype=XSD.float)))
    
    if similarity_method == 'cosine' :
        g.add((encoder, CURRENT_NS.similarityMethod, URIRef(semapv.SemanticSimilarityThresholdMatching)))
    else:
        raise ValueError(f"Unknown similarity method: {similarity_method}")

    total_triplets = 0
    abstracts_processed = len(results_complete_similarities)

    for doi, complete_similarities in results_complete_similarities.items():
        doi_uri = URIRef(urllib.parse.quote(f"https://doi.org/{doi}"))
        for tag, similarity in complete_similarities.items():
            tag_uri = URIRef(tag)
            annotation_node = BNode()
            g.add((doi_uri, DC.subject, annotation_node))
            g.add((annotation_node, SKOS.closeMatch, tag_uri))
            g.add((annotation_node, CURRENT_NS.similarityScore, Literal(similarity, datatype=XSD.float)))
            total_triplets += 1

    g.add((annotation_graph, CURRENT_NS.totalTriplets, Literal(total_triplets, datatype=XSD.integer)))
    g.add((annotation_graph, CURRENT_NS.abstractsProcessed, Literal(abstracts_processed, datatype=XSD.integer)))

    return g

def save_rdf_graph(g, output_file, format='turtle'):
    g.serialize(destination=output_file, format=format)

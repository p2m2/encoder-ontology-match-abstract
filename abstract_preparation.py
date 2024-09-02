import requests

def get_ncbi_abstracts(search_term):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_term = "plants+AND+metabolomics+AND+spring"
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmax=100&retmode=json"

    response = requests.get(search_url)
    search_results = response.json()
    id_list = search_results['esearchresult']['idlist']

    import xml.etree.ElementTree as ET

    abstracts = []
    chunk_size = 20  # Nombre d'IDs à traiter par requête

    for i in range(0, len(id_list), chunk_size):
        chunk = id_list[i:i+chunk_size]
        ids = ",".join(chunk)
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&rettype=abstract&retmode=xml"
        
        fetch_response = requests.get(fetch_url)
        root = ET.fromstring(fetch_response.content)
        
        for article in root.findall('.//PubmedArticle'):
            abstract_text = ""
            for abstract in article.findall('.//AbstractText'):
                abstract_text += abstract.text or ""
            abstracts.append({
                'pmid': article.find('.//PMID').text,
                'abstract': abstract_text
            })
        if i>1:
            break

    import json

    json_abstracts = json.dumps(abstracts, indent=4)
    print(json_abstracts)

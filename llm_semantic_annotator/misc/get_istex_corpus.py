import requests
import json
import time
import argparse
import os
from math import ceil
from tqdm import tqdm  # Import tqdm

def search_istex(query, scroll_time='30s', max_results=None):
    base_url = "https://api.istex.fr/document/"
    results = []
    
    params = {
        'q': query,
        'scroll': scroll_time,
        'size': 100,
        'output': 'title,abstract,doi,keywords.teeft'
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    #print(json.dumps(data, indent=2))  # Useful for debugging
    total_results = data['total']
    print(f"Total results found: {total_results}")
    
    results.extend(data['hits'])
    nextScrollURI = data['nextScrollURI']

    # Initialize tqdm progress bar
    with tqdm(total=total_results, desc="Retrieving results") as pbar:
        while len(results) < total_results and (max_results is None or len(results) < max_results):
            response = requests.get(nextScrollURI)
            data = response.json()
            
            #print(json.dumps(data, indent=2))

            if not data['hits']:
                break
            
            results.extend(data['hits'])
            
            # Update tqdm progress bar
            pbar.update(len(data['hits']))
            #print(f"Retrieved {len(results)} results so far...") # old print
            
            #time.sleep(0.5) #Removed sleeps, istex dont need that
    
    return results[:max_results] if max_results else results

def format_results(results):
    formatted_results = []
    for result in results:
        formatted_result = {
            "title": result.get('title', ''),
            "abstract": result.get('abstract', ''),
            "doi": result.get('doi', [None])[0],
            "keywords" : result.get('keywords', {'teeft' : []})['teeft'],
        }
        formatted_results.append(formatted_result)
    return formatted_results

def write_results_to_files(results, output_dir, articles_per_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_files = ceil(len(results) / articles_per_file)
    
    for i in range(num_files):
        start_idx = i * articles_per_file
        end_idx = min((i + 1) * articles_per_file, len(results))
        batch = results[start_idx:end_idx]
        
        filename = os.path.join(output_dir, f"results_{i+1}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(batch, f, ensure_ascii=False, indent=4)
        
        print(f"Written {len(batch)} articles to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Search ISTEX API and save results as JSON files.")
    parser.add_argument("terms", nargs='+', help="Search terms")
    parser.add_argument("-m", "--max", type=int, help="Maximum number of results to retrieve")
    parser.add_argument("-s", "--scroll", default="30s", help="Scroll time (e.g., '30s', '1m'). Default is '30s'")
    parser.add_argument("-o", "--output", default="output", help="Output directory for JSON files")
    parser.add_argument("-a", "--articles_per_file", type=int, default=100, help="Number of articles per output file")
    args = parser.parse_args()

    query = " AND ".join(args.terms)
    max_results = args.max
    scroll_time = args.scroll
    output_dir = args.output
    articles_per_file = args.articles_per_file

    print(f"Searching for: {query}")
    print(f"Scroll time: {scroll_time}")
    results = search_istex(query, scroll_time=scroll_time, max_results=max_results)
    print(f"Total results retrieved: {len(results)}")

    formatted_results = format_results(results)
    write_results_to_files(formatted_results, output_dir, articles_per_file)

if __name__ == "__main__":
    main()

import os, csv, ujson
from pathlib import Path

def save_results(data, filename):
    """
    Saves the results to a JSON file.
    """
    with open(filename, 'w') as f:
        ujson.dump(data, f)
    print(f"Results saved in {filename}")

def load_results(filename):
    """
    Loads the results from a JSON file if it exists.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return ujson.load(f)
    raise FileNotFoundError(f"The file {filename} does not exist.")

def list_of_dicts_to_csv(data, filename):
    # Check if the list is not empty
    if not data:
        print("The list is empty.")
        return

    # Get headers (all unique keys from all dictionaries)
    headers = set().union(*(d.keys() for d in data))

    # Open the file in write mode
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        # Write the headers
        writer.writeheader()
        
        # Write the data
        for row in data:
            writer.writerow(row)

def dict_to_csv(dictionary, filename):
    # Determine the headers (keys of the dictionary)
    headers = list(dictionary.keys())

    # Open the file in write mode
    with open(filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # Write the headers
        writer.writeheader()

        # Write the data
        writer.writerow(dictionary)

def get_retention_dir(config_file):
    config_base_name = os.path.basename(config_file)
    config_name_without_ext = os.path.splitext(config_base_name)[0]
    retention_dir = os.path.join(os.getcwd(), f"{config_name_without_ext}_workdir")
    if not os.path.exists(retention_dir):
        os.makedirs(retention_dir, exist_ok=True)
    return retention_dir

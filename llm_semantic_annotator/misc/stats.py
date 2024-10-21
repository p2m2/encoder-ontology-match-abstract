from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

def display_ontologies_distribution(data, keep_tag_embeddings):
    console = Console()

    # Extract key prefixes
    ontologies = []
    labels = []
    for doi, item in data.items():
        for key in item.keys():
            ontology = keep_tag_embeddings[key]['ontology']
            ontologies.append(ontology)
            labels.append(key)
    
    # Count the distribution of prefixes
    distributionOntologies = Counter(ontologies)
    distributionLabels = Counter(labels)

    # General statistics
    nb_abstracts = len(data)
    nb_annotated = sum(1 for item in data.values() if len(item) > 0)
    total_labels = sum(distributionOntologies.values())

    # Display general statistics
    console.print(Panel(
        f"[bold cyan]General Statistics[/bold cyan]\n"
        f"Total number of abstracts: [green]{nb_abstracts}[/green]\n"
        f"Number of annotated abstracts: [green]{nb_annotated}[/green]\n"
        f"Total number of labels used: [green]{total_labels}[/green]",
        title="Summary",
        expand=False
    ))

    # Table for ontology distribution
    table_onto = Table(title="Ontology Distribution")
    table_onto.add_column("Ontology", style="cyan", no_wrap=True)
    table_onto.add_column("Count", justify="right", style="magenta")

    for prefix, count in distributionOntologies.items():
        table_onto.add_row(prefix, str(count))

    console.print(table_onto)

    # Table for label distribution (top 10)
    table_labels = Table(title="Top 10 Most Used Labels")
    table_labels.add_column("Label", style="cyan", no_wrap=True)
    table_labels.add_column("Count", justify="right", style="magenta")

    sorted_distribution = sorted(distributionLabels.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_distribution[:10]:
        table_labels.add_row(label, str(count))

    console.print(table_labels)

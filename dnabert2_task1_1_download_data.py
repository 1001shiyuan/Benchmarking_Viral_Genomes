import subprocess
import os

def run_genome_updater(taxon, output_dir, genomes_per_genus=None):
    """
    Run genome_updater.sh to download genomes from RefSeq.
    
    Args:
        taxon: Taxonomy group to download (e.g., 'viral', 'bacteria')
        output_dir: Directory to save downloaded genomes
        genomes_per_genus: If specified, number of genomes to download per genus
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Base command
    command = [
        "./genome_updater.sh",
        "-d", "refseq",
        "-g", taxon,
        "-f", "genomic.fna.gz",
        "-o", output_dir,
        "-t", "120"
    ]
    
    # Add genus sampling if specified
    if genomes_per_genus is not None:
        command.extend(["-A", f"genus:{genomes_per_genus}"])
    
    # Run the command
    subprocess.run(command, check=True)

def download_genomes():
    """Download negative samples: bacterial (60 per genus) and all archaeal genomes."""
    # Create negative samples directory
    negative_output_dir = "./negative"
    os.makedirs(negative_output_dir, exist_ok=True)
    
    # Download bacterial genomes (60 per genus)
    print("Downloading bacterial genomes from RefSeq (60 per genus)")
    run_genome_updater("bacteria", negative_output_dir, genomes_per_genus=60)
    
    # Download all archaeal genomes
    print("Downloading all archaeal genomes from RefSeq")
    run_genome_updater("archaea", negative_output_dir)

if __name__ == "__main__":
    download_genomes()

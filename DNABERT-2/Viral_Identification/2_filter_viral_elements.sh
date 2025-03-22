#!/bin/bash
#SBATCH --job-name=filter_viral_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1
#SBATCH --output=filter_batch_%A_%a.out
#SBATCH --error=filter_batch_%A_%a.err
#SBATCH --array=0-59  # Process in 60 batches (can be adjusted)

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Load necessary modules
module purge
module load ncbi/blast/2.11.0
module load anaconda3

# Activate virtual environment
if [ ! -d "bert1" ]; then
    echo "Creating virtual environment bert1"
    python -m venv bert1
    source bert1/bin/activate
    pip install biopython
else
    echo "Activating existing virtual environment bert1"
    source bert1/bin/activate
fi

# Define paths
VIRAL_FASTA="/work/sgk270/dataset_for_benchmarking/identification/identification.fasta"
NEGATIVE_DIR="/work/sgk270/dnabert2_task1/negative/2025-03-01_06-40-17"
FILTERED_DIR="/work/sgk270/dnabert2_task1/filtered_negative"
BATCH_DIR="${FILTERED_DIR}/batch_${SLURM_ARRAY_TASK_ID}"

# Create output directories
mkdir -p $FILTERED_DIR
mkdir -p $BATCH_DIR

# Create BLAST database if it doesn't exist yet
if [ ! -f "./viral_db.nin" ]; then
    echo "Creating BLAST database"
    makeblastdb -in $VIRAL_FASTA -dbtype nucl -out ./viral_db
fi

# Get the total list of files
ALL_FILES=($(find $NEGATIVE_DIR -name "*.fna.gz"))
TOTAL_FILES=${#ALL_FILES[@]}
echo "Total files found: $TOTAL_FILES"

# Calculate which files this job should process
BATCH_SIZE=500
START_IDX=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END_IDX=$(( START_IDX + BATCH_SIZE - 1 ))

if [ $END_IDX -ge $TOTAL_FILES ]; then
    END_IDX=$(( TOTAL_FILES - 1 ))
fi

echo "Processing batch $SLURM_ARRAY_TASK_ID (files $START_IDX to $END_IDX)"

# Create a temporary Python script to process this batch of files
cat > process_batch.py << 'EOL'
import os
import sys
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip

def process_file(genome_file, viral_db, output_folder, evalue=1e-10):
    """Process a single genome file to filter out viral regions."""
    base_name = os.path.basename(genome_file)
    output_file = os.path.join(output_folder, f"filtered_{base_name}")
    blast_output = os.path.join(output_folder, f"{base_name}.blast")
    
    # Skip if the output file already exists (for resuming interrupted jobs)
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipping")
        return
    
    # Decompress for BLAST
    temp_input = os.path.join(output_folder, f"temp_{base_name}.fna")
    print(f"Processing {base_name}...")
    try:
        with gzip.open(genome_file, 'rt') as f_in, open(temp_input, 'w') as f_out:
            for record in SeqIO.parse(f_in, "fasta"):
                SeqIO.write(record, f_out, "fasta")
        
        # Run BLASTN
        blastn_cmd = [
            "blastn",
            "-query", temp_input,
            "-db", viral_db,
            "-out", blast_output,
            "-outfmt", "6 qseqid qstart qend",
            "-evalue", str(evalue),
            "-num_threads", "80"
        ]
        
        subprocess.run(blastn_cmd, check=True)
        
        # Process BLAST results
        viral_regions = {}
        if os.path.exists(blast_output) and os.path.getsize(blast_output) > 0:
            with open(blast_output) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        seqid, start, end = parts[0], parts[1], parts[2]
                        if seqid not in viral_regions:
                            viral_regions[seqid] = []
                        try:
                            viral_regions[seqid].append((int(start), int(end)))
                        except ValueError:
                            print(f"Warning: Could not parse coordinates in line: {line.strip()}")
        
        # Merge overlapping regions
        for seqid in viral_regions:
            viral_regions[seqid].sort()
            merged = []
            for region in viral_regions[seqid]:
                if not merged or merged[-1][1] < region[0]:
                    merged.append(region)
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], region[1]))
            viral_regions[seqid] = merged
        
        # Filter sequences
        filtered_records = []
        with gzip.open(genome_file, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                if record.id not in viral_regions:
                    filtered_records.append(record)
                else:
                    seq_str = str(record.seq)
                    current_pos = 0
                    for start, end in viral_regions[record.id]:
                        if start > current_pos:
                            new_seq = seq_str[current_pos:start]
                            if len(new_seq) >= 300:  # Only keep sequences >= 300bp
                                new_record = SeqRecord(
                                    Seq(new_seq),
                                    id=f"{record.id}_{current_pos}_{start}",
                                    description=f"Non-viral region from {record.id}"
                                )
                                filtered_records.append(new_record)
                        current_pos = end
                    
                    if current_pos < len(seq_str):
                        new_seq = seq_str[current_pos:]
                        if len(new_seq) >= 300:
                            new_record = SeqRecord(
                                Seq(new_seq),
                                id=f"{record.id}_{current_pos}_{len(seq_str)}",
                                description=f"Non-viral region from {record.id}"
                            )
                            filtered_records.append(new_record)
        
        # Write filtered sequences
        with gzip.open(output_file, 'wt') as f:
            SeqIO.write(filtered_records, f, "fasta")
        print(f"Wrote {len(filtered_records)} filtered sequences to {output_file}")
        
    except Exception as e:
        print(f"Error processing {base_name}: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(blast_output):
            os.remove(blast_output)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python process_batch.py [viral_db] [file_list_file] [output_dir]")
        sys.exit(1)
    
    viral_db = sys.argv[1]
    file_list = sys.argv[2]
    output_dir = sys.argv[3]
    
    with open(file_list, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(files)} files from {file_list}")
    for i, file_path in enumerate(files, 1):
        print(f"File {i}/{len(files)}: {file_path}")
        process_file(file_path, viral_db, output_dir)
    
    print(f"Batch processing complete. Processed {len(files)} files.")
EOL

# Create a file with the list of files to process in this batch
FILE_LIST="${BATCH_DIR}/file_list.txt"
for ((i=START_IDX; i<=END_IDX; i++)); do
    if [ $i -lt $TOTAL_FILES ]; then
        echo "${ALL_FILES[$i]}" >> $FILE_LIST
    fi
done

# Process this batch
echo "Starting batch processing of $(wc -l < $FILE_LIST) files"
python process_batch.py "./viral_db" "$FILE_LIST" "$BATCH_DIR"

# Move processed files to main output directory
echo "Moving processed files to main output directory"
mv $BATCH_DIR/filtered_*.fna.gz $FILTERED_DIR/

echo "Batch $SLURM_ARRAY_TASK_ID complete"
echo "Job finished at: $(date)"
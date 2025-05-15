import os
import subprocess
import glob
import gzip
import time
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def make_blast_db(viral_paths, db_prefix):
    combined_fasta = db_prefix + ".fasta"
    db_dir = os.path.dirname(db_prefix)
    os.makedirs(db_dir, exist_ok=True)

    print(f"[INFO] Combining viral FASTAs into {combined_fasta}")
    with open(combined_fasta, "w") as out_f:
        for path in viral_paths:
            matched_files = glob.glob(path)
            print(f"[INFO] Found {len(matched_files)} files for path: {path}")
            for file in matched_files:
                print(f"[INFO] Adding file to combined fasta: {file}")
                with open(file) as in_f:
                    shutil.copyfileobj(in_f, out_f)

    print(f"[INFO] Running makeblastdb...")
    result = subprocess.run([
        "makeblastdb", "-in", combined_fasta, "-dbtype", "nucl", "-out", db_prefix
    ], capture_output=True, text=True)

    print("[makeblastdb stdout]")
    print(result.stdout)
    print("[makeblastdb stderr]")
    print(result.stderr)
    result.check_returncode()

def filter_genome(input_path, db_prefix, output_path, evalue=1e-10):
    base_name = os.path.basename(input_path)
    temp_fasta = os.path.join("/tmp", base_name + ".tmp.fasta")
    blast_out = os.path.join("/tmp", base_name + ".blast")

    print(f"[INFO] Converting {input_path} to uncompressed temporary FASTA")
    with gzip.open(input_path, "rt") as fin, open(temp_fasta, "w") as fout:
        SeqIO.write(SeqIO.parse(fin, "fasta"), fout, "fasta")

    print(f"[INFO] Running BLAST on {temp_fasta}")
    subprocess.run([
        "blastn", "-query", temp_fasta, "-db", db_prefix,
        "-out", blast_out, "-outfmt", "6 qseqid qstart qend",
        "-evalue", str(evalue), "-num_threads", "80"
    ], check=True)

    regions = {}
    if os.path.exists(blast_out):
        with open(blast_out) as f:
            for line in f:
                qid, start, end = line.strip().split()
                regions.setdefault(qid, []).append((int(start), int(end)))

    for qid in regions:
        regions[qid].sort()
        merged = []
        for s, e in regions[qid]:
            if not merged or merged[-1][1] < s:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        regions[qid] = merged

    filtered = []
    with gzip.open(input_path, "rt") as f:
        for rec in SeqIO.parse(f, "fasta"):
            if rec.id not in regions:
                filtered.append(rec)
            else:
                seq = str(rec.seq)
                pos = 0
                for s, e in regions[rec.id]:
                    if s > pos:
                        frag = seq[pos:s]
                        if len(frag) >= 300:
                            filtered.append(SeqRecord(Seq(frag), id=f"{rec.id}_{pos}_{s}", description="non-viral"))
                    pos = e
                if pos < len(seq):
                    frag = seq[pos:]
                    if len(frag) >= 300:
                        filtered.append(SeqRecord(Seq(frag), id=f"{rec.id}_{pos}_{len(seq)}", description="non-viral"))

    print(f"[INFO] Writing filtered output to {output_path}")
    with gzip.open(output_path, "wt") as f:
        SeqIO.write(filtered, f, "fasta")

    os.remove(temp_fasta)
    os.remove(blast_out)

if __name__ == "__main__":
    import shutil  # Needed for make_blast_db

    negative_dir = "/work/sgk270/dnabert2_task1_new/negative"
    output_dir = "/work/sgk270/dnabert2_task1_new/filtered_negative"
    db_prefix = "/work/sgk270/dnabert2_task1_new/db/viral_db"
    viral_with_meta = "/work/sgk270/dataset_for_benchmarking/combine3/final_dataset/with_meta/*.fasta"
    viral_without_meta = "/work/sgk270/dataset_for_benchmarking/combine3/final_dataset/without_meta/*.fasta"

    os.makedirs(output_dir, exist_ok=True)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    batch_size = 500
    start = task_id * batch_size

    # Only task 0 creates the DB if not already created
    if task_id == 0 and not os.path.exists(db_prefix + ".nal"):
        print("[INFO] BLAST DB not found. Task 0 will create it.")
        make_blast_db([viral_with_meta, viral_without_meta], db_prefix)

    # Wait for DB to be ready (important for non-zero tasks)
    while not os.path.exists(db_prefix + ".nal"):
        print(f"[INFO] Task {task_id} waiting for BLAST DB to be ready...")
        time.sleep(10)

    all_files = sorted(glob.glob(os.path.join(negative_dir, "**", "*.fna.gz"), recursive=True))
    end = min(start + batch_size, len(all_files))

    print(f"[INFO] Processing files {start} to {end} (task {task_id})")
    for i in range(start, end):
        in_file = all_files[i]
        base = os.path.basename(in_file)
        out_file = os.path.join(output_dir, f"filtered_{base}")
        if not os.path.exists(out_file):
            try:
                print(f"[INFO] Filtering {base}")
                filter_genome(in_file, db_prefix, out_file)
            except Exception as e:
                print(f"[ERROR] {base}: {e}")

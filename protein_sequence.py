#!/usr/bin/env python3
"""
Recursively scan a top-level directory for PDB files matching '*_protein.pdb',
extract SEQRES sequences for each chain, and write all sequences
as one-per-line into a single 'protein_sequences.txt' file.
"""
import os
import sys
from Bio import SeqIO

# Hard-coded top-level directory containing your PDB files
SRC_DIR = "/home/ansh-meshram/Desktop/work/bio_project/PDBbind/data"
# Output file for all sequences
OUTPUT_FILE = os.path.join(SRC_DIR, "protein_sequences.txt")

def extract_sequences(src_dir, output_file):
    """Walk through src_dir, parse each *_protein.pdb, and write sequences to one file."""
    sequences = []
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if fname.endswith("_protein.pdb"):
                pdb_path = os.path.join(root, fname)
                try:
                    # Parse SEQRES records
                    for rec in SeqIO.parse(pdb_path, "pdb-seqres"):
                        # rec.id includes PDB ID and chain
                        sequences.append(str(rec.seq))
                except Exception as e:
                    print(f"Error processing {pdb_path}: {e}", file=sys.stderr)
    # Write all sequences, one per line
    with open(output_file, "w") as out_handle:
        for seq in sequences:
            out_handle.write(seq + "\n")
    print(f"Wrote {len(sequences)} sequences to {output_file}")

if __name__ == "__main__":
    extract_sequences(SRC_DIR, OUTPUT_FILE)

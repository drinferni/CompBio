#!/bin/bash

# Base directory containing all complex folders
BASE_DIR="/home/ansh-meshram/Desktop/work/bio_project/PDBbind/data"
SCRIPT_PATH="/home/ansh-meshram/Desktop/work/bio_project/PDBbind/convert.py"
echo "done"
# Loop over each subfolder
for dir in "$BASE_DIR"/*; do
  if [ -d "$dir" ]; then
    name=$(basename "$dir")
    echo "doing"

    ligand_input="$dir/${name}_ligand.mol2"
    protein_input="$dir/${name}_protein.pdb"
    ligand_output="$dir/${name}_ligand.pdb"
    protein_output="$dir/${name}_protein.mol2"

    if [[ -f "$ligand_input" && -f "$protein_input" ]]; then
      echo "Processing: $name"
      /home/ansh-meshram/Desktop/chimera/bin/chimera --nogui --script "$SCRIPT_PATH \"$ligand_input\" \"$protein_input\" \"$ligand_output\" \"$protein_output\""
    else
      echo "Skipping $name - missing ligand or protein file"
    fi
  fi
done


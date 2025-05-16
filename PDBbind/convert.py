import sys
from chimera import runCommand

def convert_files(ligand_input, protein_input, ligand_output, protein_output):
    # Open the ligand file
    runCommand('open {}'.format(ligand_input))
    # Save the ligand as PDB
    runCommand('write format pdb #0 {}'.format(ligand_output))
    # Close the ligand model
    runCommand('close #0')


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: chimera --nogui --script 'convert.py ligand_input protein_input ligand_output protein_output'")
    else:
        _, ligand_input, protein_input, ligand_output, protein_output = sys.argv
        convert_files(ligand_input, protein_input, ligand_output, protein_output)


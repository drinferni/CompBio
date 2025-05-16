import os
from Bio.PDB import PDBParser, PPBuilder
from esm import pretrained
import torch
from tqdm import tqdm
import pickle
import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from tqdm import tqdm
import traceback


# Load the big ESM-2 model
model, alphabet = pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
batch_converter = alphabet.get_batch_converter()
model.eval()

# Function to extract full sequence from all chains in a PDB
def extract_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    ppb = PPBuilder()
    sequences = []
    for model_struct in structure:
        for chain in model_struct:
            peptides = ppb.build_peptides(chain)
            if peptides:
                seq = ''.join(str(pp.get_sequence()) for pp in peptides)
                sequences.append(seq)
    full_seq = ''.join(sequences)
    return full_seq if full_seq else None

# Get embedding for one sequence
def get_protein_embedding(sequence):
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])
    token_embeddings = results["representations"][6]
    embedding = token_embeddings[0, 1:len(sequence)+1].mean(0)  # mean pool
    return embedding.cpu().numpy()

# Process multiple PDB files in a directory
def process_pdb_folder(parent_folder, save=True):
    embeddings = {}

    for subdir, dirs, files in os.walk(parent_folder):
        print(files)
        for file in files:
            if not file.endswith("protein.pdb"):
                continue
            pdb_path = os.path.join(subdir, file)
            try:
                sequence = extract_sequence_from_pdb(pdb_path)
                if not sequence:
                    print(f"⚠️ No sequence found in {pdb_path}")
                    continue
                embedding = get_protein_embedding(sequence)
                embeddings[pdb_path] = embedding
                path = os.path.join(subdir,"protein_esm.pkl")
                with open(path, "wb") as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                print(f"❌ Error with {pdb_path}: {e}")
                print(f"   {type(e).__name__}: {e}")
                traceback.print_exc()
    return embeddings


def get_morgan_fingerprint(mol, radius=2, n_bits=64):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def process_ligands_in_folders(parent_folder):
    for folder_name in tqdm(os.listdir(parent_folder), desc="Processing folders"):
        folder_path = os.path.join(parent_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue  # skip if not a folder

        # Find mol2 file ending with _ligand.mol2
        ligand_file = None
        for file in os.listdir(folder_path):
            if file.endswith("_ligand.mol2"):
                ligand_file = os.path.join(folder_path, file)
                break

        if not ligand_file:
            print(f"⚠️ No *_ligand.mol2 file found in {folder_name}")
            continue

        try:
            mol = Chem.MolFromMol2File(ligand_file, sanitize=True, removeHs=False)
            if mol is None:
                print(f"❌ Failed to load molecule from {ligand_file}")
                continue

            fp_vector = get_morgan_fingerprint(mol)

            # Save fingerprint as pickle
            output_path = os.path.join(folder_path, "ligand_embedding.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(fp_vector, f)

        except Exception as e:
            print(f"❌ Error processing {ligand_file}: {e}")


# Example usage
if __name__ == "__main__":
    pdb_dir = "/home/ansh-meshram/Desktop/work/bio_project/PDBbind/data"  # replace with your folder
    process_ligands_in_folders(pdb_dir)
    process_pdb_folder(pdb_dir)

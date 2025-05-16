import os
import csv
from scipy.spatial import cKDTree
from rdkit.Chem import rdMolDescriptors
from statistics import mean
import numpy as np
import itertools
from rdkit.Chem import SanitizeMol
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.cluster import KMeans
from collections import defaultdict
from deepchem.utils.rdkit_utils import load_molecule
from scipy.spatial.distance import cdist
import numpy as np
import pandas
from math import sqrt, acos, pi
#from Bio.PDB import PDBParser
from biopandas.mol2 import PandasMol2
from itertools import combinations
from rdkit.Chem import GetPeriodicTable
import itertools
from statistics import mean
import os
import pickle

class SIFt(object):
    def __init__(self, fn_pro_PDB, fn_lig_PDB, fn_pro_MOL2, fn_lig_MOL2, ID = None, addH = True, sant = True, int_cutoff = 4.5):
        """
        Initialize an SIFt class.
        Parameters:
            fn_pro_PDB - PDB file name of the protein
            fn_lig_PDB - PDB file name of the ligand
            fn_pro_MOL2 - MOL2 file name of the protein
            fn_lig_MOL2 - MOL2 file name of the ligand
            ID - ID of the complex
            addH - whether to add hydrogen atoms when reading in the structure file
            sant - whether to sanitize the molecule when reading in the structure file
            int_cutoff - distance threshold for identifying protein-ligand interacting atoms 
        """
        self.ID = ID if ID is not None else "PL"
       # print('Constructing an SIFt object for %s.........\n' % self.ID)
        # read in pdb coordinates and topology
        self.lig = (load_molecule(fn_lig_PDB, add_hydrogens=addH, calc_charges=False, sanitize=sant)) 
        self.pro = (load_molecule(fn_pro_PDB, add_hydrogens=addH, calc_charges=False, sanitize=sant))
        print(fn_pro_MOL2)

        print("loaded")
 
       
        # parse protein pdb file for identifying sidechain/mainchain atoms
#        parser = PDBParser()
#        self.structure = parser.get_structure(self.ID, fn_pro_PDB)
#        self.chid = self.pro[1].GetAtomWithIdx(0).GetPDBResidueInfo().GetChainId()
        # identify interacting area
        self.contact_bins = [(0, int_cutoff)]
        self.pd = cdist(self.pro[0], self.lig[0], metric = 'euclidean')
        contacts = np.nonzero(self.pd < int_cutoff)
        tmpcont = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
        self.cont = [[int(i) for i in contacts[0]], [int(j) for j in contacts[1]]]
        self.contacts = []


        self.protein_env = defaultdict(int)
        self.ligand_env = defaultdict(int)

        # Efficient neighbor search using KD-tree
        cutoff = 2

        ## Protein environment
        protein_tree = cKDTree(self.pro[0])
        protein_pairs = protein_tree.query_pairs(r=cutoff, output_type='ndarray')  # output_type avoids slow set-to-list

        for a, b in protein_pairs:
            self.protein_env[a] += 1
            self.protein_env[b] += 1

        ## Ligand environment
        ligand_tree = cKDTree(self.lig[0])
        ligand_pairs = ligand_tree.query_pairs(r=cutoff, output_type='ndarray')

        for a, b in ligand_pairs:
            self.ligand_env[a] += 1
            self.ligand_env[b] += 1

 

def process_folders(root_folder):
    """
    Traverses a root folder. For each subfolder, it opens all files,
    instantiates a MyObject for each file, collects the parameters along with some metadata,
    and writes the data to a CSV file.
    """
    results = []
    count = 0
    # Loop over entries in the root folder
    for foldername in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, foldername)
        if os.path.isdir(subfolder_path):
            # In each subfolder, iterate through the files
            filename_lig = foldername + "_ligand.pdb"
            filename_pro = foldername + "_protein.pdb"
            file_path_lig = os.path.join(subfolder_path, filename_lig)
            file_path_pro = os.path.join(subfolder_path, filename_pro)
            pickle_filename = f"{foldername}_sift.pkl"
            pickle_path = os.path.join(subfolder_path, pickle_filename)
            count += 1
            if os.path.isfile(pickle_path):
                continue
            
            try:
                sift = SIFt(file_path_pro, file_path_lig, "", "", ID=foldername, addH=True, sant=False, int_cutoff=18)
            except Exception as e:
                print(f"Skipping {foldername}: {e}")
                continue

            if sift is not None:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(sift, f)
                print(f'Stored {pickle_filename} in {pickle_path}',count,len(os.listdir(root_folder)))
  
    
if __name__ == '__main__':
    # Define your root folder (adjust the path as needed)
    root_folder = './data'
    process_folders(root_folder)

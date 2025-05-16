#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:58:12 2021

@author: debbywang
"""
import traceback
import logging
import numpy as np
import pandas as pd
import multiprocessing
from SIFt import SIFt
import random
import os
import time
import deepchem as dc
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle


def _featurize_complex(pdbinfo, log_message, para):
    """Featurizes a complex.
    First initializes an SIFt object, and then calculates the intlst.
    """   
    logging.info(log_message)
    protein_path = pdbinfo['fn_pro_PDB']
    pkl_path = protein_path.replace('_protein.pdb', '_sift.pkl')

# Check if the corresponding pickle file exists
    if os.path.isfile(pkl_path):
        pass
    else:
        return None

    # try:
    try :
        with open(pkl_path, 'rb') as f:
            pickled_sift_obj = pickle.load(f)
    except:
        return None
    sift = SIFt(pickled_sift_obj)
    if sift is None:
        print("error in retriving object")
        return None
    desp= sift.get_quasi_fragmental_desp_ext(bins = para['bins'])
    if (desp is None):
        print("wtf")
    return desp
#     except:
#         print("Error during featurization:")
# #        traceback.print_exc()
#         return None

  
def featurize_complexes(ligand_pdbfiles, 
                        protein_pdbfiles,
                        ligand_mol2files, 
                        protein_mol2files,
                        pdbids,
                        para = {'addH': True, 'sant': True,
                                'cutoff': 4.5, 'includeH': False, 'ifp_type': 'sift', 'count': 1,
                                'intlst': ['contact', 'backbone', 'sidechain', 'polar', 'nonpolar', 'donor', 'acceptor'],
                                'inttype': 'CH'}
#                        para = {'addH': True, 'sant': True,
#                                'cutoff': 4.5, 'ifp_type': 'rfscore',
#                                'bins': None, # use default contact bins
#                                'solo': 0, 'lst': []}
#                        para = {'addH': True, 'sant': False,
#                                'cutoff': 4.5, 'ifp_type': 'splif',
#                                'ecfp_radius': [1, 1],
#                                'base_prop': ['AtomicNumber', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge'],
#                                'folding_para': {'power': np.arange(6, 8, 1), 'counts': 1}}
                        ):
    """Obtains SIFts of a group of complexes.
    Parameters:
        ligand_pdbfiles - a list of ligand pdb files
        protein_pdbfiles - a list of protein pdb files
        ligand_mol2files - a list of ligand mol2 files
        protein_mol2files - a list of protein mol2 files
        pdbids - a list of pdb ids for the complexes under processing
        para - parameters for constructing SIFts
    Returns a list of the SIFts and the indices of failed complexes.
    """
    pool = multiprocessing.Pool(processes =8)
    results = []
    feat = []
    failures = []
    info = zip(ligand_pdbfiles, protein_pdbfiles, ligand_mol2files, protein_mol2files, pdbids)
    for i, (lig_pdbfile, pro_pdbfile, lig_mol2file, pro_mol2file, pdbid) in enumerate(info):
        #print(i)
        log_message = "Featurizing %d / %d complex..." % (i, len(pdbids))
        pdbinfo = {'fn_pro_PDB': pro_pdbfile, 'fn_lig_PDB': lig_pdbfile, 
                   'fn_pro_MOL2': pro_mol2file, 'fn_lig_MOL2': lig_mol2file, 'pdbid': pdbid}
        results.append(pool.apply_async(_featurize_complex, (pdbinfo, log_message, para)))      
    pool.close()  
    pool.join()
    for ind, result in enumerate(results):
        new_sift = result.get()
        if new_sift is None:
            failures.append(ind)
        else:
            feat.append(new_sift)
    
    return feat, failures


def loaddata_fromPDBbind_v2020(data_dir,
                               subset = "refined",
                               select_target = ['HIV-1 PROTEASE'],
                               randsplit_ratio = [0.5, 0.25, 0.25],
                               para = {'addH': True, 'sant': True,
                                       'cutoff': 4.5, 'ifp_type': 'rfscore',
                                       'bins': None, 'lst': [], # use default contact bins
                                       'solo': 0},
                               rand_seed = 123):
    """Load data from PDBbind sets or subsets.
    Parameters:
        data_dir - folder that stores PDBbind data
        subset - supports PDBbind refined set ('refined'), core set ('cs') and refined minus core set ('rs-cs')
        select_target - whether to select specific targets from the subset (['HIV-1 PROTEASE'], ['CARBONIC ANHYDRASE 2'] or None)
        ransplit_ratio - ratio for patition the set into training, validation and test sets
        para - a dictionary of parameters for computing ifp features (align with generate_dt function)
        rand_seed - random seed for splits
    Returns (training data, test data) from the deepchem library
    """
    dt_dir = data_dir
    os.chdir(dt_dir)
    
    # -----------------------------------------------------------------------------------------------------------------------
    # get folder containing the structural data and the index dataframe containing the pdbs/affinities
    # -----------------------------------------------------------------------------------------------------------------------
    data_folder_dict = {'refined': 'data/'}
    data_index_dict = {'refined': 'indexes/rs_index.csv'}
    for s in ['casf2016']:
        data_folder_dict[s] = 'data/'
        data_index_dict[s] = 'indexes/' + s + '_index.csv'

    if subset in ['casf2016']:
        cur_fd = data_folder_dict[subset]
        df_index = pd.read_csv(data_index_dict[subset])
    else:
        cur_fd = data_folder_dict['refined']
        df_index_all = pd.read_csv(data_index_dict['refined'])
        df_index = df_index_all
        
    # further filter the dataframe according to select_target
    pdbs_selected = df_index['id'].tolist()
    labels_selected = df_index['affinity'].astype(float).tolist()
            
    # obtain the pdb and mol2 filename lists for the complexes -----------------------------------------------------------------------
    print('Generate file names................................................................')
    protein_pdbfiles = []
    ligand_pdbfiles = []
    protein_mol2files = []
    ligand_mol2files = []
    for pdb in pdbs_selected:
        protein_pdbfiles += [os.path.join(dt_dir, cur_fd, pdb, "%s_protein.pdb" % pdb)]
        protein_mol2files += [os.path.join(dt_dir, cur_fd, pdb, "%s_protein.mol2" % pdb)]
        ligand_pdbfiles += [os.path.join(dt_dir, cur_fd, pdb, "%s_ligand.pdb" % pdb)]
        ligand_mol2files += [os.path.join(dt_dir, cur_fd, pdb, "%s_ligand.mol2" % pdb)]  
        
    # featurize complexes using SIFts and split them into train, validation and test sets --------------------------------------------
    print('Begin to featurize dataset....................................................................')
    feat_t1 = time.time()
    print("done1")
    feat, flrs = featurize_complexes(ligand_pdbfiles = ligand_pdbfiles, 
                                     protein_pdbfiles = protein_pdbfiles,
                                     ligand_mol2files = ligand_mol2files, 
                                     protein_mol2files = protein_mol2files,
                                     pdbids = pdbs_selected,
                                     para = para)
    # Delete labels and ids for failing elements
    labels_lft = np.delete(labels_selected, flrs)
    print(labels_lft[0].dtype)
    return [np.array(feat),np.array(labels_lft)]

            

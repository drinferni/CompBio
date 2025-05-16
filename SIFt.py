#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:53:50 2021

@author: debbywang
"""
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
import itertools
from statistics import mean

import re

class SIFt(object):
    def __init__(self, pickled_sift):
        
        if pickled_sift is None:
            print("ohno")
            return None

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
        self.ID = pickled_sift.ID 
       # print('Constructing an SIFt object for %s.........\n' % self.ID)
        # read in pdb coordinates and topology
        self.lig = pickled_sift.lig
        self.pro = pickled_sift.pro

 
       
        # parse protein pdb file for identifying sidechain/mainchain atoms
#        parser = PDBParser()
#        self.structure = parser.get_structure(self.ID, fn_pro_PDB)
#        self.chid = self.pro[1].GetAtomWithIdx(0).GetPDBResidueInfo().GetChainId()
        # identify interacting area
        self.contact_bins = pickled_sift.contact_bins
        self.pd = pickled_sift.pd
        self.cont = pickled_sift.cont
        self.contacts = pickled_sift.contacts

        self.protein_env = pickled_sift.protein_env
        self.ligand_env = pickled_sift.ligand_env
        
  
    def get_quasi_fragmental_desp_ext(self, bins = None):
        """
        Compute extended quasi fragmental descriptors.
        Parameters:
            bins - distance bins for extracting qf descriptors
        """
        if bins is not None:
            self.contact_bins = bins
        else:
            self.contact_bins = [(0, 12)]

        # list atom types ----------------------------------------------------
        pro_pool = [6, 7, 8, 16]
        lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        # get list of atom-pair types ----------------------------------------
        quasi_type = list(itertools.product(pro_pool, lig_pool))        
        quasi_feat = []

        for cbin in self.contact_bins:
            occur = {}
            for tp in quasi_type:
                occur[tp] = [0, []]
            # check the contacts one by one ----------------------------------
            contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
            conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
            distances = [self.pd[i, j] for (i, j) in conts]
            for ind in range(len(conts)):
                cont = conts[ind]
                cur_dist = distances[ind]
                atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                atm1_an = atm1.GetAtomicNum()
                atm2_an = atm2.GetAtomicNum()
                
                tmp = (atm1_an, atm2_an)
                if tmp in quasi_type:
                    occur[tmp][0] += 1
                    occur[tmp][1] += [cur_dist]
            
            for tp in quasi_type:
                if occur[tp][0] == 0:
                    quasi_feat += [0, 0]
                else:
                    quasi_feat += [occur[tp][0], mean(occur[tp][1])]
        return quasi_feat  

    def get_quasi_fragmental_desp_ext1b(self, bins = None):
        """
        Compute extended quasi fragmental descriptors.
        Parameters:
            bins - distance bins for extracting qf descriptors
        """
        if bins is not None:
            self.contact_bins = bins
        else:
            self.contact_bins = [(0, 12)]

        # list atom types ----------------------------------------------------
        pro_pool = [6, 7, 8, 16]
        lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        # get list of atom-pair types ----------------------------------------
        quasi_type = list(itertools.product(pro_pool, lig_pool))        
        quasi_feat = []

        for cbin in self.contact_bins:
            occur = {}
            for tp in quasi_type:
                occur[tp] = [0, []]
            # check the contacts one by one ----------------------------------
            contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
            conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
            distances = [self.pd[i, j] for (i, j) in conts]
            for ind in range(len(conts)):
                cont = conts[ind]
                cur_dist = distances[ind]
                atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                atm1_an = atm1.GetAtomicNum()
                atm2_an = atm2.GetAtomicNum()
                
                tmp = (atm1_an, atm2_an)
                if tmp in quasi_type:
                    occur[tmp][0] += 1
                    occur[tmp][1] += [cur_dist]
            
            for tp in quasi_type:
                if occur[tp][0] == 0:
                    quasi_feat += [0]
                else:
                    quasi_feat += [occur[tp][0]]
        return quasi_feat  

   


    def get_quasi_fragmental_desp_ext2a(self, bins = None):
            """
            Compute extended quasi fragmental descriptors.
            Parameters:
                bins - distance bins for extracting qf descriptors
            """
            if bins is not None:
                self.contact_bins = bins
            else:
                self.contact_bins = [(0, 12)]

            # list atom types ----------------------------------------------------
            pro_pool = [6, 7, 8, 16]
            lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
            # get list of atom-pair types ----------------------------------------
            quasi_type = list(itertools.product(pro_pool, lig_pool))        
            quasi_feat = []
            debug = defaultdict(int)
            for cbin in self.contact_bins:
                occur = {}
                for tp in quasi_type:
                    occur[tp] = [0, [],[]]
                # check the contacts one by one ----------------------------------
                contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
                conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
                distances = [self.pd[i, j] for (i, j) in conts]
                for ind in range(len(conts)):
                    cont = conts[ind]
                    cur_dist = distances[ind]
                    atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                    atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                    atm1_an = atm1.GetAtomicNum()
                    atm2_an = atm2.GetAtomicNum()
                    
                    tmp = (atm1_an, atm2_an)
                    if tmp in quasi_type:
                        occur[tmp][0] += 1
                        env1 = self.protein_env.get(cont[0], 1)
                        env2 = self.ligand_env.get(cont[1], 1)
                        debug[(env1,env2)] += 1
                        occur[tmp][1] += [ cur_dist]
                        occur[tmp][2] += [ 1/env1 + 1/env2 ]
                
                for tp in quasi_type:
                    if occur[tp][0] == 0:
                        quasi_feat += [0, 0]
                    else:
                        quasi_feat += [occur[tp][0], mean(occur[tp][2])]
            return quasi_feat  

        

    def get_quasi_fragmental_desp_ext2b(self, bins = None):
            """
            Compute extended quasi fragmental descriptors.
            Parameters:
                bins - distance bins for extracting qf descriptors
            """
            if bins is not None:
                self.contact_bins = bins
            else:
                self.contact_bins = [(0, 12)]

            # list atom types ----------------------------------------------------
            pro_pool = [6, 7, 8, 16]
            lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
            # get list of atom-pair types ----------------------------------------
            quasi_type = list(itertools.product(pro_pool, lig_pool))        
            quasi_feat = []
            debug = defaultdict(int)
            for cbin in self.contact_bins:
                occur = {}
                for tp in quasi_type:
                    occur[tp] = [0, [],[]]
                # check the contacts one by one ----------------------------------
                contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
                conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
                distances = [self.pd[i, j] for (i, j) in conts]
                for ind in range(len(conts)):
                    cont = conts[ind]
                    cur_dist = distances[ind]
                    atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                    atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                    atm1_an = atm1.GetAtomicNum()
                    atm2_an = atm2.GetAtomicNum()
                    
                    tmp = (atm1_an, atm2_an)
                    if tmp in quasi_type:
                        occur[tmp][0] += 1
                        env1 = self.protein_env.get(cont[0], 1)
                        env2 = self.ligand_env.get(cont[1], 1)
                        debug[(env1,env2)] += 1
                        occur[tmp][1] += [ ( 1/env1 ) * cur_dist]
                        occur[tmp][2] += [( 1/env1 ) ]
                
                for tp in quasi_type:
                    if occur[tp][0] == 0:
                        quasi_feat += [0, 0]
                    else:
                        quasi_feat += [occur[tp][0], sum(occur[tp][1])/ sum(occur[tp][2])]

            return quasi_feat  

    def get_quasi_fragmental_desp_ext2c(self, bins = None):
            """
            Compute extended quasi fragmental descriptors.
            Parameters:
                bins - distance bins for extracting qf descriptors
            """
            if bins is not None:
                self.contact_bins = bins
            else:
                self.contact_bins = [(0, 12)]

            # list atom types ----------------------------------------------------
            pro_pool = [6, 7, 8, 16]
            lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
            # get list of atom-pair types ----------------------------------------
            quasi_type = list(itertools.product(pro_pool, lig_pool))        
            quasi_feat = []
            debug = defaultdict(int)
            for cbin in self.contact_bins:
                occur = {}
                for tp in quasi_type:
                    occur[tp] = [0, [],[]]
                # check the contacts one by one ----------------------------------
                contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
                conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
                distances = [self.pd[i, j] for (i, j) in conts]
                for ind in range(len(conts)):
                    cont = conts[ind]
                    cur_dist = distances[ind]
                    atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                    atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                    atm1_an = atm1.GetAtomicNum()
                    atm2_an = atm2.GetAtomicNum()
                    
                    tmp = (atm1_an, atm2_an)
                    if tmp in quasi_type:
                        occur[tmp][0] += 1
                        env1 = self.protein_env.get(cont[0], 1)
                        env2 = self.ligand_env.get(cont[1], 1)
                        debug[(env1,env2)] += 1
                        occur[tmp][1] += [ cur_dist]
                        occur[tmp][2] += [( 1/env1 + 1/env2) ]
                
                for tp in quasi_type:
                    if occur[tp][0] == 0:
                        quasi_feat += [0, 0,0]
                    else:
                        quasi_feat += [occur[tp][0], mean(occur[tp][1]),mean(occur[tp][2])]

            return quasi_feat  

    def get_quasi_fragmental_desp_ext2d(self, bins = None):
            """
            Compute extended quasi fragmental descriptors.
            Parameters:
                bins - distance bins for extracting qf descriptors
            """
            if bins is not None:
                self.contact_bins = bins
            else:
                self.contact_bins = [(0, 12)]

            # list atom types ----------------------------------------------------
            pro_pool = [6, 7, 8, 16]
            lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
            # get list of atom-pair types ----------------------------------------
            quasi_type = list(itertools.product(pro_pool, lig_pool))        
            quasi_feat = []
            debug = defaultdict(int)
            for cbin in self.contact_bins:
                occur = {}
                for tp in quasi_type:
                    occur[tp] = [0, [], [], []]
                # check the contacts one by one ----------------------------------
                contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
                conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
                distances = [self.pd[i, j] for (i, j) in conts]
                for ind in range(len(conts)):
                    cont = conts[ind]
                    cur_dist = distances[ind]
                    atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                    atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                    atm1_an = atm1.GetAtomicNum()
                    atm2_an = atm2.GetAtomicNum()
                    
                    tmp = (atm1_an, atm2_an)
                    if tmp in quasi_type:
                        occur[tmp][0] += 1
                        env1 = self.protein_env.get(cont[0], 1)
                        env2 = self.ligand_env.get(cont[1], 1)
                        debug[(env1,env2)] += 1
                        occur[tmp][1] += [cur_dist]
                        occur[tmp][2] += [ 1/env1 ]
                        occur[tmp][3] += [ env2 ]
                
                for tp in quasi_type:
                    if occur[tp][0] == 0:
                        quasi_feat += [0,0,0]
                    else:
                        quasi_feat += [occur[tp][0], mean(occur[tp][1]),mean(occur[tp][2])]

            return quasi_feat  

    def get_quasi_fragmental_desp_ext3(self, bins = None):
            """
            Compute extended quasi fragmental descriptors.
            Parameters:
                bins - distance bins for extracting qf descriptors
            """
            if bins is not None:
                self.contact_bins = bins
            else:
                self.contact_bins = [(0, 12)]

            # list atom types ----------------------------------------------------
            pro_pool = [6, 7, 8, 16]
            lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
            # get list of atom-pair types ----------------------------------------
            quasi_type = list(itertools.product(pro_pool, lig_pool))        
            quasi_feat = []
            debug = defaultdict(int)
            for cbin in self.contact_bins:
                occur = {}
                for tp in quasi_type:
                    occur[tp] = [0, [], [], []]
                # check the contacts one by one ----------------------------------
                contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
                conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
                distances = [self.pd[i, j] for (i, j) in conts]
                for ind in range(len(conts)):
                    cont = conts[ind]
                    cur_dist = distances[ind]
                    atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                    atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                    atm1_an = atm1.GetAtomicNum()
                    atm2_an = atm2.GetAtomicNum()
                    
                    tmp = (atm1_an, atm2_an)
                    if tmp in quasi_type:
                        occur[tmp][0] += 1
                        env1 = self.protein_env.get(cont[0], 1)
                        env2 = self.ligand_env.get(cont[1], 1)
                        debug[(env1,env2)] += 1
                        occur[tmp][1] += [cur_dist]
                        occur[tmp][2] += [ 1/env1 ]
                        occur[tmp][3] += [ env2 ]
                
                for tp in quasi_type:
                    if occur[tp][0] == 0:
                        quasi_feat += [0,0,0,0]
                    else:
                        mean_dis = mean(occur[tp][2])
                        dists = occur[tp][1]
                        vals = occur[tp][2]

                        count1 = dist1 = 0
                        count2 = dist2 = 0

                        for d, v in zip(dists, vals):
                            if v <= mean_dis:
                                count1 += 1
                                dist1 += d
                            else:
                                count2 += 1
                                dist2 += d

                        if count1 == 0:
                            if count2 == 0:
                                quasi_feat += [0, 0, 0, 0]
                            else:
                                quasi_feat += [0, 0, count2, dist2 / count2]
                        else:
                            if count2 == 0:
                                quasi_feat += [count1, dist1 / count1, 0, 0]
                            else:
                                quasi_feat += [count1, dist1 / count1, count2, dist2 / count2]


            return quasi_feat  
    
    def get_quasi_fragmental_desp_ext4(self, bins = None):
            """
            Compute extended quasi fragmental descriptors.
            Parameters:
                bins - distance bins for extracting qf descriptors
            """
            if bins is not None:
                self.contact_bins = bins
            else:
                self.contact_bins = [(0, 12)]

            # list atom types ----------------------------------------------------
            pro_pool = [6, 7, 8, 16]
            lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
            # get list of atom-pair types ----------------------------------------
            quasi_type = list(itertools.product(pro_pool, lig_pool))        
            quasi_feat = []
            debug = defaultdict(int)
            for cbin in self.contact_bins:
                occur = {}
                for tp in quasi_type:
                    occur[tp] = [0, [], [], []]
                # check the contacts one by one ----------------------------------
                contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
                conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
                distances = [self.pd[i, j] for (i, j) in conts]
                for ind in range(len(conts)):
                    cont = conts[ind]
                    cur_dist = distances[ind]
                    atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                    atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                    atm1_an = atm1.GetAtomicNum()
                    atm2_an = atm2.GetAtomicNum()
                    
                    tmp = (atm1_an, atm2_an)
                    if tmp in quasi_type:
                        occur[tmp][0] += 1
                        env1 = self.protein_env.get(cont[0], 1)
                        env2 = self.ligand_env.get(cont[1], 1)
                        debug[(env1,env2)] += 1
                        occur[tmp][1] += [cur_dist]
                        occur[tmp][2] += [ 1/env1 ]
                        occur[tmp][3] += [ env2 ]
                
                for tp in quasi_type:
                    if occur[tp][0] == 0:
                        quasi_feat += [0,0,0,0,0,0]
                    else:
                        mean_dis = mean(occur[tp][2])
                        dists = occur[tp][1]
                        vals = occur[tp][2]
                        ev = occur[tp][3]

                        count1 = dist1 = en1 = 0
                        count2 = dist2 = en2 = 0

                        for d, v,e in zip(dists, vals,ev):
                            if v <= mean_dis:
                                count1 += 1
                                dist1 += d
                                en1 += e
                            else:
                                count2 += 1
                                dist2 += d
                                en2 += e

                        if count1 == 0:
                            if count2 == 0:
                                quasi_feat += [0, 0, 0, 0,0,0]
                            else:
                                quasi_feat += [0, 0,0, count2, dist2 / count2,en2/count2]
                        else:
                            if count2 == 0:
                                quasi_feat += [count1, dist1 / count1,en1/count1, 0, 0,0]
                            else:
                                quasi_feat += [count1, dist1 / count1,en1/count1, count2, dist2 / count2,en2/count2]


            return quasi_feat  
    
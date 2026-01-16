from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdchem import BondType

import os
import numpy as np
import pandas as pd
import torch
# import torch_geometric
from torch_geometric.data import Data
from dgllife.utils import *

import re

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    8 features are canonical, 2 features are from OGB
    """
    def atom_gasteiger_charge(atom):
        # Directly read pre-calculated charge values
        try:
            return [atom.GetDoubleProp("_GasteigerCharge")]
        except KeyError:
            return [0.0]  # Double check to prevent attribute missing

    def atom_h_bond_donor(atom):
        return [int(atom.GetTotalNumHs() > 0 and atom.GetAtomicNum() in [7, 8])]
    
    def atom_h_bond_acceptor(atom):
        return [int(atom.GetAtomicNum() in [7, 8])]
    
    featurizer_funcs = ConcatFeaturizer([
        atom_type_one_hot,                # Atom type (Required)
        atom_implicit_valence_one_hot,    # Implicit valence
        atom_formal_charge,               # Formal charge
        atom_hybridization_one_hot,       # Hybridization type
        atom_is_aromatic,                 # Aromaticity
        lambda a: [a.GetTotalNumHs()],    # Total number of H (scalar)
        atom_is_in_ring,                  # Atom in ring
        atom_chirality_type_one_hot,      # Chirality type
        atom_gasteiger_charge,            # Gasteiger charge
        atom_h_bond_donor,                # Hydrogen bond donor
        atom_h_bond_acceptor              # Hydrogen bond acceptor
    ])
    atom_feature = featurizer_funcs(atom)
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    # Define atom electronegativity dictionary
    electronegativity_dict = {
        # Main group elements
        1: 2.20,   # H (Hydrogen)
        2: 3.00,   # He (Helium, noble gas, generally not found in organic molecules)
        3: 0.98,   # Li (Lithium)
        4: 1.57,   # Be (Beryllium)
        5: 2.04,   # B (Boron)
        6: 2.55,   # C (Carbon)
        7: 3.04,   # N (Nitrogen)
        8: 3.44,   # O (Oxygen)
        9: 4.00,   # F (Fluorine)
        10: 4.00,  # Ne (Neon)
        11: 0.93,  # Na (Sodium)
        12: 1.31,  # Mg (Magnesium)
        13: 1.61,  # Al (Aluminum)
        14: 1.90,  # Si (Silicon)
        15: 2.19,  # P (Phosphorus)
        16: 2.58,  # S (Sulfur)
        17: 3.16,  # Cl (Chlorine)
        
        # Common organic elements in Period 4
        19: 0.82,  # K (Potassium)
        20: 1.00,  # Ca (Calcium)
        32: 2.01,  # Ge (Germanium)
        33: 2.18,  # As (Arsenic)
        34: 2.55,  # Se (Selenium)
        35: 2.96,  # Br (Bromine)
        53: 2.66,  # I (Iodine)
        
        # Transition metals (common in complexes)
        26: 1.83,  # Fe (Iron)
        28: 1.91,  # Ni (Nickel)
        29: 1.90,  # Cu (Copper)
        30: 1.65,  # Zn (Zinc)
        47: 1.93,  # Ag (Silver)
        78: 2.28,  # Pt (Platinum)
        79: 2.54,  # Au (Gold)
        
        # Other possible elements
        38: 0.95,  # Sr (Strontium)
        56: 0.89,  # Ba (Barium)
        80: 2.00   # Hg (Mercury)
    }
    
    def bond_polarity(bond):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        en1 = electronegativity_dict.get(a1.GetAtomicNum(), 2.0)
        en2 = electronegativity_dict.get(a2.GetAtomicNum(), 2.0)
        return [en1 - en2]
    
    def is_rotatable_bond(bond):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        is_sp3 = lambda a: a.GetHybridization() == Chem.HybridizationType.SP3
        return [int(not bond.IsInRing() and 
                    bond.GetBondType() == BondType.SINGLE and 
                    is_sp3(a1) and is_sp3(a2))]
    
    featurizer_funcs = ConcatFeaturizer([
        bond_type_one_hot,                # Bond type
        bond_is_conjugated,               # Conjugated
        bond_is_in_ring,                  # Bond in ring
        bond_stereo_one_hot,              # Stereochemistry (combined directionality)
        bond_polarity,                    # Polarity
        is_rotatable_bond                 # Rotatability
    ])
    bond_feature = featurizer_funcs(bond)

    return bond_feature

# Choose to use CanonicalSMILES
def smiles2graph(mol):
    """
    Converts SMILES string or rdkit's mol object to graph Data object without remove salt
    :input: SMILES string (str) or rdkit's mol object
    :return: graph object
    """
    # Handle case where input is a SMILES string
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            raise ValueError("Invalid SMILES string")

    # Unify Gasteiger charge calculation
    try:
        Chem.SanitizeMol(mol)  # Steps that might raise exceptions are checked separately
    except Exception as e:
        # Sanitization failure indicates invalid molecule, block subsequent process
        raise ValueError(f"Invalid molecule, cannot sanitize: {str(e)}") from e
    
    # Only executed if sanitization succeeds
    ComputeGasteigerCharges(mol)  # Subsequent operations

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = torch.tensor(atom_features_list, dtype=torch.float)  # Create tensor directly from list

    # bonds
    # num_bond_features = 14  # bond type, bond stereo, is_conjugated
    dummy_bond = Chem.MolFromSmiles("C-C").GetBondWithIdx(0)
    num_bond_features = len(bond_to_feature_vector(dummy_bond))
    # print(num_bond_features)
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # Create tensor directly from list
        edge_index = torch.tensor(edges_list, dtype=torch.long).T  # COO format [2, num_edges]
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)  # [num_edges, num_edge_features]

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph

def normalize_name(name):
    """Standardize column names: uppercase and remove spaces, -, _, /, etc."""
    normalized = name.upper()
    normalized = re.sub(r'[-_()\s/]+', '', normalized)
    return normalized

def get_drug_graph_ndrug(smiles_file, select_drug, drug_smiles):
    """
    1. Reads the CSV file.
    2. Converts ALL drugs in the CSV to graphs and stores them in a dictionary.
    3. Converts the manually provided (select_drug, drug_smiles) to a graph and adds it to the dictionary.
    4. Returns the complete dictionary.
    """
    # Read CSV
    smiles_df = pd.read_csv(smiles_file, header=0)
    
    # Normalize names in the DataFrame
    smiles_df.iloc[:, 0] = smiles_df.iloc[:, 0].apply(normalize_name)
    
    drug_dict = {}
    
    # 1. Process all drugs from the CSV
    # print("Processing existing drugs from CSV...")
    for i in range(len(smiles_df)):
        try:
            name = smiles_df.iloc[i, 0]
            # Assuming 'isosmiles' is the column name for SMILES in the CSV
            smi = smiles_df.loc[i, 'isosmiles']
            drug_dict[name] = smiles2graph(smi)
        except Exception as e:
            print(f"Warning: Failed to process CSV drug at row {i} ({name}). Error: {e}")

    # 2. Process the specific input drug
    try:
        # print(f"Processing input drug: {select_drug}")
        drug_dict[select_drug] = smiles2graph(drug_smiles)
    except Exception as e:
        print(f"Error processing input drug {select_drug} with SMILES {drug_smiles}. Error: {e}")
        
    return drug_dict
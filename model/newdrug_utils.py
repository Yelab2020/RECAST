# This code is adapted from the TGSA repository:
# https://github.com/violet-sto/TGSA

import os
import csv
import scipy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy

from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import scanpy as sc

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold

# 1. Store standardized highly variable gene expression info & gene names
def get_highvar_gene(exp_path, des_path='./', num=3000):
    # Ensure C layout when reading data
    exp = pd.read_csv(exp_path, index_col=0)
    
    # Explicitly specify layout when creating AnnData object
    bulk_adata = sc.AnnData(
        X = exp.values.copy(order='C'),  # Explicitly specify C layout
        obs = pd.DataFrame({'cell_id': list(exp.index)}, index=exp.index),
        var = pd.DataFrame({'gene_id': list(exp.columns)}, index=exp.columns),
        dtype=np.float32  # Use single precision float
    )
    
    # Highly variable gene selection
    sc.pp.highly_variable_genes(
        bulk_adata,
        flavor="seurat",
        n_top_genes=num,
        subset=True,
        inplace=True  # Modify adata object in-place
    )
    
    # Get selected highly variable genes
    columns_chosen = bulk_adata.var_names
    cell_names = bulk_adata.obs_names
    
    # Get highly variable gene expression matrix (ensure C layout)
    highvar_exp = bulk_adata.X.copy(order='C')  # Explicitly copy as C layout
    
    # Standardization
    scaler = StandardScaler()
    highvar_exp = scaler.fit_transform(highvar_exp).astype(np.float32)
    
    # Missing value imputation
    imp_mean = SimpleImputer()
    highvar_exp = imp_mean.fit_transform(highvar_exp)
    
    # Ensure C layout when creating DataFrame
    highvar_exp = pd.DataFrame(
        highvar_exp, 
        index=cell_names, 
        columns=columns_chosen,
        dtype=np.float32  # Maintain single precision
    )
    
    # Build cell dictionary (ensure tensor uses C layout)
    cell_dict = {}
    for i in cell_names:
        # Ensure C layout and convert to PyTorch tensor
        cell_data = highvar_exp.loc[i].values.copy(order='C')
        cell_tensor = torch.tensor(cell_data, dtype=torch.float32).view(-1, 1)
        cell_dict[i] = Data(x=cell_tensor)
    
    # Save results (use C layout)
    np.save(os.path.join(des_path, 'cell_exp_std_genename.npy'), columns_chosen.values)
    np.save(os.path.join(des_path, 'cell_exp_std.npy'), cell_dict)
    
    return cell_dict, columns_chosen.values

# 2. Get STRING-based gene network
def ensp_to_hugo_map():
    with open('./data/9606.protein.info.v12.0.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        ensp_map = {row[0]: row[1] for row in csv_reader if row[0] != ""}

    return ensp_map

def get_STRING_graph(gene_name, drug_name, des_path='./', thresh=0.95, num = 500):
    save_path = os.path.join(des_path, 'edge_index_PPI.npy')

    if not os.path.exists(save_path):
        # gene_list
        gene_list = gene_name.tolist()

        # load STRING
        ensp_map = ensp_to_hugo_map()
        edges = pd.read_csv('./data/9606.protein.links.detailed.v12.0.txt', sep=' ')

        # edge_index
        selected_edges = edges['combined_score'] > (thresh * 1000)
        edge_list = edges[selected_edges][["protein1", "protein2"]].values.tolist()

        edge_list = [[ensp_map[edge[0]], ensp_map[edge[1]]] for edge in edge_list if
                     edge[0] in ensp_map.keys() and edge[1] in ensp_map.keys()]


        edge_index = []
        for i in edge_list:
            if (i[0] in gene_list) & (i[1] in gene_list):
                edge_index.append((gene_list.index(i[0]), gene_list.index(i[1])))
                edge_index.append((gene_list.index(i[1]), gene_list.index(i[0])))
        edge_index = list(set(edge_index))
        edge_index = np.array(edge_index, dtype=np.int64).T

        np.save(save_path, edge_index)
    else:
        edge_index = np.load(save_path)

    return edge_index

# 3. Data loading
class MyDataset(Dataset):
    def __init__(self, drug_dict, cell_dict, IC, edge_index):
        super(MyDataset, self).__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)
        
        # Ensure C layout data structures
        self.drug_name = IC['Drug'].values.copy(order='C')
        self.Cell_line_name = IC['Cell_line'].values.copy(order='C')
        self.response = IC['Response'].values.astype(np.int64).copy(order='C')
        
        # Ensure edge_index is C layout
        self.edge_index = torch.tensor(
            edge_index.copy(order='C'), 
            dtype=torch.long
        ).contiguous()
        
        # Preprocess cell data (minimal modification)
        self.preprocessed_cell = {}
        for cell_line in np.unique(self.Cell_line_name):
            # Process only actually used cell lines
            cell_data = cell_dict[cell_line]
            # Create preprocessed copy
            self.preprocessed_cell[cell_line] = Data(
                x=cell_data.x.clone().to(torch.float32),
                edge_index=self.edge_index.clone()
            )
        
    def __len__(self):
        return len(self.response)

    def __getitem__(self, index):
        # Directly get preprocessed cell data
        cell_data = self.preprocessed_cell[self.Cell_line_name[index]]
        
        # Get drug data and ensure type
        drug_data = self.drug[self.drug_name[index]]
        # drug_data.x = drug_data.x.to(torch.float32)
        
        return drug_data, cell_data, self.response[index]

def _collate(samples):
    drugs, cells, labels = map(list, zip(*samples))

    batched_drug = Batch.from_data_list(drugs)
    batched_cell = Batch.from_data_list(cells)

    # Modification: Label uses long type (integer)
    labels_tensor = torch.tensor(labels, dtype=torch.long).contiguous()
    
    # Note: edge_index does not need conversion as it is already integer type
    return batched_drug, batched_cell, labels_tensor

def load_data(IC, drug_dict, cell_dict, edge_index, batch_size, seed, test_drug):
    # Ensure input data is fully sorted
    IC = IC.sort_values(by=['Cell_line', 'Drug']).reset_index(drop=True)
    
    # Separate test set
    test_set = pd.DataFrame({
        'Cell_line': IC['Cell_line'].unique(),
        'Drug': test_drug,
        'Response': 1 # 0 or 1, arbitrary placeholder not used later
    })
    test_set = test_set.sort_values(by=['Cell_line', 'Drug']).reset_index(drop=True)
    
    # Use indices instead of DataFrame for splitting
    indices = np.arange(len(IC))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.1,
        random_state=seed,
        stratify=IC['Cell_line'].values
    )
    
    train_set = IC.iloc[train_idx].reset_index(drop=True)
    val_set = IC.iloc[val_idx].reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)
    
    # 4. Create datasets
    train_dataset = MyDataset(drug_dict, cell_dict, train_set, edge_index)
    val_dataset = MyDataset(drug_dict, cell_dict, val_set, edge_index)
    test_dataset = MyDataset(drug_dict, cell_dict, test_set, edge_index)
    
    # Modify DataLoader creation part
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Important: Must disable shuffle
        collate_fn=_collate, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=_collate, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=len(test_dataset), 
        shuffle=False, 
        collate_fn=_collate, 
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader
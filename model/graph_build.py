### Create Graph Neural Network Edges

import numpy as np
import pandas as pd
import os
import csv
import scipy
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
# from torch_geometric.nn import graclus, max_pool
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import scipy.sparse as sp

# Graph computation for cell line data
def get_bulk_genes_graph(bulk_exp, save_path, method='pearson', thresh=0.90):
    """
    """

    # Calculate correlation matrix (column-wise)
    if method == 'pearson':
        genes_exp_corr = np.corrcoef(bulk_exp.T, rowvar=False)
    elif method == 'spearman':
        genes_exp_corr, _ = spearmanr(bulk_exp, axis=1)
    else:
        raise ValueError("Invalid correlation method. Use 'pearson' or 'spearman'.")
    
    genes_exp_corr = np.abs(genes_exp_corr)
    
    adj = np.where(genes_exp_corr > thresh, 1, 0)

    # Adjacency matrix excludes self-loops
    adj = adj - np.eye(genes_exp_corr.shape[0], dtype=int)
    print("links:", adj.sum()) # Final number of links

    edge_index = np.nonzero(adj)
    
    if save_path is not None:
        final_path = os.path.join(save_path, 'bulk_edge_index_{}_{}.npy').format(method, thresh)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if not os.path.exists(final_path):
            np.save(final_path, edge_index)

    return edge_index


# Graph computation for spatial data
def get_spatial_genes_graph(spatial_exp, coordinate, save_path, name='spatial', method='pearson', thresh=0.3, k=19):
    #-----1. Gene Graph-----#
    if method == 'pearson':
        genes_exp_corr = np.corrcoef(spatial_exp.T, rowvar=False)
    elif method == 'spearman':
        genes_exp_corr, _ = spearmanr(spatial_exp, axis=1)
    else:
        raise ValueError("Invalid correlation method. Use 'pearson' or 'spearman'.")

    adj_exp = np.where(genes_exp_corr > thresh, 1, 0)
    # Create identity matrix to remove self-loops
    adj_exp = adj_exp - np.eye(genes_exp_corr.shape[0], dtype=int)
    edge_index_exp = np.nonzero(adj_exp)

    #-----2. Proximity Graph-----#
    df = coordinate[["imagerow", "imagecol"]]

    closest = NearestNeighbors(n_neighbors=k).fit(df).kneighbors_graph(df).toarray()

    adj_k = closest - np.eye(closest.shape[0], dtype=int)
    edge_index_k = np.nonzero(adj_k)

    #-----3. Intersection of Two Graphs-----#
    # Expression graph
    transposed_arrays_exp = np.column_stack(edge_index_exp)
    result_exp = [tuple(row) for row in transposed_arrays_exp.tolist()]

    # Coordinate graph
    transposed_arrays_k = np.column_stack(edge_index_k)
    result_k = [tuple(row) for row in transposed_arrays_k.tolist()]

    # Convert tuples to sets
    set_exp = set(result_exp)
    set_k = set(result_k)

    # Calculate intersection
    intersection = set_exp.intersection(set_k)

    # Get final graph
    tmp = np.array(list(intersection)).T
    edge_index = (tmp[0,:], tmp[1,:])
    print("link:", len(edge_index[0]))

    save_path = os.path.join(save_path, '2.graph')
    final_path = os.path.join(save_path, '{}_edge_index_{}_{}_{}.npy').format(name, method, thresh, k)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(final_path):
            np.save(final_path, edge_index)

    return edge_index

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj) # Convert to SciPy COO sparse matrix format
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt) # D^-1/2 * A * D^-1/2
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return adj_normalized 

def get_spatial_graph(coordinate, 
                      choose,
                      save_path=None,
                      name='spatial', 
                      normalize = False,
                      bidirection = False,
                      k=3):
    
    df = coordinate[["x", "y"]]
    closest = NearestNeighbors(n_neighbors=k+1).fit(df).kneighbors_graph(df).toarray()
    adj_k = closest - np.eye(closest.shape[0], dtype=int)
    
    adj_k = adj_k[np.ix_(choose, choose)]

    if bidirection:
        adj = adj_k + adj_k.T
        adj = np.where(adj > 1, 1, adj)
    else:
        adj = adj_k

    if normalize:
        adj_normalized = preprocess_adj(adj)
        adj_final = np.nonzero(adj_normalized)
    else:
        adj_final = np.nonzero(adj)
    
    # Save
    if save_path is not None:
        save_path = os.path.join(save_path, '2.graph')
        if not os.path.exists(save_path):
            os.makedirs(save_path) 

        if bidirection:
            final_path = os.path.join(save_path, '{}_edge_index_{}_bidirection.npy').format(name, k)
        else:
            final_path = os.path.join(save_path, '{}_edge_index_{}_onedirection.npy').format(name, k)

        if not os.path.exists(final_path):
            np.save(final_path, adj_final)

    return adj_final
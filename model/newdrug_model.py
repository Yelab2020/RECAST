# This code is adapted from the TGSA repository:
# https://github.com/violet-sto/TGSA

import os
import random
 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime

import torch_geometric
from torch_geometric.nn import GINConv, GINEConv, JumpingKnowledge, global_max_pool, global_mean_pool, GATConv, GATv2Conv, max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool  
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from scipy.stats import pearsonr

from tqdm import tqdm

import pickle, io, json, sys

# 1. Classify genes based on the gene interaction graph obtained from STRING, and pool the classification results in each round
def get_predefine_cluster(edge_index, num_features, net_layer, save_path, thresh, device):
    save_path = os.path.join(save_path, 'cluster_predefine_PPI.npy')
    if not os.path.exists(save_path):
        g = Data(edge_index=torch.tensor(edge_index, dtype=torch.long), x=torch.zeros(num_features, 1))
        g = Batch.from_data_list([g])
        cluster_predefine = {}
        for i in range(net_layer):
            cluster = graclus(g.edge_index, None, g.x.size(0))
            print(len(cluster.unique()))
            g = max_pool(cluster, g, transform=None)
            cluster_predefine[i] = cluster
        np.save(save_path, cluster_predefine)
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}
    else:
        cluster_predefine = np.load(save_path, allow_pickle=True).item()
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}

    return cluster_predefine


# 2. GNN_cell
class GNN_cell(torch.nn.Module):
    def __init__(self, num_feature, layer_cell, dim_cell, cluster_predefine):
        super().__init__()
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.cluster_predefine = cluster_predefine
        # self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())
        self.final_node = len(self.cluster_predefine[0])
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()

        for i in range(self.layer_cell):
            if i:
                conv = GATv2Conv(self.dim_cell, self.dim_cell)
            else:
                conv = GATv2Conv(num_feature, self.dim_cell)

            # affine=False means Batch Normalization only normalizes means and variances without further linear transformation
            bn = torch.nn.BatchNorm1d(self.dim_cell)  

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

    def forward(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            # num_graphs is essentially the batch size
            # num_node is the number of nodes (genes) per graph. Since cell data is loaded via DataLoader as a stack of all graphs in the batch, divide by batch size to get node count per graph
            num_node = int(cell.x.size(0) / cell.num_graphs) 
            
            cell.x = self.bns_cell[i](cell.x)

        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        return node_representation # Each row represents the collection of all features for each graph. The result is a matrix of size batch * all_features (=self.final_node * self.dim_cell)
                                   # Because rows (genes) are essentially merged during multi-round pooling

# 3. GNN_drug
class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug, drugFunc):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.drugFunc = drugFunc  # Save drugFunc for subsequent checks

        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(64, self.dim_drug))
            
            if drugFunc == "GIN":
                conv = GINConv(block)
            elif drugFunc == "GINE":
                conv = GINEConv(block, edge_dim=14) 
                
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)
    
    def forward(self, drug):
        x, edge_index, edge_attr, batch = drug.x, drug.edge_index, drug.edge_attr, drug.batch
        # edge_attr = edge_attr.to(torch.float32)
        x_drug_list = []
        for i in range(self.layer_drug):
            conv = self.convs_drug[i]
            # Determine whether to pass edge_attr based on convolution type
            
            if self.drugFunc == "GINE":
                x = conv(x, edge_index, edge_attr)
            else:  # E.g., GIN or other types that do not require edge_attr
                x = conv(x, edge_index)
            x = F.relu(x)
            x = self.bns_drug[i](x)
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list) # Concatenate column-wise
        x_drug = global_mean_pool(node_representation, batch) 
        return x_drug
        
# 4. ndrug
class ndrug(nn.Module):
    def __init__(self, cluster_predefine, layer_drug, drugFunc, dim_drug, num_feature, layer, hidden_dim, dropout_ratio):
        super().__init__()
        self.layer_cell = layer
        self.dim_cell = hidden_dim
        self.dropout_ratio = dropout_ratio
        
        # drug graph branch
        self.GNN_drug = GNN_drug(layer_drug, dim_drug, drugFunc)

        self.drug_emb = nn.Sequential(
            nn.Linear(dim_drug * layer_drug, 256),
            nn.ReLU(),
            # nn.Dropout(p=self.dropout_ratio),
        )

        # cell graph branch
        self.GNN_cell = GNN_cell(num_feature, self.layer_cell, self.dim_cell, cluster_predefine)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.GNN_cell.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )

        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 2)
        )

    def forward(self, drug, cell):
        # forward drug
        x_drug = self.GNN_drug(drug)
        x_drug = self.drug_emb(x_drug)

        # forward cell
        x_cell = self.GNN_cell(cell)
        x_cell = self.cell_emb(x_cell)

        # combine drug feature and cell line feature
        x = torch.cat([x_drug, x_cell], -1) 
        x = self.regression(x)

        return x

# 5. Early stopping settings
class EarlyStopping():

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            # print(
            #     f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict()}, os.path.join('results', self.filename))

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(os.path.join('results', self.filename))['model_state_dict'])

# 6. Validation
def validate(model, train_loader, val_loader, test_loader, device):
    model.eval()

    # Use pre-allocated arrays instead of list appending
    def evaluate_loader(loader):
        total_loss = 0.0
        true_arr = np.empty(len(loader.dataset), dtype=np.int64)
        pred_labels = np.empty(len(loader.dataset), dtype=np.int64)
        pred_probs = np.empty(len(loader.dataset), dtype=np.float32)
        
        start_idx = 0
        with torch.no_grad():
            for data in loader:
                drug, cell, label = data
                
                # Batch data transfer to device
                if isinstance(cell, list):
                    drug, cell, label = drug.to(device), [feat.to(device) for feat in cell], label.to(device)
                else:
                    drug, cell, label = drug.to(device), cell.to(device), label.to(device)
                
                output = model(drug, cell)
                batch_size = label.size(0)
                
                # Accumulate loss
                total_loss += F.cross_entropy(output, label, reduction='sum').item()
                
                # Directly store predicted categories
                batch_pred = output.argmax(dim=1)
                
                end_idx = start_idx + batch_size
                true_arr[start_idx:end_idx] = label.cpu().numpy()
                pred_labels[start_idx:end_idx] = batch_pred.cpu().numpy()
                pred_probs[start_idx:end_idx] = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
                start_idx = end_idx
        
        return total_loss / len(loader.dataset), true_arr, pred_labels, pred_probs

    def test_output(loader):
        pred_labels = np.empty(len(loader.dataset), dtype=np.int64)
        pred_probs = np.empty(len(loader.dataset), dtype=np.float32)
        
        start_idx = 0
        with torch.no_grad():
            for data in loader:
                drug, cell, label = data
                
                # Batch data transfer to device
                if isinstance(cell, list):
                    drug, cell, label = drug.to(device), [feat.to(device) for feat in cell], label.to(device)
                else:
                    drug, cell, label = drug.to(device), cell.to(device), label.to(device)
                
                output = model(drug, cell)
                batch_size = label.size(0)
                
                # Directly store predicted categories
                batch_pred = output.argmax(dim=1)
                
                end_idx = start_idx + batch_size
                pred_labels[start_idx:end_idx] = batch_pred.cpu().numpy()
                pred_probs[start_idx:end_idx] = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
                start_idx = end_idx
        
        return pred_labels, pred_probs
    
    # Process all datasets at once
    train_mean_ce, train_true, train_pred, train_prob = evaluate_loader(train_loader)
    val_mean_ce, val_true, val_pred, val_prob = evaluate_loader(val_loader)
    test_pred, test_prob = test_output(test_loader) 

    # Calculate metrics
    def calc_metrics(true, pred, prob):
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average='binary', pos_label=1)
        auc = roc_auc_score(true, prob)
        ap = average_precision_score(true, prob)
        return acc, f1, auc, ap

    train_acc, train_f1, train_auc, train_ap = calc_metrics(train_true, train_pred, train_prob)
    val_acc, val_f1, val_auc, val_ap = calc_metrics(val_true, val_pred, val_prob)

    return (
        train_acc, train_f1, train_auc, train_ap, train_mean_ce,
        val_acc, val_f1, val_auc, val_ap, val_mean_ce,
        test_pred, test_prob
    )

# 7. Training
def train(model, loader, opt, device):
    model.train()
    
    # for idx, data in enumerate(tqdm(loader, desc='Iteration')):
    for idx, data in enumerate(loader):                
        drug, cell, label = data

        if isinstance(cell, list):
            drug, cell, label = drug.to(device), [feat.to(device) for feat in cell], label.to(device)
        else:
            drug, cell, label = drug.to(device), cell.to(device), label.to(device)

        output = model(drug, cell)
        
        loss = F.cross_entropy(output, label)  # Requires label to be long
        opt.zero_grad()
        loss.backward()
        
        opt.step() # Perform parameter update
        
    loss_value = loss.item()
    # print(f"Loss: {loss_value:.6f}")
    return loss
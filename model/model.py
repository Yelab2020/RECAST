import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import LayerNorm, TransformerConv, GATConv, GCNConv, GATv2Conv, GINConv, GINEConv, JumpingKnowledge, global_max_pool
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from torch.autograd import Function
import os
import numpy as np

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
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(64, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))

            if drugFunc == "GIN":
                conv = GINConv(block)
            elif drugFunc == "GINE":
                conv = GINEConv(block, edge_dim=14) 
                
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug):
        x, edge_index, edge_attr, batch = drug.x, drug.edge_index, drug.edge_attr, drug.batch
        edge_attr = edge_attr.to(torch.float32)
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

        node_representation = self.JK(x_drug_list)
        x_drug = global_max_pool(node_representation, batch) 
        return x_drug

class Feature_extractor(nn.Module):
    def __init__(self, in_dim=15962, num_hiddens=[512, 64], graphFunc=TransformerConv, 
                 dropout_rate=0.5, use_layer_norm=True, bulk_drop=0):
        super().__init__()
        self.num_layers = len(num_hiddens)
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropout_rate = dropout_rate
        
        # The input dimension of the first layer is in_dim
        self.convs.append(graphFunc(in_dim, num_hiddens[0]))
        
        self.dp = nn.Dropout(bulk_drop)
        
        # The input dimension of subsequent layers is the output dimension of the previous layer
        for i in range(1, self.num_layers):
            self.convs.append(graphFunc(num_hiddens[i-1], num_hiddens[i]))
            
        if use_layer_norm:
            for i in range(self.num_layers):
                self.layer_norms.append(LayerNorm(num_hiddens[i]))
        
        self.activate = F.elu

    def forward(self, x, edge_index, isdp = False):
        if isdp:
            x = self.dp(x)
            
        downstream = None
        for i, conv in enumerate(self.convs):
            # Apply convolution
            x = conv(x, edge_index)
            
            # Apply LayerNorm if specified
            if self.layer_norms:
                x = self.layer_norms[i](x)   
                
        # Apply activation except for the last iteration
        if i < len(self.convs) - 1:
            x = self.activate(x)
            
            # Apply Dropout except for the last iteration
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return x

class Classifier_Linear(nn.Module):
    def __init__(self, out_dim, num_hiddens=[512, 64, 32], dropout_rate=0.5, use_layer_norm=True):
        super().__init__()
        self.num_layers = len(num_hiddens)
        self.linear_classifier = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropout_rate = dropout_rate

        # Initialize the layers
        for i in range(1, self.num_layers):
            self.linear_classifier.append(nn.Linear(num_hiddens[i-1], num_hiddens[i]))
            
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(num_hiddens[i]))

        self.fc_out = nn.Linear(num_hiddens[-1], out_dim)
        self.activate = nn.ELU()

    def forward(self, x):
        for i, linear in enumerate(self.linear_classifier):
            x = linear(x)            
            
            if self.layer_norms:
                x = self.layer_norms[i](x)
                
            x = self.activate(x)

            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        output = self.fc_out(x)
        return output 

# Attention mechanism fusion
class FeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Input feature normalization
        self.expr_norm = nn.LayerNorm(feature_dim)
        self.drug_norm = nn.LayerNorm(feature_dim)
        self.final_norm = nn.LayerNorm(feature_dim)
        
        # Drug feature modulator
        self.drug_modulator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.ReLU()
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()  # Add non-linear activation
        )
        
        self.activation = nn.LeakyReLU()
                    
    def forward(self, expr_feat, drug_feat):
        """
        Input:
        expr_feat: (num_cells, feat_dim)
        drug_feat: (feat_dim,)
        Output:
        fused: (num_cells, feat_dim)
        """
        # Ensure drug features are 2D
        if drug_feat.dim() == 1:
            drug_feat = drug_feat.unsqueeze(0)  # (1, feat_dim)
            
        # Normalize input features
        expr_feat_n = self.expr_norm(expr_feat)
        drug_feat_n = self.drug_norm(drug_feat)
        
        # Generate modulation parameters
        gamma_beta = self.drug_modulator(drug_feat_n)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        
        # Apply feature modulation
        modulated = expr_feat_n * gamma + beta
            
        # Apply fusion layer
        fused = self.fusion(modulated)
        fused = self.final_norm(fused)
        fused = self.activation(fused)
        
        return fused
        
# Linear classification version
class main_classifier(nn.Module):
    def __init__(self, num_classes=2, in_dim=15962, 
                 hiddens_graph=[512,256,128], hiddens_linear=[128,64,32],
                 layer_drug=3,
                 graphFunc=TransformerConv, 
                 drugFunc = "GINE",
                 dropout_rate=0.5, use_layer_norm=True, bulk_drop = 0):
        super().__init__()
        self.graphNet = Feature_extractor(in_dim, num_hiddens=hiddens_graph, 
                                          graphFunc=graphFunc,
                                          dropout_rate=dropout_rate, 
                                          use_layer_norm=use_layer_norm, 
                                          bulk_drop = bulk_drop)
        
        self.drugNet = GNN_drug(layer_drug, hiddens_linear[0]//layer_drug, drugFunc)

        self.attentionFusion = FeatureFusion(hiddens_linear[0])
        
        self.classifier = Classifier_Linear(out_dim=num_classes, 
                                            num_hiddens=hiddens_linear, 
                                            dropout_rate=dropout_rate, 
                                            use_layer_norm=use_layer_norm)

        self.softmax = nn.Softmax(dim=1) 
        self.classes = num_classes

    def forward(self, src_x, src_edge, tar_x, tar_edge, drug, alpha=1):
        device = next(self.parameters()).device
        
        src_emb = self.graphNet(src_x, src_edge, isdp = True)
        tar_emb = self.graphNet(tar_x, tar_edge, isdp = False)        
        
        drug_emb = self.drugNet(drug)

        src_fuse = self.attentionFusion(src_emb, drug_emb)
        tar_fuse = self.attentionFusion(tar_emb, drug_emb)
        
        src_pred = self.classifier(src_fuse) 
        tar_pred = self.classifier(tar_fuse)
    
        tar_max_prob, tar_pred_loc = torch.max(F.softmax(tar_pred), dim=-1)

        src_mid_ident = src_fuse / (torch.norm(src_fuse, dim = -1).reshape(src_fuse.shape[0], 1))
        tar_mid_ident = tar_fuse / (torch.norm(tar_fuse, dim = -1).reshape(tar_fuse.shape[0], 1))

        src_mid_ident_gram = torch.clamp(src_mid_ident.mm(tar_mid_ident.transpose(dim0 = 1, dim1 = 0)),-0.99999999,0.99999999)
        tar_mid_ident_gram = torch.clamp(tar_mid_ident.mm(tar_mid_ident.transpose(dim0 = 1, dim1 = 0)),-0.99999999,0.99999999)

        tar_label = (torch.nn.functional.one_hot(tar_pred_loc, self.classes) - 1 / float(self.classes)).to(device)
        
        # Kernel method propagates label information from target samples to source samples
        tar_diag = 0.001 * torch.eye(tar_mid_ident_gram.shape[0]).to(device)
        src_label_tarpred = src_mid_ident_gram.mm(torch.inverse(tar_mid_ident_gram + tar_diag)).mm(tar_label)
        
        return src_pred, src_label_tarpred, tar_pred
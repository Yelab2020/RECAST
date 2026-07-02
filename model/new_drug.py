import os
import re
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np

# Assuming these modules are in the model package
import model.newdrug_utils
import model.newdrug_model

def normalize_name(name):
    """Helper function: Standardize name"""
    if not isinstance(name, str):
        return str(name)
    normalized = name.upper()
    normalized = re.sub(r'[-_()\s/]+', '', normalized)
    return normalized

def run_training(drug_dict, device, test_drug, current_logs, output_lines, savedir):
    """
    Encapsulated training function
    Args:
        drug_dict: Graph data
        device: torch device
        test_drug: Current drug name processed (normalized)
        args: Object containing parameters like gene_num, threshold, lr
        current_logs: List to store logs
    Returns:
        test_pred: Test set prediction results
    """

    # 1. Define parameter grid
    net_thresh = 0.95  # Network threshold parameter
    drugFunc = "GINE" # Drug function parameter
    dropout_ratio = 0.3  # Dropout ratio
    lr = 0.0001  # Learning rate
    weight_decay = 0.0001  # Weight decay

    # Fixed parameters
    features = 500
    net_layer = 3
    batch_size = 512
    layer_drug = 3
    dim_drug = 128
    layer = 3
    hidden_dim = 3
    epochs = 300


    bulk_exp_path = './data/bulk_data/train_geneexp.csv' # bulk RNA-Seq expression
    bulk_label_path = './data/bulk_data/IC50_binary_df.csv'

    output_lines.append("============================== parameter-step1 ==============================")
    
    # Write specific parameters line by line
    output_lines.append(f"drug_name: {test_drug}")
    output_lines.append(f"drugFunc: {drugFunc}")
    output_lines.append(f"net_thresh: {net_thresh}")
    output_lines.append(f"dropout_ratio: {dropout_ratio}")
    output_lines.append(f"lr: {lr}")
    output_lines.append(f"weight_decay: {weight_decay}")
    output_lines.append(f"gene_num: {features}")
    output_lines.append(f"layer_drug: {layer_drug}")    

    # Dictionary of expression for each cell line; Gene names [Reusable]
    cell_dict, genenanme = model.newdrug_utils.get_highvar_gene(bulk_exp_path, des_path='./preprocess_result', num=features)

    # Save to ./preprocess_result/edge_index_PPI.npy. [Reusable]
    edge_index = model.newdrug_utils.get_STRING_graph(genenanme, drug_name=test_drug, des_path='./preprocess_result', thresh=net_thresh, num=features)

    drug_sen = pd.read_csv(bulk_label_path, index_col=0)
    
    melted_drug_sen = drug_sen.melt(
        ignore_index=False,
        var_name='Drug',
        value_name='Response'
    ).reset_index()
    
    melted_drug_sen = melted_drug_sen.rename(columns={'index': 'Cell_line'})
    melted_drug_sen = melted_drug_sen.dropna(subset=['Response']).reset_index(drop=True)
    
    melted_drug_sen.loc[:, "Drug"] = melted_drug_sen.loc[:, "Drug"].apply(normalize_name)
    
    melted_drug_sen['Response'] = melted_drug_sen['Response'].map({'S': 1, 'R': 0})
    
    IC = melted_drug_sen
    
    # 4. Data loader
    train_loader, val_loader, test_loader = model.newdrug_utils.load_data(
        IC, drug_dict, cell_dict, edge_index,
        batch_size=batch_size, seed=114514, test_drug=test_drug
    )
    
    # 5. Pre-define clustering layer
    cluster_predefine = model.newdrug_model.get_predefine_cluster(
        edge_index, num_features=features, net_layer=net_layer,
        save_path='./preprocess_result', thresh=net_thresh, device=device
    )
    
    # Re-initialize model
    ndrug = model.newdrug_model.ndrug(
        cluster_predefine, 
        layer_drug=layer_drug, 
        drugFunc=drugFunc,
        dim_drug=dim_drug, 
        num_feature=1, 
        layer=layer, 
        hidden_dim=hidden_dim, 
        dropout_ratio=dropout_ratio
    ).to(device)
    
    def model_init(model):
        """Layer-wise parameter initialization strategy"""
        for name, module in model.named_modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                
    model_init(ndrug)
    
    # Optimizer
    opt = torch.optim.Adam(ndrug.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    
    # Model filename contains parameter information
    
    stopper = model.newdrug_model.EarlyStopping(mode='higher', patience=30, filename=os.path.join(savedir, f"{test_drug}_newdrug.pth"))
    scheduler = lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.8, patience=20, verbose=False, 
        threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-09
    )
    
    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = model.newdrug_model.train(ndrug, train_loader, opt, device)
        
        train_acc, train_f1, train_auc, train_ap, train_mean_ce, \
        val_acc, val_f1, val_auc, val_ap, val_mean_ce, \
        test_pred, test_prob  = model.newdrug_model.validate(
            ndrug, train_loader, val_loader, test_loader, device
        )
    
        # print(f"Epoch {epoch}, Current LR: {opt.param_groups[0]['lr']:.10f}", end='\r')
        scheduler.step(val_f1)
        
        early_stop = stopper.step(val_f1, ndrug)
        if early_stop:
            break

    ndrug = stopper.load_checkpoint()
    ndrug = ndrug.to(device)
    
    # Final validation
    train_acc, train_f1, train_auc, train_ap, train_mean_ce, \
    val_acc, val_f1, val_auc, val_ap, val_mean_ce, \
    test_pred, test_prob = model.newdrug_model.validate(
        ndrug, train_loader, val_loader, test_loader, device
    )

    output_lines.append("============================== result-step1 ==============================")
    
    # Write result metrics line by line    
    output_lines.append(f"Val_AUC: {val_auc:.4f}")
    output_lines.append(f"Val_AP: {val_ap:.4f}")
    output_lines.append(f"Val_Acc: {val_acc:.4f}")
    output_lines.append(f"Val_F1: {val_f1:.4f}")

    return test_pred
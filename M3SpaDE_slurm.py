import os

# Unified seed value
seed = 114514

# Environment variables setting
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import argparse
import re
import gc
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from typing import Sequence, List, Union
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import torch_geometric
from torch_geometric.data import Data

import model.data_preprocess
import model.graph_build
import model.smiles2graph
from model.vae_model import train_and_generate
from model.train_warmup import train_epoch
import model.new_drug

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch_geometric.seed_everything(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    torch.use_deterministic_algorithms(True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description="Drug Response Prediction Training Script")
    
    # --- Required Arguments ---
    parser.add_argument('--drug_name', type=str, required=True,
                        help='Name of the drug(s). For multiple drugs, separate with commas (e.g., "Gefitinib,Docetaxel").')
    parser.add_argument('--species', type=str, required=True, choices=['hs', 'mus'],
                        help='Species of the spatial data: "hs" (Human) or "mus" (Mouse).')
    
    # --- Hardware and Path Arguments ---
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'],
                        help='Device to use: "gpu" (cuda:0) or "cpu". Default: gpu.')
    parser.add_argument('--spatial_count_path', type=str, required=True, default='./data/spatial_data/count.csv',
                        help='Path to the spatial count CSV file or H5 file.')
    parser.add_argument('--spatial_coord_path', type=str, required=True, default='./data/spatial_data/category_coord.csv',
                        help='Path to the spatial category coordinate CSV file.')
    
    # --- Model Architecture Arguments ---
    parser.add_argument('--hiddens_graph', type=str, default='1024,512,128',
                        help='Hidden layer dimensions for the graph network.')
    parser.add_argument('--hiddens_linear', type=str, default='64,16',
                        help='Hidden layer dimensions for the linear network.')
    parser.add_argument('--drugFunc', type=str, default='GIN', choices=['GIN', 'GINE'],
                        help='GNN function type for the drug graph.')

    # --- Training Hyperparameters ---
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value.')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience.')
    parser.add_argument('--warm_up', type=int, default=10, help='Warm-up epochs.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Graph dropout rate.')
    parser.add_argument('--alpha', type=float, default=1.8, help='TsallisEntropy parameter 1.')
    parser.add_argument('--temperature', type=float, default=2.5, help='TsallisEntropy parameter 2.')
    parser.add_argument('--threshold', type=float, default=0.95, help='Threshold parameter.')
    
    # --- Data Processing Arguments ---
    parser.add_argument('--gene_num', type=int, default=500, help='Number of highly variable genes to select.')
    parser.add_argument('--k_neigh', type=int, default=6, help='Number of neighbors for spatial graph.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split proportion.')
    parser.add_argument('--sampling', type=str, default='SMOTE', help='Sampling strategy.')
    
    # --- Flags ---
    parser.add_argument('--save_mid', type=str2bool, default=False,
                        help='Flag to save intermediate calculation results (True/False).')
    parser.add_argument('--perform_normalize', type=str2bool, default=True,
                        help='Whether to perform filtering and log-normalization on spatial data.')

    return parser.parse_args()

def load_and_process_spatial_raw(args, logs):
    """
    Loads spatial data, optionally normalizes it using sparse operations if possible, 
    and returns a dense DataFrame for downstream processing.
    """
    logs.append("Loading spatial data...")

    # --- Part 1: Load Data into AnnData Object ---
    # Unified loading of data into AnnData object. 
    # If H5AD and sparse, do not uncompress yet to save memory.
    if args.spatial_count_path.endswith('.h5ad'):
        logs.append(f"Detected H5AD file: {args.spatial_count_path}")
        spatial_adata = sc.read_h5ad(args.spatial_count_path)
        # Ensure var_names and obs_names are unique to avoid errors when converting to DataFrame
        spatial_adata.var_names_make_unique() 
        spatial_adata.obs_names_make_unique()
    else:
        # CSV reading logic
        logs.append(f"Reading CSV file: {args.spatial_count_path}")
        df_temp = pd.read_csv(args.spatial_count_path, index_col=0)
        # Convert to AnnData immediately to reuse downstream logic
        spatial_adata = sc.AnnData(
            X=df_temp.values,
            obs=pd.DataFrame(index=df_temp.index),
            var=pd.DataFrame(index=df_temp.columns)
        )
        del df_temp
        gc.collect()

    # --- Part 2: Optional Normalization and Filtering ---
    # If spatial_adata.X is sparse, Scanpy automatically uses optimized sparse algorithms.
    if args.perform_normalize:
        logs.append("Performing filtering and normalization (Scanpy optimized)...")
        
        # 1. Filter: Equivalent to np.sum(X!=0, axis=0) >= 5
        # inplace=True modifies spatial_adata directly to save memory.
        sc.pp.filter_genes(spatial_adata, min_cells=5)
        
        # 2. Normalization
        sc.pp.normalize_total(spatial_adata, target_sum=1e6)
        
        # 3. Log transformation
        sc.pp.log1p(spatial_adata)
    else:
        logs.append("Skipping normalization (using data as provided).")

    # --- Part 3: Convert to DataFrame for Downstream Compatibility ---
    # Output unified as Dense DataFrame for species mapping and PyTorch processing.
    logs.append("Converting final data to DataFrame...")
    
    Xsp = spatial_adata.X
    
    if sp.issparse(Xsp):
        # Convert to Dense array here if it is a sparse matrix.
        # Data has been filtered (reduced columns), so memory pressure is much lower than converting at the start.
        data_values = Xsp.toarray()
    else:
        data_values = Xsp

    X_data_spatial = pd.DataFrame(
        data_values,
        index=spatial_adata.obs_names,
        columns=spatial_adata.var_names
    )

    # Clean up AnnData memory
    del spatial_adata, Xsp
    gc.collect()

    # --- Part 4: Species Mapping (No changes to logic, just applying to DF) ---
    if args.species == 'hs':
        # Ensure column names are strings and clean them
        clean_cols = X_data_spatial.columns.astype(str).str.replace(r"[-_]", ".", regex=True)
        X_data_spatial.columns = clean_cols.values
    
    elif args.species == 'mus':
        logs.append("Converting Mouse genes to Human homologs...")
        X_data_spatial.columns = spatial_adata.var['gene_id'].values 
        
        if not os.path.exists(args.mapping_path):
            raise FileNotFoundError(f"Mapping file not found at {args.mapping_path}")
            
        hsa_gene = pd.read_csv(args.mapping_path)

        hsa_gene.rename(columns={
            hsa_gene.columns[0]: 'external_gene_name',
            hsa_gene.columns[1]: 'hsapiens_homolog_associated_gene_name'
        }, inplace=True)

        X_data_spatial = X_data_spatial.loc[:, X_data_spatial.columns.isin(hsa_gene['external_gene_name'])]
        
        indices = [hsa_gene['external_gene_name'].tolist().index(x) for x in X_data_spatial.columns]
        hsa_gene = hsa_gene.iloc[indices, :].reset_index(drop=True)
        
        X_data_spatial.columns = hsa_gene['hsapiens_homolog_associated_gene_name'].values

        mean_values = {}
        unique_columns = X_data_spatial.columns.unique().values
        
        for col in unique_columns:
            if isinstance(X_data_spatial[col], pd.Series):
                mean_values[col] = X_data_spatial[col]
            else:
                # If multiple columns, revert log, average in linear space, then log again
                mean_values[col] = np.log((np.exp(X_data_spatial[col])-1).mean(axis=1)+1)
        
        X_data_spatial = pd.DataFrame(mean_values)
        logs.append(f"Conversion complete. Features reduced to {X_data_spatial.shape[1]} human homologs.")
    
    return X_data_spatial

def plot_spatial_sensitivity(loc_data, binary_pred, drug_name, save_dir='results'):
    """
    Plots the spatial sensitivity of cells based on binary predictions.
    """
    df = loc_data.copy()
    
    # Map predictions to labels (0: Resistant, 1: Sensitive)
    new_labels = np.where(binary_pred == 0, 'Resistant', 'Sensitive')
    
    mask = df['cell_type'] == 'Cancer'
    df.loc[mask, 'cell_type'] = new_labels
    df.loc[~mask, 'cell_type'] = 'Others'

    # --- Plotting Configuration ---
    plt.figure(figsize=(4.5, 3.8))
    ax = plt.gca()
    plt.grid(False)

    # Color palette
    color_palette = {
        'Sensitive': '#e7211a',
        'Resistant': '#18499e',
        'Others': '#F0F0F0'
    }

    # Draw order and z-index
    draw_order = ['Others', 'Resistant', 'Sensitive']
    z_orders = {'Others': 1, 'Resistant': 2, 'Sensitive': 3}

    legend_handles = []

    for cell_type in draw_order:
        if cell_type not in df['cell_type'].unique():
            continue
            
        subset = df[df['cell_type'] == cell_type]
        
        # Style settings per category
        if cell_type == 'Others':
            size = 10
            alpha = 0.9
            edgecolor = 'none'
        else:  # Sensitive/Resistant
            size = 10
            alpha = 1.0
            edgecolor = 'black'
        
        plt.scatter(
            x=subset['x'],
            y=subset['y'],
            c=color_palette[cell_type],
            s=size,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=0.3,
            zorder=z_orders[cell_type],
            label=cell_type
        )

    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    max_range = max(x_max - x_min, y_max - y_min) * 1.05

    plt.xlim(x_center - max_range/2, x_center + max_range/2)
    plt.ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_aspect('equal')

    for cell_type in ['Sensitive', 'Resistant', 'Others']:
        if cell_type not in df['cell_type'].unique():
            continue
            
        size = 40
        linewidth = 0.6
        
        handle = plt.scatter(
            [], [],
            c=color_palette[cell_type],
            s=size,
            alpha=1.0,
            edgecolor='black' if cell_type != 'Others' else 'black',
            linewidth=linewidth,
            label=cell_type
        )
        legend_handles.append(handle)

    legend = plt.legend(
        handles=legend_handles,
        title='Cell Type',
        bbox_to_anchor=(1.5, 0.3),
        loc='lower right',
        frameon=True,
        framealpha=0.85,
        edgecolor='#CCCCCC',
        fontsize=10,
        title_fontsize=11
    )
    legend.get_title().set_weight('bold')

    plt.xlabel('X Coordinate', fontsize=11)
    plt.ylabel('Y Coordinate', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.title(drug_name + ' Sensitivity', fontsize=13, pad=15)

    plt.tight_layout()
    plt.subplots_adjust(right=0.82)

    # Save output
    save_path = os.path.join(save_dir, f"{drug_name}_sensitivity.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
    plt.close() # Close plot to free memory
    print(f"Plot saved to {save_path}")

def normalize_name(name):
    """Normalize column name: Convert to uppercase and remove spaces, -, _, /, etc."""
    normalized = name.upper()
    normalized = re.sub(r'[-_()\s/]+', '', normalized)
    return normalized

def main():
    # 1. Parse arguments
    args = get_args()
    logs = []
    
    # 2. Configure Device
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    logs.append(f"Running on: {device}")

    # 3. Initialize Seed
    seed_everything(seed)

    # 4. Parse network structure
    hiddens_graph = [int(x) for x in args.hiddens_graph.split(',')]
    hiddens_linear = [int(x) for x in args.hiddens_linear.split(',')]
    hiddens_linear = [hiddens_graph[-1]] + hiddens_linear

    # --- Prepare Common Data --- #
    try:
        X_data_spatial_base = load_and_process_spatial_raw(args, logs)
    except Exception as e:
        print(f"Error loading spatial data: {e}")
        return

    # Load Location and Category Data
    loc_category_data_spatial = pd.read_csv(args.spatial_coord_path)
    loc_category_data_spatial.columns = ['cell', 'x', 'y', 'cell_type']
    
    choose = (loc_category_data_spatial['cell_type']=='Cancer').values
    
    # Build Spatial Edge Index
    spatial_edge_index = model.graph_build.get_spatial_graph(
        loc_category_data_spatial, 
        choose,
        save_path=None,
        normalize=False,
        bidirection=True,
        k=args.k_neigh
    )
    
    preprocess_save_path = './preprocess_result' if args.save_mid else None
    
    if not os.path.exists('./preprocess_result'):
        os.makedirs('./preprocess_result', exist_ok=True)
        
    # --- Loop over Drugs --- #
    drug_list = [d.strip() for d in args.drug_name.split(',')]
    print(f"Drugs to process: {drug_list}")

    smiles_df = pd.read_csv('./data/drug_data/train_drug.csv', header=0)
    smiles_df.iloc[:, 0] = smiles_df.iloc[:, 0].apply(normalize_name)

    ####### Define a supplementary dictionary #######
    MANUAL_DICT = {
        'Metformin': 'CN(C)C(=N)N=C(N)N'
    }
    MANUAL_DICT = {normalize_name(k): v for k, v in MANUAL_DICT.items()}
    
    for drug_idx, drug_name in enumerate(drug_list):
        print(f"\n{'='*20} Processing Drug: {drug_name} ({drug_idx+1}/{len(drug_list)}) {'='*20}")
        current_logs = logs.copy() # Start with common logs
        seed_everything(seed)
        
        # Check if the drug is in the library and build the drug graph
        drug_name_norm = normalize_name(drug_name)

        output_filename = os.path.join("results", f"{drug_name}_output.txt")
        lines = []
        
        if drug_name_norm in smiles_df.iloc[:, 0].values:
            
            drug_smiles = smiles_df.loc[smiles_df['Name'] == drug_name_norm, 'isosmiles'].values[0]

            drug_graph = model.smiles2graph.smiles2graph(drug_smiles)
            
            X_train, X_valid, Y_train, Y_valid = model.data_preprocess.bulk_preprocess(
                './data/bulk_data/train_geneexp.csv', 
                './data/bulk_data/IC50_binary_df.csv', 
                drug_name,
                save_path=preprocess_save_path,
                test_size=args.test_size,
                sampling=args.sampling,
                seed=seed
            )
            
        elif drug_name_norm in MANUAL_DICT:
            print(f"Found {drug_name} in manual hardcoded dictionary.")
            drug_smiles = MANUAL_DICT[drug_name_norm]
            
            current_logs.append(f"SMILES for {drug_name} provided manually.")
            print(f"[SUCCESS] Manually recorded SMILES for '{drug_name}'.")

            drug_dict = model.smiles2graph.get_drug_graph_ndrug('./data/drug_data/train_drug.csv', drug_name_norm, drug_smiles)

            test_pred = model.new_drug.run_training(
                drug_dict=drug_dict,
                device=device,
                test_drug=drug_name_norm,
                current_logs=current_logs,
                output_lines=lines
            )
  
            X_train, X_valid, Y_train, Y_valid = model.data_preprocess.bulk_preprocess_ndrug(
                './data/bulk_data/train_geneexp.csv',
                test_pred,
                drug_name_norm,
                save_path=preprocess_save_path,
                test_size=args.test_size,
                sampling=args.sampling,
                seed=seed)

            drug_graph = model.smiles2graph.smiles2graph(drug_smiles)

        #### Enter common calculation process ####

        X_data_bulk = pd.concat([X_train, X_valid], ignore_index=True)
        Y_data_bulk = pd.concat([Y_train, Y_valid], ignore_index=True)
        
        idx_train = list(range(len(X_train)))
        idx_test = list(range(len(X_train), len(X_train)+len(X_valid)))

        # -------- Step 2. Intersection & VAE -------- #
        X_data_spatial_curr = X_data_spatial_base.copy()
        intersection = sorted(set(X_data_bulk.columns).intersection(set(X_data_spatial_curr.columns)))
        current_logs.append(f"[{drug_name}] Number of intersection genes: {len(intersection)}")
        X_data_bulk = X_data_bulk[list(intersection)]
        X_data_spatial_curr = X_data_spatial_curr[list(intersection)]

        vae_meta = loc_category_data_spatial.loc[loc_category_data_spatial['cell_type']=='Cancer',:].reset_index(drop=True)
        vae_data = X_data_spatial_curr.loc[vae_meta['cell'].values,:]

        seed_everything(seed)
        vae_file_path = os.path.join('./preprocess_result', f"vae_generated_{drug_name}.parquet")
        
        if args.save_mid and os.path.isfile(vae_file_path):
            generate_vae_data = pd.read_parquet(vae_file_path)
        else:
            generate_vae_data = train_and_generate(
                data=vae_data,
                save_model=False,
                seed=seed,
                latent_dim=128,
                batch_size=128,
                epochs=10000,
                lr=0.0001,
                device=device 
            )
            if args.save_mid:
                generate_vae_data.to_parquet(vae_file_path)
            seed_everything(seed)

        keep_columns = generate_vae_data.columns[(generate_vae_data != 0).any()]
        vae_data_test = vae_data.loc[:, keep_columns]
        
        spatial_adata_hv = sc.AnnData(
            X = vae_data_test.values,
            obs = pd.DataFrame({'cell_id': list(vae_data_test.index)}),
            var = pd.DataFrame({'gene_id': list(vae_data_test.columns)})
        )
        
        spatial_highvar = sc.pp.highly_variable_genes(
            spatial_adata_hv,
            flavor="seurat",
            n_top_genes=args.gene_num,
            subset=True,
            inplace=False,
        )
        
        columns_chosen = vae_data_test.columns[spatial_highvar.index.astype(int)]
        
        st_raw_final = vae_data.loc[:, columns_chosen]
        st_twin_final = generate_vae_data.loc[:, columns_chosen]
        bulk_raw_final = X_data_bulk.loc[:, columns_chosen]
        st_twin_final.index = [f"{idx}-twin" for idx in st_raw_final.index]
        
        z_exp_bulk = np.ascontiguousarray(bulk_raw_final.values)
        z_exp_spatial = np.ascontiguousarray(st_raw_final.values)
        z_exp_twin = np.ascontiguousarray(st_twin_final.values)

        # -------- Step 3. Graph Construction -------- #
        
        bulk_train_graph = Data(edge_index=torch.LongTensor(np.empty((2, 0))), 
                                x=torch.FloatTensor(z_exp_bulk),
                                y=torch.LongTensor(Y_data_bulk.iloc[:,0].values.reshape(-1)),
                                train_idx = idx_train,
                                test_idx = idx_test)

        spatial_train_graph = Data(edge_index=torch.LongTensor(spatial_edge_index),
                                   x=torch.FloatTensor(z_exp_spatial),
                                   test_idx = list(range(len(z_exp_spatial))))
        
        spatial_train_graph_twin = Data(edge_index=torch.LongTensor(spatial_edge_index),
                                        x=torch.FloatTensor(z_exp_twin),
                                        test_idx = list(range(len(z_exp_spatial))))  

        # drug_graph = model.smiles2graph.get_drug_graph('./data/drug_data/train_drug.csv', drug_name)

        # -------- Step 4. Training -------- #
        
        bulk_drop = 1 - np.median((z_exp_spatial != 0).sum(axis=1)) / args.gene_num
        seed_everything(seed)
        
        results = train_epoch(
            bulk_train_graph, spatial_train_graph, spatial_train_graph_twin, drug_graph,
            loc_category_data_spatial, num_classes=2, in_dim=z_exp_bulk.shape[1], 
            hiddens_graph=hiddens_graph, hiddens_linear=hiddens_linear, layer_drug=2,
            graphFunc="GATv2Conv", drugFunc=args.drugFunc, epoches=500,
            lr=args.lr, patience=args.patience, weight_decay=args.weight_decay, grad_clip=args.grad_clip, warm_up=args.warm_up, 
            dropout_rate=args.dropout_rate, use_layer_norm=True, bulk_drop=bulk_drop,
            temperature=args.temperature, alpha=args.alpha, threshold=args.threshold,
            save_path='results', drug_name=drug_name
        )
        
        # -------- Step 5. Save Results & Plot -------- #  

        # Parameters
        lines.append("============================== parameter ==============================")
        lines.append(f"drug_name: {drug_name}")
        lines.append(f"hiddens_graph: {args.hiddens_graph}")
        lines.append(f"hiddens_linear: {args.hiddens_linear}")
        lines.append(f"drugFunc: {args.drugFunc}")
        lines.append(f"lr: {args.lr}")
        lines.append(f"weight_decay: {args.weight_decay}")
        lines.append(f"grad_clip: {args.grad_clip}")
        lines.append(f"dropout_rate: {args.dropout_rate}")
        lines.append(f"bulk_drop: {bulk_drop:.4f}")
        lines.append(f"patience: {args.patience}")
        lines.append(f"warm_up: {args.warm_up}")
        lines.append(f"alpha: {args.alpha}")
        lines.append(f"temperature: {args.temperature}")
        lines.append(f"threshold: {args.threshold}")
        lines.append(f"sampling: {args.sampling}")
        lines.append(f"gene_num: {args.gene_num}")
        lines.append(f"k_neigh: {args.k_neigh}")
        
        # Results
        lines.append("============================== result ==============================")
        lines.append(f"best_loss: {results['best_loss']:.4f}")
        lines.append(f"b_valid_auc: {results['b_valid_auc']:.4f}")
        lines.append(f"b_valid_ap: {results['b_valid_ap']:.4f}")        
        lines.append(f"b_valid_acc: {results['b_valid_acc']:.4f}")
        lines.append(f"b_valid_f1: {results['b_valid_f1']:.4f}")
        lines.append(f"b_sen_prop: {results['b_sen_prop']:.4f}")
        lines.append(f"b_bb: {results['b_bb']:.2f}")
        lines.append(f"b_e_bb: {results['b_e_bb']:.2f}")
        lines.append(f"b_sig_bb: {results['b_sig_bb']:.4f}")
        lines.append(f"best_epoch: {results['best_epoch']}")

        # Write logs and results to file
        with open(output_filename, 'w') as f:
            if current_logs:
                f.write("============================== runtime logs ==============================\n")
                f.write('\n'.join(current_logs))
                f.write('\n\n')
            f.write('\n'.join(lines))

        # Plot for single drug
        # Load the saved NPY to ensure we visualize exactly what was saved
        pred_path = os.path.join("results", f"{drug_name}_best.npy")
        if os.path.exists(pred_path):
            pred_arr = np.load(pred_path)
            plot_spatial_sensitivity(loc_category_data_spatial, pred_arr, drug_name, save_dir='results')
        
        print(f"Finished processing {drug_name}. Result saved to {output_filename}")

    # =========================================================================
    #                    Processing All Drugs Combination
    # =========================================================================
    if len(drug_list) > 1:
        print(f"\n{'='*20} Processing All Drugs Combination {'='*20}")
        
        # 1. Sort drug names to ensure unique combination key (e.g., A+B)
        sorted_drugs = sorted(drug_list)
        combined_key = "+".join(sorted_drugs)
        print(f"Generating combination result for: {combined_key}")
        
        combined_arr = None
        processed_count = 0
        
        for drug in sorted_drugs:
            file_path = os.path.join("results", f"{drug}_best.npy")
            if not os.path.exists(file_path):
                print(f"Warning: Result file for {drug} not found at {file_path}. Skipping this drug in combination.")
                continue
                
            current_arr = np.load(file_path)
            
            if combined_arr is None:
                combined_arr = current_arr
            else:
                if combined_arr.shape != current_arr.shape:
                    print(f"Error: Shape mismatch for {drug}. Stopping.")
                    combined_arr = None
                    break
                combined_arr = combined_arr + current_arr
            
            processed_count += 1
            
        if combined_arr is not None and processed_count > 0:
            # Logic OR: any drug effective = sensitive
            combined_arr[combined_arr > 1] = 1
            
            # Save Combined NPY
            np.save(os.path.join("results", f"{combined_key}_best.npy"), combined_arr)
            
            # Save Combined Report
            sensitive_ratio = np.sum(combined_arr) / combined_arr.size
            save_txt_path = os.path.join("results", f"{combined_key}_output.txt")
            
            lines = []
            lines.append("============================== Combination Result ==============================")
            lines.append(f"combination_key: {combined_key}")
            lines.append(f"included_drugs: {', '.join(sorted_drugs)}")
            lines.append(f"total_cells: {combined_arr.size}")
            lines.append(f"sensitive_cells: {np.sum(combined_arr)}")
            lines.append(f"sensitive_ratio: {sensitive_ratio:.4f}")
            lines.append("================================================================================")
            
            with open(save_txt_path, 'w') as f:
                f.write('\n'.join(lines))
                
            print(f"Combined Text Report saved to: {save_txt_path}")
            
            # Plot for Combination
            plot_spatial_sensitivity(loc_category_data_spatial, combined_arr, combined_key, save_dir='results')
            print(f"Combined results and plot saved for {combined_key}")
            
        else:
            print("Combination failed due to missing files or shape mismatch.")
    else:
        print("\nSkipping combination step: only one drug in list.")


if __name__ == "__main__":
    main()
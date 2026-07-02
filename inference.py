import os

# Unified seed value
seed = 114514

# Environment variables setting
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import gc
import re
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import matplotlib.pyplot as plt

import torch
import torch_geometric

from torch_geometric.data import Data

from libpysal.weights import DistanceBand
from esda.join_counts import Join_Counts

warnings.filterwarnings("ignore")

import model.data_preprocess
import model.graph_build
import model.smiles2graph
import model.newdrug_utils
import model.newdrug_model

from model.vae_model import train_and_generate

# =========================================================
# utils
# =========================================================

def seed_everything(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch_geometric.seed_everything(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):

    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', '1'):
        return True

    elif v.lower() in ('no', 'false', 'f', '0'):
        return False

    raise argparse.ArgumentTypeError('Boolean expected')


def normalize_name(name):

    normalized = name.upper()

    normalized = re.sub(
        r'[-_()\s/]+',
        '',
        normalized
    )

    return normalized
# =========================================================
# parser
# =========================================================

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--drug_name', type=str, required=True)
    parser.add_argument('--pth_path',  type=str, required=True)
    parser.add_argument('--newdrug_pth_path', type=str, default='')
    parser.add_argument('--species', type=str, required=True, choices=['hs', 'mus'])
    parser.add_argument('--spatial_count_path', type=str, required=True)
    parser.add_argument('--spatial_coord_path', type=str, required=True)
    parser.add_argument('--savedir', type=str, default='results')
    parser.add_argument('--save_mid', type=str2bool, default=False)
    parser.add_argument('--vae_file_path', type=str, default='')
    parser.add_argument('--perform_normalize', type=str2bool, default=True)

    return parser.parse_args()

# =========================================================
# spatial preprocess
# =========================================================

def load_spatial_anndata(args):

    if args.spatial_count_path.endswith('.h5ad'):

        adata = sc.read_h5ad(
            args.spatial_count_path
        )

    else:

        df = pd.read_csv(
            args.spatial_count_path,
            index_col=0
        )

        adata = sc.AnnData(
            X=sp.csr_matrix(df.values),
            obs=pd.DataFrame(index=df.index),
            var=pd.DataFrame(index=df.columns)
        )

    if args.perform_normalize:

        sc.pp.filter_genes(adata, min_cells=5)

        sc.pp.normalize_total(
            adata,
            target_sum=1e6
        )

        sc.pp.log1p(adata)

    return adata


def extract_spatial_genes(
        spatial_adata,
        bulk_genes,
        species):
    
    final_df = None

    if species == 'hs':

        spatial_genes = spatial_adata.var_names.astype(str)
        spatial_genes = spatial_genes.str.replace(r"[-_]", ".", regex=True)

        gene_map = dict(zip(spatial_genes, spatial_adata.var_names))

        intersection = sorted(
            list(
                set(bulk_genes).intersection(
                    set(spatial_genes)
                )
            )
        )

        if len(intersection) == 0:

            raise ValueError(
                "No intersection found between "
                "Bulk and Spatial genes (Human)."
            )

        original_genes_to_extract = [gene_map[g] for g in intersection]

        subset_adata = spatial_adata[:,original_genes_to_extract]

        Xsp = subset_adata.X

        if sp.issparse(Xsp):

            final_df = pd.DataFrame.sparse.from_spmatrix(
                Xsp,
                index=subset_adata.obs_names,
                columns=intersection
            )

        else:

            final_df = pd.DataFrame(
                Xsp,
                index=subset_adata.obs_names,
                columns=intersection
            )

    elif species == 'mus':

        hsa_gene_map = pd.read_csv('./data/mus_to_hs_mapping.csv')

        hsa_gene_map.rename(
            columns={
                hsa_gene_map.columns[0]:'external_gene_name',
                hsa_gene_map.columns[1]:'hsapiens_homolog_associated_gene_name'
                },
            inplace=True
        )

        # -------------------------------------------------
        # Keep genes existing in bulk
        # -------------------------------------------------

        relevant_mapping = hsa_gene_map[hsa_gene_map['hsapiens_homolog_associated_gene_name'].isin(bulk_genes)]

        # -------------------------------------------------
        # Keep genes existing in spatial
        # -------------------------------------------------

        valid_mouse_genes = set(spatial_adata.var_names)

        relevant_mapping = relevant_mapping[relevant_mapping['external_gene_name'].isin(valid_mouse_genes)]

        if relevant_mapping.empty:

            raise ValueError(
                "No intersection found after "
                "Mouse->Human mapping."
            )

        mouse_genes_to_extract = relevant_mapping['external_gene_name'].unique()

        subset_adata = spatial_adata[:,mouse_genes_to_extract]

        Xsp = subset_adata.X

        if sp.issparse(Xsp):

            temp_df = pd.DataFrame.sparse.from_spmatrix(
                Xsp,
                index=subset_adata.obs_names,
                columns=mouse_genes_to_extract
            )

        else:

            temp_df = pd.DataFrame(
                Xsp,
                index=subset_adata.obs_names,
                columns=mouse_genes_to_extract
            )

        # -------------------------------------------------
        # Mouse -> Human rename
        # -------------------------------------------------

        mouse_to_human = dict(
            zip(relevant_mapping['external_gene_name'],
                relevant_mapping['hsapiens_homolog_associated_gene_name']
                )
            )

        temp_df = temp_df.loc[:,temp_df.columns.isin(mouse_to_human.keys())]

        temp_df.columns = temp_df.columns.map(mouse_to_human)

        # -------------------------------------------------
        # Aggregate duplicated human homologs
        # -------------------------------------------------

        if temp_df.columns.duplicated().any():

            def exp_mean_log(x):
                if isinstance(x, pd.Series):
                    return x

                return np.log(np.expm1(x).mean(axis=1) + 1)

            final_df = temp_df.groupby(temp_df.columns, axis=1).agg(exp_mean_log)

        else:
            final_df = temp_df

    # =====================================================
    # Invalid species
    # =====================================================

    else:

        raise ValueError(
            f"Unsupported species: {args.species}"
        )

    return final_df

# =========================================================
# validate inference
# =========================================================

def validate_inference(
        model_obj,
        tar,
        drug,
        loc_data):

    model_obj.eval()

    with torch.no_grad():

        _, _, tar_pred = model_obj(
            tar.x,
            tar.edge_index,
            tar.x,
            tar.edge_index,
            drug
        )

    choose = tar_pred.argmax(
        dim=1
    ).cpu().numpy()

    sen_prop = np.sum(choose) / len(choose)

    cancer_loc = loc_data.query(
        "cell_type == 'Cancer'"
    )

    coordinates = cancer_loc[
        ['x', 'y']
    ].values

    w = DistanceBand(
        coordinates,
        threshold=5000,
        binary=True
    )

    if len(np.unique(choose)) < 2:

        bb = 0
        mean_bb = 0
        p_sim_bb = 0

    else:

        jc = Join_Counts(choose, w)

        bb = jc.bb
        mean_bb = jc.mean_bb
        p_sim_bb = jc.p_sim_bb

    return sen_prop, bb, mean_bb, p_sim_bb, choose


# =========================================================
# plot
# =========================================================

def plot_spatial_sensitivity(
        loc_data,
        binary_pred,
        drug_name,
        save_dir):

    df = loc_data.copy()

    labels = np.where(
        binary_pred == 0,
        'Resistant',
        'Sensitive'
    )

    mask = df['cell_type'] == 'Cancer'

    df.loc[mask, 'cell_type'] = labels

    df.loc[~mask, 'cell_type'] = 'Others'

    plt.figure(figsize=(4.5, 3.8))

    palette = {
        'Sensitive': '#e7211a',
        'Resistant': '#18499e',
        'Others': '#F0F0F0'
    }

    for ct in ['Others', 'Resistant', 'Sensitive']:

        subset = df[df['cell_type'] == ct]

        plt.scatter(
            subset['x'],
            subset['y'],
            c=palette[ct],
            s=10
        )

    plt.title(f"{drug_name} Sensitivity")

    save_path = os.path.join(
        save_dir,
        f"{drug_name}_sensitivity.pdf"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# =========================================================
# newdrug prediction
# =========================================================

def load_newdrug_prediction(
        drug_name,
        drug_smiles,
        checkpoint_path,
        device):

    drug_name_norm = normalize_name(drug_name)

    drug_dict = model.smiles2graph.get_drug_graph_ndrug(
        './data/drug_data/train_drug.csv',
        drug_name_norm,
        drug_smiles
    )

    bulk_exp_path = './data/bulk_data/train_geneexp.csv'

    features = 500
    net_thresh = 0.95
    batch_size = 512

    cell_dict, genename = \
        model.newdrug_utils.get_highvar_gene(
            bulk_exp_path,
            des_path='./preprocess_result',
            num=features
        )

    edge_index = model.newdrug_utils.get_STRING_graph(
        genename,
        drug_name=drug_name_norm,
        des_path='./preprocess_result',
        thresh=net_thresh,
        num=features
    )

    drug_sen = pd.read_csv(
        './data/bulk_data/IC50_binary_df.csv',
        index_col=0
    )

    melted = drug_sen.melt(
        ignore_index=False,
        var_name='Drug',
        value_name='Response'
    ).reset_index()

    melted = melted.rename(
        columns={'index': 'Cell_line'}
    )

    melted = melted.dropna(
        subset=['Response']
    )

    melted['Drug'] = melted['Drug'].apply(
        normalize_name
    )

    melted['Response'] = melted['Response'].map(
        {'S': 1, 'R': 0}
    )

    train_loader, val_loader, test_loader = \
        model.newdrug_utils.load_data(
            melted,
            drug_dict,
            cell_dict,
            edge_index,
            batch_size=batch_size,
            seed=seed,
            test_drug=drug_name_norm
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    ndrug = checkpoint['model']

    ndrug = ndrug.to(device)

    _, _, _, _, _, \
    _, _, _, _, _, \
    test_pred, _ = model.newdrug_model.validate(
        ndrug,
        train_loader,
        val_loader,
        test_loader,
        device
    )

    return test_pred


# =========================================================
# main
# =========================================================

def main():

    args = get_args()

    seed_everything(seed)

    os.makedirs(args.savedir, exist_ok=True)

    device = torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )

    spatial_adata = load_spatial_anndata(args)

    loc_data = pd.read_csv(
        args.spatial_coord_path
    )

    loc_data.columns = ['cell', 'x', 'y', 'cell_type']

    cancer_mask = (loc_data['cell_type'] == 'Cancer').values

    # spatial_edge_index = model.graph_build.get_spatial_graph(
    #     loc_data,
    #     choose,
    #     save_path=None,
    #     normalize=False,
    #     bidirection=True,
    #     k=6)

    smiles_df = pd.read_csv('./data/drug_data/train_drug.csv')

    smiles_df.iloc[:, 0] = smiles_df.iloc[:, 0].apply(normalize_name)

    drug_list = [
        x.strip()
        for x in args.drug_name.split(',')
    ]

    pth_list = [
        x.strip()
        for x in args.pth_path.split(',')
    ]

    newdrug_pth_list = []

    if args.newdrug_pth_path != '':

        newdrug_pth_list = [
            x.strip()
            for x in args.newdrug_pth_path.split(',')
        ]

    vae_file_list = []
    if args.vae_file_path != '':

        vae_file_list = [
            x.strip()
            for x in args.vae_file_path.split(',')
        ]
    
    for idx, drug_name in enumerate(drug_list):
        seed_everything(seed)

        drug_name_norm = normalize_name(drug_name)

        checkpoint = torch.load(pth_list[idx], map_location=device)

        main_model = checkpoint['model']
        main_model = main_model.to(device)

        train_args = checkpoint['train_args']

        results = checkpoint['results']

        if drug_name_norm in smiles_df.iloc[:,0].values:

            drug_smiles = smiles_df.loc[
                smiles_df['Name'] == drug_name_norm,
                'isosmiles'
            ].values[0]

            drug_graph = model.smiles2graph.smiles2graph(drug_smiles)

            X_train, X_valid, Y_train, Y_valid = \
                model.data_preprocess.bulk_preprocess(
                    './data/bulk_data/train_geneexp.csv',
                    './data/bulk_data/IC50_binary_df.csv',
                    drug_name,
                    save_path=None,
                    test_size=0.2,
                    sampling='SMOTE',
                    seed=seed
                )

        else:

            drug_smiles = input(
                f'SMILES for {drug_name}: '
            )

            test_pred = load_newdrug_prediction(
                drug_name,
                drug_smiles,
                newdrug_pth_list[idx],
                device
            )

            X_train, X_valid, Y_train, Y_valid = \
                model.data_preprocess.bulk_preprocess_ndrug(
                    './data/bulk_data/train_geneexp.csv',
                    test_pred,
                    drug_name_norm,
                    save_path=None,
                    test_size=0.2,
                    sampling='SMOTE',
                    seed=seed
                )

            drug_graph = model.smiles2graph.smiles2graph(drug_smiles)

        X_bulk = pd.concat(
            [X_train, X_valid],
            ignore_index=True
        )

        bulk_genes = X_bulk.columns.tolist()

        X_spatial = extract_spatial_genes(
            spatial_adata,
            bulk_genes,
            args.species
        )

        intersection = sorted(
            set(bulk_genes).intersection(
                set(X_spatial.columns)
            )
        )

        X_bulk = X_bulk[intersection]

        X_spatial = X_spatial[intersection]

        cancer_meta = loc_data.loc[
            loc_data['cell_type'] == 'Cancer'
        ].reset_index(drop=True)

        vae_data = X_spatial.loc[cancer_meta['cell'].values,:]

        if idx < len(vae_file_list) and vae_file_list[idx].strip():
        
            if os.path.exists(vae_file_list[idx]):
                print(f"Loading VAE result from: {vae_file_list[idx]}")
                generate_vae_data = pd.read_parquet(vae_file_list[idx])
                generate_vae_data.columns = generate_vae_data.columns.astype(str)
                generate_vae_data.index = generate_vae_data.index.astype(str)
        
            else:
                raise FileNotFoundError(f"{vae_file_list[idx]} not found")
        
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
                vae_save_path = os.path.join(args.savedir,f"vae_generated_{drug_name}.parquet")
                generate_vae_data.to_parquet(vae_save_path)
                print(f"Saved VAE result to: {vae_save_path}")
        
            seed_everything(seed)

        keep_columns = generate_vae_data.columns[(generate_vae_data != 0).any()]

        vae_data_test = vae_data.loc[:,keep_columns]

        spatial_adata_hv = sc.AnnData(X=vae_data_test.values)

        spatial_highvar = sc.pp.highly_variable_genes(
            spatial_adata_hv,
            flavor="seurat",
            n_top_genes=train_args['gene_num'],
            subset=True,
            inplace=False,
        )

        columns_chosen = vae_data_test.columns[spatial_highvar.index.astype(int)]

        spatial_edge_index = model.graph_build.get_spatial_graph(
            loc_data,
            cancer_mask,
            save_path=None,
            normalize=False,
            bidirection=True,
            k=train_args['k_neigh'])

        
        st_raw_final = vae_data.loc[:,columns_chosen]

        spatial_graph = Data(edge_index=torch.LongTensor(spatial_edge_index),
                             x=torch.FloatTensor(np.ascontiguousarray(st_raw_final.values)))

        sen_prop, bb, mean_bb, p_sim_bb, choose = validate_inference(
            main_model,
            spatial_graph,
            drug_graph,
            loc_data
        )

        np.save(os.path.join(args.savedir, f"{drug_name}_best.npy"), choose)

        plot_spatial_sensitivity(
            loc_data,
            choose,
            drug_name,
            args.savedir
        )

        lines = []

        lines.append("============================== parameter ==============================")

        for k, v in train_args.items():
            lines.append(f"{k}: {v}")

        lines.append("============================== result ==============================")

        lines.append(f"b_sen_prop: {sen_prop:.4f}")
        lines.append(f"b_bb: {bb:.2f}")
        lines.append(f"b_e_bb: {mean_bb:.2f}")
        lines.append(f"b_sig_bb: {p_sim_bb:.4f}")

        # for k, v in results.items():

        #     if k not in [
        #         'b_sen_prop',
        #         'b_bb',
        #         'b_e_bb',
        #         'b_sig_bb'
        #     ]:
        #         lines.append(f"{k}: {v}")

        with open(os.path.join(args.savedir, f"{drug_name}_output.txt"), 'w') as f:
            f.write('\n'.join(lines))

        # seed_everything(seed)

    # =====================================================
    # combination
    # =====================================================

    if len(drug_list) > 1:

        combined_key = "+".join(
            sorted(drug_list)
        )

        combined_arr = None

        for drug in sorted(drug_list):

            arr = np.load(os.path.join(args.savedir, f"{drug}_best.npy"))

            if combined_arr is None:
                combined_arr = arr

            else:
                combined_arr += arr

        combined_arr[combined_arr > 1] = 1

        np.save(os.path.join(args.savedir, f"{combined_key}_best.npy"), combined_arr)

        sensitive_ratio = np.sum(combined_arr) / combined_arr.size

        lines = []

        lines.append(
            "============================== Combination Result =============================="
        )

        lines.append(f"combination_key: {combined_key}")
        lines.append(f"sensitive_ratio: {sensitive_ratio:.4f}")

        with open(os.path.join(args.savedir, f"{combined_key}_output.txt"),'w') as f:
            f.write('\n'.join(lines))

        plot_spatial_sensitivity(
            loc_data,
            combined_arr,
            combined_key,
            args.savedir
        )


if __name__ == "__main__":

    main()

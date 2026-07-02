# M<sup>3</sup>SpaDE
[![DOI image](https://zenodo.org/badge/DOI/10.5281/zenodo.18211668.svg)](https://zenodo.org/records/18211668)  
## Introduction
**M<sup>3</sup>SpaDE** (<ins>**M**</ins>ulti-<ins>**M**</ins>odal <ins>**M**</ins>odel for predicting <ins>**Spa**</ins>tial <ins>**D**</ins>rug <ins>**E**</ins>fficacy) is a versatile computational framework for predicting drug sensitivity from spatial transcriptomics data. It is resolution-agnostic, accommodating inputs ranging from subcellular to spot-level resolution, and supports generalizable prediction of responses to unseen compounds based on their chemical structures.

M<sup>3</sup>SpaDE enables the following tasks:

*   **Binarized Sensitivity Prediction**  
    Performs binary classification of drug sensitivity at the single-cell or spot level (Sensitive vs. Resistant).

*   **Combinatorial Therapy Assessment**  
    Predicts and evaluates drug sensitivity outcomes for drug combinations.

*   **Spatial Autocorrelation Analysis**  
    Quantifies global spatial dependency and clustering patterns using **Join Count** statistics.

<img src="img/model.png" width="95%" alt="model architecture">

## Requirements

> **Note:** We strongly recommend running M<sup>3</sup>SpaDE on a GPU for optimal performance.

Our experiments were conducted using Python 3.8.20 with CUDA 11.8. We recommend using Anaconda or Miniconda to create an isolated conda environment for running M<sup>3</sup>SpaDE.

**1. Create a virtual environment:**
```bash
conda create -n M3SpaDE python==3.8.20
```
**2. Activate the environment:**
```bash
conda activate M3SpaDE
```
**3. Install required packages:**

> We provide a shell script for dependency installation at [`env/pkg_install.sh`](env/pkg_install.sh).  

## Data Availability

To ensure reproducibility and facilitate easy testing of M<sup>3</sup>SpaDE, we provide comprehensive access to both the training resources and a diverse collection of spatial transcriptomics datasets for demonstration.

### 1. Training Resources
We provide all necessary data to retrain the model or reproduce our benchmarks, including:
*   **Bulk RNA-Seq Data:** Pre-processed gene expression profiles from CCLE.
*   **Response Labels:** Binarized drug sensitivity metrics (IC50) for corresponding cell lines.
*   **Drug Representations:** Chemical structures formatted as IsoSMILES.

### 2. Example Datasets (Pre-processed)
To demonstrate how to use M<sup>3</sup>SpaDE for predicting spatial drug responses, we have curated **10 example datasets**. These include standard 10x Visium slides and high-resolution Visium HD data:

*   **MC38 Model:** 3 slides (10x Visium)
*   **B16 Model:** 3 slides (10x Visium)
*   **Human Colorectal Cancer (hCRC):** 3 slides (10x Visium)
*   **hCRC High-Definition:** 1 slide (Visium HD)

### Download & Setup
All example datasets, pre-trained models, and prediction results are available on Zenodo: [**Download Here**](https://zenodo.org/records/18211668)

Please download the datasets and unzip them. Ensure the `data` directory is placed in the same root directory as the `M3SpaDE.py` script. Your directory structure should look like this:

```bash
M3SpaDE-main/
├── M3SpaDE.py
├── data/                  # Place the downloaded data folder here
│   ├── bulk_data/
│   ├── drug_data/
│   ├── spatial_data/
│   └── ...
└── ...
└── README.md
```

## Usage
```bash
Usage: M3SpaDE.py [options]

Required:
      --drug_name STRING: Name of the drug(s). For multiple drugs, separate with commas (e.g., "Gefitinib,Docetaxel")
      --species STRING: Species of the spatial data: "hs" (Human) or "mus" (Mouse)
      --spatial_count_path STRING: Path to the spatial count CSV file or H5 file
      --spatial_coord_path STRING: Path to the spatial category coordinate CSV file

Optional:
    # Environment & Output
      --device STRING: Device to use: "gpu" or "cpu" (default: gpu)
      --savedir STRING: Directory to save training results and outputs (default: results)
      --save_mid BOOL: Flag to save intermediate calculation results (True/False) (default: False)
    
    # Model Architecture
      --hiddens_graph STRING: Hidden layer dimensions for the graph network (default: 1024,512,128)
      --hiddens_linear STRING: Hidden layer dimensions for the linear network (default: 64,16)
      --drugFunc STRING: GNN function type for the drug graph: "GIN" or "GINE" (default: GIN)
      --k_neigh INT: Number of neighbors for spatial graph (default: 6)

    # Data Processing
      --gene_num INT: Number of highly variable genes to select (default: 500)
      --test_size FLOAT: Test split proportion (default: 0.2)
      --sampling STRING: Sampling strategy (default: SMOTE)
      --perform_normalize BOOL: Whether to perform filtering and log-normalization on spatial data (True/False) (default: True)
      
    # Training Hyperparameters
      --epoch INT: Training epoches (default: 500)
      --lr FLOAT: Learning rate (default: 0.001)
      --weight_decay FLOAT: Weight decay (default: 0.0001)
      --grad_clip FLOAT: Gradient clipping value (default: 1.0)
      --patience INT: Early stopping patience (default: 25)
      --warm_up INT: Warm-up epochs (default: 10)
      --dropout_rate FLOAT: Graph dropout rate (default: 0.3)

    # Loss Function (Tsallis Entropy)
      --alpha FLOAT: TsallisEntropy parameter 1 (default: 1.8)
      --temperature FLOAT: TsallisEntropy parameter 2 (default: 2.5)
      --threshold FLOAT: Threshold parameter (default: 0.95)

```

*   We evaluated the model using pseudo-spatial data generated from six gold-standard single-cell datasets using STEM ( [*Hao et al., 2024*](https://doi.org/10.1038/s42003-023-05640-1) ). The default parameters provided in this repository were selected based on their robust performance across these tests. These settings are suitable for most tasks and serve as an excellent starting point for personalized parameter tuning.

### Examples

#### 1. Single Drug Prediction
To predict the response for a single drug (e.g., Afatinib):

```bash
python M3SpaDE.py --drug_name Afatinib --species hs --device gpu --spatial_count_path ./data/spatial_data/CRC_P6/count.csv --spatial_coord_path ./data/spatial_data/CRC_P6/category_coord.csv
```

#### 2. Combination Therapy Prediction
To predict the response for a combination of drugs (e.g., Afatinib and Oxaliplatin):

```bash
python M3SpaDE.py --drug_name Afatinib,Oxaliplatin --species hs --device gpu --spatial_count_path ./data/spatial_data/CRC_P6/count.csv --spatial_coord_path ./data/spatial_data/CRC_P6/category_coord.csv
```

#### 3. Inference
To perform inference using pretrained M<sup>3</sup>SpaDE models (e.g., Afatinib and Oxaliplatin):

```bash
python inference.py --drug_name Afatinib,Oxaliplatin --pth_path ./results/Afatinib_best.pth,./results/Oxaliplatin_best.pth --species hs --spatial_count_path ./data/spatial_data/crc_6/count.csv --spatial_coord_path ./data/spatial_data/crc_6/category_coord.csv --savedir ./results --vae_file_path ./preprocess_results/vae_generated_Afatinib.parquet,./preprocess_results/vae_generated_Oxaliplatin.parquet
```

---

### Important Notes

#### Note 1: Predicting Unseen Drugs
For any drug, regardless of whether it exists in our internal database, the implementation of M<sup>3</sup>SpaDE remains consistent. 
*   **Interactive Mode:** If a drug is not found in the database, M<sup>3</sup>SpaDE will prompt the user to input the IsoSMILES structure of the drug interactively. Once entered, the model automatically performs prediction.
*   **Non-Interactive Mode (HPC/SLURM):** For users running on clusters (e.g., SLURM) where interactive input is not feasible, we provide `M3SpaDE_noninteractive.py`. Users must edit the `MANUAL_DICT` dictionary within this script to manually define the IsoSMILES for unseen drugs before execution.

#### Note 2: IC50 Data Processing
As large-scale drug screening data expands, new IC50 data may become available. To enhance model scalability, we implemented an IC50 binarization method based on established theoretical research ( [*Iorio et al., 2016*](https://doi.org/10.1016/j.cell.2016.06.017), [*Knijnenburg et al., 2016*](https://doi.org/10.1038/srep36812) ).
*   The script is available here: [`preprocess/IC50_binarize.R`](preprocess/IC50_binarize.R)

#### Note 3: High-Resolution Data
For high-resolution spatial transcriptomics data, we recommend aggregating spots into "metaspots" using SuperSpot ( [*Telemanet al., 2024*](https://doi.org/10.1093/bioinformatics/btae734) ) before running M<sup>3</sup>SpaDE. This step reduces data sparsity and file size, thereby improving computational efficiency.
*   The tutorial script is available here: [`preprocess/superspot_tutorial.R`](preprocess/superspot_tutorial.R)

#### Note 4: Input File Specifications
M<sup>3</sup>SpaDE requires two primary input files for spatial transcriptomics data. For concrete examples of the required file structures, please refer to the sample files provided in the `data/` directory.

1.  Spatial Expression Matrix (`--spatial_count_path`)
    *   Format: Supports CSV (`.csv`) or H5AD (`.h5ad`) files.
    *   Content: The input can be Raw Counts (recommended) or Pre-normalized Data.
    *   Normalization: By default, M<sup>3</sup>SpaDE performs internal filtering and log-normalization. If you provide user-customized normalized data, please set `--perform_normalize False` to disable the built-in processing steps.

2.  Spatial Coordinates & Annotations (`--spatial_coord_path`)
    *   Format: A CSV file.
    *   Content: This file must contain cell identifiers, spatial coordinates (x, y), and cell type annotations.
    *   Requirement: Please ensure that the cell IDs in this file match the cell IDs in the expression matrix.
---

## Outputs

M<sup>3</sup>SpaDE generates the following files in the results directory.

**For Single Drug Prediction:**
```text
results
├── {drug_name}_best.npy          # Best model prediction results (NumPy array)
├── {drug_name}_best.pth          # Saved model weights (PyTorch)
├── {drug_name}_output.txt        # Prediction logs and text output
└── {drug_name}_sensitivity.pdf   # Visualization of drug sensitivity
```

**For Combination Therapy Prediction**
(Includes all single drug files plus the following combination files in the same directory) :
```text
results
├── {drug_name}_best.npy          # ... (Individual drug files generated as above)
│
├── {combine_drugs}_best.npy          # Combined response predictions
├── {combine_drugs}_output.txt        # Combined response logs
└── {combine_drugs}_sensitivity.pdf   # Visualization of combined sensitivity
```

**Mid Files**
(If specify `--save_mid True`):
```text
preprocess_resluts
├── bulk_exp_reindex.csv
├── bulk_label_binary.csv
├── bulk_train_exp_SMOTE.csv
├── bulk_train_label_binary_SMOTE.csv
├── bulk_valid_exp_SMOTE.csv
├── bulk_valid_label_binary_SMOTE.csv
└── vae_generated_{drug_name}.parquet    # VAE result
```

## Citation
(Unpublished now)
```bibtex
@article{M³SpaDE,
    title={M³SpaDE: A Multi-Modal Deep Learning Framework for Predicting Spatially Resolved Drug Responses},
    author={Zihao Zhang#, Xinyu Cui#, Zhengke Lian#, Xiufeng Pang*, Youqiong Ye*, Cizhong Jiang*},
    journal={XX},
    year={2026},
    doi={xx}
}
```
## Contacts
If you have any problem about our code, feel free to contact
- 2110819@tongji.edu.cn
- youqiong.ye@shsmu.edu.cn

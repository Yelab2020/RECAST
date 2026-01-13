# M<sup>3</sup>Spade
[![DOI image](https://zenodo.org/badge/DOI/10.5281/zenodo.18211668.svg)](https://zenodo.org/records/18211668)  
## Introduction
**M<sup>3</sup>Spade** (<ins>**M**</ins>ulti-<ins>**M**</ins>odal <ins>**M**</ins>odel for predicting <ins>**Spa**</ins>tial <ins>**D**</ins>rug <ins>**E**</ins>fficacy) is a versatile computational framework designed for predicting drug sensitivity in spatial transcriptomics data. It is resolution-agnostic, capable of processing data ranging from single-cell to spot-level resolutions, and supports generalizable prediction of responses to previously unseen drugs based on their chemical structures.

M<sup>3</sup>Spade enables the following tasks:

*   **Binarized Sensitivity Prediction**  
    Performs binary classification of drug sensitivity at the single-cell or spot level (Sensitive vs. Resistant).

*   **Spatial Autocorrelation Analysis**  
    Quantifies global spatial dependency and clustering patterns using **Join Count statistics**.

*   **Combinatorial Therapy Assessment**  
    Predicts and evaluates drug sensitivity outcomes for drug combinations.

<img src="img/model.png" width="80%" alt="model architecture">

## Requirements

> **Note:** We strongly recommend running M<sup>3</sup>Spade on a GPU for optimal performance.

Our experiments were conducted using Python 3.8.20 with CUDA 11.8. We recommend using Anaconda or Miniconda to create an isolated conda environment for running M<sup>3</sup>Spade.

**1. Create a virtual environment:**
```bash
conda create -n M3Spade python==3.8.20
```
**2. Activate the environment:**
```bash
conda activate M3Spade
```
**3. Install required packages:**

> We provide a shell script for dependency installation at [`env/pkg_install.sh`](env/pkg_install.sh).  

## Data Availability

We provide ten example datasets to demonstrate how to use M<sup>3</sup>Spade for predicting spatial transcriptomic drug responses. These include:
*   **MC38, B16, hCRC:** Three slides each.
*   **hCRC Visium HD:** One high-resolution slide.

All example datasets, pre-trained models, and prediction results are available on Zenodo: [**Download Here**](https://zenodo.org/records/18211668)

Please download the datasets and place the `data` directory in the same directory as `M3Spade.py`.

## Usage
```bash
Usage: M3Spade.py [options]

Required:
      --drug_name STRING: Name of the drug(s). For multiple drugs, separate with commas (e.g., "Gefitinib,Docetaxel")
      --species STRING: Species of the spatial data: "hs" (Human) or "mus" (Mouse)
      --spatial_count_path STRING: Path to the spatial count CSV file or H5 file
      --spatial_coord_path STRING: Path to the spatial category coordinate CSV file

Optional:
    # Hardware and Path
      --device STRING: Device to use: "gpu" or "cpu" (default: gpu)
    
    # Model Architecture
      --hiddens_graph STRING: Hidden layer dimensions for the graph network (default: 1024,512,128)
      --hiddens_linear STRING: Hidden layer dimensions for the linear network (default: 64,16)
      --drugFunc STRING: GNN function type for the drug graph: "GIN" or "GINE" (default: GIN)

    # Data Processing
      --gene_num INT: Number of highly variable genes to select (default: 500)
      --k_neigh INT: Number of neighbors for spatial graph (default: 6)
      --test_size FLOAT: Test split proportion (default: 0.2)
      --sampling STRING: Sampling strategy (default: SMOTE)
      
    # Training Hyperparameters
      --lr FLOAT: Learning rate (default: 0.001)
      --weight_decay FLOAT: Weight decay (default: 0.0001)
      --grad_clip FLOAT: Gradient clipping value (default: 1.0)
      --patience INT: Early stopping patience (default: 25)
      --warm_up INT: Warm-up epochs (default: 10)
      --dropout_rate FLOAT: Graph dropout rate (default: 0.3)
      --alpha FLOAT: TsallisEntropy parameter 1 (default: 1.8)
      --temperature FLOAT: TsallisEntropy parameter 2 (default: 2.5)
      --threshold FLOAT: Threshold parameter (default: 0.95)

    # Flags
      --save_mid BOOL: Flag to save intermediate calculation results (True/False) (default: False)
      --perform_normalize BOOL: Whether to perform filtering and log-normalization on spatial data (True/False) (default: True)
```

P.S. We evaluated the model using pseudo-spatial data generated from six gold-standard single-cell datasets. The default parameters provided in this repository were selected based on their robust performance across these tests. These settings are suitable for most tasks and serve as an excellent starting point for personalized parameter tuning.

### Examples

#### 1. Single Drug Prediction
To predict the response for a single drug (e.g., Afatinib):

```bash
python M3Spade.py --drug_name Afatinib,Oxaliplatin --species hs --device gpu --spatial_count_path ./data/spatial_data/CRC_P6/count.csv --spatial_coord_path ./data/spatial_data/CRC_P6/category_coord.csv
```

#### 2. Combination Therapy Prediction
To predict the response for a combination of drugs (e.g., Afatinib and Oxaliplatin):

```bash
python M3Spade.py --drug_name Afatinib, --species hs --device gpu --spatial_count_path ./data/spatial_data/CRC_P6/count.csv --spatial_coord_path ./data/spatial_data/CRC_P6/category_coord.csv
```

---

### Important Notes

#### Note 1: Predicting Unseen Drugs
For any drug, regardless of whether it exists in our internal database, the implementation of M3Spade remains consistent. 
*   **Interactive Mode:** If a drug is not found in the database, M3Spade will prompt the user to input the IsoSMILES structure of the drug interactively. Once entered, the model automatically performs prediction.
*   **Non-Interactive Mode (HPC/SLURM):** For users running on clusters (e.g., SLURM) where interactive input is not feasible, we provide `M3Spade_noninteractive.py`. Users must edit the `MANUAL_DICT` dictionary within this script to manually define the IsoSMILES for unseen drugs before execution.

#### Note 2: IC50 Data Processing
As large-scale drug screening data expands, new IC50 data may become available. To enhance model scalability, we implemented an IC50 binarization method based on established theoretical research (e.g., *[Insert Citation Here]*).
*   The script is available here: [`preprocess/IC50_binarize.R`](preprocess/IC50_binarize.R)

#### Note 3: High-Resolution Data (Superspot)
For high-resolution spatial transcriptomics data, we recommend aggregating spots into "superspots" before running M3Spade. This reduces data sparsity and file size, thereby improving computational efficiency.
*   The tutorial script is available here: [`preprocess/superspot_tutorial.R`](preprocess/superspot_tutorial.R)

---

## Outputs

M3Spade generates the following files in the output directory.

**For Single Drug Prediction:**
```text
Output Directory
├── {drug_name}_best.npy          # Best model prediction results (NumPy array)
├── {drug_name}_best.pth          # Saved model weights (PyTorch)
├── {drug_name}_output.txt        # Prediction logs and text output
└── {drug_name}_sensitivity.pdf   # Visualization of drug sensitivity
```

**For Combination Therapy Prediction:**
(Includes all single drug files plus the following combination files in the same directory)
```text
Output Directory
├── {drug_name}_best.npy          # ... (Individual drug files generated as above)
│
├── {combine_drugs}_best.npy          # Combined response predictions
├── {combine_drugs}_output.txt        # Combined response logs
└── {combine_drugs}_sensitivity.pdf   # Visualization of combined sensitivity
```

## Citation
(Unpublished now)
```bibtex
@article{M$^{3}Spade,
    title={M$^{3}$Spade: A Multi-Modal Deep Learning Framework for Predicting Spatially Resolved Drug Responses},
    author={Zihao Zhang and Xinyu Cui and Zhengke Lian and Xiufeng Pang and Youqiong Ye and Cizhong Jiang},
    journal={XX},
    year={2025},
    doi={xx}
}
```
## Contacts

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

## Usage
```bash
Usage: M3Spade.py [options]

Required:
      --drug_name STRING: Name of the drug(s). For multiple drugs, separate with commas (e.g., "Gefitinib,Docetaxel")
      --species STRING: Species of the spatial data: "hs" (Human) or "mus" (Mouse)
      --spatial_count_path STRING: Path to the spatial raw count CSV file 
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
      --save_mid BOOL: Flag to save intermediate calculation results (True/False) (default: False)
      
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
```

我们基于六个单细胞金标准数据集产生了pseudo-spatial数据进行了模型测试，选择了均获得良好效果的公共参数作为default参数设置，可适应大部分任务，也可作为个性化参数调整的起始点。

We provided nine examples 分别为MC38、B16、hCRC各自三张片子show you how to use M<sup>3</sup>Spade to 预测空间转录组药物响应。




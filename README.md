# M<sup>3</sup>Spade
## Introduction
**M<sup>3</sup>Spade** (<ins>**M**</ins>ulti-<ins>**M**</ins>odal <ins>**M**</ins>odel for predicting <ins>**Spa**</ins>tial <ins>**D**</ins>rug <ins>**E**</ins>fficacy) is a versatile framework designed for predicting drug sensitivity within spatial transcriptomics data. It is resolution-agnostic, capable of processing data ranging from single-cell to spot-level resolutions, and supports generalizable prediction of responses to previously unseen drugs based on their chemical structures.

M<sup>3</sup>Spade facilitates the following analyses:

*   **Binarized Sensitivity Prediction**  
    Performs binary classification of drug sensitivity at the individual cell or spot level (Sensitive vs. Resistant).

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
conda activate M3Spade
```







我们在六个单细胞金标准数据集测试了模型，选择了均获得良好效果的公共参数，如default设置，可适应大部分任务，也可作为调整参数的起始点。

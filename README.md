# M<sup>3</sup>Spade
M<sup>3</sup>Spade (<ins>**M**</ins>ulti-<ins>**M**</ins>odal <ins>**M**</ins>odel for predicting <ins>**Spa**</ins>tial <ins>**D**</ins>rug <ins>**E**</ins>fficacy) is a versatile framework designed for predicting drug sensitivity within spatial transcriptomics data. It is resolution-agnostic, capable of processing data ranging from single-cell to spot-level resolutions.

## Key Features

M3Spade facilitates the following analyses:

*   **Binarized Sensitivity Prediction**  
    Performs binary classification of drug sensitivity at the individual cell or spot level (Sensitive vs. Resistant).

*   **Spatial Autocorrelation Analysis**  
    Quantifies global spatial dependency and clustering patterns using **Join Count statistics**.

*   **Combinatorial Therapy Assessment**  
    Predicts and evaluates drug sensitivity outcomes for drug combinations.

<img src="img/model.png" width="80%" alt="model architecture">


我们在六个单细胞金标准数据集测试了模型，选择了均获得良好效果的公共参数，如default设置，可适应大部分任务，也可作为调整参数的起始点。
We strongly recommend running the M3Spade model on a GPU.

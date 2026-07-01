# ==============================================================================
# SuperSpot (Meta-Spot) Analysis and H5AD Generation Pipeline
# ==============================================================================
#
# ==============================================================================
# Program Description & Usage Guide
# ==============================================================================
#
# Reference: https://github.com/GfellerLab/SuperSpot
# [Program Function]
# This pipeline performs "SuperSpot" (Meta-Spot) aggregation on high-resolution 
# spatial transcriptomics data (e.g., Visium HD). It merges spatially adjacent 
# and transcriptionally similar spots/cells into Metaspots to reduce sparsity 
# and computational burden, and finally converts the result to H5AD format.
#
# [Key Algorithm Parameters]
# The following parameters in Step 2 determine the granularity of SuperSpots:
# 1. g_val (Gamma) : Granularity level. Higher values result in fewer, larger spots.
# 2. n_pc_use      : Number of Principal Components used for similarity calculation.
# 3. k_knn         : Number of neighbors used for graph construction.
#
# [Input Requirements (Step 1)]
# The script requires the user to prepare three specific R objects in Step 1:
# 1. normcounts_mat  : A gene expression matrix (Genes x Cells).
# 2. normcounts_meta : A metadata DataFrame with row names matching cells, 
#                      containing at least a "DefineTypes" column for annotations.
# 3. normcounts_coord: A coordinate DataFrame (row names = cells) with columns 
#                      "imagerow" and "imagecol".
#
# [Output Files]
# The script saves intermediate results and final training data:
# 1. MC_centroids.rds   : Spatial coordinates (X, Y) of the generated SuperSpots.
# 2. MC_ge.rds          : Aggregated gene expression matrix of SuperSpots.
# 3. MC_DefineTypes.rds : Dominant cell type annotation for each SuperSpot.
# 4. MC_membership.rds  : Mapping relationship between original cells and SuperSpots.
#
# *** [CRITICAL INPUTS FOR M3Spade] ***
# 5. category_coord.csv : CSV file containing spatial coordinates (x,y) and labels.
# 6. spatial_data.h5ad  : The final Anndata object (converted from Seurat) 
#                         containing normalized expression data.
# ==============================================================================

rm(list = ls())

# Load required libraries
library(Seurat)
library(SuperSpot)
library(SuperCell)
library(tidyr)
library(tidyverse)
library(igraph)
library(Matrix)
library(SeuratDisk)

# ==============================================================================
# Step 1: Data Preparation
# ==============================================================================

# Note: Please modify the loading paths below to fit your actual data sources.
# Ensure the final objects (mat, meta, coord) match the format described above.

# --- 1. Load Expression Matrix ---
# Example: Reading from a processed Seurat RDS file
input_rds_path <- 'bin2cell_DefineTypes.rds.gz'
hd_data <- readRDS(input_rds_path)
DefaultAssay(hd_data) <- "RNA"
normcounts_mat <- GetAssayData(hd_data, assay = "RNA", slot = "counts")

# --- 2. Prepare Metadata ---
# Extracting cell type annotations
normcounts_meta <- hd_data@meta.data[, c("nFeature_RNA", "DefineTypes")]

# --- 3. Prepare Spatial Coordinates ---
# Example: Reading from a CSV file
coord_path <- "spatial_coordinates.csv"
hd_coord <- read.csv(coord_path)

# Formatting coordinates to match the required interface
# Ensuring cell names match the expression matrix
hd_coord[,1] <- paste0("Cell_", hd_coord[,1]) 
colnames(hd_coord) <- c("names", "imagerow", "imagecol")

normcounts_coord <- dplyr::select(hd_coord, c("names", "imagerow", "imagecol")) %>% 
  column_to_rownames("names")

# Clean up intermediate objects to save memory
rm(hd_data, hd_coord)
gc()

# ==============================================================================
# Step 2: Metaspot (SuperSpot) Generation
# ==============================================================================

# --- Configuration Parameters ---
g_val <- 40       # Gamma: Granularity level for SuperSpot (higher = fewer spots)
n_pc_use <- 1:5   # Number of Principal Components to use
k_knn <- 16       # Number of neighbors for graph construction

# --- Run SuperSpot Analysis ---
# SCimplify_SpatialDLS: Main function to coarsen the graph
hddata_DLS <- SCimplify_SpatialDLS(
  X = normcounts_mat,
  spotPositions = normcounts_coord,
  method_similarity = "1",
  split_not_connected = TRUE,          # Split metaspots that are spatially disconnected
  genes.use = rownames(normcounts_mat),
  gamma = g_val,
  n.pc = n_pc_use,
  method_knn = "1",
  k.knn = k_knn,
  method_normalization = "log_normalize",
  cell.annotation = normcounts_meta$DefineTypes
)

# --- Assign Membership and Calculate Purity ---
# Store membership info in metadata
normcounts_meta[, str_c("MC_membership_", g_val)] <- as.character(hddata_DLS$membership)

# Calculate purity of the resulting metaspots based on original cell types
# Options: "max_proportion" or "entropy"
method_purity <- "max_proportion" 
hddata_DLS$purity <- supercell_purity(
  clusters = normcounts_meta$DefineTypes,
  supercell_membership = hddata_DLS$membership, 
  method = method_purity
)

print(paste0("Mean purity of SuperSpots: ", mean(hddata_DLS$purity)))

# Assign the dominant cell type to each SuperSpot
hddata_DLS$DefineTypes <- supercell_assign(
  clusters = normcounts_meta$DefineTypes,
  supercell_membership = hddata_DLS$membership,
  method = "absolute"
)

# ==============================================================================
# Step 3: Visualization and Refinement
# ==============================================================================

# --- Generate Shapes ---
hddata_DLS$polygons <- supercell_metaspots_shape(
  MC = hddata_DLS,
  spotpositions = normcounts_coord,
  annotation = "DefineTypes",
  concavity = 2,
  membership_name = "membership"
)

# Initial Plot
plot1 <- SpatialDimPlotSC(
  original_coord = normcounts_coord,
  MC = hddata_DLS,
  sc.col = "DefineTypes",
  sc.col2 = str_c("MC_membership_", g_val),
  polygons_col = "polygons",
  meta_data = normcounts_meta
) + NoLegend()
# print(plot1)

# --- Refinement: Split Unconnected Components ---
# This ensures that a single SuperSpot ID does not span across separated tissue regions
hddata_DLS.spl <- split_unconnected(hddata_DLS)

# Update membership in metadata
normcounts_meta[, str_c("MC_membership_spl_", g_val)] <- as.character(hddata_DLS.spl$membership)

# Re-assign cell types to the split SuperSpots
hddata_DLS.spl$DefineTypes <- supercell_assign(
  clusters = normcounts_meta$DefineTypes,
  supercell_membership = hddata_DLS.spl$membership,
  method = "absolute"
)

# Re-calculate shapes
hddata_DLS.spl$polygons <- supercell_metaspots_shape(
  MC = hddata_DLS.spl,
  spotpositions = normcounts_coord,
  annotation = "DefineTypes",
  concavity = 2,
  membership_name = "membership"
)

# Final Visualization
SpatialDimPlotSC(
  original_coord = normcounts_coord,
  MC = hddata_DLS.spl,
  sc.col = "DefineTypes",
  sc.col2 = str_c("MC_membership_spl_", g_val),
  polygons_col = "polygons",
  meta_data = normcounts_meta
) +
  NoLegend() + 
  theme(
    plot.background = element_rect(fill = 'black'),
    panel.background = element_rect(fill = 'black'),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

# --- Data Extraction ---
# 1. Centroids
MC_centroids <- supercell_spatial_centroids(hddata_DLS.spl, spotPositions = normcounts_coord)
# 2. Aggregated Expression
MC_ge <- superspot_GE(
  MC = hddata_DLS.spl,
  ge = as.matrix(normcounts_mat),
  groups = as.numeric(hddata_DLS.spl$membership),
  mode = "sum"
)

# ==============================================================================
# Step 4: Save Intermediate Results
# ==============================================================================

save_dir <- "output dir" # Update output path for intermediate files
if(!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)

saveRDS(MC_centroids, file = file.path(save_dir, "MC_centroids.rds"))
saveRDS(MC_ge, file = file.path(save_dir, "MC_ge.rds"))
saveRDS(hddata_DLS.spl$DefineTypes, file = file.path(save_dir, "MC_DefineTypes.rds"))
saveRDS(hddata_DLS.spl$membership, file = file.path(save_dir, "MC_membership.rds"))

print("Intermediate SuperSpot data saved.")

# ==============================================================================
# Step 5: Reload Data (Optional Checkpoint)
# ==============================================================================

# --- Memory Cleanup ---
# Remove large objects from previous steps to free up RAM before reloading
rm(hddata_DLS, hddata_DLS.spl, normcounts_mat, normcounts_coord, normcounts_meta)
gc() # Force garbage collection

MC_centroids <- readRDS(file.path(save_dir, "MC_centroids.rds"))
MC_ge <- readRDS(file.path(save_dir, "MC_ge.rds"))
MC_DefineTypes <- readRDS(file.path(save_dir, "MC_DefineTypes.rds"))
MC_membership <- readRDS(file.path(save_dir, "MC_membership.rds"))

# ==============================================================================
# Step 6: Create Training Data (H5AD format)
# ==============================================================================

# Set output directory for M3Spade inputs
out_dir <- file.path(save_dir, "traindata")
if(!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# --- 1.Metadata Preparation ---
category_coord <- data.frame(MC_centroids[, 1:3], MC_DefineTypes)
colnames(category_coord) <- c("name", "x", "y", "cell_type")

# Simplified labeling (Customize as needed)
category_coord$cell_type <- ifelse(category_coord$cell_type == "TumorCells", "Cancer", "Other")
category_coord$name <- paste0("Cell_", category_coord$name) # Using your setting

# --- Save Coordinates for M3Spade (Critical Input for M3Spade) ---
# Note: Saving to 'out_dir' to keep paired with the h5ad file
write.csv(category_coord, 
          file = file.path(out_dir, "category_coord.csv"), 
          row.names = FALSE, 
          quote = FALSE)

# --- 2.Expression Matrix Preparation ---
MC_ge_transposed <- t(MC_ge)
rownames(MC_ge_transposed) <- category_coord$name
counts_mat <- t(MC_ge_transposed)

# --- Create Seurat Object ---
hd_data <- CreateSeuratObject(
  counts = counts_mat,
  assay = "RNA",
  project = "high_reso",
  meta.data = NULL
)
DefaultAssay(hd_data) <- "RNA"

# --- 3.Gene Filtering ---
# Calculate the number of cells (SuperSpots) in which each gene is expressed (non-zero count)
counts_mat_seurat <- GetAssayData(hd_data, assay = "RNA", slot = "counts")
gene_nonzero_cells <- rowSums(counts_mat_seurat != 0)

# Identify genes expressed in at least 5 cells
genes_keep <- names(gene_nonzero_cells)[gene_nonzero_cells >= 5]

# Subset the Seurat object to keep only valid genes
hd_data <- subset(hd_data, features = genes_keep)
message(paste("Genes retained after filtering:", length(genes_keep)))

# --- 4.Normalization ---
# Apply LogNormalize with a scale factor of 1e6
# (This corresponds to 'target_sum = 1e6' in Python Scanpy workflows)
hd_data <- NormalizeData(
  hd_data,
  normalization.method = "LogNormalize",
  scale.factor = 1e6
)

# Verify that the 'data' slot has been generated successfully
if (!is.null(GetAssayData(hd_data, assay = "RNA", slot = "data")) &&
    length(GetAssayData(hd_data, assay = "RNA", slot = "data")) > 0) {
  dd <- GetAssayData(hd_data, assay = "RNA", slot = "data")
  message("Normalization successful. Data dimensions: ", paste(dim(dd), collapse = " x "))
} else {
  stop("Error: @data slot missing. Please check NormalizeData execution.")
}

# --- 5.Save as H5AD ---

h5seurat_file <- file.path(out_dir, "spatial_data.h5Seurat")

# Fix for Seurat v5 object compatibility with SeuratDisk
hd_data[["RNA"]] <- as(object = hd_data[["RNA"]], Class = "Assay")

# Save as h5Seurat
SaveH5Seurat(hd_data, filename = h5seurat_file, overwrite = TRUE)

# Convert h5Seurat to h5ad format
Convert(h5seurat_file, dest = "h5ad", overwrite = TRUE)

print(paste("All steps completed. M3Spade inputs saved to:", out_dir))

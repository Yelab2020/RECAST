# ==============================================================================
# IC50 Data Binarization and Sensitivity Analysis
# ==============================================================================

# ==============================================================================
# Program Description & Usage Guide
# ==============================================================================
#
# [Modifications Required]
# 1. Input File Path: Update 'read.csv("input file", ...)' with the actual path to your dataset.
# 2. Output Directory: Update 'setwd("output dir")' with the path to the folder where you want to save results.
#
# [Input File]
# - Format: CSV file
# - Required Columns:
#   1. ModelID (Cell Line ID)
#   2. Name (Drug Name)
#   3. LN_IC50 (Natural Logarithm of IC50)
#
# [Output Files]
# 1. IC50_df.csv        : Processed IC50 numeric matrix (Rows = Cell Lines, Columns = Drugs).
# 2. IC50_binary_df.csv : Binarized sensitivity matrix ('R' = Resistant, 'S' = Sensitive).
# 3. cutoff_dict.RData  : RData file storing the calculated binarization thresholds for each drug.
# 4. [DrugName]_plot.pdf: Individual visualization plots showing density distribution, curve fitting, and thresholds.
# 5. Resistant_Proportion_Boxplot.pdf: Boxplot summarizing the proportion of resistant cell lines across all drugs.
#
# ==============================================================================

rm(list = ls())

# Load Input Data
IC50 <- read.csv("input file", header = TRUE)
# Input Data Demo Structure:
# ModelID	Name	PubCHEM	Drug.Id	SampleCollectionSite	LN_IC50	AUC	Z_SCORE
# ACH-001711	Gefitinib	123631	1010	central_nervous_system	2.100251	0.953363	-0.862438
# ACH-001711	Paclitaxel	36314	1080	central_nervous_system	-4.03161	0.916581	-0.532133
# ACH-001711	AZD3759	78209992	1915	central_nervous_system	3.107089	0.987487	0.555013

# Set Working Directory
setwd("output dir")

# Initialize list to store cutoff values
cutoff_dict <- list()

# Get list of unique drug names
name_list <- unique(IC50$Name)

# ==============================================================================
# Step 1: Data Binarization
# Note: Thresholds are calculated based on the Natural Logarithm (LN) of IC50
# ==============================================================================

for (drug in name_list) {
  
  # Extract IC50 data for the current drug
  assigned_data <- IC50[IC50$Name == drug, "LN_IC50"]
  
  assigned_data <- as.numeric(na.omit(assigned_data))
  
  # ----------------------------------------------------------------------------
  # 1.1 Up-sampling
  # ----------------------------------------------------------------------------
  set.seed(20240803)
  
  # Parameters for data augmentation
  num_points <- 1000  # Number of random points per cell line
  sd_value <- 0.2     # Standard deviation for normal distribution
  
  # Generate normal distribution around existing data points
  data_points <- lapply(assigned_data, function(ic50) {
    rnorm(num_points, mean = ic50, sd = sd_value)
  })
  
  # Flatten list to vector
  all_points <- unlist(data_points)
  
  # ----------------------------------------------------------------------------
  # 1.2 Kernel Density Estimation (KDE)
  # ----------------------------------------------------------------------------
  density_est <- density(all_points, bw = 0.5, n = 10000)
  
  # ----------------------------------------------------------------------------
  # 1.3 Modeling Resistant Cell Lines (Finding Peak and Inflection Points)
  # ----------------------------------------------------------------------------
  
  # Identify Mu (μ): The peak of the density function
  mu <- density_est$x[which.max(density_est$y)]
  
  # Calculate first derivative (f')
  f_prime <- density_est$y
  x_values <- density_est$x
  f_prime_values <- diff(c(0, f_prime)) / diff(x_values)[1]
  
  # Identify Theta (θ): The inflection point satisfying specific rules
  theta <- NULL
  integral_threshold <- 0.05
  
  # Rule 1: Search for theta based on first derivative properties
  for (i in 1:(length(f_prime_values) - 1)) {
    if (x_values[i] < mu && 
        f_prime[i] < 0.8 * max(f_prime) && 
        sum(f_prime[1:i]) * diff(x_values)[1] > integral_threshold && 
        f_prime_values[i+1] * f_prime_values[i] < 0) {
      theta <- x_values[i]
    }
  }
  
  # Rule 2: If Rule 1 fails, search based on third derivative (change in curvature)
  if (is.null(theta)) {
    f_double_prime <- diff(f_prime_values) / diff(x_values)[1]
    f_treble_prime <- diff(f_double_prime) / diff(x_values)[1]
    
    for (i in 1:(length(f_treble_prime) - 1)) {
      if (x_values[i] < mu && 
          f_treble_prime[i] > 0 &&
          f_prime[i] < 0.8 * max(f_prime)  && 
          sum(f_prime[1:i]) * diff(x_values)[1] >= integral_threshold && 
          f_double_prime[i+1] * f_double_prime[i] < 0) {
        theta <- x_values[i]
      }
    }
  }
  
  # Fallback: If no theta found, use the minimum X value
  if (is.null(theta)) {
    theta <- min(x_values)
  }
  
  # Calculate Sigma (σ): Median Absolute Deviation (MAD) for the resistant population
  points_in_range <- all_points[all_points >= theta & all_points <= mu]
  sigma <- median(abs(points_in_range - mu))
  
  # Debug Output
  print(paste0("[", drug, "] Mu: ", round(mu, 3), " | Theta: ", round(theta, 3), " | Sigma: ", round(sigma, 3)))
  
  # ----------------------------------------------------------------------------
  # 1.4 Determine Binarization Threshold (t)
  # ----------------------------------------------------------------------------
  # Calculate the 3rd percentile (p=0.03) of the estimated normal distribution
  b <- qnorm(0.03, mean = mu, sd = sigma)
  
  cutoff_dict[[drug]] <- b
  
  # ----------------------------------------------------------------------------
  # 1.5 Visualization
  # ----------------------------------------------------------------------------
  # Sanitize filename (remove characters that might cause issues)
  safe_filename <- gsub("[^A-Za-z0-9]", "_", drug)
  pdf(paste0(safe_filename, "_distribution_plot.pdf"), width = 7.11, height = 3.92)
  
  # Plot Histogram
  hist(all_points, breaks = seq(min(all_points), max(all_points) + 0.1, by = 0.1),
       freq = FALSE,
       main = paste0("Density Distribution: ", drug),
       xlab = "LN(IC50) Values", ylab = "Frequency",
       xlim = c(-7, 7),
       ylim = c(0, 0.8),
       col = "#F0F0F0", border = "gray")
  
  # Add KDE line
  lines(density_est, col = "blue", lwd = 3)
  
  # Mark Mu
  abline(v = mu, col = "#CA8787", lty = 2, lwd = 3)
  text(mu, 0.4, "mu", pos = 2, offset = 0.5, cex = 1.2, font = 2, col = "#CA8787")
  
  # Mark Theta
  abline(v = theta, col = "#F19ED2", lty = 2, lwd = 2)
  text(theta, 0, "theta", pos = 2, offset = 0.5, cex = 1.2, font = 2, col = "#F19ED2")
  
  # Plot Estimated Normal Distribution
  normal_density <- dnorm(x_values, mean = mu, sd = sigma)
  lines(x_values, normal_density, col = "#55AD9B", lty = 1, lwd = 3)
  
  # Mark Threshold (b)
  abline(v = b, col = "#FF8000", lty = 1, lwd = 3)
  text(b, 0.1, "t=0.03", pos = 2, offset = 0.5, cex = 1.2, font = 2, col = "#FF8000")
  
  dev.off()
}

# Save the dictionary of cutoffs
save(cutoff_dict, file = "cutoff_dict.RData")


# ==============================================================================
# Step 2: Create Sensitivity Data Matrix
# ==============================================================================

# Initialize DataFrame
IC50_df <- data.frame(matrix(nrow = length(unique(IC50$ModelID)), 
                             ncol = length(unique(IC50$Name))))
rownames(IC50_df) <- unique(IC50$ModelID)
colnames(IC50_df) <- unique(IC50$Name) # Use original names first to match cutoff_dict

# Populate Matrix
# Note: Iterating over unique names is faster than iterating over every row
unique_drugs <- unique(IC50$Name)
for (drug in unique_drugs) {
  tmp_df <- IC50[IC50$Name == drug, c("ModelID", "LN_IC50")]
  # Match rows
  if(nrow(tmp_df) > 0){
    IC50_df[tmp_df$ModelID, drug] <- as.numeric(tmp_df$LN_IC50)
  }
}

# Apply Binarization (Resistant 'R' vs Sensitive 'S')
IC50_binary_df <- IC50_df
for (col_name in names(cutoff_dict)) {
  # If value >= cutoff, it is Resistant
  IC50_binary_df[, col_name] <- ifelse(IC50_binary_df[, col_name] >= cutoff_dict[[col_name]], 'R', 'S')
}

# Optional: Rename columns to safe format after processing
name_trans <- gsub("-", "_", colnames(IC50_df))
name_trans <- gsub(" ", "*", name_trans)
name_trans <- gsub("/", "+", name_trans)
colnames(IC50_df) <- name_trans
colnames(IC50_binary_df) <- name_trans


# ==============================================================================
# Step 3: Save Sensitivity Data
# ==============================================================================

# IC50 Data (Continuous)
write.csv(IC50_df, file = "IC50_df.csv", row.names = TRUE, quote = FALSE)

# IC50 Binarized Data (Discrete)
write.csv(IC50_binary_df, file = "IC50_binary_df.csv", row.names = TRUE, quote = FALSE)


# ==============================================================================
# Step 4: Proportion Analysis & Boxplot
# ==============================================================================

# Calculate proportion of Resistant ('R') cell lines per drug
prop_result <- apply(IC50_binary_df, 2, function(x) {
  # Handle cases where 'R' might not exist in the column
  tbl <- table(x)
  if("R" %in% names(tbl)) {
    return(prop.table(tbl)['R'])
  } else {
    return(0)
  }
})

# --- Save Boxplot (Requested Feature) ---

# Save as PDF (Vector graphic, best for publication)
pdf("Resistance_Proportion_Boxplot.pdf", width = 6, height = 6)

boxplot(prop_result, 
        main = "Distribution of Resistant Cell Line Proportions",
        ylab = "Proportion of Resistant Lines",
        col = "#C8E8F6",
        border = "#404040", # Dark grey border
        outcex = 0.8,       # Outlier point size
        pch = 19,           # Solid circle for outliers
        lwd = 1.5)          # Line width

grid(nx = NA, ny = NULL, col = "lightgray", lty = "dotted") # Add grid lines for readability

dev.off()

print("Analysis Complete. All files saved.")
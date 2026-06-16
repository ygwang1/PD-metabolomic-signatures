# PCA and OPLS-DA analysis for PD metabolomics
# Reference: "Metabolomic signatures for diagnosis and clinical severity in Parkinson's disease"

# ============================================================================
# Environment setup
# ============================================================================
rm(list = ls())

# Set working directory to the location of input files
# setwd("path/to/data")

library(ggpubr)
library(ggprism)
library(patchwork)
library(ggplot2)
library(ropls)
library(factoextra)
library(vegan)
library(dplyr)
library(tidyr)
library(reshape2)
library(pheatmap)
library(RColorBrewer)

# Color palette for PD vs NC groups
group_colors <- c("#4B74B2", "#E73847")

# ============================================================================
# Data loading
# ============================================================================
# Log10-transformed and Z-score standardized metabolomics data per cohort
main_data <- read.csv("data/example_cohort1_data.csv",
                      sep = ",", encoding = "UTF-8", header = TRUE,
                      check.names = FALSE, na.strings = c("NA", "", "N/F"))

# Load QC-passed metabolites
all_met <- read.csv("data/metabolite_list.csv")$metabolite

# Prepare data matrices
metabolites_data_log10_scaled <- main_data[all_met]
data_matrix <- metabolites_data_log10_scaled
sample_metadata <- main_data[, c("age", "sex", "bmi", "diagnosis")]
sample_metadata$diagnosis <- as.factor(sample_metadata$diagnosis)

# ============================================================================
# PCA
# ============================================================================
pca_result <- prcomp(data_matrix, center = FALSE, scale = FALSE)

# Explained variance
var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cum_var <- cumsum(var_explained)
k_80 <- which(cum_var >= 0.8)[1]
k_90 <- which(cum_var >= 0.9)[1]

# ---- Figure 1: Scree plot with cumulative variance ----
df <- data.frame(
  PC = 1:length(var_explained),
  Variance = var_explained,
  Cumulative = cum_var
)

p_scree <- ggplot(df, aes(x = PC)) +
  geom_bar(aes(y = Variance), stat = "identity", fill = "#4B74B2") +
  geom_line(aes(y = Cumulative), linewidth = 1, color = "#DB3124") +
  geom_point(aes(y = Cumulative), size = 1.5, color = "#DB3124") +
  geom_hline(yintercept = 0.8, linetype = "dashed") +
  geom_vline(xintercept = k_80, linetype = "dashed") +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  geom_vline(xintercept = k_90, linetype = "dashed") +
  annotate(
    "text", x = k_80, y = 0.3,
    label = paste0("80% variance (PC", k_80, ")"),
    angle = 90, vjust = -0.5
  ) +
  annotate(
    "text", x = k_90, y = 0.3,
    label = paste0("90% variance (PC", k_90, ")"),
    angle = 90, vjust = -0.5
  ) +
  labs(
    x = "Principal Component",
    y = "Explained Variance",
    title = "Scree Plot with Cumulative Variance"
  ) +
  theme_bw()

ggsave("fig_pca_scree.pdf", plot = p_scree, width = 10, height = 6, device = "pdf")

# ---- Figure 2: PCA score plot (PC1 vs PC2) ----
pca_scores <- as.data.frame(pca_result$x)
pca_scores$group <- sample_metadata$diagnosis
variance_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2) * 100

pca_plot <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = group)) +
  geom_point(size = 3.5, alpha = 0.7) +
  stat_ellipse(
    geom = "polygon", level = 0.95,
    linetype = 2, linewidth = 0.5, aes(fill = group),
    alpha = 0.1, show.legend = TRUE
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  scale_color_manual(values = group_colors) +
  scale_fill_manual(values = group_colors) +
  labs(
    x = paste0("PC1 (", round(variance_explained[1], 2), "%)"),
    y = paste0("PC2 (", round(variance_explained[2], 2), "%)")
  ) +
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.position = c(0.85, 0.1),
    legend.text = element_text(color = "black", size = 12),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    axis.title.x = element_text(color = "black", size = 14, face = "bold"),
    axis.title.y = element_text(color = "black", size = 14, face = "bold", angle = 90),
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 12),
    axis.ticks = element_line(color = "black")
  )

ggsave("fig_pca_score.pdf", plot = pca_plot, width = 6, height = 6, device = "pdf")

# ============================================================================
# OPLS-DA
# ============================================================================
# orthoI = NA: automatically determines the number of orthogonal components via CV
oplsda <- opls(
  data_matrix, sample_metadata$diagnosis,
  predI = 1, orthoI = NA,
  permI = 999,
  scaleC = "none",
  crossvalI = 7
)

# ---- Figure 3: OPLS-DA score plot ----
oplsda_scores <- as.data.frame(oplsda@scoreMN)
o1 <- oplsda@orthoScoreMN[, 1]
oplsda_scores$o1 <- o1
oplsda_scores$group <- sample_metadata$diagnosis
oplsda_scores$samples <- rownames(oplsda_scores)

x_lab <- oplsda@modelDF[1, "R2X"] * 100

p_oplsda <- ggplot(oplsda_scores, aes(x = p1, y = o1, color = group)) +
  geom_point(size = 3.5, alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.5) +
  stat_ellipse(
    geom = "polygon", level = 0.95,
    linetype = 2, linewidth = 0.5, aes(fill = group),
    alpha = 0.1, show.legend = TRUE
  ) +
  scale_color_manual(values = group_colors) +
  scale_fill_manual(values = group_colors) +
  labs(
    x = paste0("P1 (", x_lab, "%)"),
    y = "to1"
  ) +
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.position = c(0.85, 0.1),
    legend.text = element_text(color = "black", size = 12),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    axis.title.x = element_text(color = "black", size = 14, face = "bold"),
    axis.title.y = element_text(color = "black", size = 14, face = "bold", angle = 90),
    axis.text.y = element_text(size = 12),
    axis.text.x = element_text(size = 12),
    axis.ticks = element_line(color = "black")
  )

ggsave("fig_oplsda_score.pdf", plot = p_oplsda, width = 6, height = 6, device = "pdf")

# ---- OPLS-DA model summary ----
message("\nOPLS-DA Model Summary:")
summary(oplsda)

# ---- Figure 4: Permutation test ----
perm_results <- oplsda@suppLs$permMN

true_R2Y <- perm_results[1, "R2Y(cum)"]
true_Q2 <- perm_results[1, "Q2(cum)"]

perm_data <- data.frame(
  Q2 = perm_results[-1, "Q2(cum)"],
  R2Y = perm_results[-1, "R2Y(cum)"]
)

# Calculate p-values (proportion of permuted values >= true value)
p_R2Y <- sum(perm_data$R2Y >= true_R2Y) / nrow(perm_data)
p_Q2 <- sum(perm_data$Q2 >= true_Q2) / nrow(perm_data)

perm_data_long <- perm_data %>%
  pivot_longer(cols = everything(), names_to = "Metric", values_to = "Value")

p_perm <- ggplot(perm_data_long, aes(x = Value, fill = Metric)) +
  geom_histogram(
    position = "identity", alpha = 0.8, bins = 50, color = "dimgrey"
  ) +
  scale_fill_manual(values = c("Q2" = "#5E4FA2", "R2Y" = "#DF7A5E")) +
  geom_vline(
    xintercept = true_R2Y, color = "#DF7A5E",
    linetype = "dashed", linewidth = 1
  ) +
  geom_vline(
    xintercept = true_Q2, color = "#5E4FA2",
    linetype = "dashed", linewidth = 1
  ) +
  annotate(
    "text", x = true_R2Y + 0.05, y = 300,
    label = paste0(
      "R2Y: ", round(true_R2Y, 3), "  p < 0.001",
      " (", sum(perm_data$R2Y >= true_R2Y), "/", nrow(perm_data), ")"
    ),
    color = "black", hjust = 0.5, angle = 90, size = 4
  ) +
  annotate(
    "text", x = true_Q2 - 0.05, y = 300,
    label = paste0(
      "Q2: ", round(true_Q2, 3), "  p < 0.001",
      " (", sum(perm_data$Q2 >= true_Q2), "/", nrow(perm_data), ")"
    ),
    color = "black", hjust = 0.5, angle = 90, size = 4
  ) +
  labs(x = "Permuted Values", y = "Frequency") +
  theme_minimal() +
  theme(
    axis.title = element_text(color = "black", size = 12, face = "bold"),
    axis.text = element_text(size = 10)
  ) +
  scale_x_continuous(limits = c(-1, 1))

ggsave("fig_oplsda_permutation_test.pdf", plot = p_perm,
       width = 6, height = 5, device = "pdf")

# ============================================================================
# VIP scores, p(corr), and bootstrap stability
# ============================================================================
# Predictive component score
t_pred <- oplsda@scoreMN[, 1]

# p(corr): correlation of each metabolite with the predictive component
pcorr <- apply(data_matrix, 2, function(x) cor(x, t_pred))

# VIP scores from OPLS-DA model
vip_scores <- oplsda@vipVn

# Bootstrap stability of VIP scores
n_boot <- 1000
vip_mat <- matrix(NA, nrow = ncol(data_matrix), ncol = n_boot)
rownames(vip_mat) <- colnames(data_matrix)

set.seed(66)

for (i in 1:n_boot) {
  idx <- sample(1:nrow(data_matrix), replace = TRUE)
  X_boot <- data_matrix[idx, ]
  y_boot <- sample_metadata$diagnosis[idx]

  model_boot <- opls(
    X_boot, y_boot,
    predI = 1, orthoI = NA,
    permI = 0,
    scaleC = "none",
    crossvalI = 7
  )

  vip_mat[, i] <- model_boot@vipVn
}

# Stability metrics
vip_stability1 <- rowMeans(vip_mat > 1, na.rm = TRUE)
vip_stability2 <- rowMeans(vip_mat > 2, na.rm = TRUE)
vip_ci_upper <- apply(vip_mat, 1, function(x) quantile(x, 0.975, na.rm = TRUE))
vip_ci_lower <- apply(vip_mat, 1, function(x) quantile(x, 0.025, na.rm = TRUE))
vip_mean <- rowMeans(vip_mat, na.rm = TRUE)
vip_sd <- apply(vip_mat, 1, sd, na.rm = TRUE)

vip_results <- data.frame(
  metabolite = rownames(vip_mat),
  VIP = vip_scores,
  VIP_mean = vip_mean,
  VIP_sd = vip_sd,
  CI_lower = vip_ci_lower,
  CI_upper = vip_ci_upper,
  VIP_stability1 = vip_stability1,
  VIP_stability2 = vip_stability2,
  pcorr = pcorr
)

write.csv(vip_results, "oplsda_vip_results.csv", row.names = FALSE)

# ---- Figure 5: VIP heatmap for top metabolites ----
top_metabolites <- c(
  "12-HETE", "Adenosine", "Niacinamide", "Spermine",
  "Ile-Pro", "Spermidine", "3-Allylphenol sulfate",
  "Erucic acid (FFA(22:1n9))", "p-Cresol glucuronide",
  "4-Acetylphenol sulfate", "Chenodeoxycholic acid glycine conjugate",
  "Glycocholic acid"
)

vip_sub <- vip_mat[top_metabolites, ]
vip_stability <- rowMeans(vip_sub > 1)
annotation_row <- data.frame(Stability = vip_stability)
rownames(annotation_row) <- rownames(vip_sub)

p_heatmap <- pheatmap(
  vip_sub,
  color = colorRampPalette(brewer.pal(n = 9, name = "YlGnBu"))(10),
  annotation_row = annotation_row,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  border_color = NA,
  main = "VIP Stability Across Bootstrap Models"
)

ggsave("fig_vip_heatmap.pdf", plot = p_heatmap,
       bg = "white", width = 10, height = 5, device = "pdf")

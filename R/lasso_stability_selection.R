# LASSO feature selection with stability analysis for PD metabolomics
# Reference: "Metabolomic signatures for diagnosis and clinical severity in Parkinson's disease"

# ============================================================================
# Environment setup
# ============================================================================
rm(list = ls())

# Set working directory to the location of input files
# Users should adjust this path or set working directory before running
# setwd("path/to/data")

library(glmnet)
library(data.table)
library(psych)
library(fastDummies)
library(ggplot2)
library(reshape2)
library(pheatmap)
library(RColorBrewer)
library(dplyr)

# ============================================================================
# Data loading and preprocessing
# ============================================================================
# Training data was split 80:20 (stratified by diagnosis, fixed seed) in Python
train_data <- read.csv("train_data.csv",
                       sep = ",", header = TRUE,
                       check.names = FALSE, na.strings = c("NA", "", "N/F"))

# Load all QC-passed metabolites (n = 746)
metab_list <- read.csv("data/metabolite_list.csv")$metabolite

# ============================================================================
# Variable setup
# ============================================================================
# log10 transform metabolites
train_data[metab_list] <- lapply(train_data[metab_list], function(x) log10(x))

y <- train_data$diagnosis

# Clinical covariates included as unpenalized variables
covariates <- c("age", "sex", "bmi", "HBP", "diabetes")

all_vars <- c(covariates, metab_list)
missing_vars <- setdiff(all_vars, colnames(train_data))
if (length(missing_vars) > 0) {
  stop(paste("Missing variables:", paste(missing_vars, collapse = ", ")))
}

X_raw <- train_data[, all_vars]

# Standardize continuous variables (metabolites, age, BMI)
X_scaled <- as.data.frame(X_raw)
scale_vars <- c("age", "bmi", metab_list)
X_scaled[, scale_vars] <- scale(X_scaled[, scale_vars])
X <- as.matrix(X_scaled)

# Penalty factor: 0 for covariates (unpenalized), 1 for metabolites (penalized)
penalty_factor <- c(rep(0, length(covariates)), rep(1, length(metab_list)))

# ============================================================================
# LASSO cross-validation
# ============================================================================
set.seed(123)
cvfit <- cv.glmnet(
  x = X, y = y,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  penalty.factor = penalty_factor,
  type.measure = "auc",
  standardize = FALSE
)

lambda_min <- cvfit$lambda.min
lambda_1se <- cvfit$lambda.1se
auc_mean <- cvfit$cvm
auc_sd <- cvfit$cvsd

# Non-zero feature counts across lambda path
lambda_seq <- cvfit$lambda
nonzero_count <- apply(
  coef(cvfit, s = lambda_seq)[-1, , drop = FALSE], 2,
  function(col) sum(col != 0)
)
cv_df <- data.frame(
  lambda = lambda_seq,
  mean_auc = auc_mean,
  auc_sd = auc_sd,
  nonzero = nonzero_count
)

# ============================================================================
# Extract LASSO-selected variables (lambda.1se)
# ============================================================================
coef_1se <- coef(cvfit, s = "lambda.1se")
coef_df_1se <- data.frame(
  feature = rownames(coef_1se),
  coef = as.numeric(coef_1se)
)
coef_df_1se <- subset(coef_df_1se, coef != 0 & feature != "(Intercept)")
selected_metabs_1se <- coef_df_1se$feature[coef_df_1se$feature %in% metab_list]
message("Number of metabolites selected by lambda.1se: ", length(selected_metabs_1se))

# Save LASSO-selected metabolites
write.csv(selected_metabs_1se, "lasso_selected_1se.csv", row.names = FALSE)

# ============================================================================
# Stability selection (subsampling with LASSO)
# ============================================================================
set.seed(123)

n_iterations <- 200
subsample_ratio <- 0.5

n <- nrow(X)
p <- ncol(X)

selection_matrix <- matrix(0, nrow = p, ncol = n_iterations)
rownames(selection_matrix) <- colnames(X)

for (b in 1:n_iterations) {
  message("Stability selection iteration: ", b, "/", n_iterations)

  # Subsample without replacement
  idx <- sample(1:n, size = floor(subsample_ratio * n), replace = FALSE)
  X_sub <- X[idx, ]
  y_sub <- y[idx]

  # CV to find optimal lambda
  cvfit_sub <- cv.glmnet(
    x = X_sub, y = y_sub,
    family = "binomial",
    alpha = 1,
    nfolds = 10,
    penalty.factor = penalty_factor,
    type.measure = "auc",
    standardize = FALSE
  )

  lambda_sub <- cvfit_sub$lambda.1se

  # Fit LASSO with selected lambda
  fit_sub <- glmnet(
    x = X_sub, y = y_sub,
    family = "binomial",
    alpha = 1,
    lambda = lambda_sub,
    penalty.factor = penalty_factor,
    standardize = FALSE
  )

  # Record which variables were selected (non-zero coefficients)
  coef_sub <- as.matrix(coef(fit_sub))[-1, , drop = FALSE]
  selection_matrix[, b] <- as.numeric(coef_sub != 0)
}

# ============================================================================
# Compute selection probabilities
# ============================================================================
selection_prob <- rowMeans(selection_matrix)

selection_df <- data.frame(
  feature = names(selection_prob),
  prob = selection_prob
)

# Retain only metabolite entries
selection_df_metab <- subset(selection_df, feature %in% metab_list)

# ============================================================================
# Select stable metabolites (selection probability >= 0.85)
# ============================================================================
threshold <- 0.85
stable_metabs <- selection_df_metab$feature[selection_df_metab$prob > threshold]

message("Number of stable metabolites (prob >= ", threshold, "): ", length(stable_metabs))

# Save results
write.csv(selection_df_metab, "lasso_stability_probabilities.csv", row.names = FALSE)
write.csv(stable_metabs, "data/selected_metabolites.csv", row.names = FALSE)

# ============================================================================
# Figure 1: Stability selection probability bar plot (top 50)
# ============================================================================
selection_df_metab <- selection_df_metab[order(-selection_df_metab$prob), ]
selection_df_top50 <- selection_df_metab[1:min(50, nrow(selection_df_metab)), ]

p_stability <- ggplot(
  selection_df_top50,
  aes(x = reorder(feature, prob), y = prob, fill = prob)
) +
  geom_bar(stat = "identity", width = 0.8) +
  coord_flip() +
  geom_hline(
    yintercept = threshold, linetype = "dashed",
    color = "firebrick3", linewidth = 0.7
  ) +
  scale_fill_gradient(low = "#AED6F1", high = "#1B4F72") +
  labs(
    title = "Stability Selection Probability of Metabolites",
    x = NULL,
    y = "Selection Probability"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", hjust = 1, size = 12),
    axis.text.y = element_text(size = 10)
  )

ggsave("fig_stability_selection_probability.pdf", p_stability,
       width = 5, height = 8, device = "pdf")

# ============================================================================
# Figure 2: LASSO CV curve with non-zero metabolite count
# ============================================================================
coef_all <- as.matrix(coef(cvfit, s = lambda_seq))[-1, , drop = FALSE]
metab_idx <- which(rownames(coef_all) %in% metab_list)
nonzero_metab <- apply(coef_all[metab_idx, , drop = FALSE], 2, function(x) sum(x != 0))

ord <- order(log(lambda_seq))
loglam <- log(lambda_seq)[ord]
auc <- cvfit$cvm[ord]
auc_lower <- (cvfit$cvm - cvfit$cvsd)[ord]
auc_upper <- (cvfit$cvm + cvfit$cvsd)[ord]
cv_plot_df <- data.frame(
  log_lambda = loglam,
  auc = auc,
  lower = auc_lower,
  upper = auc_upper,
  n_metabs = nonzero_metab[ord]
)
scale_factor <- max(cv_plot_df$auc) / max(cv_plot_df$n_metabs)

p_cv <- ggplot(cv_plot_df, aes(x = log_lambda)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.4, fill = "grey60") +
  geom_line(aes(y = auc, color = "AUC"), linewidth = 1) +
  geom_point(aes(y = auc, color = "AUC"), size = 0.8) +
  geom_line(
    aes(y = n_metabs * scale_factor, color = "# Selected Metabolites"),
    linewidth = 1, linetype = "solid"
  ) +
  geom_vline(
    xintercept = log(lambda_min), color = "firebrick3",
    linetype = "dashed", linewidth = 0.8
  ) +
  geom_vline(
    xintercept = log(lambda_1se), color = "firebrick3",
    linetype = "dotdash", linewidth = 0.8
  ) +
  annotate(
    "text", x = log(lambda_min) - 1.1, y = max(auc) - 0.05,
    label = paste("lambda min =", round(lambda_min, 4)),
    hjust = -0.1, size = 3, color = "firebrick3"
  ) +
  annotate(
    "text", x = log(lambda_1se), y = max(auc) - 0.05,
    label = paste("lambda 1se =", round(lambda_1se, 4)),
    hjust = -0.1, size = 3, color = "firebrick3"
  ) +
  scale_y_continuous(
    name = "Cross-validated AUC",
    breaks = seq(0, 1, by = 0.2),
    sec.axis = sec_axis(~ . / scale_factor, name = "Number of selected metabolites")
  ) +
  scale_color_manual(
    values = c("AUC" = "navy", "# Selected Metabolites" = "#238B45")
  ) +
  labs(x = expression(log(lambda)), title = "LASSO Cross-Validation") +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
    axis.text.y = element_text(size = 10),
    legend.title = element_blank(),
    legend.position = "bottom"
  )

ggsave("fig_lasso_cv_curve.pdf", p_cv, width = 5, height = 4, device = "pdf")

# ============================================================================
# Figure 3: LASSO coefficient path (highlighting stable metabolites)
# ============================================================================
final_metabs <- stable_metabs
fit_path <- glmnet(
  X, y, family = "binomial", alpha = 1,
  penalty.factor = penalty_factor, standardize = FALSE
)
coef_mat <- as.matrix(coef(fit_path))[-1, , drop = FALSE]
lambda_path <- fit_path$lambda
coef_df_path <- as.data.frame(t(coef_mat))
colnames(coef_df_path) <- rownames(coef_mat)
coef_df_path$log_lambda <- log(lambda_path)
coef_long <- reshape2::melt(
  coef_df_path, id.vars = "log_lambda",
  variable.name = "feature", value.name = "coefficient"
)

coef_long$highlight <- ifelse(
  coef_long$feature %in% final_metabs, "Selected", "Other"
)
safe_log_min <- if (lambda_min > 0) log(lambda_min) else NA
safe_log_1se <- if (lambda_1se > 0) log(lambda_1se) else NA

n_final <- length(final_metabs)
if (n_final > 0) {
  if (n_final <= 9) {
    brewer_colors <- brewer.pal(n_final, "Set1")
  } else {
    brewer_colors <- colorRampPalette(brewer.pal(9, "Set1"))(n_final)
  }
} else {
  brewer_colors <- NULL
}

p_path <- ggplot(coef_long, aes(x = log_lambda, y = coefficient, group = feature)) +
  geom_line(
    data = subset(coef_long, highlight == "Other"),
    color = "gray70", alpha = 0.5, size = 0.6
  ) +
  geom_line(
    data = subset(coef_long, highlight == "Selected"),
    aes(color = feature), size = 1.0
  ) +
  scale_color_manual(name = NULL, values = brewer_colors) +
  geom_vline(
    xintercept = safe_log_min, color = "firebrick3",
    linetype = "dashed", size = 0.8
  ) +
  geom_vline(
    xintercept = safe_log_1se, color = "firebrick3",
    linetype = "dotdash", size = 0.8
  ) +
  labs(
    x = expression(log(lambda)), y = "Coefficients",
    title = "LASSO Coefficient Path (Stable Metabolites Highlighted)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )
p_path <- p_path + guides(color = guide_legend(ncol = 3, byrow = TRUE))

ggsave("fig_lasso_coefficient_path.pdf", p_path,
       width = 5, height = 4, device = "pdf")

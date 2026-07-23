library(data.table)
library(ggplot2)

models = c('model1_fold0', 'model1_fold1')

sim_ids = list()

## for simulated data
sim_ids[[1]] = c('jz57c1l5', 'eomgity2', 'yfomzbcr', 'yqiy96v3')
sim_ids[[2]] = c('h804eyib', 'zxwd3e8p', 'sfvnn618', '2hp5oakm')

## real data
ids = list()
ids[[1]] = c('iczglgc5', 'c9rht185', 'llbmsuz6', 'zx8ebya4')
ids[[2]] = c('rofgk1v9', 'sxqk103a', '7r2wmwkn', '7ant8vq7')

# prediction
pred = lapply(1:2, function(i){
    id = ids[[i]]
    fold = i-1
    do.call('rbind', lapply(id, function(j)
    {
        fread(paste0('results/attn2_1_pred_norm_3_tissue_discrete/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', j, '/Prediction_Results_19_in_valid_donors.csv'))
    }))
})

# metrics
metrics <- lapply(1:2, function(i) {
    id <- ids[[i]]
    fold = i-1
    do.call("rbind", lapply(id, function(j) {
        filepath <- paste0(
            "results/attn2_1_pred_norm_3_tissue_discrete/MultiGene/rain_filter_egenes_5K/",
            "Fold-", fold, "/", j, "/"
        )
        files <- list.files(filepath, pattern = "^CrossIndivMetrics_valid_donors_Epoch19_rank.*\\.csv$")
        mm <- read.csv(file.path(filepath, files[1]))
        mm$model <- models[i]
        return(mm)
    }))
})

train_genes = unique(metrics[[1]]$gene_name[metrics[[1]]$gene_split == 'train'])
test_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'valid', 'gene_name'])

entropy_pred <- function(pred_valid, suffix = 'train_genes',
                         bins = c(-1.4657382, -0.9538726, -0.5449254, -0.1777120, 0.1777120, 0.5449254, 0.9538726, 1.4657382),
                         bin_means = c(-2.1982420, -1.1966764, -0.7441936, -0.3592933, 0.0000000, 0.3592933, 0.7441936, 1.1966764, 2.1982420))
{
  pred_valid$entropy = -rowSums(log(pred_valid[,3:11]) * pred_valid[,3:11])
  pred_valid$y_pred = colSums(t(pred_valid[,3:11]) * (bin_means + t(pred_valid[,12:20])))
  pred_valid$y_pred_max = apply(pred_valid[,3:7], 1, which.max) - 1
  pred_valid$std = sqrt(colSums(t(pred_valid[,3:11]) * (bin_means + t(pred_valid[,12:20]))^2) - pred_valid$y_pred^2 + 1e-6)
  pred_valid$y_true_bin <- as.integer(cut(
      pred_valid$y_true,
      breaks = c(-Inf, bins, Inf),
      labels = FALSE,
      include.lowest = TRUE,
      right = TRUE
    )) - 1
  pred_valid$y_pred_bin = max.col(pred_valid[, 3:11], ties.method = "first") - 1

  if(suffix == 'train_genes')
  {
    pred2plot = pred_valid[sample(.N, ceiling(0.2 * .N))]
  }else{
    pred2plot = pred_valid
  }

  pred2plot$bin_diff <- abs(pred2plot$y_true_bin - pred2plot$y_pred_bin)
  pred2plot$bin_diff_grp <- ifelse(pred2plot$bin_diff >= 5, ">=5", as.character(pred2plot$bin_diff))
  pred2plot$bin_diff_grp <- factor(pred2plot$bin_diff_grp, levels = c("0", "1", "2", "3", "4", ">=5"))

  return(pred2plot)
}

# Process train and test genes separately
pred2plot_list = list()
pred2plot_list[[1]] = entropy_pred(pred[[1]][gene %in% train_genes], 'train_genes')
pred2plot_list[[2]] = entropy_pred(pred[[1]][gene %in% test_genes], 'test_genes')
pred2plot_list[[1]]$geneset = 'Train genes'
pred2plot_list[[2]]$geneset = 'Test genes'
pred2plot = do.call('rbind', pred2plot_list)

# Build combined data crossing: include/exclude mid bin x train/test genes
build_combined <- function(dat) {
  all_pred = copy(dat)
  all_pred$mid_bin_filter = 'Include mid bin'

  excl_pred = copy(dat[y_pred_bin != 4])
  excl_pred$mid_bin_filter = 'Exclude mid bin'

  combined = rbind(all_pred, excl_pred)
  combined$pred_correct = factor(
    combined$y_true_bin == combined$y_pred_bin,
    levels = c(FALSE, TRUE),
    labels = c('Incorrect', 'Correct')
  )
  return(combined)
}

entropy_compare = build_combined(pred2plot)
entropy_compare$mid_bin_filter = factor(entropy_compare$mid_bin_filter,
                                        levels = c('Include mid bin', 'Exclude mid bin'))
entropy_compare$geneset = factor(entropy_compare$geneset,
                                  levels = c('Train genes', 'Test genes'))

# # Plot 1: facet by tissue (rows) and mid_bin_filter (cols), x = geneset, fill = pred_correct
# dir.create('Rscripts/plots/real_data/discrete', recursive = TRUE, showWarnings = FALSE)

# pdf('Rscripts/plots/real_data/discrete/entropy_vs_pred_correct_geneset_midbin_combined.pdf', width = 10, height = 8)
# ggplot(entropy_compare, aes(x = geneset, y = entropy, fill = pred_correct)) +
#   geom_boxplot(outlier.shape = NA) +
#   facet_grid(tissue ~ mid_bin_filter) +
#   theme_bw() + theme(text = element_text(size = 18),
#                       axis.text.x = element_text(angle = 30, hjust = 1)) +
#   labs(x = '', y = 'Entropy', fill = 'Prediction')
# dev.off()

# # Plot 2: facet by tissue (rows) and geneset (cols), x = pred_correct, fill = mid_bin_filter
# pdf('Rscripts/plots/real_data/discrete/entropy_vs_pred_correct_midbin_geneset_combined.pdf', width = 10, height = 8)
# ggplot(entropy_compare, aes(x = pred_correct, y = entropy, fill = mid_bin_filter)) +
#   geom_boxplot(outlier.shape = NA) +
#   facet_grid(tissue ~ geneset) +
#   theme_bw() + theme(text = element_text(size = 18)) +
#   labs(x = '', y = 'Entropy', fill = 'Filter')
# dev.off()

# # Plot 3: single panel, x = interaction of geneset and mid_bin_filter, fill = pred_correct
# entropy_compare$group_label = paste(entropy_compare$geneset, '\n', entropy_compare$mid_bin_filter)
# entropy_compare$group_label = factor(entropy_compare$group_label,
#   levels = c(
#     paste('Train genes', '\n', 'Include mid bin'),
#     paste('Train genes', '\n', 'Exclude mid bin'),
#     paste('Test genes', '\n', 'Include mid bin'),
#     paste('Test genes', '\n', 'Exclude mid bin')
#   ))

# pdf('Rscripts/plots/real_data/discrete/entropy_vs_pred_correct_all_combined.pdf', width = 12, height = 8)
# ggplot(entropy_compare, aes(x = group_label, y = entropy, fill = pred_correct)) +
#   geom_boxplot(outlier.shape = NA) +
#   facet_grid(tissue ~ .) +
#   theme_bw() + theme(text = element_text(size = 18),
#                       axis.text.x = element_text(angle = 30, hjust = 1)) +
#   labs(x = '', y = 'Entropy', fill = 'Prediction')
# dev.off()

# # Plot 4: std version of Plot 1
# pdf('Rscripts/plots/real_data/discrete/std_vs_pred_correct_geneset_midbin_combined.pdf', width = 10, height = 8)
# ggplot(entropy_compare, aes(x = geneset, y = std, fill = pred_correct)) +
#   geom_boxplot(outlier.shape = NA) +
#   facet_grid(tissue ~ mid_bin_filter) +
#   theme_bw() + theme(text = element_text(size = 18),
#                       axis.text.x = element_text(angle = 30, hjust = 1)) +
#   labs(x = '', y = 'Standard Deviation', fill = 'Prediction')
# dev.off()

# Combined train+test genes: real vs simulated panels
sim_pred = lapply(1:2, function(i){
    id = sim_ids[[i]]
    fold = i-1
    do.call('rbind', lapply(id, function(j)
    {
        fread(paste0('results/attn2_1_pred_norm_3_tissue_discrete/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', j, '/Prediction_Results_19_in_valid_donors.csv'))
    }))
})

tissue_map = c(Whole_Blood = 'Blood', Muscle_Skeletal = 'Muscle', Adipose_Subcutaneous = 'Adipose',
               blood = 'Blood', muscle = 'Muscle', adipose = 'Adipose')

real_pred_unified = copy(pred[[1]])
real_pred_unified[, tissue := tissue_map[tissue]]

sim_pred_unified = copy(sim_pred[[1]])
sim_pred_unified[, tissue := tissue_map[tissue]]

real_train = entropy_pred(real_pred_unified[gene %in% train_genes], 'train_genes')
real_train$geneset = 'Train genes'
real_test = entropy_pred(real_pred_unified[gene %in% test_genes], 'test_genes')
real_test$geneset = 'Test genes'
real_combined = rbind(real_train, real_test)
real_combined$data_type = 'GTEx data'

sim_train = entropy_pred(sim_pred_unified[gene %in% train_genes], 'train_genes')
sim_train$geneset = 'Train genes'
sim_test = entropy_pred(sim_pred_unified[gene %in% test_genes], 'test_genes')
sim_test$geneset = 'Test genes'
sim_combined = rbind(sim_train, sim_test)
sim_combined$data_type = 'Simulated data'

combined_all = rbind(real_combined, sim_combined)
combined_all$data_type = factor(combined_all$data_type, levels = c('GTEx data', 'Simulated data'))
combined_all$geneset = factor(combined_all$geneset, levels = c('Train genes', 'Test genes'))
combined_all$pred_correct = factor(
    combined_all$y_true_bin == combined_all$y_pred_bin,
    levels = c(FALSE, TRUE),
    labels = c('Incorrect', 'Correct')
)

pred_colors = c('Incorrect' = '#4393C3', 'Correct' = '#D6604D')

# Plot 5: include mid bin — real vs simulated panels, train+test on x-axis
pdf('Rscripts/plots/real_data/discrete/entropy_vs_pred_correct_combined_genes_include_midbin.pdf', width = 9, height = 8)
ggplot(combined_all, aes(x = geneset, y = entropy, fill = pred_correct)) +
  geom_boxplot(outlier.shape = NA) +
  scale_fill_manual(values = pred_colors) +
  facet_grid(tissue ~ data_type) +
  theme_bw() + theme(text = element_text(size = 18),
                      strip.text = element_text(size = 16),
                      legend.position = 'bottom') +
  labs(x = '', y = 'Entropy', fill = 'Prediction')
dev.off()

# Plot 6: exclude mid bin — separate y-axis ranges for GTEx vs simulated
library(cowplot)

excl_dat = combined_all[y_pred_bin != 4]

p_gtex = ggplot(excl_dat[data_type == 'GTEx data'], aes(x = geneset, y = entropy, fill = pred_correct)) +
  geom_boxplot(outlier.shape = NA) +
  scale_fill_manual(values = pred_colors) +
  coord_cartesian(ylim = c(2, 2.2)) +
  facet_grid(tissue ~ data_type) +
  theme_bw() + theme(text = element_text(size = 18),
                      strip.text = element_text(size = 16),
                      legend.position = 'none') +
  labs(x = '', y = 'Entropy', fill = 'Prediction')

p_sim = ggplot(excl_dat[data_type == 'Simulated data'], aes(x = geneset, y = entropy, fill = pred_correct)) +
  geom_boxplot(outlier.shape = NA) +
  scale_fill_manual(values = pred_colors) +
  facet_grid(tissue ~ data_type) +
  theme_bw() + theme(text = element_text(size = 18),
                      strip.text = element_text(size = 16),
                      axis.title.y = element_blank(),
                      legend.position = 'none') +
  labs(x = '', fill = 'Prediction')

# legend = get_legend(
#   p_sim + theme(legend.position = 'bottom')
# )

pdf('Rscripts/plots/real_data/discrete/entropy_vs_pred_correct_combined_genes_exclude_midbin.pdf', width = 11, height = 8)
#plot_grid(
  plot_grid(p_gtex, p_sim, nrow = 1, align = 'v') #,
#  legend, ncol = 1, rel_heights = c(1, 0.06)
#)
dev.off()

# Plot 7: std vs bin_diff_grp — combining train+test genes, faceted by tissue and data_type
pdf('Rscripts/plots/simulated_data/discrete/std_vs_pred_error_boxplot_combined_genes.pdf', width = 9, height = 6)
ggplot(combined_all[data_type == 'Simulated data'], aes(x = bin_diff_grp, y = std, fill = geneset)) +
  geom_boxplot(outlier.shape = NA) +
  scale_fill_manual(values = c('Train genes' = '#66C2A5', 'Test genes' = '#FC8D62')) +
  coord_cartesian(ylim = c(0, 3)) +
  theme_bw() + theme(text = element_text(size = 18),
                      strip.text = element_text(size = 16),
                      legend.position = 'bottom') +
  labs(x = 'Bin difference', y = 'Predicted Standard Deviation', fill = '')
dev.off()

# Plot 8: pearsonr vs std density — separate panels for data_type x geneset
# Load simulated metrics
sim_metrics <- lapply(1:2, function(i) {
    id <- sim_ids[[i]]
    fold = i-1
    do.call("rbind", lapply(id, function(j) {
        filepath <- paste0(
            "results/attn2_1_pred_norm_3_tissue_discrete/MultiGene/rain_filter_egenes_5K/",
            "Fold-", fold, "/", j, "/"
        )
        files <- list.files(filepath, pattern = "^CrossIndivMetrics_valid_donors_Epoch19_rank.*\\.csv$")
        mm <- read.csv(file.path(filepath, files[1]))
        mm$model <- models[i]
        return(mm)
    }))
})

# Add std column via entropy_pred (need to compute it first)
real_ep = entropy_pred(real_pred_unified, 'combined')
sim_ep = entropy_pred(sim_pred_unified, 'combined')
real_std = real_ep[, list(std_mean = mean(std)), by = c('gene', 'tissue')]
sim_std = sim_ep[, list(std_mean = mean(std)), by = c('gene', 'tissue')]

# Build per-gene summaries: merge metrics with mean std
real_met = as.data.table(metrics[[1]])
real_met[, tissue := tissue_map[tissue]]
real_res = merge(real_met, real_std, by.x = c('gene_name', 'tissue'), by.y = c('gene', 'tissue'))
real_res$data_type = 'GTEx data'
real_res$geneset = ifelse(real_res$gene_name %in% train_genes, 'Train genes', 'Test genes')

sim_met = as.data.table(sim_metrics[[1]])
sim_met[, tissue := tissue_map[tissue]]
sim_res = merge(sim_met, sim_std, by.x = c('gene_name', 'tissue'), by.y = c('gene', 'tissue'))
sim_res$data_type = 'Simulated data'
sim_res$geneset = ifelse(sim_res$gene_name %in% train_genes, 'Train genes', 'Test genes')

res_all = rbind(real_res, sim_res, fill = TRUE)
res_all$data_type = factor(res_all$data_type, levels = c('Simulated data', 'GTEx data'))
res_all$geneset = factor(res_all$geneset, levels = c('Train genes', 'Test genes'))

cor_labels = res_all[, list(cor_val = paste('r =', round(cor(pearsonr, std_mean, use = 'complete'), 2))),
                     by = c('data_type', 'geneset')]

library(hexbin)

pdf('Rscripts/plots/real_data/discrete/pearsonr_vs_std_density_combined.pdf', width = 9, height = 8)
ggplot(res_all, aes(x = pearsonr, y = std_mean)) +
  geom_hex(bins = 80) +
  scale_fill_viridis_c(breaks = scales::pretty_breaks(n = 2)) +
  geom_text(data = cor_labels, aes(label = cor_val), x = -Inf, y = Inf,
            hjust = -0.1, vjust = 1.5, size = 5, inherit.aes = FALSE) +
  facet_grid(data_type ~ geneset) +
  theme_bw() + theme(text = element_text(size = 18),
                      strip.text = element_text(size = 16),
                      legend.position = 'bottom') +
  labs(x = 'Pearson r', y = 'Mean Predicted Std', fill = 'Count')
dev.off()

# Plot 9: pearsonr vs std scatterplot — separate panels for data_type x geneset
pdf('Rscripts/plots/real_data/discrete/pearsonr_vs_std_scatter_combined.pdf', width = 10, height = 8)
ggplot(res_all, aes(x = pearsonr, y = std_mean)) +
  geom_point(size = 0.5, alpha = 0.3) +
  geom_text(data = cor_labels, aes(label = cor_val), x = -Inf, y = Inf,
            hjust = -0.1, vjust = 1.5, size = 8, inherit.aes = FALSE) +
  facet_grid(data_type ~ geneset) +
  theme_bw() + theme(text = element_text(size = 18),
                      strip.text = element_text(size = 16)) +
  labs(x = 'Pearson r', y = 'Mean Predicted Std')
dev.off()

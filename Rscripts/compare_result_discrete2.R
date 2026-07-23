library(data.table) 
#library(patchwork)
library(ggplot2)
# For simulated data, test run, combine prediction to compute entropy
models = c('model1_fold0', 'model1_fold1')

ids = list()

# real data
ids[[1]] = 'iczglgc5'     
ids[[2]] = 'c92158vx'

# prediction
pred = lapply(1:2, function(i){
    id = ids[[i]]
    fold = i-1
    if(fold == 1)
    {
      dat = fread(paste0('results/attn2_1_pred_norm_3_tissue_discrete/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', id, '/test_genes_sub_fold0/Prediction_Results_-1_in_test_donors.csv'))
    }else{
      dat = fread(paste0('results/attn2_1_pred_norm_3_tissue_discrete/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', id, '/test_genes/Prediction_Results_-1_in_test_donors.csv'))
    }
    dat$fold = fold
    return(dat)
})


donor_pairs = unique(pred[[1]]$donor)
donor_pairs2 = unique(pred[[2]]$donor)

# metrics
metrics <- lapply(1:2, function(i) {
    id <- ids[[i]]
    fold = i-1
    if(fold == 1)
    {
      dat = fread(paste0('results/attn2_1_pred_norm_3_tissue_discrete/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', id, '/test_genes_sub_fold0/CrossIndivMetrics_test_donors_Epoch-1_rank0.csv'))
    }else{
      dat = fread(paste0('results/attn2_1_pred_norm_3_tissue_discrete/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', id, '/test_genes/CrossIndivMetrics_test_donors_Epoch-1_rank0.csv'))
    }
    dat$fold = fold
    return(dat)
})

train_genes = unique(metrics[[1]]$gene_name[metrics[[1]]$gene_split == 'train']) # 
test_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'test', 'gene_name']) #1538
tissues = unique(metrics[[1]]$tissue)


# prediction entropy vs error
entropy_pred <- function(pred_valid, suffix = 'train_genes_fold0', bins = c(-1.4657382, -0.9538726, -0.5449254, -0.1777120, 0.1777120, 0.5449254, 0.9538726, 1.4657382), 
                         bin_means =c(-2.1982420, -1.1966764, -0.7441936, -0.3592933, 0.0000000, 0.3592933, 0.7441936, 1.1966764, 2.1982420))
{
  pred_valid = pred_valid[, 1:23]
  pred_valid$entropy = -rowSums(log(pred_valid[,3:11]) * pred_valid[,3:11])
  pred_valid$y_pred = colSums(t(pred_valid[,3:11]) * (bin_means + t(pred_valid[,12:20])))
  #pred_valid$y_pred_cat = round(pred_valid$y_pred)
  pred_valid$std = sqrt(colSums(t(pred_valid[,3:11]) * (bin_means + t(pred_valid[,12:20]))^2) - pred_valid$y_pred^2 + 1e-6)
  pred_valid$y_true_bin <- as.integer(cut(
      pred_valid$y_true,
      breaks = c(-Inf, bins, Inf),
      labels = FALSE,
      include.lowest = TRUE,
      right = TRUE
    )) - 1
  pred_valid$y_pred_bin = max.col(pred_valid[, 3:11], ties.method = "first") - 1

  return(pred_valid)
}

pred_valid = lapply(pred, entropy_pred)

probability_distance <- function(prob_run1, prob_run2, eps = 1e-12)
{
  prob_run1 = as.matrix(prob_run1)
  prob_run2 = as.matrix(prob_run2)

  prob_run1 = pmax(prob_run1, eps)
  prob_run2 = pmax(prob_run2, eps)
  prob_run1 = prob_run1 / rowSums(prob_run1)
  prob_run2 = prob_run2 / rowSums(prob_run2)

  mean_prob = (prob_run1 + prob_run2) / 2
  js_divergence = 0.5 * rowSums(prob_run1 * log(prob_run1 / mean_prob) + prob_run2 * log(prob_run2 / mean_prob))
  total_variation = 0.5 * rowSums(abs(prob_run1 - prob_run2))
  l2_distance = sqrt(rowSums((prob_run1 - prob_run2)^2))

  data.table(
    prob_dist_tv = total_variation,
    prob_dist_l2 = l2_distance,
    prob_dist_js = sqrt(pmax(js_divergence, 0)),
    pred_stability = 1 - total_variation
  )
}


# combine two runs
pred_valid = cbind(pred_valid[[1]], pred_valid[[2]])
names(pred_valid) = make.unique(names(pred_valid), sep = "_")
all(pred_valid$donor == pred_valid$donor_1)
all(pred_valid$gene == pred_valid$gene_1)

# compute overall std and entropy
col1 = paste0('y_pred',1:9 )
col2 = paste0('y_pred',1:9,'_1' )
pred_valid$entropy_all = -rowSums(log((pred_valid[,..col1] + pred_valid[,..col2])/2) * (pred_valid[, ..col1] + pred_valid[, ..col2])/2)

pred_valid$std_all = sqrt(pred_valid$std^2 + pred_valid$std_1^2 + ( pred_valid$y_pred - pred_valid$y_pred_1 )^2)

pred_valid = cbind(pred_valid, probability_distance(pred_valid[, ..col1], pred_valid[, ..col2]))

  ### subsample 
  ##pred2plot = pred_valid[sample(.N, ceiling(0.2 * .N))] 
  pred2plot = pred_valid
  print(dim(pred2plot))


  # Plot distribution of entropy and entropy_1 for correct and incorrect prediction
  library(reshape2)
  entropy_df <- data.frame(
    entropy = pred2plot[y_true_bin == y_pred_bin & y_pred_bin_1 != y_pred_bin]$entropy,
    entropy_1 = pred2plot[y_true_bin == y_pred_bin & y_pred_bin_1 != y_pred_bin]$entropy_1
  )
  entropy_long <- melt(entropy_df)
  ggplot(entropy_long, aes(x = value, fill = variable)) +
    geom_density(alpha = 0.5) +
    labs(title = "Distribution of entropy and entropy_1",
         x = "Entropy value",
         y = "Density",
         fill = "Variable") +
    theme_bw() +
    theme(text = element_text(size = 18))

# same prediction vs different prediction
    pdf('Rscripts/plots/simulated_data/discrete/hist_entropy_same_prediction_exclude_mid_bin.pdf')
    ggplot(pred2plot[y_pred_bin!=4], aes(x = entropy, fill = y_pred_bin_1 == y_pred_bin)) +
    geom_density(alpha = 0.5) +
    labs(title = "Distribution of entropy",
         x = "Entropy value",
         y = "Density",
         fill = "Same predicted bin") +
    theme_bw() +
    theme(text = element_text(size = 18))
    dev.off()

    pdf('Rscripts/plots/simulated_data/discrete/hist_std_same_prediction_exclude_mid_bin.pdf')
    ggplot(pred2plot[y_pred_bin!=4], aes(x = std, fill = y_pred_bin_1 == y_pred_bin)) +
    geom_density(alpha = 0.5) +
    labs(title = "Distribution of standard deviation",
         x = "Standard deviation",
         y = "Density",
         fill = "Same predicted bin") +
    theme_bw() +
    theme(text = element_text(size = 18))
    dev.off()


pdf('Rscripts/plots/simulated_data/discrete/hist_entropy_same_prediction_exclude_mid_bin.pdf')
 ggplot(pred2plot[y_pred_bin!=4], aes(x = entropy, fill = interaction(y_pred_bin_1 == y_pred_bin, y_pred_bin == y_true_bin))) +
    geom_density(alpha = 0.5) +
    labs(title = "Distribution of entropy",
         x = "Entropy value",
         y = "Density",
         fill = "Same predicted bin") +
    theme_bw() + theme(text = element_text(size = 18))
  dev.off()


pred2plot$category <- interaction(
  Stability = pred2plot$y_pred_bin_1 == pred2plot$y_pred_bin,
  Correctness = pred2plot$y_pred_bin == pred2plot$y_true_bin,
  sep = "_"
)

# Rename the levels of the category for better readability
levels(pred2plot$category) <- c(
  "Unstable_Incorrect",  # Stability = FALSE, Correctness = FALSE
  "Stable_Incorrect",    # Stability = TRUE, Correctness = FALSE
  "Unstable_correct",    # Stability = FALSE, Correctness = TRUE
  "Stable_Correct"       # Stability = TRUE, Correctness = TRUE
)

pdf('Rscripts/plots/real_data/discrete/boxplot_entropy_category_real.pdf')
ggplot(pred2plot, aes(x = category, y = entropy, fill = category)) +
  geom_boxplot(alpha = 0.5,  outlier.shape = NA) +
  labs(title = "Entropy by Correctness and Stability",
       x = "",
       y = "Entropy",
       fill = "Category") +
  coord_cartesian(ylim = c(2.1, 2.2))  + 
  theme_bw() +
  theme(text = element_text(size = 18), 
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none" )
dev.off()


pdf('Rscripts/plots/real_data/discrete/boxplot_std_category_real.pdf')
ggplot(pred2plot, aes(x = category, y = std, fill = category)) +
  geom_boxplot(alpha = 0.5, outlier.shape = NA) +
  labs(title = "Standard Deviation by Correctness and Stability",
       x = "",
       y = "Standard Deviation",
       fill = "Category") +
  coord_cartesian(ylim = c(1.1, 1.4))  + 
  theme_bw() +
  theme(text = element_text(size = 18), 
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none" )
dev.off()


aa = xtabs(~pred2plot[['category']])
aa[c(2,4)]/sum(aa[c(2,4)])
aa[c(1,3)]/sum(aa[c(1,3)])
sum(aa[c(1,3)])/sum(aa)
chisq.test(matrix(aa, nrow = 2))

mean(pred2plot$y_pred_bin_1 == pred2plot$y_pred_bin)


ggplot(pred2plot[sample(.N, ceiling(0.2 * .N))], aes(x = abs(y_pred_1 - y_pred), y = std_all)) + geom_point() + 
  geom_smooth(method = 'lm') + 
    theme_bw() + theme(text = element_text(size = 18))

# get average entropy per gene vs metrics
# entropy_metrics <- function(pred, res)
# {
#   pred$entropy = -rowSums(log(pred[,3:7]) * pred[,3:7])
#   pred$y_pred = colSums(t(pred[,3:7]) * 0:4)
#   pred$variance = colSums(t(pred[,3:7]) * (0:4)^2) - pred$y_pred^2
#   entropy = pred[, list('entropy' = mean(entropy), 'variance' = mean(variance)), by = gene]
#   res = merge(res, entropy, by.x ='gene_name', by.y = 'gene' )
#   par(mfrow = c(2,3))
#   plot(res$pearsonr, res$entropy, cex.lab=1.5, cex.axis=1.5)
#   mtext(paste('cor:', round(cor(res$pearsonr, res$entropy),2)), side =3)
#   plot(res$r2, res$entropy, cex.lab=1.5, cex.axis=1.5)
#   mtext(paste('cor:',round(cor(res$r2, res$entropy),2)), side =3)
#   plot(res$accuracy, res$entropy, cex.lab=1.5, cex.axis=1.5)
#   mtext(paste('cor:',round(cor(res$accuracy, res$entropy),2)), side =3)
#   plot(res$pearsonr, res$variance, cex.lab=1.5, cex.axis=1.5)
#   mtext(paste('cor:',round(cor(res$pearsonr, res$variance),2)), side =3)
#   plot(res$r2, res$variance, cex.lab=1.5, cex.axis=1.5)
#   mtext(paste('cor:',round(cor(res$r2, res$variance),2)), side =3)
#   plot(res$accuracy, res$variance, cex.lab=1.5, cex.axis=1.5)
#   mtext(paste('cor:',round(cor(res$accuracy, res$variance),2)), side =3)

#   return(res)
# }

met_all = cbind(metrics[[1]], metrics[[2]])
names(met_all) = make.unique(names(met_all), sep = "_")
all(met_all$gene_name == met_all$gene_name_1)
all(met_all$tissue == met_all$tissue_1)

plot(met_all$pearsonr, met_all$pearsonr_1)
plot(met_all$r2, met_all$r2_1)
plot(met_all$accuracy, met_all$accuracy_1)

met_all$accuracy_all = ( met_all$accuracy + met_all$accuracy_1 ) /2
met_all$pearsonr_all = ( met_all$pearsonr + met_all$pearsonr_1 ) /2
met_all$r2_all = ( met_all$r2 + met_all$r2_1 ) /2

entropy = pred_valid[, list('entropy' = mean(entropy), 'std' = mean(std), 
                            'entropy_1' = mean(entropy_1), 'std_1' = mean(std_1), 
                            'entropy_all' = mean(entropy_all), 'std_all' = mean(std_all), 
                            'consist' = mean(y_pred_bin == y_pred_bin_1), 
                            'diff' = mean(abs(y_pred - y_pred_1)),
                            'prob_dist_tv' = mean(prob_dist_tv),
                            'prob_dist_l2' = mean(prob_dist_l2),
                            'prob_dist_js' = mean(prob_dist_js),
                            'pred_stability' = mean(pred_stability)), by = c('gene', 'tissue')]


res = merge(met_all, entropy, by.x = c('gene_name', 'tissue'), by.y = c('gene', 'tissue'))


# r2 vs entropy vs std vs consistent
plot_accuracy_vs_entropy <- function(res, filename) {
    cor_val <- round(cor(res$accuracy_all, res$prob_dist_js, use = 'complete'), 2)
    #round(cor(res$accuracy_all, res$prob_dist_tv, use = 'complete'), 2)
    #round(cor(res$accuracy_all, res$prob_dist_l2, use = 'complete'), 2)
    #round(cor(res$accuracy_all, res$pred_stability, use = 'complete'), 2)
    p <- ggplot(res, aes(x = accuracy_all, y = prob_dist_js)) +
      geom_point() +
      theme_bw(base_size = 18) +
      labs(x = "Accuracy", y = "Jensen-Shannon divergence") +
      annotate("text", x = max(res$accuracy_all, na.rm=TRUE), y = max(res$prob_dist_js, na.rm=TRUE),  # max for simulated data
               label = paste("cor:", cor_val), hjust = 1, vjust = 1, size = 6)
    ggsave(filename, plot = p, width = 6, height = 5)
  }

plot_pearsonr_vs_std <- function(res, filename) {
    cor_val <- round(cor(res$pearsonr_all, res$std_all, use = 'complete'), 2)
    p <- ggplot(res, aes(x = pearsonr_all,  y = std_all)) +
      geom_point() + #geom_smooth(method = 'lm') + 
      theme_bw(base_size = 18) +
      labs(x = "Pearsonr", y = "Standard Deviation") +
      annotate("text", x = min(res$pearsonr, na.rm=TRUE), y = min(res$std_all, na.rm=TRUE) + 0.2, # max for simulated data
               label = paste("cor:", cor_val), hjust = 0, vjust = 1, size = 6) 
    ggsave(filename, plot = p, width = 6, height = 5)
  }
plot_accuracy_vs_entropy(res, paste0('Rscripts/plots/real_data/discrete/accuracy_vs_JS_test_genes_real.pdf'))
plot_pearsonr_vs_std(res, paste0('Rscripts/plots/real_data/discrete/pearsonr_vs_std_all_test_genes_real.pdf'))





plots <- function(results, metrics, gene)
{
  for(g in gene)
  {
    dat2plot = results[results$gene==g, ]
    dat2plot$y_pred = colSums(t(dat2plot[,3:7]) * 0:4)
    met = metrics[metrics$gene_name == g, ]
    #dat2plot$expr = factor(dat2plot$expr, levels = annot)
    p = ggplot(dat2plot, aes(x = y_true, y = y_pred, group = y_true)) + geom_boxplot() + theme(text = element_text(size = 18)) +
      annotate("text", x = min(dat2plot$y_true), y = max(dat2plot$y_pred), 
               label = paste0('Gene: ', g, '\n',  'Pearsonr: ', round(met$pearsonr,2), ', ',
                              'r2: ', round(met$r2,2), ', ',
                              'accuracy: ', round(met$accuracy,2)),
               hjust = -0.5, vjust = 1, size = 4) + ylab('y_pred')  #geom_density2d()
    p1 = ggplot(dat2plot, aes(x = y_true)) + geom_histogram() + theme(text = element_text(size = 18))
    print(p/p1)
    #print(ggMarginal(p, type = "histogram", margins = 'x'))
    
    # heatmap
    dat2plot = dat2plot[order(dat2plot$y_true, dat2plot$y_pred), ]
    colnames(dat2plot)[3:7] = paste0("y_pred",0:4)
    ann_colors = list(
      y_true = c("#1B9E77","#90EE90", "white",  "#D95F02","firebrick" )
    )
    pheatmap::pheatmap(dat2plot[, 3:7], cluster_cols = F, cluster_rows = T, annotation_colors = ann_colors,
                       show_rownames = F, annotation_row = dat2plot[, 8, drop=F],
                       breaks = seq(0,1,length.out = 101))
    pheatmap::pheatmap(dat2plot[, 3:7], cluster_cols = F, cluster_rows = F, annotation_colors = ann_colors,
                       show_rownames = F, annotation_row = dat2plot[, 8, drop=F],
                       breaks = seq(0,1,length.out = 101))
  }
}
plots(pred_test, res1, gene)
plots(pred_valid, res, gene[7])
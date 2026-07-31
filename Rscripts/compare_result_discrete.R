library(data.table) 
#library(patchwork)
library(ggplot2)

# paths come from the YAML config (CREAM/configs/defaults.yaml)
source("Rscripts/config.R")
cream <- cream_config()
# For simulated data, valid genes
models = c('model1_fold0', 'model1_fold1')

ids = list()

## for simualted data    
ids[[1]] = c('jz57c1l5', 'eomgity2', 'yfomzbcr', 'yqiy96v3')
ids[[2]] = c('h804eyib', 'zxwd3e8p', 'sfvnn618', '2hp5oakm')
     
## real data
ids[[1]] = c('iczglgc5', 'c9rht185', 'llbmsuz6', 'zx8ebya4' )
ids[[2]] = c('rofgk1v9', 'sxqk103a', '7r2wmwkn', '7ant8vq7' )

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

train_genes = unique(metrics[[1]]$gene_name[metrics[[1]]$gene_split == 'train']) # 4949
test_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'valid', 'gene_name']) #1047
tissues = unique(metrics[[1]]$tissue)


# prediction entropy vs error
entropy_pred <- function(pred_valid, suffix = 'train_genes_fold0', bins = c(-1.4657382, -0.9538726, -0.5449254, -0.1777120, 0.1777120, 0.5449254, 0.9538726, 1.4657382), 
                         bin_means =c(-2.1982420, -1.1966764, -0.7441936, -0.3592933, 0.0000000, 0.3592933, 0.7441936, 1.1966764, 2.1982420))
{
  pred_valid$entropy = -rowSums(log(pred_valid[,3:11]) * pred_valid[,3:11])
  pred_valid$y_pred = colSums(t(pred_valid[,3:11]) * (bin_means + t(pred_valid[,12:20])))
  #pred_valid$y_pred_cat = round(pred_valid$y_pred)
  pred_valid$y_pred_max = apply(pred_valid[,3:7], 1, which.max)-1
  pred_valid$std = sqrt(colSums(t(pred_valid[,3:11]) * (bin_means + t(pred_valid[,12:20]))^2) - pred_valid$y_pred^2 + 1e-6)
  pred_valid$y_true_bin <- as.integer(cut(
      pred_valid$y_true,
      breaks = c(-Inf, bins, Inf),
      labels = FALSE,
      include.lowest = TRUE,
      right = TRUE
    )) - 1
  pred_valid$y_pred_bin = max.col(pred_valid[, 3:11], ties.method = "first") - 1
  # subsample 
  if(suffix == 'train_genes_fold0')
  {
    pred2plot = pred_valid[sample(.N, ceiling(0.2 * .N))] 
  }else{
    pred2plot = pred_valid
  }

  print(dim(pred2plot))

  pred2plot$bin_diff <- abs(pred2plot$y_true_bin - pred2plot$y_pred_bin)
  pred2plot$bin_diff_grp <- ifelse(pred2plot$bin_diff >= 5, ">=5", as.character(pred2plot$bin_diff))
  pred2plot$bin_diff_grp <- factor(pred2plot$bin_diff_grp, levels = c("0", "1", "2", "3", "4", ">=5"))

  return(pred2plot) 
}

  # plot std vs prediction error
  pdf(paste0('Rscripts/plots/real_data/discrete/std_vs_pred_error_scatter_', suffix,'.pdf'))
  ggplot(pred2plot[sample(.N, ceiling(0.2 * .N))], aes(x = abs(y_true - y_pred), y = std)) + geom_point() + 
    geom_smooth(method = 'lm') + 
    facet_grid(tissue~.) + 
      theme_bw() + theme(text = element_text(size = 18))
  dev.off()


  pdf(paste0('Rscripts/plots/real_data/discrete/std_vs_pred_error_boxplot_', suffix,'.pdf'))
  ggplot(pred2plot, aes(x = bin_diff_grp, y = std)) + geom_boxplot(outlier.shape  = NA) + # factor(bin_diff)
  facet_grid(tissue~.) + ylim(0,4) +
    theme_bw() + theme(text = element_text(size = 18))
  dev.off()

  # plot entropy vs prediction
  entropy_all = copy(pred2plot)
  entropy_all$filter = 'All predictions'
  entropy_exclude_mid = copy(pred2plot[y_pred_bin != 4])
  entropy_exclude_mid$filter = 'Exclude mid bin'
  entropy_compare = rbind(entropy_all, entropy_exclude_mid)
  entropy_compare$pred_correct = factor(
    entropy_compare$y_true_bin == entropy_compare$y_pred_bin,
    levels = c(FALSE, TRUE),
    labels = c('Incorrect', 'Correct')
  )

  pdf(paste0('Rscripts/plots/simulated_data/discrete/entropy_vs_pred_correct_combined_', suffix,'.pdf'))
  ggplot(entropy_compare, aes(x = pred_correct, y = entropy, fill = filter)) +
    geom_boxplot(outlier.shape = NA) +
    facet_grid(tissue~.) +
    theme_bw() + theme(text = element_text(size = 18))
  dev.off()

pdf(paste0('Rscripts/plots/real_data/discrete/entropy_vs_pred_correct_exclude_mid_', suffix,'.pdf'))
  ggplot(pred2plot[y_pred_bin != 4], aes(x = y_true_bin == y_pred_bin, y = entropy)) +
    geom_boxplot(outlier.shape = NA) +
    facet_grid(tissue~.) + coord_cartesian(ylim = c(2., 2.2)) + 
    theme_bw() + theme(text = element_text(size = 18))
  dev.off()

  pdf(paste0('Rscripts/plots/real_data/discrete/entropy_vs_pred_bin_', suffix,'.pdf'))
  ggplot(pred2plot, aes(x = y_pred_bin, y = entropy, group  = y_pred_bin)) + geom_boxplot(outlier.shape  = NA) + 
    facet_grid(tissue~.) + 
      theme_bw() + theme(text = element_text(size = 18))
  dev.off()
   


pred_valid = pred[[1]][gene %in% train_genes]
pred_valid = pred[[1]][gene %in% test_genes]
suffix = 'train_genes_fold0'

pred2plot = entropy_pred(pred[[1]][gene %in% train_genes], 'train_genes_fold0')
pred2plot = entropy_pred(pred[[1]][gene %in% test_genes], 'test_genes_fold0')

pred2plot = list()
pred2plot[[1]] = entropy_pred(pred[[1]][gene %in% train_genes], 'train_genes_fold0')
pred2plot[[2]] = entropy_pred(pred[[1]][gene %in% test_genes], 'test_genes_fold0')
pred2plot[[1]]$geneset = 'train_genes'
pred2plot[[2]]$geneset = 'test_genes'
pred2plot = do.call('rbind', pred2plot)

pdf(paste0('Rscripts/plots/simulated_data/discrete/entropy_vs_pred_bin_both.pdf'))
ggplot(pred2plot, aes(x = factor(y_pred_bin), y = entropy, fill = geneset)) + geom_boxplot(outlier.shape  = NA) + 
  facet_grid(tissue~.) + 
    theme_bw() + theme(text = element_text(size = 18))
dev.off()

pdf(paste0('Rscripts/plots/real_data/discrete/entropy_both.pdf'))
ggplot(pred2plot, aes(x = tissue, y = entropy, fill = geneset)) + geom_boxplot(outlier.shape  = NA) + 
    theme_bw() + theme(text = element_text(size = 18)) + coord_cartesian(ylim = c(2., 2.2))
dev.off()

pdf(paste0('Rscripts/plots/real_data/discrete/std_both.pdf'))
ggplot(pred2plot, aes(x = tissue, y = std, fill = geneset)) + geom_boxplot(outlier.shape  = NA) + 
    theme_bw() + theme(text = element_text(size = 18)) + coord_cartesian(ylim = c(1, 1.5))
dev.off()

pred2plot$pred_correct = factor(
  pred2plot$y_true_bin == pred2plot$y_pred_bin,
  levels = c(FALSE, TRUE),
  labels = c('Incorrect', 'Correct')
)
pdf(paste0('Rscripts/plots/real_data/discrete/entropy_vs_pred_correct_exclude_mid_both.pdf'))
  ggplot(pred2plot[y_pred_bin != 4], aes(x = geneset, y = entropy, fill = pred_correct)) +
    geom_boxplot(outlier.shape = NA) +
    facet_grid(tissue~.) + coord_cartesian(ylim = c(2., 2.2)) + 
    theme_bw() + theme(text = element_text(size = 18))
  dev.off()


## check y_pred = 0, separate correct and incorrect prediction
pdf(paste0('Rscripts/plots/real_data/discrete/entropy_vs_pred_correct_mid_both.pdf'))
ggplot(pred2plot[y_pred_bin == 4], aes(x = geneset, y = entropy, fill = pred_correct)) +
    geom_boxplot(outlier.shape = NA) +
    facet_grid(tissue~.) + #coord_cartesian(ylim = c(2., 2.2)) + 
    theme_bw() + theme(text = element_text(size = 18))
dev.off()

# percentage of zeros in the data
mean(pred[[1]]$y_true == 0) #69%, real: 10%
mean(pred2plot$y_true_bin == 4) #simulated: 71%, real:19%

mean(pred2plot$y_pred_bin == 4)  # simulated 86% real 44%
mean(pred2plot$y_pred_bin[pred2plot$geneset == 'train_genes'] == 4) # simulated: 73% real 43%
mean(pred2plot$y_pred_bin[pred2plot$geneset == 'test_genes'] == 4) # simulated: 98% real 45%

# Does the incorrect prediction to zero is becuase of that genes have excessive zeros (low snps counts or magnitude?)

# read in eQTL file
tissue_id = read.csv(cream$eqtl_tissue_label_file)
tissue = c('Whole_Blood','Muscle_Skeletal', 'Adipose_Subcutaneous')
ids = tissue_id[match(tissue, tissue_id$data_tissue), 'path']
tissue_name = c("blood", "muscle", "adipose")
coef = lapply(1:length(ids), function(i){
  coef = fread(file.path(cream$eqtl_dir, ids[i]))
  coef$beta = coef$beta * coef$pip
  coef$tissue = tissue_name[i]
  return(coef)
})
coef = do.call('rbind', coef)
coef[, c("chr", "pos", "ref", "alt") := tstrsplit(variant, "_", fixed = TRUE, keep = 1:4)]

coef[, var_type := fifelse(
  nchar(ref) == 1 & nchar(alt) == 1, "SNP",
  fifelse(nchar(ref) != nchar(alt), "INDEL", "MNP")
)]
coef = coef[var_type == 'SNP']
coef$pos = as.integer(coef$pos)

intervals = fread(cream$genomic_intervals_file)
intervals[, gene_id := sub("\\..*$", "", gene_id)]

eqtls_in_interval <- coef[
  intervals,
  on = .(gene_id, pos >= starts, pos <= ends),
  nomatch = 0L,
  allow.cartesian = TRUE
]

hist(eqtls_in_interval[, .N, by = c('gene_id', 'tissue') ][['N']])
bb = eqtls_in_interval[, .N, by = c('gene_id', 'tissue') ]
aa = pred2plot[, list('pred_zero' = mean(y_pred_bin ==4), 'true_std' = sd(y_true), 'true_zero' = mean(y_true == 0), 'geneset' = geneset[1]), by = c('gene', 'tissue')]
aa = merge(aa, bb, by.x = 'gene', by.y = 'gene_id')
# setorder(aa, -V1) 
# gn = aa$gene[1]
# metrics[[1]][metrics[[1]]$gene_name == aa$gene[1], ]

# plot errors with true std
# alternative: point-wise local density coloring via ggpointdensity
if (!requireNamespace("ggpointdensity", quietly = TRUE)) {
  stop("Package 'ggpointdensity' is required. Install with: install.packages('ggpointdensity')")
}

pdf('Rscripts/plots/real_data/discrete/pred_zero_true_zero_ggpointdensity.pdf', width = 14)
ggplot(aa, aes(x = pred_zero, y = true_zero)) + facet_grid(~geneset) + 
  ggpointdensity::geom_pointdensity(size = 1) +
  scale_color_viridis_c(name = "Point density") +
  labs(x = "Fraction predicted as mid bin", y = "Percentage of true zeros") +
  theme_bw(base_size = 18)
dev.off()

pdf('Rscripts/plots/real_data/discrete/pred_zero_std_ggpointdensity.pdf')
ggplot(aa, aes(x = pred_zero, y = true_std)) + #facet_grid(~geneset) + 
  ggpointdensity::geom_pointdensity(size = 1) +
  scale_color_viridis_c(name = "Point density") +
  labs(x = "Fraction predicted as mid bin (pred_zero)", y = "SD of true expression (true_std)") +
  theme_bw(base_size = 18)
dev.off()

## number of eQTL per gene
ggplot(aa[geneset == 'train_genes'], aes(x = pred_zero, y = N)) + 
  ggpointdensity::geom_pointdensity(size = 1) +
  scale_color_viridis_c(name = "Point density") +
  labs(x = "Fraction predicted as mid bin (pred_zero)", y = "SD of true expression (true_std)") +
  theme_bw(base_size = 18)



# Boxplot for y_pred1 to y_pred10 for the specified subset
# library(reshape2) # for melt
# subset_data <- pred2plot[y_pred_bin == 4 & y_true_bin != 4]
# cols_to_plot <- paste0("y_pred", 1:9)
# if (all(cols_to_plot %in% colnames(subset_data))) {
#   melted_data <- melt(subset_data[, ..cols_to_plot])
#   #pdf('Rscripts/plots/real_data/discrete/y_pred1_10_boxplot_incorrect_mid_bin.pdf')
#   ggplot(melted_data, aes(x = variable, y = value)) +
#     geom_boxplot(outlier.shape = NA) +
#     theme_bw(base_size = 18) +
#     labs(title = "Boxplot of y_pred1 to y_pred10 (Incorrect Mid Bin)", x = "y_pred", y = "Value")
#   #dev.off()
# } else {
#   warning("Some y_pred columns are missing in pred2plot.")
# }




# get average entropy per gene vs metrics
entropy_metrics <- function(pred_valid, res, bin_means =c(-2.1982420, -1.1966764, -0.7441936, -0.3592933, 0.0000000, 0.3592933, 0.7441936, 1.1966764, 2.1982420))
{
  pred_valid$entropy = -rowSums(log(pred_valid[,3:11]) * pred_valid[,3:11])
  pred_valid$y_pred = colSums(t(pred_valid[,3:11]) * (bin_means + t(pred_valid[,12:20])))
  pred_valid$std = sqrt(colSums(t(pred_valid[,3:11]) * (bin_means + t(pred_valid[,12:20]))^2) - pred_valid$y_pred^2 + 1e-6)

  entropy = pred_valid[, list('entropy' = mean(entropy), 'std' = mean(std)), by = c('gene', 'tissue')]
  res = merge(res, entropy, by.x = c('gene_name', 'tissue'), by.y = c('gene', 'tissue'))
  par(mfrow = c(2,3))
  plot(res$pearsonr, res$entropy, cex.lab=1.5, cex.axis=1.5)
  mtext(paste('cor:', round(cor(res$pearsonr, res$entropy, use = 'complete'),2)), side =3)
  plot(res$r2, res$entropy, cex.lab=1.5, cex.axis=1.5)
  mtext(paste('cor:',round(cor(res$r2, res$entropy),2)), side =3)
  plot(res$accuracy, res$entropy, cex.lab=1.5, cex.axis=1.5)
  mtext(paste('cor:',round(cor(res$accuracy, res$entropy, use = 'complete'),2)), side =3)
  plot(res$pearsonr, res$std, cex.lab=1.5, cex.axis=1.5)
  mtext(paste('cor:',round(cor(res$pearsonr, res$std, use = 'complete'),2)), side =3)
  plot(res$r2, res$std, cex.lab=1.5, cex.axis=1.5, xlim = c(-1,1))
  mtext(paste('cor:',round(cor(res$r2, res$std),2)), side =3)
  plot(res$accuracy, res$std, cex.lab=1.5, cex.axis=1.5)
  mtext(paste('cor:',round(cor(res$accuracy, res$std),2)), side =3)

  return(res)
}
met = data.table(metrics[[1]])
res = entropy_metrics(pred[[1]][gene %in% train_genes], met[gene_name %in% train_genes])
plot_accuracy_vs_entropy <- function(res, filename) {
    cor_val <- round(cor(res$accuracy, res$entropy, use = 'complete'), 2)
    p <- ggplot(res, aes(x = accuracy, y = entropy)) +
      geom_point() +
      theme_bw(base_size = 18) +
      labs(x = "Accuracy", y = "Entropy") +
      annotate("text", x = max(res$accuracy, na.rm=TRUE), y = max(res$entropy, na.rm=TRUE),  # max for simulated data
               label = paste("cor:", cor_val), hjust = 1, vjust = 1, size = 6) 
    ggsave(filename, plot = p, width = 6, height = 5)
  }
plot_pearsonr_vs_std <- function(res, filename) {
    cor_val <- round(cor(res$pearsonr, res$std, use = 'complete'), 2)
    p <- ggplot(res, aes(x = pearsonr, y = std)) +
      geom_point() +
      theme_bw(base_size = 18) +
      labs(x = "Pearsonr", y = "Standard Deviation") +
      annotate("text", x = min(res$pearsonr, na.rm=TRUE), y = min(res$std, na.rm=TRUE) + 0.2, # max for simulated data
               label = paste("cor:", cor_val), hjust = 0, vjust = 1, size = 6) 
    ggsave(filename, plot = p, width = 6, height = 5)
  }
plot_accuracy_vs_entropy(res, paste0('Rscripts/plots/real_data/discrete/accuracy_vs_entropy_test_genes_real.pdf'))
plot_pearsonr_vs_std(res, paste0('Rscripts/plots/real_data/discrete/pearsonr_vs_std_train_genes_real.pdf'))


entropy_metrics(pred[[1]][gene %in% train_genes], met[gene_name %in% train_genes])







pred_test$entropy = -rowSums(log(pred_test[,3:7]) * pred_test[,3:7])
plot(density(pred_test$entropy[pred_test$gene %in% train_genes]))
lines(density(pred_test$entropy[pred_test$gene %in% test_genes]), col = 2)

dat2plot = pred_valid[, list('mean_prob' = colMeans(.SD), 'y_pred'=0:4), by = y_true, .SDcols = 3:7] #gene == gene[1]
dat2plot$y_true = factor(dat2plot$y_true, levels = 0:4)
setorder(dat2plot, 'y_true')
ggplot(dat2plot, aes(x = y_pred, group = y_pred, y = mean_prob, fill = y_true)) + geom_bar(stat = 'identity', position = position_dodge2()) 
dat2plot = melt(pred_valid[, 3:8], id.vars = 'y_true')
ggplot(dat2plot, aes(x = variable, group = variable, y = value)) + geom_boxplot() + facet_grid(~y_true) 


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
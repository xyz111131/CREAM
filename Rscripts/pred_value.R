# read in prediction results
prediction2 = read.csv('results/shift_diff_multi_attn2_pred/MultiGene/blood_train_egenes_3K/Fold-0/xdsnow9b/Prediction_Results_19_in_valid_donors.csv')
prediction0 = read.csv('results/shift_diff_multi_attn2_pred/MultiGene/blood_train_egenes_3K/Fold-0/f05cqjnr/Prediction_Results_19_in_valid_donors.csv')
prediction1 = read.csv('results/shift_diff_multi_attn2_pred/MultiGene/blood_train_egenes_3K/Fold-0/11vjcl9h/Prediction_Results_19_in_valid_donors.csv')

# prediction2 = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/993714h9/Prediction_Results_19_in_valid_donors.csv')
# prediction0 = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/3cxzcjmn/Prediction_Results_19_in_valid_donors.csv')
# prediction1 = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/1c8jeh46/Prediction_Results_19_in_valid_donors.csv')
# prediction3 = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/m6derjox/Prediction_Results_19_in_valid_donors.csv')

prediction = rbind(prediction0, prediction1, prediction2)

metrics2 = read.csv('results/shift_diff_multi_attn2_pred/MultiGene/blood_train_egenes_3K/Fold-0/xdsnow9b/CrossIndivMetrics_valid_donors_Epoch19_rank2.csv')
metrics0 = read.csv('results/shift_diff_multi_attn2_pred/MultiGene/blood_train_egenes_3K/Fold-0/f05cqjnr/CrossIndivMetrics_valid_donors_Epoch19_rank0.csv')
metrics1 = read.csv('results/shift_diff_multi_attn2_pred/MultiGene/blood_train_egenes_3K/Fold-0/11vjcl9h/CrossIndivMetrics_valid_donors_Epoch19_rank1.csv')

metrics = rbind(metrics2, metrics1, metrics0)

metrics2 = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/993714h9/CrossIndivMetrics_valid_donors_Epoch19_rank0.csv')
metrics0 = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/3cxzcjmn/CrossIndivMetrics_valid_donors_Epoch19_rank2.csv')
metrics1 = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/1c8jeh46/CrossIndivMetrics_valid_donors_Epoch19_rank1.csv')
metrics3 = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/m6derjox/CrossIndivMetrics_valid_donors_Epoch19_rank3.csv')

metrics_std = rbind(metrics2, metrics1, metrics0, metrics3)


# compute std
library(data.table)
prediction = data.table(prediction)
gene_sd = prediction[, sd(y_true), by  = gene]
gene_sd = merge(gene_sd, metrics[, c('gene_name', 'pearsonr','r2', 'gene_split')], by.x = 'gene', by.y= 'gene_name')
gene_sd = merge(gene_sd, metrics_std[, c('gene_name', 'pearsonr','r2', 'gene_split')], by.x = 'gene', by.y= 'gene_name')

# compute std for observed gene expresion values
expr_raw = fread('data/gtex_eqtl_expression_matrix/gene_log_tpm_2017-06-05_v8_Whole_Blood.gct')
gene_sd0 = matrixStats::rowSds(as.matrix(expr_raw[,-1:-3]))
gene_sd0 = data.frame('gene' = sapply(expr_raw$Name, function(x) strsplit(x, '.', fixed = T)[[1]][1]), 'sd_raw' = gene_sd0)
gene_sd = merge(gene_sd, gene_sd0, by = 'gene')

plot(gene_sd$sd_raw[gene_sd$gene_split == 'valid'], gene_sd$V1[gene_sd$gene_split == 'valid'], ylab = 'standard deviation of expression under linear model in valid_donors', 
xlab = 'standard deviation of expression')
hist(gene_sd$sd_raw, main="")

plot(gene_sd$sd_raw[gene_sd$gene_split == 'valid'], gene_sd$pearsonr[gene_sd$gene_split == 'valid'], xlab = 'standard deviation of expression', ylab = 'pearson R')

par(mfrow = c(2,2))
plot(gene_sd$V1[gene_sd$gene_split.x == 'valid'], gene_sd$pearsonr.x[gene_sd$gene_split.x == 'valid'], xlab = 'standard deviation of expression under linear model in valid_donors', ylab = 'pearson R')
plot(gene_sd$V1[gene_sd$gene_split.x == 'valid'], gene_sd$pearsonr.y[gene_sd$gene_split.x == 'valid'], xlab = 'standard deviation of expression under linear model in valid_donors', ylab = 'pearson R (scaled expression)')
plot(gene_sd$pearsonr.x[gene_sd$gene_split.x == 'valid'], gene_sd$pearsonr.y[gene_sd$gene_split.x == 'valid'], xlab = 'pearson R', ylab = 'pearson R (scaled expression)')
abline(c(0,1))
plot(gene_sd$V1[gene_sd$gene_split.x == 'valid'], gene_sd$pearsonr.y[gene_sd$gene_split.x == 'valid'] - gene_sd$pearsonr.x[gene_sd$gene_split.x == 'valid'], main = '', 
     xlab = 'standard deviation of expression under linear model in valid_donors', ylab = 'diff pearson R of genes')
hist(gene_sd$pearsonr.y[gene_sd$gene_split.x == 'valid']- gene_sd$pearsonr.x[gene_sd$gene_split.x == 'valid'],, main = '', xlab = 'diff pearson')

plot(gene_sd$sd_raw[gene_sd$gene_split == 'train'], gene_sd$V1[gene_sd$gene_split == 'train'], ylab = 'standard deviation of expression under linear model in valid_donors', 
xlab = 'standard deviation of expression')
plot(gene_sd$sd_raw[gene_sd$gene_split == 'train'], gene_sd$pearsonr[gene_sd$gene_split == 'train'], xlab = 'standard deviation of expression', ylab = 'pearson R')
par(mfrow = c(2,2))
plot(gene_sd$V1[gene_sd$gene_split.x == 'train'], gene_sd$pearsonr.x[gene_sd$gene_split.x == 'train'], xlab = 'standard deviation of expression under linear model in valid_donors', ylab = 'pearson R')
plot(gene_sd$V1[gene_sd$gene_split.x == 'train'], gene_sd$pearsonr.y[gene_sd$gene_split.x == 'train'], xlab = 'standard deviation of expression under linear model in valid_donors', ylab = 'pearson R (scaled expression)')
plot(gene_sd$pearsonr.x[gene_sd$gene_split.x == 'train'], gene_sd$pearsonr.y[gene_sd$gene_split.x == 'train'], xlab = 'pearson R', ylab = 'pearson R (scaled expression)')
abline(c(0,1))
plot(gene_sd$V1[gene_sd$gene_split.x == 'train'], gene_sd$pearsonr.y[gene_sd$gene_split.x == 'train'] - gene_sd$pearsonr.x[gene_sd$gene_split.x == 'train'], main = '', 
     xlab = 'standard deviation of expression under linear model in valid_donors', ylab = 'diff pearson R of genes')

#hist(gene_sd$pearsonr.y[gene_sd$gene_split.x == 'train' & gene_sd$V1 > 0.1], main = '', xlab = 'pearson R of genes with SD > 0.1 (scaled)')
hist(gene_sd$pearsonr.y[gene_sd$gene_split.x == 'train' & gene_sd$V1 > 0.1] - gene_sd$pearsonr.x[gene_sd$gene_split.x == 'train' & gene_sd$V1 > 0.1], main = '', xlab = 'diff pearson R of genes with SD > 0.1')

# check the range of prediction and true
prediction = merge(prediction, gene_sd[, c('gene', 'gene_split')])
hist(prediction$y_true[prediction$gene_split == 'valid'])
hist(prediction$y_pred[prediction$gene_split == 'valid'])

# select genes with low/high sd
gene_low = gene_sd[V1 < 0.001]
gene_high = gene_sd[V1 > 1]

#gn = 'ENSG00000173597' #'ENSG00000225523' #'ENSG00000088387'
gn = 'ENSG00000179344' #'ENSG00000272221'
gene_sd[gn,]
plot(prediction[gene == gn, ]$y_pred, prediction[gene == gn, ]$y_true)

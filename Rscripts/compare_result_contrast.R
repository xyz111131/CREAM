library(data.table)
# compare enformer results

setwd('/Users/zhu/Gladstone\ Dropbox/Zhirui\ Hu/Enformer')
old = read.csv('en_whole_blood.csv', row.names = 1)
train = read.csv('300_train_genes.txt', header=F)
test = read.csv('test_genes.txt',header=F)
mean(old[old$fold==0, 'pearsonr'])

old = as.data.table(old)
summary(old[gene %in% train[,1] & fold == 0,])
summary(old[gene %in% test[,1] & fold == 0,])

org_test = fread('results/300Gene/4fplk4yy/CrossIndivMetrics_test_donors_Epoch100.csv')
org_valid = fread('results/300Gene/4fplk4yy/CrossIndivMetrics_valid_donors_Epoch99.csv')
contrast_test = fread('results/300Gene/contrast/9hrtx3w8/CrossIndivMetrics_test_donors_Epoch100.csv')
contrast_valid = fread('results/300Gene/contrast/9hrtx3w8/CrossIndivMetrics_valid_donors_Epoch99.csv')
contrast2_test = fread('results/300Gene/contrast/md0rfiqj_lr2/CrossIndivMetrics_test_donors_Epoch100.csv') #u3790ioj_2
contrast2_valid = fread('results/300Gene/contrast/md0rfiqj_lr2/CrossIndivMetrics_valid_donors_Epoch99.csv')
contrast2_best = fread('results/300Gene/contrast/md0rfiqj_lr2/CrossIndivMetrics_valid_donors_Epoch70.csv') #85
contrast3_test = fread('results/300Gene/contrast_raw/xvjtfr0b/CrossIndivMetrics_test_donors_Epoch100.csv') #raw expr
contrast3_valid = fread('results/300Gene/contrast_raw/xvjtfr0b/CrossIndivMetrics_valid_donors_Epoch99.csv')
contrast3_best = fread('results/300Gene/contrast_raw/xvjtfr0b/CrossIndivMetrics_valid_donors_Epoch96.csv')

scatterplot_metrics <- function(res, res1,  xlab = 'valid donors', ylab = 'test donors')
{
  par(mfrow=c(2,2))
  plot(res$pearsonr, res1[match(res$gene_name, gene_name), ]$pearsonr, xlab = xlab, ylab = ylab)
  abline(c(0,1))
  abline(c(-0.2,1))
  abline(c(0.2,1))
  mtext(paste('pearsonr:', round(cor(res$pearsonr, res1[match(res$gene_name, gene_name), ]$pearsonr),2)), side=3)
  ll = list(res$pearsonr, res1[match(res$gene_name, gene_name), ]$pearsonr)
  names(ll) = c(xlab, ylab)
  boxplot(ll)
  plot(res$r2, res1[match(res$gene_name, gene_name), ]$r2, xlab = xlab, ylab = ylab)
  mtext(paste('r2:', round(cor(res$r2, res1[match(res$gene_name, gene_name), ]$r2),2)), side=3)
  abline(c(0,1))
  ll = list(res$r2, res1[match(res$gene_name, gene_name), ]$r2)
  names(ll) = c(xlab, ylab)
  boxplot(ll)
}
# compare valid or test donors
scatterplot_metrics(org_valid, org_test) # contrast has the highest correlation for r2
# compare different epoch
scatterplot_metrics(contrast2_best, contrast2_valid, xlab = 'best epoch', ylab = 'final epoch')
# compare between runs
# train genes
scatterplot_metrics(org_valid[gene_split=='train'], contrast2_best[gene_split=='train'], xlab = 'org_train', ylab = 'contrast_train') 
scatterplot_metrics(org_test[gene_split=='train'], contrast2_test[gene_split=='train'], xlab = 'org_train', ylab = 'contrast2_train') 
scatterplot_metrics(org_valid[gene_split=='train'], contrast3_best[gene_split=='train'], xlab = 'org_train', ylab = 'contrast3_train') 
scatterplot_metrics(contrast2_test[gene_split=='train'], contrast3_test[gene_split=='train'], xlab = 'contrast2_train', ylab = 'contrast3_train') 
# test genes
scatterplot_metrics(org_test[gene_split=='test'], contrast2_test[gene_split=='test'], xlab = 'org_test', ylab = 'contrast2_test') # contrast and contrast2 is more similar to org, contrast2 have wider range of peearsonr for different test genes 
scatterplot_metrics(org_test[gene_split=='test'], contrast3_test[gene_split=='test'], xlab = 'org_test', ylab = 'contrast3_test')


# gn = intersect(res$gene_name[res$gene_split=='train'], old$gene)
# old = old[fold == 0]
# plot(old[match(gn, gene), ]$pearsonr, res[match(gn, gene_name), ]$pearsonr)
# abline(c(0,1))
# boxplot(list(old[match(gn, gene), ]$pearsonr, res[match(gn, gene_name), ]$pearsonr))
# 
# plot(old[match(gn, gene), ]$r2, res[match(gn, gene_name), ]$r2)
# abline(c(0,1))
# boxplot(list(old[match(gn, gene), ]$r2, res[match(gn, gene_name), ]$r2))
# abline(c(0,1))

# prediction
org_pred_test = read.csv('results/300Gene/4fplk4yy/Prediction_Results_100_in_test_donors.csv')
org_pred_valid = read.csv('results/300Gene/4fplk4yy/Prediction_Results_99_in_valid_donors.csv')
contrast_pred_test = read.csv('results/300Gene/contrast/9hrtx3w8/Prediction_Results_100_in_test_donors.csv')
contrast_pred_valid = read.csv('results/300Gene/contrast/9hrtx3w8/Prediction_Results_99_in_valid_donors.csv')
contrast2_pred_test = read.csv('results/300Gene/contrast/md0rfiqj_lr2/Prediction_Results_100_in_test_donors.csv') #u3790ioj_2
contrast2_pred_valid = read.csv('results/300Gene/contrast/md0rfiqj_lr2/Prediction_Results_99_in_valid_donors.csv')
contrast2_pred_best = read.csv('results/300Gene/contrast/md0rfiqj_lr2/Prediction_Results_70_in_valid_donors.csv') #85
contrast3_pred_test = read.csv('results/300Gene/contrast_raw/xvjtfr0b/Prediction_Results_100_in_test_donors.csv') 
contrast3_pred_valid = read.csv('results/300Gene/contrast_raw/xvjtfr0b/Prediction_Results_99_in_valid_donors.csv')
contrast3_pred_best = read.csv('results/300Gene/contrast_raw/xvjtfr0b/Prediction_Results_96_in_valid_donors.csv') 
#diff = res$pearsonr -  res1[match(res$gene_name, gene_name), ]$pearsonr
#gene = res$gene_name[which(diff > 0.4)]
#res[gene_name %in% gene];res1[gene_name %in% gene]
library(ggExtra)
library(ggplot2)
scatterplot_true <- function(results, metrics, gene)
{
  for(g in gene)
  {
    dat2plot = results[results$gene==g, ]
    met = metrics[metrics$gene_name == g, ]
    #dat2plot$expr = factor(dat2plot$expr, levels = annot)
    p = ggplot(dat2plot, aes(x = y_true, y = y_pred)) + geom_point(alpha = 0.5) + geom_abline(slope = 1, intercept = 0)+
      annotate("text", x = min(dat2plot$y_true), y = max(dat2plot$y_pred), 
               label = paste0('Gene: ', g, '\n',  'Pearsons: ', round(met$pearsonr,2), ', ',
                              'r2: ', round(met$r2,2)),
               hjust = 0, vjust = 1, size = 4) + ylab('y_pred')  #geom_density2d()
    #print(p)
    print(ggMarginal(p, type = "histogram", margins = 'x'))
  }
}

scatterplot_pred_true <- function(res, res1, xlab = 'org', ylab = 'contrast')
{
  par(mfrow=c(2,2))
  res = res[order(res$gene, res$donor), ]
  res1 = res1[order(res1$gene, res1$donor), ]
  print(all(res$donor == res1$donor)); print(all(res$gene == res1$gene)); 
  plot(res$y_pred, res1$y_true, xlab = xlab, ylab = ylab, pch = 20, col = factor(res$gene), cex = 0.2) #scales::alpha('red', 0.5)
  abline(c(0,1))
  abline(c(-1,1))
  abline(c(1,1))
  mtext(paste('pearsonr:', round(cor(res$y_pred, res1$y_pred),2)), side=3)
  # ll = list(res$y_pred, res1$y_pred)
  # names(ll) = c(xlab, ylab)
  # boxplot(ll)
  hist(res$y_pred, main = xlab, xlab = 'y_pred')
  hist(res1$y_true, main = ylab, xlab = 'y_true')
  plot(density(res$y_pred), main = 'y_pred'); lines(density(res1$y_true), col=2)
}
scatterplot_pred <- function(res, res1, xlab = 'org', ylab = 'contrast')
{
  par(mfrow=c(2,2))
  res = res[order(res$gene, res$donor), ]
  res1 = res1[order(res1$gene, res1$donor), ]
  print(all(res$donor == res1$donor)); print(all(res$gene == res1$gene)); 
  plot(res$y_pred, res1$y_pred, xlab = xlab, ylab = ylab, pch = 20, col = factor(res$gene), cex = 0.2) #scales::alpha('red', 0.5)
  abline(c(0,1))
  abline(c(-1,1))
  abline(c(1,1))
  mtext(paste('pearsonr:', round(cor(res$y_pred, res1$y_pred),2)), side=3)
  # ll = list(res$y_pred, res1$y_pred)
  # names(ll) = c(xlab, ylab)
  # boxplot(ll)
  hist(res$y_pred, main = xlab, xlab = 'y_pred')
  hist(res1$y_pred, main = ylab, xlab = 'y_pred')
  plot(density(res$y_pred), main = 'y_pred'); lines(density(res1$y_pred), col=2)
}
# for example genes
scatterplot_true(pred_test, res1, gene)
scatterplot_true(pred_valid, res, gene)

# to compare prediction between methods
test_genes = org_test$gene_name[org_test$gene_split == 'test']
train_genes = org_test$gene_name[org_test$gene_split == 'train']
scatterplot_pred(org_pred_test[org_pred_test$gene %in% test_genes, ], contrast2_pred_test[contrast2_pred_test$gene %in% test_genes, ], 
                    xlab = 'org_test_genes', ylab = 'contrast2_test_genes') #0.16
scatterplot_pred(org_pred_test[org_pred_test$gene %in% train_genes, ], contrast2_pred_test[contrast2_pred_test$gene %in% train_genes, ], 
                 xlab = 'org_train_genes', ylab = 'contrast2_train_genes') #0.9

scatterplot_pred(org_pred_test[org_pred_test$gene %in% test_genes, ], contrast3_pred_test[contrast3_pred_test$gene %in% test_genes, ], 
                 xlab = 'org_test_genes', ylab = 'contrast3_test_genes') 
scatterplot_pred(org_pred_test[org_pred_test$gene %in% train_genes, ], contrast3_pred_test[contrast3_pred_test$gene %in% train_genes, ], 
                 xlab = 'org_train_genes', ylab = 'contrast3_train_genes') 
scatterplot_pred_true(contrast3_pred_test[contrast3_pred_test$gene %in% test_genes, ], contrast3_pred_test[contrast3_pred_test$gene %in% test_genes, ], 
                xlab = 'contrast3_test_genes', ylab = 'test_genes') 

scatterplot_pred(contrast_pred_test[contrast_pred_test$gene %in% test_genes, ], contrast2_pred_test[contrast2_pred_test$gene %in% test_genes, ],)
scatterplot_pred(contrast2_pred_valid, contrast2_pred_best,)
scatterplot_pred(org_pred_valid, contrast2_pred_valid)


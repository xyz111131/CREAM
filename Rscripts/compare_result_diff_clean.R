library(data.table) 
library(colorout)
options(vsc.dev.args = list(width=1900, height=1400, pointsize=5, res=300))
# compare differential expression results (with coordinate shifts)
# test genes only
setwd('Rscripts/')
models = read.csv('wandb_export_org_models.csv')
diff_models = read.csv('wandb_export_diff_models1.csv')
valid = T
if(!valid) test_genes = read.csv('test_genes.txt',header=F)$V1 else test_genes = read.csv('valid_genes.txt',header=F)$V1

#donors = unique(org[['pred']][[1]]$donor)
#write.table(donors, file = 'test_donors.txt', col.names =F, quote = F, row.names = F)
donors = read.table('test_donors.txt')$V1

readResults <- function(models, nameidx = c(2,5), valid = T)
{
  org = list()
  prefix = ""
  if(valid)  prefix = "/valid_genes" 
  for(i in 1:nrow(models))
  {
    meta = models[i,]
    id = paste(strsplit(meta$Name, '_')[[1]][nameidx], collapse = '_')
    if(valid)  ep = -1 else ep = meta$max_epoch
    if(!file.exists(paste0(meta$save_dir, prefix, '/CrossIndivMetrics_test_donors_Epoch', ep, '.csv'))) next
    org[['metrics']][[id]] = fread(paste0(meta$save_dir, prefix, '/CrossIndivMetrics_test_donors_Epoch', ep, '.csv'))
    org[['pred']][[id]] = read.csv(paste0(meta$save_dir, prefix, '/Prediction_Results_', ep,'_in_test_donors', '.csv'))
  }
  
  return(org)
}

# convert diff expression to expression
convert_expr <- function(dat, donors)
{
  dat = unique(dat)
  #print(paste0('number of samples: ', nrow(dat)))
  X = matrix(0, nrow(dat), length(donors))
  colnames(X) = donors
  dat$donor1 = sapply(dat$donor, function(x) strsplit(x, ':')[[1]][1])
  dat$donor2 = sapply(dat$donor, function(x) strsplit(x, ':')[[1]][2])
  for(i in 1:nrow(X))
  {
    X[i, dat$donor1[i]] = 1
    X[i, dat$donor2[i]] = -1
  }
  X = rbind(X, rep(1, ncol(X)))
  res = glm.fit(X, c(dat$y_pred,0))
  #res = cv.glmnet(X, c(dat$y_pred,0),intercept=FALSE)
  #plot(res)
  return(res$coefficients) # #coef(res)[-1]
}

compute_metrics <- function(contrast_pred_test2, org_pred_test, donors,test_genes, contrast_metrics = NULL, toplot = F){
  metrics = predicts = NULL
  for(gene in test_genes){
    dat = contrast_pred_test2[contrast_pred_test2$gene == gene, c('donor', 'y_pred', 'gene') ]
    y_true = org_pred_test[org_pred_test$gene == gene, ]
    y_true = y_true[match(donors, y_true$donor),'y_true' ]
    y_pred = convert_expr(dat, donors)
    temp = cbind('pred' = y_pred, 'true' = y_true, 'donor' = donors, 'gene' = gene)
    predicts = rbind(predicts, temp)
    #plot(y_true,y_pred)
    metrics = rbind(metrics,c(cor(y_true, y_pred), 1-sum((y_true - y_pred)^2)/sum(y_true^2)))
  }
  predicts = data.frame(predicts)
  if(toplot ) #& !is.null(contrast_metrics)
  {
    #par(mfrow=c(1,2))
    #plot(unlist(contrast_metrics[match(test_genes, gene_name), 'pearsonr']), metrics[1,], xlab = 'Delta expression', ylab = 'Recovered expression', main = 'Pearson r' );abline(c(0,1)) 
    #plot(unlist(contrast_metrics[match(test_genes, gene_name), 'r2']), metrics[2,], xlab = 'Delta expression', ylab = 'Recovered expression', main = 'R2'  );abline(c(0,1))
    plot(predicts$true, predicts$pred, xlab = 'true', ylab = 'pred', pch = 20, col = factor(predicts$gene), cex = 0.2) #scales::alpha('red', 0.5)
    abline(c(0,1))
    mtext(paste('pearsonr:', round(cor(as.numeric(predicts$pred), as.numeric(predicts$true)),2)), side=3)
  }
  metrics = data.table(metrics)
  metrics$gene_name = test_genes
  metrics$gene_split = 'test'
  colnames(metrics)[1:2] = c('pearsonr', 'r2')
  print(summary(metrics[,1:2]))
  return(metrics)
}


org = readResults(models, nameidx = c(2), valid = valid)
diff = readResults(diff_models, nameidx = c(4), valid = valid)
org_pred = org[['pred']][[1]]  # any prediction file from org

diff_metrics = lapply(diff[['pred']], function(x) compute_metrics(x, org_pred, donors, test_genes, toplot = F))

boxplots <- function(org_metrics, diff_metrics, mets = 'pearsonr', valid = 'valid')
{
  pearsonrs = lapply(c(org_metrics, diff_metrics), function(x) x[gene_split == 'test'][[mets]])  
  boxplot(pearsonrs, col = c(rep('grey', length(org_metrics)), rep(2, length(diff_metrics))), outline = F, ylab = mets)
  abline(h=median(pearsonrs[['org']]), lty=2,col=3)
  pval = wilcox.test(org_metrics[['org']][[mets]],
  diff_metrics[['attn']][match(org_metrics[['org']]$gene_name, gene_name), ][[mets]], paired = T, alternative = 'less')
  mtext(paste(valid, 'genes, contrast attn vs org p-value:', round(pval$p.value,4)), side=3)
}

boxplots(org[['metrics']], diff_metrics, valid = ifelse(valid, 'valid', 'test'))

# recompute r2 for org
recompute_r2 <- function(org_pred)
{
  org_pred_test = data.table(org_pred)
  org_pred_test = org_pred_test[gene %in% test_genes]
  org_pred_test[, list(r2 = {
    y_pred = y_pred - mean(y_pred)
    1-sum((y_true - y_pred)^2)/sum(y_true^2)
  }, gene_split = 'test'), by = gene]
}
org_r2s = lapply(org[['pred']], function(x) recompute_r2(x))
boxplots(org_r2s, diff_metrics, mets = 'r2')


####################

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
  pval = wilcox.test(res$pearsonr, res1[match(res$gene_name, gene_name), ]$pearsonr, paired = T, alternative = 'less')
  mtext(paste('p-value:', round(pval$p.value,4)), side=3)
  plot(res$r2, res1[match(res$gene_name, gene_name), ]$r2, xlab = xlab, ylab = ylab)
  mtext(paste('r2:', round(cor(res$r2, res1[match(res$gene_name, gene_name), ]$r2),2)), side=3)
  abline(c(0,1))
  ll = list(res$r2, res1[match(res$gene_name, gene_name), ]$r2)
  names(ll) = c(xlab, ylab)
  boxplot(ll)
}




contrast_2K_metrics20 = compute_metrics(contrast_2K_pred0, org_pred, donors, test_genes, toplot = F) #contrast_2K_metrics0, 
contrast_1K_no_attn_metrics2 = compute_metrics(contrast_1K_no_attn_pred, org_pred, donors, test_genes,  toplot = T) #contrast_1K_metrics,
contrast_1K_metrics2 = compute_metrics(contrast_1K_pred, org_pred, donors, test_genes,  toplot = T) #contrast_1K_metrics,
contrast_metrics2 = compute_metrics(contrast_pred, org_pred, donors, test_genes) #, contrast_metrics

scatterplot_metrics(org_metrics[gene_split=='test'], contrast_2K_metrics20, xlab = 'org_test', ylab = 'contrast_2K')
scatterplot_metrics(contrast_2K_metrics2, contrast_2K_metrics20, xlab = 'contrast', ylab = 'contrast_2K')

# recompute r2 for org
recompute_r2 <- function(org_pred)
{
  org_pred_test = data.table(org_pred)
  org_pred_test = org_pred_test[gene %in% test_genes]
  org_pred_test[, list(r2 = {
    y_pred = y_pred - mean(y_pred)
    1-sum((y_true - y_pred)^2)/sum(y_true^2)
  }), by = gene]
}

## boxplots comparing methods
par(mfrow = c(2,1))
org_metrics_test = org_metrics[gene_split=='test', ]
org_1K_metrics_test = org_1K_metrics[gene_split=='test', ]
boxplot(list('org' = org_metrics_test$pearsonr, 'org_1K' = org_1K_metrics_test$pearsonr, 
             'constrast' = contrast_metrics2$pearsonr,'constrast_1K' = contrast_1K_no_attn_metrics2$pearsonr, 'constrast_attn_1K' = contrast_1K_metrics2$pearsonr, 
             'constrast_attn_2K' = contrast_2K_metrics20$pearsonr))
abline(h=median(org_metrics_test$pearsonr), lty=2,col=2)
pval = wilcox.test(org_metrics_test$pearsonr,contrast_2K_metrics20[match(org_metrics_test$gene_name, gene_name), ]$pearsonr, paired = T, alternative = 'less')
mtext(paste('contrast 2K vs org p-value:', round(pval$p.value,4)), side=3)

for(oo in org[['pred']])
{
  org_pred_test = recompute_r2(oo)
  #org_1K_pred_test = recompute_r2(org_1K_pred)
}

boxplot(list('org' = org_pred_test$r2, 'org_1K' = org_1K_pred_test$r2, 
  'constrast' = contrast_metrics2$r2, 'constrast_1K' = contrast_1K_no_attn_metrics2$r2, 'constrast_attn_1K' = contrast_1K_metrics2$r2, 
  'constrast_attn_2K' = contrast_2K_metrics20$r2),outline = F )
abline(h=median(org_pred_test$r2), lty=2,col=2)
pval = wilcox.test(org_pred_test$r2,contrast_2K_metrics20[match(contrast_metrics2$gene_name, gene_name), ]$r2, paired = T, alternative = 'less')
mtext(paste('contrast 2K vs org p-value:', round(pval$p.value,4)), side=3)



######################

contrast_metrics = fread('results/300Gene/shift_diff/qhxnix3g/CrossIndivMetrics_test_donors_Epoch-1.csv') # faster, pearson r is better but r2 is worse
contrast_pred = read.csv('results/300Gene/shift_diff/qhxnix3g/Prediction_Results_-1_in_test_donors.csv')

contrast_1K_no_attn_metrics = fread('results/1000Gene/shift_diff/gvebm8sj_no_attn/CrossIndivMetrics_test_donors_Epoch20.csv') 
contrast_1K_no_attn_pred = read.csv('results/1000Gene/shift_diff/gvebm8sj_no_attn/Prediction_Results_20_in_test_donors.csv')

contrast_1K_metrics = fread('results/1000Gene/shift_diff/5adchjbf_attn/CrossIndivMetrics_test_donors_Epoch20.csv') # last epoch, better
contrast_1K_pred = read.csv('results/1000Gene/shift_diff/5adchjbf_attn/Prediction_Results_20_in_test_donors.csv')

#contrast_1K_metrics = fread('results/1000Gene/shift_diff/7u9z3gmb_sym/CrossIndivMetrics_test_donors_Epoch-1.csv') 
#contrast_1K_pred = read.csv('results/1000Gene/shift_diff/7u9z3gmb_sym/Prediction_Results_-1_in_test_donors.csv')

#contrast_2K_metrics = fread('results/2000Gene/shift_diff/ygepm1pu_20/CrossIndivMetrics_test_donors_Epoch-1.csv')
#contrast_2K_pred = read.csv('results/2000Gene/shift_diff/ygepm1pu_20/Prediction_Results_-1_in_test_donors.csv')

contrast_2K_metrics0 = fread('results/2000Gene/shift_diff/qlygtpab_wdcay_30/CrossIndivMetrics_test_donors_Epoch-1.csv') # better
contrast_2K_pred0 = read.csv('results/2000Gene/shift_diff/qlygtpab_wdcay_30/Prediction_Results_-1_in_test_donors.csv')
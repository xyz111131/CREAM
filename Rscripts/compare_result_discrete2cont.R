library(data.table)
# compare with discrete results

setwd('/Users/zhu/Gladstone\ Dropbox/Zhirui\ Hu/Enformer')

# metrics
org_test = fread('results/300Gene/4fplk4yy/CrossIndivMetrics_test_donors_Epoch100.csv')
org_valid = fread('results/300Gene/4fplk4yy/CrossIndivMetrics_valid_donors_Epoch99.csv')
res = fread('results/300Gene/discrete/h83xoa9n/CrossIndivMetrics_valid_donors_Epoch99.csv')
res1 = fread('results/300Gene/discrete/h83xoa9n/CrossIndivMetrics_test_donors_Epoch100.csv')


# prediction
org_pred_test = read.csv('results/300Gene/4fplk4yy/Prediction_Results_100_in_test_donors.csv')
org_pred_valid = read.csv('results/300Gene/4fplk4yy/Prediction_Results_99_in_valid_donors.csv')
pred_valid = read.csv('results/300Gene/discrete/h83xoa9n/Prediction_Results_99_in_valid_donors.csv')
pred_test = read.csv('results/300Gene/discrete/h83xoa9n/Prediction_Results_100_in_test_donors.csv')

# convert prediction to continuous
mean(org_pred_valid$y_true[org_pred_valid$y_true < Inf & org_pred_valid$y_true > 1])
mean(org_pred_test$y_true[org_pred_test$y_true < Inf & org_pred_test$y_true > 1])
pred_valid$y_pred = colSums(t(pred_valid[,3:7]) * c(-1.5, -0.62, 0, 0.62, 1.5)) # bins -1, -0.3, 0.3, 1
pred_test$y_pred = colSums(t(pred_test[,3:7]) * c(-1.5, -0.62, 0, 0.62, 1.5)) # bins -1, -0.3, 0.3, 1

# compute metrics
all(pred_valid$gene == org_pred_valid$gene)
all(pred_valid$donor == org_pred_valid$donor)
pred_valid$y_true2 = org_pred_valid$y_true
pred_valid = data.table(pred_valid)
discrete_valid = pred_valid[, list('pearsonr' = cor(y_true2, y_pred), 'r2' = 1-sum((y_true2 - y_pred)^2)/sum(y_true2^2)), by = gene]
colnames(discrete_valid)[1] = 'gene_name' 

all(pred_test$gene == org_pred_test$gene)
all(pred_test$donor == org_pred_test$donor)
pred_test$y_true2 = org_pred_test$y_true
pred_test = data.table(pred_test)
discrete_test = pred_test[, list('pearsonr' = cor(y_true2, y_pred), 'r2' = 1-sum((y_true2 - y_pred)^2)/sum(y_true2^2)), by = gene]
colnames(discrete_test)[1] = 'gene_name' 

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
  print(mean(res1[match(res$gene_name, gene_name), ]$pearsonr))
  plot(res$r2, res1[match(res$gene_name, gene_name), ]$r2, xlab = xlab, ylab = ylab)
  mtext(paste('r2:', round(cor(res$r2, res1[match(res$gene_name, gene_name), ]$r2),2)), side=3)
  abline(c(0,1))
  ll = list(res$r2, res1[match(res$gene_name, gene_name), ]$r2)
  names(ll) = c(xlab, ylab)
  boxplot(ll)
  print(mean(res1[match(res$gene_name, gene_name), ]$r2))
}

scatterplot_metrics(org_valid, discrete_valid, xlab = 'org_train', ylab = 'discrete_train') 
scatterplot_metrics(org_test[gene_split=='test'], discrete_test, xlab = 'org_test', ylab = 'discrete_test') # contrast and contrast2 is more similar to org, contrast2 have wider range of peearsonr for different test genes 

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
  mtext(paste('pearsonr:', round(cor(res$y_pred, res1$y_true),2)), side=3)
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
test_genes = org_test$gene_name[org_test$gene_split == 'test']
train_genes = org_test$gene_name[org_test$gene_split == 'train']
scatterplot_pred(org_pred_test[org_pred_test$gene %in% test_genes, ], pred_test[pred_test$gene %in% test_genes, ], 
                 xlab = 'org_test_genes', ylab = 'discrete_test_genes') 
scatterplot_pred(org_pred_test[org_pred_test$gene %in% train_genes, ], pred_test[pred_test$gene %in% train_genes, ], 
                 xlab = 'org_train_genes', ylab = 'discrete_train_genes') 

scatterplot_pred_true(org_pred_test[org_pred_test$gene %in% test_genes, ], org_pred_test[org_pred_test$gene %in% test_genes, ], 
                 xlab = 'org_test_genes', ylab = 'org_test_genes_true') 
scatterplot_pred_true(org_pred_test[org_pred_test$gene %in% train_genes, ], org_pred_test[org_pred_test$gene %in% train_genes, ], 
                 xlab = 'org_train_genes', ylab = 'org_train_genes_true') 

scatterplot_pred_true(pred_test[pred_test$gene %in% test_genes, ], org_pred_test[org_pred_test$gene %in% test_genes, ], 
                     xlab = 'test_genes', ylab = 'test_genes_true') 
scatterplot_pred_true(pred_test[pred_test$gene %in% train_genes, ], org_pred_test[org_pred_test$gene %in% train_genes, ], 
                      xlab = 'train_genes', ylab = 'train_genes_true') 


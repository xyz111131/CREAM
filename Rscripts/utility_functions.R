# convert diff expression to absolute expression
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


##contrast_1K_pred2 = convert_expr(contrast_1K_pred[contrast_1K_pred$gene == 'PPP5C',], donors)


compute_metrics <- function(contrast_pred_test2, org_pred_test, donors,test_genes, tissues = NULL, contrast_metrics = NULL, toplot = T){
  metrics = predicts = NULL
  if(!is.null(tissues))
  {
    predicts = lapply(tissues, function(ts)
    {
      print(ts)
      results = mclapply(test_genes, function(gn){
        dat = contrast_pred_test2[contrast_pred_test2$gene == gn & contrast_pred_test2$tissue == ts, c('donor', 'y_pred', 'gene') ]
        expr_true = org_pred_test[org_pred_test$gene == gn & org_pred_test$tissue == ts, ]
        y_true = expr_true[match(donors, expr_true$donor),]$y_true
        y_pred = convert_expr(dat, donors)
        data.table('y_pred' = y_pred, 'y_true' = y_true, 'donor' = donors, 'gene_name' = gn, 'tissue' = ts)
        #plot(y_true,y_pred)
        
      }, mc.cores = 8)
      do.call(rbind, results)
    })
    predicts = do.call(rbind, predicts)
    metrics = predicts[, list('pearsonr'= cor(y_true, y_pred), 'r2' = 1-sum((y_true - y_pred)^2)/sum(y_true^2)), 
          by = c('tissue', 'gene_name')]
  }else{
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
    predicts = data.table(predicts)
    metrics = data.frame(metrics)
    metrics$gene_name = test_genes
    colnames(metrics)[1:2] = c('pearsonr', 'r2')
  }

  if(toplot ) #& !is.null(contrast_metrics)
  {
    #par(mfrow=c(1,2))
    #plot(unlist(contrast_metrics[match(test_genes, gene_name), 'pearsonr']), metrics[1,], xlab = 'Delta expression', ylab = 'Recovered expression', main = 'Pearson r' );abline(c(0,1)) 
    #plot(unlist(contrast_metrics[match(test_genes, gene_name), 'r2']), metrics[2,], xlab = 'Delta expression', ylab = 'Recovered expression', main = 'R2'  );abline(c(0,1))
    plot(predicts$true, predicts$pred, xlab = 'true', ylab = 'pred', pch = 20, col = factor(predicts$gene), cex = 0.2) #scales::alpha('red', 0.5)
    abline(c(0,1))
    mtext(paste('pearsonr:', round(cor(as.numeric(predicts$pred), as.numeric(predicts$true)),2)), side=3)
  }
  
  print(summary(metrics[!is.na(metrics$pearsonr),c('pearsonr', 'r2')]))
  return(list(metrics, predicts))
}


scatterplot_metrics <- function(res, res1,  xlab = 'valid donors', ylab = 'test donors')
{
  par(mfrow=c(2,2))
  plot(res$pearsonr, res1[match(res$gene_name, gene_name), ]$pearsonr, xlab = xlab, ylab = ylab)
  abline(c(0,1))
  #abline(c(-0.2,1))
  #abline(c(0.2,1))
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

# compute metrics with a mismatch tissue
# read in gene expression
genes_id_mapping = read.csv('gtex_eqtl_expression_matrix/gene_id_mapping.csv', row.names=1)
cortex = fread('gtex_eqtl_expression_matrix/Muscle_Skeletal.v8.normalized_expression.bed.gz') 
cortex$gene_name = genes_id_mapping[match(cortex$gene_id, genes_id_mapping$Name), 'Description']
donors_overlap = intersect(donors, colnames(cortex)) # 57, 49
length(intersect(test_genes, cortex$gene_name)) # 90, 100
cortex_test = as.data.frame(cortex[gene_name %in% test_genes, ])
cortex_test = reshape2::melt(cortex_test[, c('gene_name', donors_overlap)], id.vars = 'gene_name')
colnames(cortex_test) = c('gene', 'donor', 'true2')
predicts = merge(contrast_1K_metrics2[[2]], cortex_test, by = c('gene', 'donor'))

compare_metrics_boxplot <- function(predicts)
{
  predicts = as.data.table(predicts)
  predicts[, true := as.numeric(true)]
  predicts[, pred := as.numeric(pred)]
  metrics = predicts[, list('cor' = cor(true, pred), 'r2' = 1-sum((true - pred)^2)/sum(true^2), 
                            'cor_muscle' = cor(true2, pred), 'r2_muscle' = 1-sum((true2 - pred)^2)/sum(true2^2)), by = gene]
  par(mfrow=c(1,2))
  boxplot(metrics[,c(2,4)]); abline(h=0, lty=2)
  boxplot(metrics[,c(3,5)]); abline(h=0, lty=2)
}



# plot prediction for a gene and compare with linear model
gn =  'SIGLEC12'
predicts = contrast_1K_metrics2[[2]]
predicts = predicts[predicts$gene == gn,]
plot(predicts$true, predicts$pred, xlab = 'true', ylab = 'pred', pch = 20, cex = 0.5) #scales::alpha('red', 0.5)
abline(c(0,1))

# compare with prediction from linear model
lm_pred0 = read.table('genotype/combined.tsv', header=T)
lm_pred = lm_pred0[lm_pred0$fold == 0 & lm_pred$gene == gn, ]
predicts = merge(predicts, lm_pred, by.y = 'person_id', by.x = 'donor')
plot(predicts$pred, predicts$y_hat, xlab = 'DL', ylab = 'LM', pch = 20, cex = 0.5) #scales::alpha('red', 0.5)
abline(c(0,1))




# recompute r2 for org
recompute_r2 <- function(org_pred, test_genes)
{
  org_pred_test = data.table(org_pred)
  org_pred_test = org_pred_test[gene_name %in% test_genes]
  if('tissue' %in% colnames(org_pred))
  {
      org_pred_test[, list(r2 = {
      y_pred = y_pred - mean(y_pred)
      1-sum((y_true - y_pred)^2)/sum(y_true^2)
    }), by = c('gene_name', 'tissue')]
  }else{
    org_pred_test[, list(r2 = {
      y_pred = y_pred - mean(y_pred)
      1-sum((y_true - y_pred)^2)/sum(y_true^2)
    }), by = gene_name]
  }
}


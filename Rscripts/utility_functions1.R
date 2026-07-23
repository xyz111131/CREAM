# convert diff expression to absolute expression
convert_expr <- function(dat, donors)
{
  dat = unique(dat)
  if(nrow(dat) == 0)
  {
    # No pairwise contrast rows for this gene/tissue; propagate NA predictions.
    return(rep(NA_real_, length(donors)))
  }
  #print(paste0('number of samples: ', nrow(dat)))
  X = matrix(0, nrow(dat), length(donors))
  colnames(X) = donors
  dat$donor1 = sapply(dat$donor, function(x) strsplit(x, ':')[[1]][1])
  dat$donor2 = sapply(dat$donor, function(x) strsplit(x, ':')[[1]][2])
  for(i in seq_len(nrow(X)))
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
  org_gene_col = if('gene' %in% colnames(org_pred_test)) 'gene' else 'gene_name'
  if(!is.null(tissues))
  {
    predicts = lapply(tissues, function(ts)
    {
      print(ts)
      results = mclapply(test_genes, function(gn){
        tryCatch({
          dat = contrast_pred_test2[contrast_pred_test2$gene == gn & contrast_pred_test2$tissue == ts, c('donor', 'y_pred', 'gene') ]
          expr_true = org_pred_test[org_pred_test[[org_gene_col]] == gn & org_pred_test$tissue == ts, ]
          y_true = expr_true[match(donors, expr_true$donor),]$y_true
          y_pred = convert_expr(dat, donors)
          data.table('y_pred' = y_pred, 'y_true' = y_true, 'donor' = donors, 'gene_name' = gn, 'tissue' = ts)
        }, error = function(e) {
          message(paste0('compute_metrics error | tissue=', ts, ' gene=', gn, ' | ', conditionMessage(e)))
          data.table('y_pred' = rep(NA_real_, length(donors)),
                     'y_true' = rep(NA_real_, length(donors)),
                     'donor' = donors,
                     'gene_name' = gn,
                     'tissue' = ts,
                     'error_msg' = conditionMessage(e))
        })
        
      }, mc.cores = 8)
      data.table::rbindlist(results, fill = TRUE)
    })
    predicts = data.table::rbindlist(predicts, fill = TRUE)
    metrics = predicts[, list('pearsonr'= cor(y_true, y_pred, use = 'na.or.complete'), 
                    'r2' = 1-sum((y_true - y_pred)^2, na.rm=T)/sum(y_true^2, na.rm = T)), 
          by = c('tissue', 'gene_name')]
  }else{
    for(gene in test_genes){
      dat = contrast_pred_test2[contrast_pred_test2$gene == gene, c('donor', 'y_pred', 'gene') ]
      y_true = org_pred_test[org_pred_test[[org_gene_col]] == gene, ]
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
  mtext(paste('pearsonr:', round(cor(res$pearsonr, res1[match(res$gene_name, gene_name), ]$pearsonr, use = 'complete'),2)), side=3)
  ll = list(res$pearsonr, res1[match(res$gene_name, gene_name), ]$pearsonr)
  names(ll) = c(xlab, ylab)
  boxplot(ll, outline = FALSE)
  pval = wilcox.test(res$pearsonr, res1[match(res$gene_name, gene_name), ]$pearsonr, paired = T) #, alternative = 'less')
  mtext(paste('p-value:', round(pval$p.value,4)), side=3)
  plot(res$r2, res1[match(res$gene_name, gene_name), ]$r2, xlab = xlab, ylab = ylab)
  mtext(paste('r2:', round(cor(res$r2, res1[match(res$gene_name, gene_name), ]$r2, use = 'complete'),2)), side=3)
  abline(c(0,1))
  ll = list(res$r2, res1[match(res$gene_name, gene_name), ]$r2)
  names(ll) = c(xlab, ylab)
  boxplot(ll, outline = FALSE)
  pval = wilcox.test(res$r2, res1[match(res$gene_name, gene_name), ]$r2, paired = T) #, alternative = 'less')
  mtext(paste('p-value:', round(pval$p.value,4)), side=3)
}


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


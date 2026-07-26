# compare the model prediction of rare variants with ground truth
library(data.table)
library(tidyr)
library(ggplot2)
runids = c('1s4gqfj7', '1zw1vgdj', 'b4ihfjbn', 'aaipwoio')
res_dir = 'results/CREAM_ISM_rarevarints/Whole_BloodModels/MultiGene/'
dat_dir = 'data/rare_variants_folds/'
results_all = NULL
for(i in 1:4)
{
    results = read.csv(paste0(res_dir, runids[i], '/', runids[i], '_ContrastMultiAttention_49152bp_Rare_variants_Fold-', i, '.csv')) # 
    results = unique(results)
    ground_truth = read.csv(paste0(dat_dir, 'unseen_variants-fold', i, '.csv'), header = T, row.names = NULL)
    genotype = read.table(paste0(dat_dir, 'select_genotype-fold', i, '.txt'), header = T, sep = '\t')
    homo_donors = separate_rows(genotype[genotype$hom_donors!='', c('variant_id', 'hom_donors')],'hom_donors', sep = ",")
    homo_donors$multiplier = 2
    results$variant_id = paste(results$chrom, results$pos0 + 1, results$ref, results$alt, 'b38', sep = '_')
    results$coef = ground_truth[match(results$variant_id, ground_truth$variant_id), 'en_avg_coeff']
    results = merge(results, homo_donors, by.x = c('variant_id', 'donor'), by.y = c('variant_id', 'hom_donors'), all.x = T)
    results$multiplier[is.na(results$multiplier)] = 1
    results$multiplier[results$donor == 'ref'] = 2
    results_all = rbind(results_all, results)
}


col = rep(1, nrow(results_all)) # 450 rows, 377 unique variant + donor
col[results_all$donor=='ref'] = 2


## plot predicted vs truth
# double the diff for homozygotes and reverse sign for ref
ind = which(results_all$multiplier == 2)
results_all[ind,  'diff_pred'] = results_all[ind,  'diff_pred'] /2
results_all[results_all$donor!='ref','diff_pred'] = -results_all[results_all$donor!='ref','diff_pred']

#results$coef[is.na(results$coef)] = 0 # coef 0 might be false negative, because not enough samples with that variants
plot(results_all$coef[!is.na(results_all$coef)], results_all$diff_pred[!is.na(results_all$coef)], col = col); abline(h = 0)



### compute correlation between ISM and true coef on sample means
dat2plot = data.table(results_all)
dat2plot = dat2plot[, list('diff_pred' = mean(diff_pred), 'coef' = mean(coef)), by = "variant_id"]
ggplot(dat2plot, aes(x = coef, y = diff_pred)) + geom_point() + geom_smooth(method='lm', formula= y~x, se = FALSE)+
    theme_bw()
cor(dat2plot$coef, dat2plot$diff_pred, use = 'complete.obs', method = 'pearson') # 0.3, 0.11
cor(dat2plot$coef, dat2plot$diff_pred, use = 'complete.obs', method = 'kendall')  # 0.2, 0.01

dat2plot = data.table(results_all[results_all$donor != 'ref', ])
dat2plot = dat2plot[, list('diff_pred' = mean(diff_pred), 'coef' = mean(coef)), by = "variant_id"]
ggplot(dat2plot, aes(x = coef, y = diff_pred)) + geom_point() + geom_smooth(method='lm', formula= y~x, se = FALSE)+
    theme_bw()

cor(dat2plot$coef, dat2plot$diff_pred, use = 'complete.obs', method = 'pearson') # 0.3, 0.11
cor(dat2plot$coef, dat2plot$diff_pred, use = 'complete.obs', method = 'kendall')  # 0.2, 0.04

### compute correlation between ISM and true coef on refseq
dat2plot2 = data.table(results_all[results_all$donor == 'ref', ])
dat2plot2 = dat2plot2[, list('diff_pred' = mean(diff_pred), 'coef' = mean(coef)), by = "variant_id"]
cor(dat2plot2$coef, dat2plot2$diff_pred, use = 'complete.obs', method = 'pearson') # -0.3, 0.04
cor(dat2plot2$coef, dat2plot2$diff_pred, use = 'complete.obs', method = 'kendall')  # -0.15, 0.04

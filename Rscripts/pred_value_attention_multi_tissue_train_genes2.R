## load baseline and models, add constrast results
## real data

library(data.table)
library(parallel)
library(ggplot2)

# paths come from the YAML config (CREAM/configs/defaults.yaml)
source("Rscripts/config.R")
cream <- cream_config()


source('Rscripts/utility_functions1.R')

fold = 0

ids = c('s2hdcef3', '2k2p0thu', 'krx1eucp', '5mnn0ol7', '3qoubimu', 'bhp8ez3y') #, 'a4roqha5', model2_atten 
models = c('model2', 'contrast', 'baseline', 'model3', 'model3_abs', 'model2_atten')

config = 'train' # 'test' 'train'

output_dir = 'Rscripts/plots/real_data/'

#### load predictions

pred_base = fread(paste0('results/baseline_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', ids[3], '/Prediction_Results_20_in_test_donors.csv'))

setnames(pred_base, old = 'gene', new = 'gene_name') 
pred_base = pred_base[, list(y_pred = mean(y_pred), y_true = mean(y_true)), by = c('donor', 'gene_name','tissue')]# compute each donor/gene/tissue multiple times

pred = fread(paste0('results/attn2_2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', ids[1], '/Prediction_Results_20_in_test_donors.csv'))

#pred2 = fread(paste0('results/attn3_1_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', ids[5], '/Prediction_Results_20_in_test_donors.csv'))
#pred3 = fread(paste0('results/attn3_1_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', ids[6], '/Prediction_Results_20_in_test_donors.csv'))
pred3 = fread(paste0('results/attn3_2_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', ids[6], '/', config, '_genes//Prediction_Results_-1_in_test_donors.csv'))


#pred = fread(paste0('results/attn2_2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', ids[1], '/', config, '_genes/Prediction_Results_-1_in_test_donors.csv'))


metrics <- read.csv(paste0('results/attn2_2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', ids[1], '/CrossIndivMetrics_test_donors_Epoch20_rank0.csv'))

# need to recompute metrics
metrics_base = pred_base[, list('pearsonr'= cor(y_true, y_pred), 'r2' = 1-sum((y_true - y_pred)^2)/sum(y_true^2)), 
          by = c('tissue', 'gene_name')]


# add prediction results form contrast model
pred_contrast = fread(paste0('results/attn0_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', ids[2], '/Prediction_Results_20_in_test_donors.csv'))



#### convert diff expression to absolute expression, compare with base model ####
donors = unique(pred_base$donor) #42
train_genes = unique(metrics$gene_name[metrics$gene_split == 'train']) # 4529
test_genes = unique(metrics[metrics$gene_split == 'test', 'gene_name']) #1538
tissues = unique(metrics$tissue)

if(config == 'train')
{
   output_genes = train_genes
}else {
   output_genes = test_genes
}

# do not run if preload, takes too long
metrics_all2 = list()
preds_all2 = list()

# attention model
results = compute_metrics(pred, pred_base, donors, output_genes,tissues = tissues, toplot = F) # for baselime, some donors will be missing for each gene, because of batch size

metrics_all2[[1]] = results[[1]]
metrics_all2[[1]]$model = 'model2'
preds_all2[[1]] = results[[2]]
preds_all2[[1]]$model = 'model2'

# add contrast to results
results = compute_metrics(pred_contrast, pred_base, donors, output_genes, tissues = tissues, toplot = F)
metrics_all2[[2]] = results[[1]]
metrics_all2[[2]]$model = 'contrast'
preds_all2[[2]] = results[[2]]
preds_all2[[2]]$model = 'contrast'



metrics_all2[[3]] = metrics_base[gene_name %in% output_genes,c('tissue', 'gene_name', 'pearsonr', 'r2') ]
metrics_all2[[3]]$model = 'baseline'
#r2_base = recompute_r2(pred_base, output_genes) # doesn't change much
#all(metrics_all2[[3]]$gene_name == r2_base$gene_name)
#all(metrics_all2[[3]]$tissue == r2_base$tissue)
#metrics_all2[[3]]$r2 = r2_base$r2

print(summary(metrics_all2[[3]][!is.na(pearsonr) & gene_name %in% output_genes,c('pearsonr', 'r2')]))

preds_all2[[3]] = pred_base[gene_name %in% output_genes, c("y_pred", "y_true", "donor", "gene_name", "tissue" )]
preds_all2[[3]]$model = 'baseline'

# # adding model3 and model3_abs
# results = compute_metrics(pred2, pred_base, donors, output_genes,tissues = tissues, toplot = F) # for baselime, some donors will be missing for each gene, because of batch size
# metrics_all2[[4]] = results[[1]]
# metrics_all2[[4]]$model = 'model3'
# preds_all2[[4]] = results[[2]]
# preds_all2[[4]]$model = 'model3'

results = compute_metrics(pred3, pred_base, donors, output_genes,tissues = tissues, toplot = F) # for baselime, some donors will be missing for each gene, because of batch size
metrics_all2[[4]] = results[[1]]
metrics_all2[[4]]$model = 'model2_atten'
preds_all2[[4]] = results[[2]]
preds_all2[[4]]$model = 'model2_atten'

saveRDS(preds_all2, file = paste0('Rscripts/results/real_data/prediction_abs_expr_test_donor_', config, '_genes.rds'))
saveRDS(metrics_all2, file = paste0('Rscripts/results/real_data/metrics_abs_expr_test_donor_epoch20_', config, '_genes.rds'))

metrics_all2 = readRDS(paste0('Rscripts/results/real_data/metrics_abs_expr_test_donor_epoch20_', config, '_genes.rds'))
preds_all2 = readRDS(paste0('Rscripts/results/real_data/prediction_abs_expr_test_donor_', config, '_genes.rds'))


##### plotting
metrics_all2 = do.call('rbind', metrics_all2)
ind = which(metrics_all2$r2 != -Inf & metrics_all2$r2!=1) # remove genes that do not exist in particular tissue


pdf(paste0('Rscripts/plots/real_data/pearsonr_boxplot_3_tissues_', config,'_genes_abs_expr2_real.pdf'))
if(config == 'train')
{
    ggplot(metrics_all2, aes(x = tissue, y = pearsonr, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(-0.5,0.8)) + theme(text = element_text(size = 16))
}else{
    ggplot(metrics_all2, aes(x = model, y = pearsonr, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(-0.8,0.8)) + theme(text = element_text(size = 16)) #outlier.shape = NA
}
dev.off()

pdf(paste0('Rscripts/plots/real_data/r2_boxplot_3_tissues_', config,'_genes_abs_expr2_real.pdf'))
if(config == 'train')
{ 
    ggplot(metrics_all2[r2!= -Inf & r2!=1, ], aes(x = tissue, y = r2, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(-0.3,0.3)) + theme(text = element_text(size = 16)) #
}else{
   ggplot(metrics_all2[r2!= -Inf, ], aes(x = tissue, y = r2, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
      coord_cartesian(ylim = c(-0.3,0.3)) + theme(text = element_text(size = 16)) #
}
dev.off()


##### separate test donors with vs without de novo (novel) SNVs #####
# A novel SNV = a SNP in the gene's 49,152 bp model-input (TSS-centered) window that is
# carried (het or hom) by >=1 held-out valid donor but 0 training donors of this CV fold.
# Tables are precomputed from the GTEx WGS VCF by Rscripts/novel_variants.py (Fold-0).
# NOTE: 'valid' donors here are the held-out evaluation donors (what we call test donors).
novel_pair    = fread('Rscripts/results/simulated_data/novel_variants_per_donor_gene_test_fold0.csv') # gene_name, donor, n_novel
novel_gene    = fread('Rscripts/results/simulated_data/novel_variants_per_gene_test_fold0.csv')
novel_variant = fread('Rscripts/results/simulated_data/novel_variants_unique_test_fold0.csv')         # variant_id, gene_name, n_val_carriers

# how many novel variants in test donors?
donor_novel_tot = novel_pair[, .(n_novel = sum(n_novel)), by = donor]
message('=== de novo (novel) SNVs in held-out test donors, Fold-', fold, ' ===')
message('  test genes covered: ', uniqueN(novel_gene$gene_name))
message('  unique novel SNVs across test-gene windows: ', uniqueN(novel_variant$variant_id))
message('  novel SNVs summed over gene windows: ', sum(novel_gene$n_novel_variants))
message('  test genes with >=1 novel SNV: ', novel_gene[n_novel_variants > 0, .N], '/', uniqueN(novel_gene$gene_name))
message('  test donors carrying >=1 novel SNV (any gene): ', donor_novel_tot[n_novel > 0, .N], '/', uniqueN(novel_pair$donor))
message('  novel SNVs per donor: median=', median(donor_novel_tot$n_novel),
        ' mean=', round(mean(donor_novel_tot$n_novel), 1),
        ' range=', min(donor_novel_tot$n_novel), '-', max(donor_novel_tot$n_novel))
fwrite(donor_novel_tot[order(-n_novel)],
       paste0('Rscripts/results/simulated_data/novel_snv_count_per_test_donor_', config, '.csv'))

# flag each (gene, donor) prediction by whether that donor carries a novel SNV in the gene window,
# then recompute per gene/tissue/model pearsonr & r2 WITHIN each donor group
preds_dt = rbindlist(preds_all2, use.names = TRUE, fill = TRUE)
preds_dt[novel_pair, on = .(gene_name, donor), n_novel := i.n_novel]
preds_dt[is.na(n_novel), n_novel := 0L] # donors/genes with no window overlap -> no novel SNV
preds_dt[, novel_group := factor(fifelse(n_novel > 0, 'has novel SNV', 'no novel SNV'),
                                 levels = c('no novel SNV', 'has novel SNV'))]

min_donors = 5 # need enough donors in a group for a stable per-gene correlation
# r2 centers y_pred within each group first (as recompute_r2 does for baseline), so the
# absolute-expression offset does not dominate; pearsonr is shift/scale invariant already.
metrics_novel = preds_dt[!is.na(y_true) & !is.na(y_pred),
    .(pearsonr = if (.N >= min_donors) cor(y_true, y_pred) else NA_real_,
      r2       = if (.N >= min_donors) { yp = y_pred - mean(y_pred); 1 - sum((y_true - yp)^2) / sum(y_true^2) } else NA_real_,
      n_donors = .N),
    by = .(gene_name, tissue, model, novel_group)]
metrics_novel[!is.finite(r2), r2 := NA_real_] # genes with all-zero y_true in a group give -Inf
fwrite(metrics_novel, paste0('Rscripts/results/real_data/metrics_by_novelSNV_', config, '.csv'))

pdf(paste0('Rscripts/plots/real_data/pearsonr_boxplot_3_tissues_', config, '_genes_abs_expr4_by_novelSNV.pdf'), width = 11)
print(
    ggplot(metrics_novel[!is.na(pearsonr)], aes(x = model, y = pearsonr, fill = novel_group)) +
        geom_boxplot(outlier.shape = NA)  + theme_bw() + #+ facet_wrap(~tissue)
        coord_cartesian(ylim = if (config == 'train') c(-1, 1) else c(-1, 1)) +
        theme(text = element_text(size = 18), axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(fill = 'test donors', title = 'Per-gene pearsonr split by de novo SNV in test donors')
)
dev.off()

pdf(paste0('Rscripts/plots/real_data/r2_boxplot_3_tissues_', config, '_genes_abs_expr4_by_novelSNV.pdf'), width = 11)
print(
    ggplot(metrics_novel[!is.na(r2) & r2 != -Inf], aes(x = model, y = r2, fill = novel_group)) +
        geom_boxplot(outlier.shape = NA) + theme_bw() + # + facet_wrap(~tissue)
        coord_cartesian(ylim = if (config == 'train') c(-0.3, 0.3) else c(-0.3, 0.3)) +
        theme(text = element_text(size = 18), axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(fill = 'test donors', title = 'Per-gene r2 split by de novo SNV in test donors')
)
dev.off()

# paired comparison (per gene/tissue) of has-novel vs no-novel, within each model
novel_wide = dcast(metrics_novel, gene_name + tissue + model ~ novel_group, value.var = c('pearsonr', 'r2'))
novel_test = novel_wide[, {
    p_r  = tryCatch(wilcox.test(`pearsonr_no novel SNV`, `pearsonr_has novel SNV`, paired = TRUE)$p.value, error = function(e) NA_real_)
    p_r2 = tryCatch(wilcox.test(`r2_no novel SNV`, `r2_has novel SNV`, paired = TRUE)$p.value, error = function(e) NA_real_)
    .(n_genes           = sum(!is.na(`pearsonr_no novel SNV`) & !is.na(`pearsonr_has novel SNV`)),
      med_pearsonr_none = median(`pearsonr_no novel SNV`, na.rm = TRUE),
      med_pearsonr_novel= median(`pearsonr_has novel SNV`, na.rm = TRUE),
      wilcox_p_pearsonr = p_r,
      med_r2_none       = median(`r2_no novel SNV`, na.rm = TRUE),
      med_r2_novel      = median(`r2_has novel SNV`, na.rm = TRUE),
      wilcox_p_r2       = p_r2)
}, by = model]
print(novel_test)
fwrite(novel_test, paste0('Rscripts/results/real_data/metrics_by_novelSNV_wilcox_', config, '.csv'))

# wilcoxon text
wilcox.test(unlist(metrics_all2[metrics_all2$model == 'model3_abs', 'pearsonr']), unlist(metrics_all2[metrics_all2$model == 'baseline', 'pearsonr']))
wilcox.test(unlist(metrics_all2[metrics_all2$model == 'contrast', 'r2']), unlist(metrics_all2[metrics_all2$model == 'model3_abs', 'r2']))

# scatterplot
metrics_all2 = metrics_all2[ind, ]
pdf('Rscripts/plots/real_data/scatterplot_metrics_test_genes_model23_2.pdf')
scatterplot_metrics(metrics_all2[metrics_all2$model == 'model2',], metrics_all2[metrics_all2$model == 'model2_atten',],
    xlab = 'model2', ylab = 'model2_atten')
dev.off()
# compare abs gene expression between baseline and new model
#scatterplot_metrics(metrics_all2[model == 'model1'], metrics_all2[model == 'baseline'])
preds_all2 = do.call('rbind', preds_all2)

# select genes
#head(metrics_all2[order(gene_name), ],12)
gn = 'ENSG00000011007' #test_genes[456]
metrics_all2[gene_name == gn, ]

pdf('Rscripts/plots/simulated_data/scatterplot_3_tissues_train_genes_abs_expr3.pdf')
ggplot(preds_all2[gene_name == gn], aes(x = y_true, y = y_pred, color = model)) + geom_point() + 
    theme_bw() + theme(text = element_text(size = 16), legend.position = "none") + facet_grid(model ~ tissue, scales = 'free') +
    ggtitle(gn)
dev.off()

# cross tissue gene expression correlation
preds_all2 = readRDS(paste0('Rscripts/results/real_data/prediction_abs_expr_test_donor_', config, '_genes.rds'))
preds_wide = lapply(preds_all2[1:3], function(pred2){
    pred_long <- melt(
                        pred2,
                        id.vars = c('donor', 'gene_name', 'tissue'),
                        measure.vars = c('y_pred', 'y_true'),
                        variable.name = 'value_type',
                        value.name = 'value'
                    )

    pred_wide = dcast(
        pred_long,
        donor + gene_name ~ tissue + value_type,
        value.var = 'value'
    )

    pred_wide
})

cor_by_gene <- lapply(preds_wide, function(pred_wide) {
    value_cols <- setdiff(names(pred_wide), c('donor', 'gene_name'))
    value_pairs <- combn(value_cols, 2, simplify = FALSE)
    pair_names <- vapply(value_pairs, function(pair) paste(pair, collapse = '__'), character(1))

    pred_wide[, {
        cors <- vapply(value_pairs, function(pair) {
            cor(.SD[[pair[1]]], .SD[[pair[2]]])
        }, numeric(1))

        as.list(setNames(cors, pair_names))
    }, by = gene_name, .SDcols = value_cols]
})

boxplot(cor_by_gene[[1]][,-1])

# only keep genes with at least one tissue-specific eQtLs
tissue_specific_genes = intersect(tissue_specific_genes, output_genes) # 1703

heatmap_cor_data <- rbindlist(Map(function(cor_dt, pred2, pred_wide) {
    value_cols <- setdiff(names(pred_wide), c('donor', 'gene_name'))
    pred_cols <- grep('y_pred', value_cols, value = TRUE)
    true_cols <- grep('y_true', value_cols, value = TRUE)
    pair_cols <- setdiff(names(cor_dt), 'gene_name')
    model_name <- unique(pred2$model)[1]

    mean_cor <- cor_dt[gene_name %in% tissue_specific_genes, lapply(.SD, mean, na.rm = TRUE), .SDcols = pair_cols]
    pair_dt <- data.table(
        pair = pair_cols,
        mean_cor = as.numeric(mean_cor[1])
    )
    pair_dt[, c('var1', 'var2') := tstrsplit(pair, '__', fixed = TRUE)]

    # keep only cross-type pairs (one y_pred, one y_true)
    cross_pairs <- pair_dt[
        (var1 %in% pred_cols & var2 %in% true_cols) |
        (var1 %in% true_cols & var2 %in% pred_cols)
    ]
    # rows = pred, columns = true
    cross_pairs[, pred_var := ifelse(var1 %in% pred_cols, var1, var2)]
    cross_pairs[, true_var := ifelse(var1 %in% true_cols, var1, var2)]

    heatmap_dt <- cross_pairs[, .(
        var1 = factor(true_var, levels = true_cols),        # x-axis = true
        var2 = factor(pred_var, levels = rev(pred_cols)),   # y-axis = pred
        mean_cor
    )]
    heatmap_dt[, model := model_name]
    heatmap_dt
}, cor_by_gene, preds_all2[1:3], preds_wide), use.names = TRUE)

# heatmap: y_true cross-tissue correlations (same for all models since y_true is shared)
heatmap_true_data <- {
    pred_wide <- preds_wide[[1]]
    value_cols <- setdiff(names(pred_wide), c('donor', 'gene_name'))
    true_cols <- grep('y_true', value_cols, value = TRUE)

    cor_true <- pred_wide[, {
        mat <- cor(.SD, use = 'pairwise.complete.obs')
        pairs <- as.data.table(as.table(mat), responseName = 'mean_cor')
        setnames(pairs, c('var1', 'var2', 'mean_cor'))
        pairs
    }, .SDcols = true_cols, by = gene_name]

    cor_true_mean <- cor_true[gene_name %in% tissue_specific_genes, .(mean_cor = mean(mean_cor, na.rm = TRUE)), by = .(var1, var2)]
    cor_true_mean[, var1 := factor(var1, levels = true_cols)]
    cor_true_mean[, var2 := factor(var2, levels = rev(true_cols))]
    cor_true_mean
}

pdf(paste0('Rscripts/plots/real_data/mean_correlation_heatmap_ytrue_', config, '_tissue_specific_genes.pdf'), width = 6, height = 5)
ggplot(heatmap_true_data, aes(x = var1, y = var2, fill = mean_cor)) +
    geom_tile() +
    geom_text(aes(label = round(mean_cor, 2)), size = 3) +
    scale_fill_gradient2(low = 'steelblue', mid = 'white', high = 'firebrick', midpoint = 0, limits = c(-1, 1)) +
    coord_fixed() +
    theme_bw() +
    theme(
        text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank()
    ) +
    xlab(NULL) + ylab(NULL) + ggtitle('Mean cross-tissue y_true correlation')
dev.off()

pdf(paste0('Rscripts/plots/real_data/mean_correlation_heatmap_', config, '_tissue_specific_genes.pdf'), width = 12, height = 8)
ggplot(heatmap_cor_data, aes(x = var1, y = var2, fill = mean_cor)) +
    geom_tile() +
    facet_wrap(~model) +
    scale_fill_gradient2(low = 'steelblue', mid = 'white', high = 'firebrick', midpoint = 0, limits = c(-0.1, 0.25)) +
    coord_fixed() +
    theme_bw() +
    theme(
        text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank()
    ) +
    xlab(NULL) +
    ylab(NULL)
dev.off()



# obtain ground truth eQTLs for each tissue
tissue_id = read.csv(cream$eqtl_tissue_label_file)
tissue = c('Whole_Blood','Muscle_Skeletal', 'Adipose_Subcutaneous')
ids = tissue_id[match(tissue, tissue_id$data_tissue), 'path']
tissue_name = tissue_id[match(tissue, tissue_id$data_tissue), 'tabix_tissue']
tissue_name[3] = 'adipose'
egenes_all = list()
for(i in 1:length(ids))
{
    id = ids[i]
    ts = tissue[i]
    egenes = fread(file.path(cream$eqtl_dir, id))
    egenes$beta2 = egenes$beta * egenes$pip
    egenes$pos = as.numeric(sapply(egenes$variant, function(x) strsplit(x, '_')[[1]][2]))
    egenes$pos0 = egenes$pos - 1
    egenes$tissue = ts
    egenes_all[[ts]] = egenes
}





# identify tissue-specific eQTL genes (variant-level: significant variants not seen in other tissues for same gene)
pip_thr <- 0.5

eqtl_sig <- rbindlist(lapply(names(egenes_all), function(ts) {
    unique(egenes_all[[ts]][pip > pip_thr, .(gene = gene_id, variant, tissue = ts)])
}), use.names = TRUE, fill = TRUE)

# variant is tissue-specific for a gene if it appears in only one tissue for that gene
eqtl_sig[, n_tissues_per_variant := uniqueN(tissue), by = .(gene, variant)]
eqtl_sig[, is_tissue_specific_variant := n_tissues_per_variant == 1L]


gene_tissue_specific <- unique(eqtl_sig[is_tissue_specific_variant == TRUE, .(gene, tissue)])
#a = gene_tissue_specific[,.N, by = gene][N == 3]
#length(intersect(a$gene, output_genes))
# # optional stricter gene-level definition: all significant variants of the gene are unique to one tissue
# gene_strict_specific <- eqtl_sig[, .(
#     n_sig_variants = uniqueN(variant),
#     n_shared_variants = uniqueN(variant[is_tissue_specific_variant == FALSE]),
#     n_tissues_with_sig_eqtl = uniqueN(tissue)
# ), by = gene]
# gene_strict_specific[, strict_tissue_specific := (n_shared_variants == 0L) & (n_tissues_with_sig_eqtl == 1L)]

tissue_specific_genes <- sort(unique(gene_tissue_specific$gene))
#strict_tissue_specific_genes <- sort(unique(gene_strict_specific[strict_tissue_specific == TRUE, gene]))

message('pip threshold = ', pip_thr)
message('genes with >=1 tissue-specific eQTL variant: ', length(tissue_specific_genes))
#message('strict tissue-specific genes (sig eQTLs in one tissue only): ', length(strict_tissue_specific_genes))

atten_list = list()

id =  's2hdcef3'
atten_list[[1]] = fread(paste0('results/attn2_2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/', config, '_genes/Prediction_Results_-1_in_test_donors.csv'))

# id =  '5mnn0ol7'
# atten = fread(paste0('results/attn3_1_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/', config, '_genes/Prediction_Results_-1_in_test_donors.csv'))

id =  'bhp8ez3y' #'a4roqha5' #'3qoubimu'
atten_list[[2]] = fread(paste0('results/attn3_2_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/', config, '_genes/Prediction_Results_-1_in_test_donors.csv'))

# read intervals
intervals = read.csv(cream$genomic_intervals_file)
intervals_dt = as.data.table(intervals)
intervals_dt[, gene_id := sub('\\..*$', '', gene_id)]
setkey(intervals_dt, gene_id)

# merge intervals with egenes_all and keep only eQTLs within interval bounds
egenes_in_interval_all = lapply(egenes_all, function(eg) {
    eg = copy(eg)
    eg[, chr := tstrsplit(variant, '_', fixed = TRUE, keep = 1L)]
    eg[, chr := fifelse(startsWith(chr, 'chr'), chr, paste0('chr', chr))]

    merged = merge(
        eg,
        intervals_dt[, .(gene_id, seqnames, starts, ends)],
        by = 'gene_id',
        all.x = FALSE,
        all.y = FALSE
    )

    merged = merged[chr == seqnames & pos0 >= starts & pos0 <= ends]
    merged[, ind := pos0 - starts]
    merged
})

#number of eQTLs per tissue, per genes, total number of eQTLs, distribution of PIP, effect size

eqtl_plot_source = if (exists('egenes_in_interval_all')) egenes_in_interval_all else egenes_all

eqtl_all_dt = rbindlist(lapply(names(eqtl_plot_source), function(ts) {
    dt = copy(eqtl_plot_source[[ts]])
    if(!('tissue' %in% names(dt))) dt[, tissue := ts]
    dt[, .(tissue, gene_id, pip, beta)]
}), use.names = TRUE, fill = TRUE)

pdf(paste0('Rscripts/plots/eqtl_pip_distribution_hist_per_tissue.pdf'), width = 11, height = 14)
ggplot(eqtl_all_dt[!is.na(pip)], aes(x = pip)) +
    geom_histogram(bins = 70, fill = 'steelblue', color = 'white') +
    facet_wrap(tissue~., scales = 'free_y', nrow = 3) +
    theme_bw() +
    theme(text = element_text(size = 18)) +
    xlab('PIP') +
    ylab('Number of eQTLs') +
    ggtitle('PIP distribution per tissue')
dev.off()

#beta_lim = as.numeric(quantile(abs(eqtl_all_dt$beta), 0.995, na.rm = TRUE))
pdf(paste0('Rscripts/plots/eqtl_beta_distribution_hist_per_tissue.pdf'), width = 11, height = 14)
ggplot(eqtl_all_dt[!is.na(beta) ], aes(x = beta)) + #& abs(beta) <= beta_lim
    geom_histogram(bins = 80, fill = 'firebrick', color = 'white') +
    facet_wrap(~tissue, scales = 'free_y', nrow = 3) +
    theme_bw() +
    theme(text = element_text(size = 18)) +
    xlab('beta') +
    ylab('Number of eQTLs') +
    ggtitle('beta distribution per tissue') # (trimmed at 99.5% abs(beta))
dev.off()

eqtl_per_gene = eqtl_all_dt[, .(n_eqtls = .N), by = .(tissue, gene_id)]
eqtl_per_gene_median = eqtl_per_gene[, .(median_n_eqtls = median(n_eqtls, na.rm = TRUE)), by = tissue]
pdf(paste0('Rscripts/plots/eqtl_count_per_gene_hist_per_tissue.pdf'), width = 11, height = 14)
ggplot(eqtl_per_gene, aes(x = n_eqtls)) +
    geom_histogram(binwidth = 1, fill = '#1B4332', color = 'white', linewidth = 0.2) +
    geom_vline(data = eqtl_per_gene_median, aes(xintercept = median_n_eqtls), linetype = 'dashed', color = 'firebrick', linewidth = 0.8, inherit.aes = FALSE) +
    geom_text(data = eqtl_per_gene_median, aes(x = median_n_eqtls, y = Inf, label = paste0('median=', median_n_eqtls)), color = 'firebrick', angle = 90, vjust = 1.2, hjust = 1.1, size = 5, inherit.aes = FALSE) +
    facet_wrap(~tissue, scales = 'free_y', nrow = 3) +
    theme_bw() +
    theme(text = element_text(size = 18)) +
    xlab('Number of eQTLs per gene') +
    ylab('Number of genes') +
    ggtitle('Histogram of eQTL count per gene per tissue')
dev.off()

eqtl_per_gene[, .N, by = tissue]
#                  tissue    N
# 1:          Whole_Blood 3743
# 2:      Muscle_Skeletal 3942
# 3: Adipose_Subcutaneous 4728


# select egenes with at least one SNP pip > 0.7
sig_genes = unique(do.call('c', sapply(egenes_all, function(x) unlist(x[which(x$pip > 0.7), 'gene_id'])))) # 3062
train_genes = unique(atten_list[[1]]$gene)
train_genes = intersect(train_genes, sig_genes) # 471, 530
print(length(train_genes))

dat2plot_all = list()
model_names = c('model2', 'model2_atten')
for(i in 1:length(atten_list))
{
    atten = atten_list[[i]]
    dat2plot <- mclapply(train_genes, function(gn) #mc
    {
        results = lapply(tissue, function(ts)
        {
            atten_temp = atten[tissue == ts]
            # select the cases where gene expression is different
            atten_temp = atten_temp[y_true!=0] # get null
            egenes = egenes_all[[ts]]
            donor_pairs = which(atten_temp$gene == gn)
            if(length(donor_pairs) == 0)
            {
                return(NULL)
            }
            eqtls = egenes[gene_id == gn]
            start = intervals[grep(gn, intervals$gene_id), 'starts']
            eqtls$ind = eqtls$pos0 - start
            dat2plot = lapply(donor_pairs, function(i){
                inds = as.numeric(strsplit(atten_temp$attn_inds[i], ',')[[1]])
                weights = as.numeric(strsplit(atten_temp$attn_weights[i], ',')[[1]])
                data.table('gene' = gn, 'inds' = inds, 'atten_weights' = weights, 'eQTL_size' = unlist(eqtls[match(inds, ind), 'beta2']), 'eQTL_pip' = unlist(eqtls[match(inds, ind), 'pip']))
            })
            dat2plot = do.call('rbind', dat2plot)
            dat2plot = dat2plot[dat2plot$atten_weights > 0, ]

            # get average attention
            ##dat2plot = dat2plot[, .(atten_weights = mean(atten_weights), eQTL_size = mean(eQTL_size), eQTL_pip = max(eQTL_pip)), by = .(inds = inds, gene)]
            dat2plot$tissue = ts
        return(dat2plot) # remove loci not different between individual pairs
        })
        do.call(rbind, results)
    } , mc.cores = 8)

    dat2plot <- do.call(rbind, dat2plot)
    dat2plot$model = model_names[i]
    dat2plot_all[[i]] = dat2plot
}
dat2plot_all = do.call(rbind, dat2plot_all)
#save(dat2plot_all, file = paste0('Rscripts/results/real_data/atten3_abs_', config, '_genes.rdat'))

length(unique(dat2plot_all[eQTL_pip > 0.7,][['gene']])) # number of positive samples: 383, 439

#boxplot(atten_weights~is.na(eQTL_size), dat2plot, outline = F)


dat2plot_all$eqtl = dat2plot_all$eQTL_pip > 0.7
dat2plot_all$eqtl[is.na(dat2plot_all$eqtl)] = FALSE
boxplot(atten_weights~eqtl, dat2plot, outline = F)

# separate tissue
# subsample negative cases to reduce plooting time
#sum(dat2plot_all[tissue == 'adipose' & model == 'model1', 'eqtl'])
dat2plot_all[, .N, by = .(eqtl,tissue,model) ]
dat2plot2 <- dat2plot_all[, .SD[sample(.N, min(.N, 34602 * 5))], by = .(eqtl,tissue,model) ] #sum(dat2plot_all$eqtl) *5
#dat2plot2 <- dat2plot_all[, .SD[sample(.N, min(.N, 44488 * 5))], by = .(eqtl,tissue) ] #sum(dat2plot_all$eqtl) *5
library(ggh4x)
pdf('Rscripts/plots/real_data/atten_weights_boxplot_3_tissues_train_genes_model2_2.pdf', width = 14) # ***1 select cases where gene expression shows difference
ggplot(dat2plot2, aes(x = eqtl, y = atten_weights)) + geom_boxplot(outlier.shape = NA) + facet_grid(model~tissue, scales = 'free_y') + theme_bw() + 
     xlab('eQTL pip > 0.7, 383 positive genes') +  #+ #0.06 
     facetted_pos_scales(
        y = list(
        model == "model2" ~ scale_y_continuous(limits = c(0, 0.06)),
        model == "model2_atten" ~ scale_y_continuous(limits = c(0, 0.7))
        )
        ) + theme(text = element_text(size = 16))
     #coord_cartesian(ylim = c(0,0.5)) 
dev.off()

pdf('Rscripts/plots/real_data/atten_weights_boxplot_3_tissues_test_genes_model23_2.pdf', width = 14) # ***1 select cases where gene expression shows difference
ggplot(dat2plot2, aes(x = eqtl, y = atten_weights)) + geom_boxplot(outlier.shape = NA) + facet_grid(model~tissue) + theme_bw() + 
    coord_cartesian(ylim = c(0,0.06)) + xlab('eQTL pip > 0.7, 439 positive genes') + theme(text = element_text(size = 16))
dev.off()

dat2plot[is.na(dat2plot[,'beta2']),'beta2'] = 0
#plot(abs(dat2plot$beta2), dat2plot$atten_weights)
boxplot(atten_weights~ round(abs(beta2),2), dat2plot, outline=F)

breaks = c(seq(0, 0.07, by = 0.01), 1)
dat2plot$beta2_cut = cut(abs(dat2plot$beta2), breaks = breaks, right = FALSE, include.lowest = TRUE)
boxplot(atten_weights~ beta2_cut, dat2plot, outline=F)
#boxplot(abs(beta2)~round(atten_weights,1), dat2plot, outline=F)
# boxplot(atten_weights~inds, dat2plot[dat2plot$inds %in% eqtls$ind, ])
# xtabs(~dat2plot[dat2plot$inds %in% eqtls$ind, 'inds'])
# print(metrics[metrics$gene_name == gn, ])



# select positive cases that eQTLs are shared by multi-tissue or unique
genes = unique(unlist(dat2plot_all[dat2plot_all$eQTL_pip > 0.9, 'gene'])) #314
#good_genes = metrics[which(metrics$pearsonr > 0.95), 'gene_name'] # select good genes


metrics <- lapply(c(1,6), function(i) {
    id <- ids[i]
    if(i == 6)
    {
        filepath <- paste0(
            "results/attn3_2",
            "_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/",
            "Fold-", fold, "/", id, "/", config, "_genes/"
        )
    }else if(i == 5)
    {
         filepath <- paste0(
            "results/attn3_1",
            "_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/",
            "Fold-", fold, "/", id, "/", config, "_genes/"
        )
    }else
    {   
        filepath <- paste0(
            "results/attn2_2",
            "_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/",
            "Fold-", fold, "/", id, "/", config, "_genes/"
        )
    }
    files <- 'CrossIndivMetrics_test_donors_Epoch-1_rank0.csv'
    mm <- read.csv(file.path(filepath, files))
    mm$model <- models[i]
    return(mm)
})

metrics_all = do.call('rbind', metrics)
metrics_all = as.data.table(metrics_all)
##dat2plot_all[model == 'contrast', 'model'] = 'model3_2_abs'
dat2plot_all[metrics_all[, .(gene_name, tissue, pearsonr, model)], 
         on = .(gene = gene_name, tissue = tissue, model = model),
         pearsonr := i.pearsonr]
genes = unlist(unique(dat2plot_all[atten_weights > 0.05 & pearsonr > 0.5 & eQTL_pip > 0.7, 'gene'])) #185  good_genes, 0.9, ##0.1, 0.7
metrics_wide = merge(metrics[[1]], metrics[[2]], by = c('gene_name', 'tissue'))
genes1 = intersect(genes, metrics_wide[metrics_wide$pearsonr.x >metrics_wide$pearsonr.y & metrics_wide$pearsonr.x > 0.5, 'gene_name'])
genes2 = intersect(genes, metrics_wide[metrics_wide$pearsonr.x <metrics_wide$pearsonr.y & metrics_wide$pearsonr.y > 0.5, 'gene_name'])

#genes = intersect(good_genes, genes)
i = 12#2,5


##setnafill(dat2plot, fill = 0, cols = c("eQTL_pip"))
gn = genes2[i] #'ENSG00000204099' # ENSG00000182378 ENSG00000136856 ENSG00000170954
metrics_all[metrics_all$gene_name == gn, ]
temp = dat2plot_all[gene == gn]

temp2 = temp[, .(atten_weights = mean(atten_weights), eQTL_size = mean(eQTL_size), eQTL_pip = max(eQTL_pip)), by = .(inds = inds, tissue, model)]


# manhattan plot
pdf(paste0('Rscripts/plots/real_data/manhanttan_plot_atten_3_tissues_',config,'_', gn, '_avg.pdf'), width = 14)
ggplot(temp2) + geom_point(size = 1, aes(x = inds, y = atten_weights, color = eQTL_size )) + facet_grid(tissue~model) + 
 geom_text(data = metrics_all[metrics_all$gene_name == gn, ], aes(x = -Inf, y = Inf, label = round(pearsonr,2)), hjust = -0.1, vjust = 1.1) + theme_bw()  +
  geom_vline(aes(xintercept = inds, color = eQTL_pip), data = temp[!is.na(eQTL_pip) & eQTL_pip > 0.7]) + ggtitle(gn) 
dev.off()


# get average attention per position for pairs without gene expression difference
dat2plot_all = lapply(2, function(i) #1:3
{
    atten_temp = atten[[i]]
    # select the cases where gene expression is different
    atten_temp = atten_temp[y_true==0] # get null

    inds_list = strsplit(atten_temp$attn_inds, ',', fixed = TRUE)
    weights_list = strsplit(atten_temp$attn_weights, ',', fixed = TRUE)
    n_per_row = lengths(inds_list)

    inds_vec = as.numeric(unlist(inds_list))
    weights_vec = as.numeric(unlist(weights_list))

    dat2plot = data.table(
        gene = rep(atten_temp$gene, n_per_row),
        tissue = rep(atten_temp$tissue, n_per_row),
        inds = inds_vec,
        atten_weights = weights_vec
    )
    dat2plot = dat2plot[atten_weights > 0]
    dat2plot$model = paste0('model', i)
    return(dat2plot)
})

dat2plot_all = rbindlist(dat2plot_all, fill = TRUE)

temp2 = dat2plot[, .(mean_atten_weights = mean(atten_weights)), by = .(inds_bin = inds %/% 50, tissue, model)]

pdf('Rscripts/plots/simulated_data/null_atten_model1.pdf', width = 14)
ggplot(temp2) + geom_point(size = 1, aes(x = inds_bin, y = mean_atten_weights)) + facet_grid(tissue~model)
dev.off()
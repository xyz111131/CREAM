## load baseline and models, add constrast results

library(data.table)
library(parallel)
library(ggplot2)


source('Rscripts/utility_functions1.R')

fold = 0
seed = 2 # 1

if(fold == 0)
{
    baseline = c('pnhnqwbc', 'b217h7ok', 'u4r9795d', 'v03gybbm')
    #ids = c('kabm6xx9', 'icb2whi4', '8puj4r29')
    ids = list()
    if(seed !=1)
    {
        ids[[1]] = c('kabm6xx9', '7ha00zo8', 'tfcy569t', 'ql7nazfo')
        ids[[2]] = c('icb2whi4', 'als1zcy4', 'qxgv61a4', '0djohyzk')
        ids[[3]] = c('8puj4r29', 'bgrf8cay', 'j4ifqr3x', 'jfvgesmj')
        ids[[4]] = c('j0garsir', 'sj4cmono', 'btkm3alj', '6s3s5s98') # model3_1_abs
        ids[[5]] = c('l418n2z7', 'j81o4cik', 'aatpygpa', '2i541ch0') # model3_1
        ids[[6]] = c('3zzlfa0b', 'uur8la8g', 'wpmrhle3', 'ydnwlbtb') # model3_2_abs
    }else{
        ids[[1]] = c('kcjbxdco', 'i1lvfjut', 'geet4y9k', 'csn3bgv7')
        ids[[2]] = c('8uh15d1j', 'azgyvgn9', 'b6n8bwvh', 'yv86rk80')
        ids[[3]] = c('8sstgcc0', 'it1lrjvn', 'nfob6o3p', 'rak0k936')
    }
}else
{
    
    baseline = c('72lzslb2', 'e512dw7d', 's8izz21u', 'wrhgpvag')
    #ids = c('kabm6xx9', 'icb2whi4', '8puj4r29')
    ids = list()
    ids[[1]] = c('c1v5bhr0', 'suogw2ob', 'vbml84ls', 'wjpyvvjm')
    ids[[2]] = c('d0sj2293', 'e5l18gs9', 'ey3hcgow', 'janqphuf')
    ids[[3]] = c('79us58oo', '8fbsnqnr', 'eiqww84q', 'kz44q6ks')

}
constr_att = c('n3c6i3ia', 'fnw3vml2', 'oij5gq16', 'c3zypcbg')



config = 'test' # 'test' 'train'

output_dir = 'Rscripts/plots/simulated_data/'

#### load base and models predictions

pred_base = lapply(baseline, function(bs)
{
    fread(paste0('results/baseline_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', bs, '/Prediction_Results_19_in_valid_donors.csv'))
})
pred_base = do.call(rbind, pred_base)
setnames(pred_base, old = 'gene', new = 'gene_name') 
pred_base = pred_base[, list(y_pred = mean(y_pred), y_true = mean(y_true)), by = c('donor', 'gene_name','tissue')]# compute each donor/gene/tissue multiple times 1493004 

# do no run
pred = lapply(1:3, function(i){
    id = ids[[i]]
    do.call('rbind', lapply(id, function(j)
    {
        fread(paste0('results/attn2_', i, '_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', j, '/Prediction_Results_19_in_valid_donors.csv'))
    }))
})

# add model 3
# pred_model3_abs = do.call('rbind', lapply(ids[[4]], function(j)
#     {
#         fread(paste0('results/attn3_1_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', j, '/Prediction_Results_18_in_valid_donors.csv'))
#     }))

pred_model3_abs = do.call('rbind', lapply(ids[[6]], function(j)
    {
        fread(paste0('results/attn3_2_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', j, '/Prediction_Results_17_in_valid_donors.csv'))
    }))

pred_model3 = do.call('rbind', lapply(ids[[5]], function(j)
    {
        fread(paste0('results/attn3_1_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-', fold, '/', j, '/Prediction_Results_19_in_valid_donors.csv'))
    }))



metrics <- lapply(1:3, function(i) {
    id <- ids[[i]]
    do.call("rbind", lapply(id, function(j) {
        filepath <- paste0(
            "results/attn2_", i,
            "_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/",
            "Fold-", fold, "/", j, "/"
        )
        files <- list.files(filepath, pattern = "^CrossIndivMetrics_valid_donors_Epoch19_rank.*\\.csv$")
        mm <- read.csv(file.path(filepath, files[1]))
        mm$model <- paste0("model", i)
        return(mm)
    }))
})

#metrics_base = fread(paste0('results/baseline_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', baseline, '/CrossIndivMetrics_valid_donors_Epoch19_rank0.csv'))
# need to recompute metrics
metrics_base = pred_base[, list('pearsonr'= cor(y_true, y_pred), 'r2' = 1-sum((y_true - y_pred)^2)/sum(y_true^2)), 
          by = c('tissue', 'gene_name')]


# load previously saved results
metrics_all2 = readRDS(paste0('Rscripts/results/simulated_data/metrics_abs_expr_valid_donor_epoch19_', config, '_genes.rds'))
preds_all2 = readRDS(paste0('Rscripts/results/simulated_data/prediction_abs_expr_valid_donor_epoch19_', config, '_genes.rds'))

# add prediction results form contrast model
pred_contrast = do.call('rbind', lapply(constr_att, function(bs)
{
    fread(paste0('results/attn0_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', bs, '/Prediction_Results_19_in_valid_donors.csv'))
}))


#### convert diff expression to absolute expression, compare with base model ####
donors = unique(pred_base$donor) #83
train_genes = unique(metrics[[1]]$gene_name[metrics[[1]]$gene_split == 'train']) # 4949
test_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'valid', 'gene_name']) #1047
tissues = unique(metrics[[1]]$tissue)

if(config == 'train')
{
   output_genes = train_genes
}else {
   output_genes = test_genes
}

# do not run if preload, takes too long
metrics_all2 = list()
preds_all2 = list()
for(i in 1:3)
{
    results = compute_metrics(pred[[i]], pred_base, donors, output_genes,tissues = tissues, toplot = F) # for baselime, some donors will be missing for each gene, because of batch size
    metrics_all2[[i]] = results[[1]]
    metrics_all2[[i]]$model = paste0('model', i)
    preds_all2[[i]] = results[[2]]
    preds_all2[[i]]$model = paste0('model', i)
}
metrics_all2[[4]] = metrics_base[gene_name %in% output_genes,c('tissue', 'gene_name', 'pearsonr', 'r2') ]
metrics_all2[[4]]$model = 'baseline'
r2_base = recompute_r2(pred_base, output_genes) # doesn't change much
all(metrics_all2[[4]]$gene_name == r2_base$gene_name)
all(metrics_all2[[4]]$tissue == r2_base$tissue)
metrics_all2[[4]]$r2 = r2_base$r2

print(summary(metrics_all2[[4]][!is.na(pearsonr) & gene_name %in% output_genes,c('pearsonr', 'r2')]))

preds_all2[[4]] = pred_base[gene_name %in% output_genes, c("y_pred", "y_true", "donor", "gene_name", "tissue" )]
preds_all2[[4]]$model = 'baseline'

# add contrast to results
results = compute_metrics(pred_contrast, pred_base, donors, output_genes, tissues = tissues, toplot = F)
metrics_all2[[5]] = results[[1]]
metrics_all2[[5]]$model = 'contrast'
preds_all2[[5]] = results[[2]]
preds_all2[[5]]$model = 'contrast'

saveRDS(preds_all2, file = paste0('Rscripts/results/simulated_data/prediction_abs_expr_valid_donor_epoch19_', config, '_genes.rds'))
saveRDS(metrics_all2, file = paste0('Rscripts/results/simulated_data/metrics_abs_expr_valid_donor_epoch19_', config, '_genes.rds'))

# add model3 to results
results = compute_metrics(pred_model3, pred_base, donors, output_genes, tissues = tissues, toplot = F)
metrics_all2[[6]] = results[[1]]
metrics_all2[[6]]$model = 'model4'
preds_all2[[6]] = results[[2]]
preds_all2[[6]]$model = 'model4'

results = compute_metrics(pred_model3_abs, pred_base, donors, output_genes, tissues = tissues, toplot = F)
metrics_all2[[7]] = results[[1]]
metrics_all2[[7]]$model = 'model2_atten'
preds_all2[[7]] = results[[2]]
preds_all2[[7]]$model = 'model2_atten'



preds_wide = lapply(preds_all2[1:5], function(pred2){
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


# separate genes with tissue specific eQTLs 

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
tissue_specific_genes = intersect(tissue_specific_genes, output_genes)



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
}, cor_by_gene, preds_all2[1:5], preds_wide), use.names = TRUE)

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

pdf(paste0('Rscripts/plots/simulated_data/mean_correlation_heatmap_ytrue_', config, '_tissue-specific-genes.pdf'), width = 6, height = 5)
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

pdf(paste0('Rscripts/plots/simulated_data/mean_correlation_heatmap_', config, '_tissue-specific-genes.pdf'), width = 12, height = 8)
ggplot(heatmap_cor_data, aes(x = var1, y = var2, fill = mean_cor)) +
    geom_tile() +
    facet_wrap(~model) +
    scale_fill_gradient2(low = 'steelblue', mid = 'white', high = 'firebrick', midpoint = 0, limits = c(-0.1, 0.1)) +
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


##### plotting
metrics_all2 = do.call('rbind', metrics_all2)
ind = which(metrics_all2$r2 != -Inf) # remove genes that do not exist in particular tissue

pdf(paste0('Rscripts/plots/simulated_data/pearsonr_boxplot_3_tissues_', config,'_genes_abs_expr4.pdf'))
if(config == 'train')
{
    ggplot(metrics_all2, aes(x = tissue, y = pearsonr, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(0.5,1)) + theme(text = element_text(size = 16))
}else{
    ggplot(metrics_all2, aes(x = model, y = pearsonr, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(-1,1)) + theme(text = element_text(size = 16)) #outlier.shape = NA
}
dev.off()

pdf(paste0('Rscripts/plots/simulated_data/r2_boxplot_3_tissues_', config,'_genes_abs_expr4.pdf'))
if(config == 'train')
{ 
    ggplot(metrics_all2[r2!= -Inf, ], aes(x = tissue, y = r2, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(0.3,1)) + theme(text = element_text(size = 16)) #
}else{
   ggplot(metrics_all2[r2!= -Inf, ], aes(x = model, y = r2, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
      coord_cartesian(ylim = c(-0.3,0.3)) + theme(text = element_text(size = 16)) #
}
dev.off()

##### eQTL-restricted version: de novo SNVs that are themselves eQTLs (all pip) #####
# Restrict a novel SNV to one that also sits in a GTEx susie credible set for the gene (any pip),
# i.e. an unseen *functional* variant. FINDING: this intersection is empty -- de novo variants are
# absent from all training donors (MAF < ~0.07%) whereas eQTL credible sets only contain common
# variants (MAF >~ 1%, even at low pip), so no de novo SNV is ever a tested/fine-mapped eQTL.
# Switching from pip>0.5 to all pip does not change this (rarity, not pip, is the limiter).
# This block documents that and still runs the split if the intersection is ever non-empty.
eqtl_dir   = '/pollard/data/projects/zhhu/GTEX/eQTL_susie/'
tissue_lab = read.csv(paste0(eqtl_dir, 'dataset_tissue_label.csv'))
eqtl_map   = data.table(data_tissue = c('Whole_Blood', 'Muscle_Skeletal', 'Adipose_Subcutaneous'),
                        tissue      = c('blood', 'muscle', 'adipose'))
eqtl_map[, path := tissue_lab$path[match(data_tissue, tissue_lab$data_tissue)]]
eqtl_sets = rbindlist(lapply(seq_len(nrow(eqtl_map)), function(k) {
    cs = fread(paste0(eqtl_dir, eqtl_map$path[k]), select = c('gene_id', 'variant')) # all credible-set variants, any pip
    unique(cs[, .(gene_name = gene_id, variant_id = variant, tissue = eqtl_map$tissue[k])])
}))

# which specific novel SNVs each test donor carries per gene (from Rscripts/novel_variants.py)
novel_carrier = fread('Rscripts/results/simulated_data/novel_variant_carriers_valid_fold0.csv') # gene_name, variant_id, donor
novel_eqtl = merge(novel_carrier, eqtl_sets, by = c('gene_name', 'variant_id'), allow.cartesian = TRUE) # (gene,variant,donor,tissue)

message('=== de novo SNVs that are also eQTLs (all pip), Fold-', fold, ' ===')
message('  novel SNVs tested: ', uniqueN(novel_carrier$variant_id),
        ' | eQTL credible-set variants (3 tissues): ', uniqueN(eqtl_sets$variant_id))
message('  novel SNVs that are eQTLs: ', uniqueN(novel_eqtl[, .(gene_name, variant_id)]),
        ' | (gene,donor,tissue) with a novel eQTL: ', uniqueN(novel_eqtl[, .(gene_name, donor, tissue)]))

if (nrow(novel_eqtl) == 0) {
    message('  -> empty: no de novo SNV is a credible-set eQTL. De novo variants (MAF < ~0.07%) are too rare ',
            'to be tested/fine-mapped as eQTLs (credible sets require MAF >~ 1%). No eQTL-restricted plot produced.')
} else {
    novel_eqtl_gdt = unique(novel_eqtl[, .(gene_name, donor, tissue)])
    novel_eqtl_gdt[, has_novel_eqtl := TRUE]
    preds_dt[novel_eqtl_gdt, on = .(gene_name, donor, tissue), has_novel_eqtl := i.has_novel_eqtl]
    preds_dt[is.na(has_novel_eqtl), has_novel_eqtl := FALSE]
    preds_dt[, eqtl_group := factor(fifelse(has_novel_eqtl, 'has novel eQTL', 'no novel eQTL'),
                                    levels = c('no novel eQTL', 'has novel eQTL'))]
    metrics_eqtl = preds_dt[!is.na(y_true) & !is.na(y_pred),
        .(pearsonr = if (.N >= min_donors) cor(y_true, y_pred) else NA_real_,
          r2       = if (.N >= min_donors) { yp = y_pred - mean(y_pred); 1 - sum((y_true - yp)^2) / sum(y_true^2) } else NA_real_,
          n_donors = .N),
        by = .(gene_name, tissue, model, eqtl_group)]
    metrics_eqtl[!is.finite(r2), r2 := NA_real_]
    fwrite(metrics_eqtl, paste0('Rscripts/results/simulated_data/metrics_by_novel_eQTL_', config, '.csv'))

    pdf(paste0('Rscripts/plots/simulated_data/pearsonr_boxplot_3_tissues_', config, '_genes_abs_expr4_by_novel_eQTL.pdf'), width = 11)
    print(ggplot(metrics_eqtl[!is.na(pearsonr)], aes(x = model, y = pearsonr, fill = eqtl_group)) +
        geom_boxplot(outlier.shape = NA) + facet_wrap(~tissue) + theme_bw() +
        coord_cartesian(ylim = if (config == 'train') c(0.5, 1) else c(-1, 1)) +
        theme(text = element_text(size = 14), axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(fill = 'test donors', title = 'Per-gene pearsonr split by de novo eQTL in donor window'))
    dev.off()
    pdf(paste0('Rscripts/plots/simulated_data/r2_boxplot_3_tissues_', config, '_genes_abs_expr4_by_novel_eQTL.pdf'), width = 11)
    print(ggplot(metrics_eqtl[!is.na(r2)], aes(x = model, y = r2, fill = eqtl_group)) +
        geom_boxplot(outlier.shape = NA) + facet_wrap(~tissue) + theme_bw() +
        coord_cartesian(ylim = if (config == 'train') c(0.3, 1) else c(-0.3, 0.3)) +
        theme(text = element_text(size = 14), axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(fill = 'test donors', title = 'Per-gene r2 split by de novo eQTL in donor window'))
    dev.off()
}

# wilcoxon text
wilcox.test(unlist(metrics_all2[metrics_all2$model == 'model4', 'pearsonr']), unlist(metrics_all2[metrics_all2$model == 'model4_abs', 'pearsonr']))
wilcox.test(unlist(metrics_all2[metrics_all2$model == 'baseline', 'r2']), unlist(metrics_all2[metrics_all2$model == 'model4_abs', 'r2']))

metrics_all2 = metrics_all2[ind, ]
pdf('Rscripts/plots/simulated_data/scatterplot_metrics_test_genes_model23.pdf')
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

ids = c('kabm6xx9', 'icb2whi4', '8puj4r29')
atten = lapply(1:3, function(i){
    id = ids[i]
    fread(paste0('results/attn2_', i, '_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/train_genes/Prediction_Results_-1_in_test_donors.csv'))
})


## read in attention
ids = c('kabm6xx9', 'icb2whi4', '8puj4r29', '2i541ch0', 'sj4cmono', '3zzlfa0b')

atten = lapply(c(2,6), function(i){ #1:3
    id = ids[i]
    if(i <= 3)
    {
        fread(paste0('results/attn2_', i, '_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/', config, '_genes/Prediction_Results_-1_in_test_donors.csv'))
    }else if(i == 4){
        fread(paste0('results/attn3_1_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/', config, '_genes/Prediction_Results_-1_in_test_donors.csv'))
    }else if(i ==5){
        fread(paste0('results/attn3_1_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/', config, '_genes/Prediction_Results_-1_in_test_donors.csv'))
    }else{
        fread(paste0('results/attn3_2_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/', config, '_genes/Prediction_Results_-1_in_test_donors.csv'))
    }
})
# obtain ground truth eQTLs for each tissue
tissue_id = read.csv('../GTEX/eQTL_susie/dataset_tissue_label.csv')  
tissue = c('Whole_Blood','Muscle_Skeletal', 'Adipose_Subcutaneous')
ids = tissue_id[match(tissue, tissue_id$data_tissue), 'path']
tissue_name = tissue_id[match(tissue, tissue_id$data_tissue), 'tabix_tissue']
tissue_name[3] = 'adipose'
egenes_all = list()
for(i in 1:length(ids))
{
    id = ids[i]
    ts = tissue_name[i]
    egenes = fread(paste0('/pollard/data/projects/zhhu/GTEX/eQTL_susie/', id))
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

# fwrite(gene_tissue_specific,
#        paste0('Rscripts/results/simulated_data/gene_tissue_specific_eqtl_pairs_pip', pip_thr, '_', config, '.tsv'),
#        sep = '\t')
# fwrite(data.table(gene = tissue_specific_genes),
#        paste0('Rscripts/results/simulated_data/tissue_specific_genes_pip', pip_thr, '_', config, '.tsv'),
#        sep = '\t')
# fwrite(gene_strict_specific,
#        paste0('Rscripts/results/simulated_data/gene_strict_tissue_specific_summary_pip', pip_thr, '_', config, '.tsv'),
#        sep = '\t')


# atten_all = list()
# for(ts in tissue_name)
# {
#     atten_all[[ts]] = atten[tissue == ts]
# }

# read intervals
intervals = read.csv("data/Gencode.v46.TSSCentered_49K_Intervals.csv")

#gn = atten$gene[1]

#genes = unique(atten$gene)

# separate genes by metrics and train/val/test
#train_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'train', 'gene_name'])
#val_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'valid', 'gene_name'])
#test_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'test', 'gene_name'])

train_genes = unique(atten[[1]]$gene)

# random select genes to plot attention
#train_genes = sample(train_genes, 500)
#val_genes = sample(val_genes, 300)
#test_genes = sample(test_genes, 300)

#gn1 = metrics[metrics$pearsonr > 0.5, 'gene_name'] for noiseless data
#gn2 = metrics[metrics$pearsonr < -0.2, 'gene_name']

# dat2plot <- mclapply(1:200, function(i) #mc
# {
#     gn = genes[i]

# select egenes with at least one SNP pip > 0.7
sig_genes = unique(do.call('c', sapply(egenes_all, function(x) unlist(x[which(x$pip > 0.7), 'gene_id'])))) # 3062
train_genes = intersect(train_genes, sig_genes) # 516
#val_genes = intersect(val_genes, sig_genes) # 335
#test_genes = intersect(test_genes, sig_genes) # 

dat2plot_all = list()

for(i in 1:length(atten))
{
    dat2plot <- mclapply(train_genes, function(gn) #mc
    {
        #print(gn)
        #print(metrics[metrics$gene_name == gn, ])
        results = lapply(tissue_name, function(ts)
        {
            atten_temp = atten[[i]][tissue == ts]
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
            #dat2plot = dat2plot[, .(atten_weights = mean(atten_weights), eQTL_size = mean(eQTL_size), eQTL_pip = max(eQTL_pip)), by = .(inds = inds, gene)]
            dat2plot$tissue = ts
        return(dat2plot) # remove loci not different between individual pairs
        })
        do.call(rbind, results)
    } , mc.cores = 8)

    dat2plot <- do.call(rbind, dat2plot)
    dat2plot$model = paste0('model', i)
    dat2plot_all[[i]] = dat2plot
}

dat2plot_all = do.call(rbind, dat2plot_all)
save(dat2plot_all, file = paste0('atten_', config, '_genes_model23.rdat')) #***1 select cases where gene expression shows difference

length(unique(dat2plot_all[eQTL_pip > 0.7,][['gene']])) # number of positive samples: 425, 267 for valid genes

#boxplot(atten_weights~is.na(eQTL_size), dat2plot, outline = F)


dat2plot_all$eqtl = dat2plot_all$eQTL_pip > 0.7
dat2plot_all$eqtl[is.na(dat2plot_all$eqtl)] = FALSE
boxplot(atten_weights~eqtl, dat2plot, outline = F)

dat2plot_all[model == 'model2', 'model'] = 'model2_atten'
dat2plot_all[model == 'model1', 'model'] = 'model2'

# separate tissue
# subsample negative cases to reduce plooting time
#sum(dat2plot_all[tissue == 'adipose' & model == 'model1', 'eqtl'])
dat2plot_all[, .N, by = .(eqtl,tissue, model) ]
if(config == 'train')
{
    dat2plot2 <- dat2plot_all[, .SD[sample(.N, min(.N, 64199 * 5))], by = .(eqtl,tissue, model) ] #sum(dat2plot_all$eqtl) *5
}else{
    #dat2plot2 <- dat2plot_all[, .SD[sample(.N, min(.N, 5000))], by = .(eqtl,tissue, model) ] 
    dat2plot2 <- dat2plot_all[, .SD[sample(.N, min(.N, 52046))], by = .(eqtl,tissue, model) ] 
}



pdf(paste0('Rscripts/plots/simulated_data/atten_weights_boxplot_3_tissues_', config, '_genes_model23.pdf')) # ***1 select cases where gene expression shows difference
ggplot(dat2plot2, aes(x = eqtl, y = atten_weights)) + geom_boxplot(outlier.shape = NA) + facet_grid(tissue ~ model) + theme_bw() + 
    coord_cartesian(ylim = c(0,0.15)) + xlab('eQTL pip > 0.7, 425 positive genes') # 425, 267 0.1
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


metrics <- lapply(c(2,6), function(i) {
    id <- ids[i]
    if(i <= 3)
    {
        filepath <- paste0(
            "results/attn2_", i,
            "_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/",
            "Fold-", fold, "/", id, "/", config, "_genes/"
        )
        files <- 'CrossIndivMetrics_test_donors_Epoch-1_rank0.csv'
        mm <- read.csv(file.path(filepath, files))
        mm$model <- paste0("model", i)
    }else{
         filepath <- paste0(
            "results/attn3_2", 
            "_pred_norm_3_tissue_abs/MultiGene/rain_filter_egenes_5K/",
            "Fold-", fold, "/", id, "/", config, "_genes/"
        )
        files <- 'CrossIndivMetrics_test_donors_Epoch-1_rank0.csv'
        mm <- read.csv(file.path(filepath, files))
        mm$model <- "model2_atten"
    }
    return(mm)
})
metrics_all = do.call('rbind', metrics)
metrics_all = as.data.table(metrics_all)
dat2plot_all[metrics_all[, .(gene_name, tissue, pearsonr, model)], 
         on = .(gene = gene_name, tissue = tissue, model = model),
         pearsonr := i.pearsonr]
genes = unlist(unique(dat2plot_all[atten_weights > 0.1 & pearsonr > 0.9 & eQTL_pip > 0.9, 'gene'])) #185  good_genes
#genes = intersect(good_genes, genes)
i = 1#2,5
metrics_all[metrics_all$gene_name == genes[i], ]

##setnafill(dat2plot, fill = 0, cols = c("eQTL_pip"))
genes = c('ENSG00000249159', 'ENSG00000204099', 'ENSG00000286288')
temp = dat2plot_all[gene == genes[i]]

temp2 = temp[, .(atten_weights = mean(atten_weights), eQTL_size = mean(eQTL_size), eQTL_pip = max(eQTL_pip)), by = .(inds = inds, tissue, model)]


# manhattan plot
pdf('Rscripts/plots/simulated_data/manhanttan_plot_atten_3_tissues_train_genes1_avg.pdf', width = 14)
ggplot(temp2) + geom_point(size = 1, aes(x = inds, y = atten_weights, color = eQTL_size )) + facet_grid(tissue~model) + 
 geom_text(data = metrics_all[metrics_all$gene_name == genes[i], ], aes(x = -Inf, y = Inf, label = round(pearsonr,2)), hjust = -0.1, vjust = 1.1) + theme_bw()  +
  geom_vline(aes(xintercept = inds, color = eQTL_pip), data = temp[!is.na(eQTL_pip) & eQTL_pip > 0.7]) + ggtitle(genes[i]) 
dev.off()


# get average attention per position for pairs without gene expression difference
dat2plot_all = lapply(1:3, function(i)
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
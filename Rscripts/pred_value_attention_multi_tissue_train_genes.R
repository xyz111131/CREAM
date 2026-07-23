library(data.table)
library(parallel)
library(ggplot2)

source('utility_functions.R')

baseline = c('pnhnqwbc', 'b217h7ok', 'u4r9795d', 'v03gybbm')
constr_att = c('n3c6i3ia', 'fnw3vml2', 'oij5gq16', 'c3zypcbg')

#ids = c('kabm6xx9', 'icb2whi4', '8puj4r29')
ids = list()
ids[[1]] = c('kabm6xx9', '7ha00zo8', 'tfcy569t', 'ql7nazfo')
ids[[2]] = c('icb2whi4', 'als1zcy4', 'qxgv61a4', '0djohyzk')
ids[[3]] = c('8puj4r29', 'bgrf8cay', 'j4ifqr3x', 'jfvgesmj')



output_dir = 'Rscripts/plots/simulated_data/'

pred_base = lapply(baseline, function(bs)
{
    fread(paste0('results/baseline_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', bs, '/Prediction_Results_19_in_valid_donors.csv'))
})
pred_base = do.call(rbind, pred_base)
setnames(pred_base, old = 'gene', new = 'gene_name') 
pred_base = pred_base[, list(y_pred = mean(y_pred), y_true = mean(y_true)), by = c('donor', 'gene_name','tissue')]# compute each donor/gene/tissue multiple times 1493004 


pred = lapply(1:3, function(i){
    id = ids[[i]]
    do.call('rbind', lapply(id, function(j)
    {
        fread(paste0('results/attn2_', i, '_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', j, '/Prediction_Results_19_in_valid_donors.csv'))
    }))
})


pred_contrast = do.call('rbind', lapply(constr_att, function(bs)
{
    fread(paste0('results/attn0_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', bs, '/Prediction_Results_19_in_valid_donors.csv'))
}))

metrics <- lapply(1:3, function(i) {
    id <- ids[[i]]
    do.call("rbind", lapply(id, function(j) {
        filepath <- paste0(
            "results/attn2_", i,
            "_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/",
            "Fold-0/", j, "/"
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

train_genes = unique(metrics[[1]]$gene_name[metrics[[1]]$gene_split == 'train']) # 4949

# run_id = c('kabm6xx9', '7ha00zo8', 'tfcy569t', 'ql7nazfo')
# for(rk in 0:3)
# {
#     id = run_id[rk+1]
#     atten = fread(paste0('results/attn2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/Prediction_Results_19_in_valid_donors.csv'))
#     metrics = read.csv(paste0('results/attn2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/CrossIndivMetrics_test_donors_Epoch19_rank', rk, '.csv'))
# }

# check the metrics per tissue, 1047 genes
# ind = which(metrics$r2 != 0) # remove genes that do not exist in particular tissue
# boxplot(metrics$r2[ind]~metrics$tissue[ind], outline = F)
# boxplot(metrics$pearsonr~metrics$tissue, outline = F)

metrics_all = do.call('rbind', metrics)
ind = which(metrics_all$r2 != 0) # remove genes that do not exist in particular tissue

pdf('Rscripts/plots/simulated_data/pearsonr_boxplot_3_tissues_train_genes.pdf')
ggplot(metrics_all, aes(x = tissue, y = pearsonr, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(0.9,1)) + theme(text = element_text(size = 16))
dev.off()

pdf('Rscripts/plots/simulated_data/r2_boxplot_3_tissues_train_genes.pdf')
ggplot(metrics_all[ind, ], aes(x = tissue, y = r2, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(0.8,1)) + theme(text = element_text(size = 16))
dev.off()

# compare metrics for pairs of model
all(metrics[[1]]$tissue == metrics[[3]]$tissue)
all(metrics[[1]]$gene == metrics[[3]]$gene)


df = data.frame('model1' = metrics[[1]]$pearsonr, 'model2' = metrics[[2]]$pearsonr, 
    'model3' = metrics[[3]]$pearsonr)

pdf('Rscripts/plots/simulated_data/scatterplot_pearsonr_3_tissues_train_genes_23.pdf')
ggplot(df, aes(x = model2, y = model3)) +
  geom_pointdensity() + geom_abline(intercept = 0, slope = 1) + ggtitle('pearson r') + 
  scale_color_viridis_c(option = "plasma") + # Use a color scale like viridis
  theme_minimal() + theme(text = element_text(size = 16))
dev.off()

#### convert diff expression to absolute expression, compare with base model ####
donors = unique(pred_base$donor) #83
train_genes = unique(metrics[[1]]$gene_name[metrics[[1]]$gene_split == 'train']) # 4949
test_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'valid', 'gene_name']) #1047
tissues = unique(metrics_base$tissue)

#abs_expr = convert_expr(atten[[1]][gene == 'ENSG00000259430' & tissue == 'blood' ,1:6], donors)
metrics_all2 = list()
preds_all2 = list()
for(i in 1:3)
{
    results = compute_metrics(pred[[i]], pred_base, donors, train_genes,tissues = tissues, toplot = F)
    metrics_all2[[i]] = results[[1]]
    metrics_all2[[i]]$model = paste0('model', i)
    preds_all2[[i]] = results[[2]]
    preds_all2[[i]]$model = paste0('model', i)
}
metrics_all2[[4]] = metrics_base[gene_name %in% train_genes,c('tissue', 'gene_name', 'pearsonr', 'r2') ]
metrics_all2[[4]]$model = 'baseline'
r2_base = recompute_r2(pred_base, train_genes) # doesn't change much
all(metrics_all2[[4]]$gene_name == r2_base$gene_name)
all(metrics_all2[[4]]$tissue == r2_base$tissue)
metrics_all2[[4]]$r2 = r2_base$r2

print(summary(metrics_all2[[4]][!is.na(pearsonr) & gene_name %in% train_genes,c('pearsonr', 'r2')]))

preds_all2[[4]] = pred_base[gene_name %in% train_genes, c("y_pred", "y_true", "donor", "gene_name", "tissue" )]
preds_all2[[4]]$model = 'baseline'
saveRDS(preds_all2, file = 'prediction_abs_expr_train_genes.rds')


saveRDS(metrics_all2, file = 'metrics_abs_expr_valid_donor_epoch19_train_genes.rds')
metrics_all2 = do.call('rbind', metrics_all2)
ind = which(metrics_all2$r2 != -Inf) # remove genes that do not exist in particular tissue

pdf('Rscripts/plots/simulated_data/pearsonr_boxplot_3_tissues_train_genes_abs_expr1.pdf')
ggplot(metrics_all2, aes(x = tissue, y = pearsonr, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(0.5,1)) + theme(text = element_text(size = 16))
ggplot(metrics_all2, aes(x = model, y = pearsonr, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(0.5,1)) + theme(text = element_text(size = 16)) #outlier.shape = NA
dev.off()

pdf('Rscripts/plots/simulated_data/r2_boxplot_3_tissues_train_genes_abs_expr1.pdf')
ggplot(metrics_all2[r2!= -Inf, ], aes(x = tissue, y = r2, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
    coord_cartesian(ylim = c(0.3,1)) + theme(text = element_text(size = 16)) #
#ggplot(metrics_all2[r2!= -Inf, ], aes(x = model, y = r2, fill = model)) + geom_boxplot(outlier.shape = NA) + theme_bw() + 
#    coord_cartesian(ylim = c(0.3,1)) + theme(text = element_text(size = 16)) #
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


## read in attention
atten = lapply(1:3, function(i){
    id = ids[i]
    fread(paste0('results/attn2_', i, '_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/train_genes/Prediction_Results_-1_in_test_donors.csv'))
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
train_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'train', 'gene_name'])
val_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'valid', 'gene_name'])
test_genes = unique(metrics[[1]][metrics[[1]]$gene_split == 'test', 'gene_name'])

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

for(i in 1:3)
{
    dat2plot <- mclapply(train_genes, function(gn) #mc
    {
        #print(gn)
        #print(metrics[metrics$gene_name == gn, ])
        results = lapply(tissue_name, function(ts)
        {
            atten = atten[[i]][tissue == ts]
            egenes = egenes_all[[ts]]
            donor_pairs = which(atten$gene == gn)
            eqtls = egenes[gene_id == gn]
            start = intervals[grep(gn, intervals$gene_id), 'starts']
            eqtls$ind = eqtls$pos0 - start
            dat2plot = lapply(donor_pairs, function(i){
                inds = as.numeric(strsplit(atten$attn_inds[i], ',')[[1]])
                weights = as.numeric(strsplit(atten$attn_weights[i], ',')[[1]])
                data.table('gene' = gn, 'inds' = inds, 'atten_weights' = weights, 'eQTL_size' = unlist(eqtls[match(inds, ind), 'beta2']), 'eQTL_pip' = unlist(eqtls[match(inds, ind), 'pip']))
            })
            dat2plot = do.call('rbind', dat2plot)
            dat2plot = dat2plot[dat2plot$atten_weights > 0, ]
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

unique(dat2plot_all[dat2plot_all$eQTL_pip > 0.7, 'gene']) # number of positive samples: 425

#boxplot(atten_weights~is.na(eQTL_size), dat2plot, outline = F)


dat2plot_all$eqtl = dat2plot_all$eQTL_pip > 0.7
dat2plot_all$eqtl[is.na(dat2plot_all$eqtl)] = FALSE
boxplot(atten_weights~eqtl, dat2plot, outline = F)

# separate tissue
# subsample negative cases to reduce plooting time
#sum(dat2plot_all[tissue == 'adipose' & model == 'model1', 'eqtl'])
dat2plot2 <- dat2plot_all[, .SD[sample(.N, min(.N, 64199 * 5))], by = .(eqtl,tissue, model) ] #sum(dat2plot_all$eqtl) *5
pdf('Rscripts/plots/simulated_data/atten_weights_boxplot_3_tissues_train_genes.pdf')
ggplot(dat2plot2, aes(x = eqtl, y = atten_weights)) + geom_boxplot(outlier.shape = NA) + facet_grid(tissue ~ model) + theme_bw() + 
    coord_cartesian(ylim = c(0,0.1)) + xlab('eQTL pip > 0.7, 425 positive genes')
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

metrics_all = as.data.table(metrics_all)
dat2plot_all[metrics_all[, .(gene_name, tissue, pearsonr)], 
         on = .(gene = gene_name, tissue = tissue),
         pearsonr := i.pearsonr]
genes = unlist(unique(dat2plot_all[atten_weights > 0.1 & pearsonr > 0.9 & eQTL_pip > 0.9, 'gene'])) #185  good_genes
#genes = intersect(good_genes, genes)
i = 5#2
metrics_all[metrics_all$gene_name == genes[i], ]

##setnafill(dat2plot, fill = 0, cols = c("eQTL_pip"))
temp = dat2plot_all[gene == genes[i]]


# manhattan plot
pdf('Rscripts/plots/simulated_data/manhanttan_plot_atten_3_tissues_train_genes5.pdf', width = 14)
ggplot(temp) + geom_point(size = 1, aes(x = inds, y = atten_weights, color = eQTL_size )) + facet_grid(tissue~model) + 
 geom_text(data = metrics_all[metrics_all$gene_name == genes[i], ], aes(x = -Inf, y = Inf, label = round(pearsonr,2)), hjust = -0.1, vjust = 1.1) + theme_bw()  +
  geom_vline(aes(xintercept = inds, color = eQTL_pip), data = temp[!is.na(eQTL_pip) & eQTL_pip > 0.7]) + ggtitle(genes[i])
dev.off()


# get average attention per position
temp2 = dat2plot[is.na(pearsonr), mean(atten_weights), by = .(inds %/% 50, tissue)]
ggplot(temp2) + geom_point(size = 1, aes(x = inds, y = V1)) + facet_grid(tissue~.)

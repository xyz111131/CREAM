library(data.table)
library(parallel)

id = 'kabm6xx9'
atten = fread(paste0('results/attn2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/test_genes/Prediction_Results_-1_in_test_donors.csv'))
metrics = read.csv(paste0('results/attn2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/test_genes/CrossIndivMetrics_test_donors_Epoch-1_rank0.csv'))

output_dir = 'Rscripts/plots/simulated_data/'
# run_id = c('kabm6xx9', '7ha00zo8', 'tfcy569t', 'ql7nazfo')
# for(rk in 0:3)
# {
#     id = run_id[rk+1]
#     atten = fread(paste0('results/attn2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/Prediction_Results_19_in_valid_donors.csv'))
#     metrics = read.csv(paste0('results/attn2_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/CrossIndivMetrics_test_donors_Epoch19_rank', rk, '.csv'))
# }

# check the metrics per tissue, 1047 genes
ind = which(metrics$r2 != 0)
boxplot(metrics$r2[ind]~metrics$tissue[ind], outline = F)
boxplot(metrics$pearsonr~metrics$tissue, outline = F)

pdf('Rscripts/plots/simulated_data/pearsonr_boxplot_3_tissues_model_1_valid_genes.pdf')
ggplot(metrics, aes(x = tissue, y = pearsonr)) + geom_boxplot(outlier.shape = NA) + theme_bw() 
dev.off()

pdf('Rscripts/plots/simulated_data/r2_boxplot_3_tissues_model_1_valid_genes.pdf')
ggplot(metrics[ind, ], aes(x = tissue, y = r2)) + geom_boxplot(outlier.shape = NA) + theme_bw() + coord_cartesian(ylim = c(-0.05,0.05))
dev.off()


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

atten_all = list()
for(ts in tissue_name)
{
    atten_all[[ts]] = atten[tissue == ts]
}

# read intervals
intervals = read.csv("data/Gencode.v46.TSSCentered_49K_Intervals.csv")

#gn = atten$gene[1]

#genes = unique(atten$gene)

# separate genes by metrics and train/val/test
train_genes = unique(metrics[metrics$gene_split == 'train', 'gene_name'])
val_genes = unique(metrics[metrics$gene_split == 'valid', 'gene_name'])
test_genes = unique(metrics[metrics$gene_split == 'test', 'gene_name'])

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
train_genes = intersect(train_genes, sig_genes) # 
val_genes = intersect(val_genes, sig_genes) # 335
test_genes = intersect(test_genes, sig_genes) # 

dat2plot <- mclapply(val_genes, function(gn) #mc
{
    #print(gn)
    #print(metrics[metrics$gene_name == gn, ])
    results = lapply(tissue_name, function(ts)
    {
        atten = atten_all[[ts]]
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

unique(dat2plot[dat2plot$eQTL_pip > 0.7, 'gene']) # number of positive samples: 267

boxplot(atten_weights~is.na(eQTL_size), dat2plot, outline = F)


dat2plot$eqtl = dat2plot$eQTL_pip > 0.7
dat2plot$eqtl[is.na(dat2plot$eqtl)] = FALSE
boxplot(atten_weights~eqtl, dat2plot, outline = F)

# separate tissue
library(ggplot2)
# subsample negative cases to reduce plooting time
dat2plot2 <- dat2plot[, .SD[sample(.N, min(.N, 136529 * 5))], by = eqtl]
pdf('Rscripts/plots/simulated_data/atten_weights_boxplot_3_tissues_model_1_valid_genes.pdf')
ggplot(dat2plot2, aes(x = eqtl, y = atten_weights)) + geom_boxplot(outlier.shape = NA) + facet_wrap(~tissue, nrow = 2) + theme_bw() + 
    coord_cartesian(ylim = c(0,0.1)) + xlab('eQTL pip > 0.7, 267 positive genes')
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
genes = unique(unlist(dat2plot[dat2plot$eQTL_pip > 0.7, 'gene']))
good_genes = metrics[which(metrics$pearsonr > 0.1), 'gene_name'] # select good genes
genes = intersect(good_genes, genes)
i = 120 #1, 5

##setnafill(dat2plot, fill = 0, cols = c("eQTL_pip"))
temp = dat2plot[gene == genes[i]]
# manhattan plot
pdf('Rscripts/plots/simulated_data/manhanttan_plot_atten_3_tissues_model_1_valid_genes.pdf')
ggplot(temp) + geom_point(size = 1, aes(x = inds, y = atten_weights, color = eQTL_size )) + facet_grid(tissue~.) + 
 geom_text(data = metrics[metrics$gene_name == genes[i], ], aes(x = -Inf, y = Inf, label = round(pearsonr,2)), hjust = -0.1, vjust = 1.1) + theme_bw()  +
  geom_vline(aes(xintercept = inds, color = eQTL_pip), data = temp[!is.na(eQTL_pip) & eQTL_pip > 0.1])
dev.off()

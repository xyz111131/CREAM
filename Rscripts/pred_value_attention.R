library(data.table)
library(parallel)
#atten = fread('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/993714h9/test_genes/Prediction_Results_-1_in_test_donors_valid_genes.csv')
#metrics = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/993714h9/test_genes/CrossIndivMetrics_test_donors_valid_genes_Epoch-1_rank0.csv')


atten = fread('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/eadyk49k/Prediction_Results_20_in_test_donors.csv')
metrics = read.csv('results/shift_diff_multi_attn2_pred_norm/MultiGene/blood_train_filter_egenes_3K/Fold-0/eadyk49k/CrossIndivMetrics_test_donors_Epoch20_rank0.csv')


# obtain ground truth eQTLs
egenes = fread('/pollard/data/projects/zhhu/GTEX/eQTL_susie/QTD000356.credible_sets.tsv.gz') 
egenes$beta2 = egenes$beta * egenes$pip
egenes$pos = as.numeric(sapply(egenes$variant, function(x) strsplit(x, '_')[[1]][2]))
egenes$pos0 = egenes$pos - 1

# read intervals
intervals = read.csv("data/Gencode.v46.TSSCentered_49K_Intervals.csv")

#gn = atten$gene[1]

genes = unique(atten$gene)

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
sig_genes = unique(unlist(egenes[which(egenes$pip > 0.7), 'gene_id']))
train_genes = intersect(train_genes, sig_genes) # 527
val_genes = intersect(val_genes, sig_genes) # 159
test_genes = intersect(test_genes, sig_genes) # 171

dat2plot <- mclapply(val_genes, function(gn) #mc
{
    print(gn)
    print(metrics[metrics$gene_name == gn, ])
    donor_pairs = which(atten$gene == gn)
    eqtls = egenes[gene_id == gn]
    start = intervals[grep(gn, intervals$gene_id), 'starts']
    eqtls$ind = eqtls$pos0 - start
    dat2plot = NULL
    for(i in donor_pairs)
    {
        inds = as.numeric(strsplit(atten$attn_inds[i], ',')[[1]])
        weights = as.numeric(strsplit(atten$attn_weights[i], ',')[[1]])
        dat2plot = rbind(dat2plot, data.frame('gene' = gn, 'inds' = inds, 'atten_weights' = weights, 'eQTL_size' = eqtls[match(inds, ind), 'beta2'], 'eQTL_pip' = eqtls[match(inds, ind), 'pip']))
    }
    return(dat2plot[dat2plot$atten_weights > 0, ]) # remove loci not different between individual pairs
} , mc.cores = 8)

dat2plot <- do.call(rbind, dat2plot)

unique(dat2plot[dat2plot$pip > 0.7, 'gene']) # number of positive samples, 5 for training, 1 for val

boxplot(atten_weights~is.na(beta2), dat2plot, outline = F)
dat2plot$eqtl = dat2plot$pip > 0.7
dat2plot$eqtl[is.na(dat2plot$eqtl)] = FALSE
boxplot(atten_weights~eqtl, dat2plot, outline = F)

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




library(data.table)
library(parallel)
library(ggplot2)

# paths come from the YAML config (CREAM/configs/defaults.yaml)
source("Rscripts/config.R")
cream <- cream_config()

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
    ts = tissue_name[i]
    egenes = fread(file.path(cream$eqtl_dir, id))
    egenes$beta2 = egenes$beta * egenes$pip
    egenes$pos = as.numeric(sapply(egenes$variant, function(x) strsplit(x, '_')[[1]][2]))
    egenes$pos0 = egenes$pos - 1
    egenes$tissue = ts
    egenes_all[[ts]] = egenes
}

# read intervals
intervals = read.csv(cream$genomic_intervals_file)

## read in attention from different runs, folds or seeds
ids = c('kabm6xx9', 'icb2whi4', '8puj4r29', 'c1v5bhr0', 'd0sj2293', '79us58oo', 'csn3bgv7', '8uh15d1j', 'nfob6o3p')
atten = lapply(1:9, function(i){
    id = ids[i]
    j = (i - 1) %% 3 + 1
    if( i %in% 4:6)
    {
        fread(paste0('results/attn2_', j, '_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-1/', id, '/train_genes/Prediction_Results_-1_in_test_donors.csv'))
    }else{
        fread(paste0('results/attn2_', j, '_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/Fold-0/', id, '/train_genes/Prediction_Results_-1_in_test_donors.csv'))
    }
})




# combine attention with eQTL
dat2plot_all <- mclapply(c('ENSG00000286288', 'ENSG00000249159', 'ENSG00000204099'), function(gn) #mc
{
    dat2plot = lapply(c(1:3,7:9), function(i)
    {
        results = lapply(tissue_name, function(ts)
        {
            atten_temp = atten[[i]][tissue == ts]
            # select the cases where gene expression is different
            #atten_temp = atten_temp[y_true!=0] # get null
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
            dat2plot$tissue = ts
            return(dat2plot) # remove loci not different between individual pairs
        })
        results = do.call(rbind, results)
        j = (i - 1) %% 3 + 1
        results$model = paste0('model', j)
        return(results)
    })
    do.call(rbind, dat2plot)
   
}, mc.cores = 3)

dat2plot_all = do.call(rbind, dat2plot_all)

dat2plot_all = dat2plot_all[, .(atten_weights = mean(atten_weights), eQTL_size = mean(eQTL_size), eQTL_pip = max(eQTL_pip)), by = .(inds = inds, gene, tissue, model)]

metrics <- lapply(c(1:3, 7:9), function(i) {
    id <- ids[i]
    j = (i - 1) %% 3 + 1
    filepath <- paste0(
        "results/attn2_", j,
        "_pred_norm_3_tissue/MultiGene/rain_filter_egenes_5K/",
        "Fold-", 0, "/", id, "/train_genes/"
    )
    files <- 'CrossIndivMetrics_test_donors_Epoch-1_rank0.csv'
    mm <- read.csv(file.path(filepath, files))
    mm$model <- paste0("model", j)
    return(mm)
})
metrics_all = do.call('rbind', metrics)
metrics_all = as.data.table(metrics_all)
metrics_all = metrics_all[, .(r2 = mean(r2), pearsonr = mean(pearsonr)), by = .(gene_name, tissue, model)]
dat2plot_all[metrics_all[, .(gene_name, tissue, pearsonr)], 
         on = .(gene = gene_name, tissue = tissue),
         pearsonr := i.pearsonr]

gn = 'ENSG00000204099'
temp = dat2plot_all[gene == gn]

# manhattan plot
pdf('Rscripts/plots/simulated_data/manhanttan_plot_atten_3_tissues_train_genes20_avg2.pdf', width = 14)
ggplot(temp) + geom_point(size = 1, aes(x = inds, y = atten_weights, color = eQTL_size )) + facet_grid(tissue~model) + 
 geom_text(aes(x = -Inf, y = Inf, label = round(pearsonr,2)), hjust = -0.1, vjust = 1.1) + 
 theme_bw()  + geom_vline(aes(xintercept = inds, color = eQTL_pip), data = temp[!is.na(eQTL_pip) & eQTL_pip > 0.7]) + ggtitle(gn)
dev.off()
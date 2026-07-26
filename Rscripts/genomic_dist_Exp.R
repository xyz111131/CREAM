library(data.table)
# read in ISM results
gns = read.table('data/genes/Whole_Blood/MultiGene/300_train_genes.txt')
# read in linear regression results
lm_res = fread('data/combined_coefs.tsv')
lm_res0 = fread('data/combined.tsv')
load('../GTEX/rare_variants/300_train_genes_genotype_v1.rdat')
genotype_all$variant_id = sub('_b38', '', genotype_all$variant_id)

# gene expression
expr_norm = fread('data/gtex_eqtl_expression_matrix/Whole_Blood.v8.normalized_expression.bed.gz')
genes_id_mapping = read.csv('data/gtex_eqtl_expression_matrix/gene_id_mapping.csv', row.names=1)
expr_norm$gene_name = genes_id_mapping[match(expr_norm$gene_id, genes_id_mapping$Name), 'Description']

expr_raw = fread('data/gtex_eqtl_expression_matrix/gene_log_tpm_2017-06-05_v8_Whole_Blood.gct')

test_donors = read.table('data/cross_validation_folds/gtex/cv_folds/person_ids-test-fold0.txt')$V1
train_donors = read.table('data/cross_validation_folds/gtex/cv_folds/person_ids-train-fold0.txt')$V1
val_donors = read.table('data/cross_validation_folds/gtex/cv_folds/person_ids-val-fold0.txt')$V1
allresults = NULL
for(gn in gns$V1)
{
    genotype = genotype_all[genotype_all$gene_name == gn, ]
    # individuals has identical SNPs
    # combine ./. with 0/0
    genotype1 = genotype[, -1: -7]
    #genotype1[genotype1 == './.'] = '0/0'
    results= matrix(0, ncol(genotype1), ncol(genotype1))
    for(i in 1:(ncol(genotype1)-1))
    {
        for(j in (i+1):ncol(genotype1))
        {
            results[i,j] = all(genotype1[, i] == genotype1[, j])
        }
    }
    colnames(results) = rownames(results) = colnames(genotype1)
    print(sum(results))
    if(sum(results) > 100)
    {
        print(gn)
        results = melt(results, na.rm =T)
        results$gene = gn
        expr1 = t(expr_norm[expr_norm$gene_name == gn, -1:-4])
        results = merge(results, expr1, by.x = 'Var1', by.y = 'row.names', all.x = T)
        results = merge(results, expr1, by.x = 'Var2', by.y = 'row.names', all.x = T)
        results$diff_expr = as.numeric(results$V1.x) - as.numeric(results$V1.y)
        boxplot(diff_expr~value, results)
        allresults = rbind(allresults, results[results])
    }
}  # partially run, ~97 genes

# obtain gene expression


par(mfrow = c(3,2), mar = c(4,4,3,1))
for(gn in gns$V1)
{
    ism = read.csv(paste0('results/CREAM_ISM/Whole_BloodModels/MultiGene/5adchjbf_1/5adchjbf_', gn, '_ContrastMultiAttention_49152bp.csv'))
    ism$variant_id = paste(ism$chrom, ism$pos0+1, ism$ref, ism$alt, sep='_')
    #ism$chrom = sub('chr', '', ism$chrom)
    #ism$chrom = as.numeric(ism$chrom)
    ism$rel_pos = ism$pos0 - ism$region_start - (ism$region_end - ism$region_start) /2

    #lm_res[, list(min(abs(coef)), .N), by = variant_id]
    # selected SNPs 
    coefs = lm_res[gene_name == gn & fold == 0 ] #lm_res[, list(sum(coef)/.N), by = variant_id]
    #coefs$mag = abs(coefs$coef)
    #setorder(coefs, -mag)
    coefs$variant_id = sub('_b38', '', coefs$variant_id)
    if(nrow(coefs) == 0) next

    ism = merge(ism, coefs[, c('variant_id', 'coef')], by = 'variant_id', all.x= T)
    ism$diff_pred = -ism$diff_pred/2

    # check genotype, add allele freq
    genotype = genotype_all[genotype_all$gene_name == gn, ]
    ism$maf = genotype[match(ism$variant_id, genotype$variant_id), 'maf']
    ism$maf[ism$maf > 0.5] = 1-ism$maf[ism$maf > 0.5]
    #plot(ism$maf[!is.na(ism$coef)], ism$coef[!is.na(ism$coef)])
    #plot(ism$maf, ism$diff_pred)

   
   # sig_variants = ism[which(abs(ism$diff_pred) > 0.1 | abs(ism$coef) > 0.1), ]
    expr = t(expr_norm[expr_norm$gene_name == gn, 5:674])


    # # compute correlation of genotypes around sig_variants
    # info = cbind(sig_variants[, c(1:6, 9:12,14:15)],genotype[genotype$variant_id %in% sig_variants$variant_id,c(3,6)])
    # info$id = 1:nrow(info)
    # dat  = genotype[genotype$variant_id %in% sig_variants$variant_id,-1:-7]
    # dat[dat == '1/1'] = 2
    # dat[dat == '0/1'] = 1
    # dat[dat == '0/0'] = 0
    # dat[dat == './.'] = NA
    # dat = t(sapply(dat, as.numeric))
    # colnames(dat) = 1:ncol(dat)
    # ind = which(info$maf > 0.1 & info$maf < 0.9)
    # cors = cor(dat[, ind], use = 'pairwise.complete.obs')
    # ##dat[which(rowSums(dat[,-2]) > 0),]
    # dat= cbind(dat, 'expr' = expr[rownames(dat),1 ])
    # boxplot(dat[,'expr']~dat[,2])

}


gn = 'CDS2'
dat = genotype_all[genotype_all$gene_name == gn, ]
ism = read.csv(paste0('results/CREAM_ISM/Whole_BloodModels/MultiGene/5adchjbf/5adchjbf_', gn, '_ContrastMultiAttention_49152bp.csv'))
dat[which(dat[,'GTEX-T5JW']!=dat[, 'GTEX-XUW1']),'pos1'] - (ism$region_start[1] + ism$region_end[1])/2

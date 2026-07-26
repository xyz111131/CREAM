from torch.utils.data.distributed import DistributedSampler


class EvalDistributedSampler(DistributedSampler):
    def __init__(self, dataset, shuffle=False, **kwargs):
        super().__init__(dataset, shuffle=shuffle, **kwargs)
        self.dataset = dataset
        self.num_total_genes = len(self.dataset.genomic_regions_df) * self.dataset.repeat
        self.num_individuals_per_gene = self.dataset.num_individuals_per_gene
        self.num_samples = self.num_total_genes * self.num_individuals_per_gene * self.dataset.n_gene_replicate_batches_per_epoch

        # self.n_gene_replicate_batches_per_epoch_per_replica = (
        #     self.dataset.n_gene_replicate_batches_per_epoch
        # )

        # self.genes_per_replica = (
        #     self.num_total_genes // self.num_replicas
        # )  # this calculcates the number of complete groups per GPU. Any extras are ignored
        # self.genes_per_replica = (
        #     self.genes_per_replica * self.n_gene_replicate_batches_per_epoch_per_replica
        # )  # With gradient accumulation, the same gene is repeated in different batches until the effective batch size is achieved. Then, the same thing could repeat multiple times for the same genes if there are enough people. Thus, if there are 3 genes but enough ppl to achieve the effective batch size 4x, it is equiavlent to ther ebeing 12 genes
        # self.num_samples_per_replica = self.genes_per_replica * self.num_individuals_per_gene

        print(f"ValidationDistributedSampler: # of samples: {self.num_samples}")

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples

from torch.utils.data.distributed import DistributedSampler


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, shuffle=False, **kwargs):
        super().__init__(dataset, shuffle=shuffle, **kwargs)
        self.dataset = dataset
        self.num_total_genes = len(self.dataset.genomic_regions_df) 
        self.num_individuals_per_gene = self.dataset.num_individuals_per_gene * self.dataset.repeat
        self.n_gene_replicate_batches_per_epoch_per_replica = (
            self.dataset.n_gene_replicate_batches_per_epoch
        )

        self.genes_per_replica = (
            self.num_total_genes // self.num_replicas
        )  # this calculcates the number of complete groups per GPU. Any extras are ignored
        self.genes_per_replica = (
            self.genes_per_replica * self.n_gene_replicate_batches_per_epoch_per_replica
        )  # With gradient accumulation, the same gene is repeated in different batches until the effective batch size is achieved. Then, the same thing could repeat multiple times for the same genes if there are enough people. Thus, if there are 3 genes but enough ppl to achieve the effective batch size 4x, it is equiavlent to ther ebeing 12 genes
        self.num_samples_per_replica = self.genes_per_replica * self.num_individuals_per_gene 

        print(
            f"Effective genes_per_replica: {self.genes_per_replica}",
            f"num_individuals_per_gene: {self.num_individuals_per_gene}",
            dataset,
        )

        print(f"CustomDistributedSampler Expected # of devices: {self.num_replicas}")

    def __iter__(self):
        """
        This is called once per epoch and is not shuffled, because the order of genes and people will be shuffled by the dataset object at each epoch. Instead, these
        indices are kept the same to ensure the data ordering by the dataset is maintained, and each batch only contains samples of one gene on each gpu

        How this works: Suppose I have 937 genes, and want each GPU to be trained using data from 6 people for the same gene per batch. And I have 4 GPUs.
        Each GPU will be assigned 937 //4 = 234 genes. One gene will be ignored.
        For a GPU 0 will be assigned indices 0 to 1403, GPU1 will get 1404 to 1807 etc Because:
        For GPU 0 the start group will be 0 * 234 = 0 and the end group will be 0 + 234 = 234. The indices will be extended by 6 for each element from 0->234, yielding a list that spans 0 to 1403 (inclusive)
        For GPU 1 the start group will be 234 and the end group will be 468. And so forth
        The final GPU will span 4212 to 5615 (inclusive) because the last gene is dropped, since there is not 3 other genes (one per GPU)
        Then, each batch, indices 0-6 will be sampled, then 7-12 and so on. The dataset is organized such that this yields samples for a given gene.


        if use_all_ppl is True, and dataset.n_gene_replicate_batches_per_epoch >1, it is possible that some genes won't see all people in the batch (But they will always see a multiple of num_individuals_per_gene).
        This is beacuse, instead of certain genes being ommitted to make things even across the replicas, certain gradient accumulated batches of length (num_individuals_per_gene) for certain genes may be ommitted instead.

        If use all_pppl is true then the same gene may appear on different GPUs, but only as different gradient accumulated effective batches
        """
        # Start and end index of groups for this replica
        start_group = self.rank * self.genes_per_replica
        end_group = start_group + self.genes_per_replica

        # indices that will only include genes (and people for those genes) that are meant to be run by one GPU. Ensures you only get one gene per batch for a GPU
        indices = []
        for group in range(start_group, end_group):
            group_start_idx = group * self.num_individuals_per_gene
            indices.extend(
                range(group_start_idx, group_start_idx + self.num_individuals_per_gene)
            )  # add next self.num_individuals_per_gene indices until you reach the end of the group

        assert (
            len(indices) == self.num_samples_per_replica
        ), f"Length of indices per GPU ({len(indices)}) is not the same as the number of Genes x number of people ({self.num_samples_per_replica}) per gene assigned to the GPU"
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica

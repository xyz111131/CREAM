from CREAM.datasets.gtex_dataset0 import GTExDataset


import numpy as np


import os


class ROSMAPDataset(GTExDataset):
    """
    Overwrite GTExDataset to account for different location of consensus fasta files for ROSMAP WGS, and the different structure of the expression data
    """

    def __init__(
        self,
        requested_regions: list,
        desired_seq_len: int,
        num_individuals_per_gene: int,
        donor_list_path: str,
        gene_expression_df: "pandas.DataFrame",
        DATA_DIR: str,
    ) -> None:
        """
        Args:
            requested_regions (list): List of desired genes.
            desired_seq_len (int): Desired length of input DNA sequences. These sequences will be TSS-centered, and this controls how big of a region around the gene's TSS will be used.
            num_individuals_per_gene (int): Number of people assigned to a gene for one effective gradient-accumulated batch. This is distinct from batch size and there can be more than one effective batch per gene. For example, if train batch size is 8 and num_individuals_per_gene is 128, then the next 128 // 8 = 16 consecutive batches will include the same gene and 128 people will be assigned to that gene for training during those 16 batches. If there are 512 total people with data for a given tissue, then there are 512 / 128 = 4 gradient accumulated batches for each gene per epoch, and these batches need not be consecutive (they occur randomly)
            donor_list_path (str): Path to a line-separated txt file denoting the GTEx donor IDs for use during training. There are multiple train/validation/test files for different cross validation splits.
            gene_expression_df (pandas.DataFrame): DataFrame containing gene expression data for the same single tissue as in tissues_to_train.
            DATA_DIR (str): Directory for data storage.
        """
        self.tissues_to_train = ["DLPFC"]  # dorsolateral prefrontal cortex
        super().__init__(
            self.tissues_to_train,
            requested_regions,
            desired_seq_len,
            num_individuals_per_gene,
            donor_list_path,
            gene_expression_df,
            DATA_DIR,
        )
        self.consensus_seq_dir = os.path.join(self.DATA_DIR, "ROSMAPConsensusSeqs_SNPsOnlyUnphased")

    def generate_train_batch_one_gene(
        self, sampled_individual, region_chr, region_start, region_end, gene_name
    ):
        """
        Overwrite to change to use self._get_single_ROSMAP_donor_expression
        self._get_single_GTEx_donor_sequence will work for ROSMAP sequences.
        """

        if (
            self.desired_seq_len != 196608
        ):  # if using shorter seq len, redefine start and end of region while keeping site of gene TSS centered
            region_center = region_end - (196608 // 2)
            region_start = region_center - (self.desired_seq_len // 2)
            region_end = region_center + (self.desired_seq_len // 2)

        gene_name = gene_name.split(
            "/"
        )  # gene_name can be a list of multiple genes overlap TSS bin in the embedding. Splitting converts to a list of 1 gene if none overlap, or multiple genes
        dna_seq = self._get_single_GTEx_donor_sequence(
            sampled_individual, region_chr, region_start, region_end
        )

        gene_expression = self._get_single_ROSMAP_donor_expression(sampled_individual, gene_name)

        return dna_seq, gene_expression

    def get_path_to_consensus_seq(self, donor_id, haplotype_num):
        """Consensus sequences have different naming scheme between GTEx and ROSMAP. Overwrite to accomodate this."""
        return os.path.join(self.consensus_seq_dir, f"{donor_id}_H{haplotype_num}.fa")

    def _get_single_ROSMAP_donor_expression(self, donor_id, gene_name):
        """
        donor_id (str): ROSMAP donor ID
        gene_name (list): Gene to get expression from, in the form of a list of length 1. In the scenario where multiple genes align to the same bin, multiple genes are passed in and their summed expression is used
        """
        gene_expression_vector = np.zeros(
            (1), dtype=np.float32
        )  # the only element in the vector will be expression in the Brain Cortex tissue. No other tissues.

        indiv_tpm = self.gene_expression_df.loc[gene_name, donor_id]
        assert (
            indiv_tpm.shape[0] > 0
        ), f"There is no gene expression data available for gene(s) at position {gene_name}"
        gene_expression_vector[0] = indiv_tpm.sum(min_count=1)
        return gene_expression_vector

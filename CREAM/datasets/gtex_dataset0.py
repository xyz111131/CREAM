import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import kipoiseq
import random
import pysam
from lightning.pytorch import LightningDataModule
from CREAM.datasets.samplers.custom_distributed_sampler import CustomDistributedSampler
from CREAM.datasets.samplers.validation_distributed_sampler import EvalDistributedSampler
from CREAM import config as cream_config


def resolve_dataset_paths(paths):
    """Accepts the dict returned by ``cream_config.dataset_paths()``, or a bare
    data directory (older call sites), which is filled in from the defaults."""
    if isinstance(paths, dict):
        return paths
    return cream_config.dataset_paths(cream_config.load_config(paths={"data_dir": str(paths)}))


class CustomDataModule(LightningDataModule):
    def __init__(self, train_dataset, valid_dataset, test_dataset, config):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = config.train_batch_size
        self.num_gpus = config.num_gpus

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        print(f"world size: {self.trainer.world_size}")

        if self.num_gpus > 1:
            sampler =  CustomDistributedSampler(self.train_dataset,shuffle=False)
        else:
            sampler = None

        return DataLoader(self.train_dataset, batch_size= self.train_batch_size, num_workers = 1, shuffle=False, sampler=sampler)

    def val_dataloader(self):
                
        if self.num_gpus > 1:
            sampler =  CustomDistributedSampler(self.valid_dataset, shuffle=False)
        else:
            sampler = None

        return DataLoader(self.valid_dataset, batch_size= self.train_batch_size, shuffle=False,num_workers = 1, sampler=sampler)

    def test_dataloader(self):
                
        if self.num_gpus > 1:
            sampler =  EvalDistributedSampler(self.test_dataset, shuffle=False)
        else:
            sampler = None

        return DataLoader(self.test_dataset, batch_size= self.train_batch_size, shuffle=False,num_workers = 1, sampler=sampler)


class GTExDataset(Dataset):
    def __init__(
        self,
        tissues_to_train: list,
        requested_regions: list,
        desired_seq_len: int,
        shift: int,
        shift_freq: float, # not used
        rc: bool,
        rc_freq: float,
        mask_dna_prop: float,
        discretize_expr: int,
        stratify_expr: int,  # batch size or 0
        batch_size: int,
        num_individuals_per_gene: int,
        donor_list_path: str,
        repeat: int,
        gene_expression_df: "pandas.DataFrame",
        true_eQTLs: dict,
        paths: dict,
    ) -> None:
        """
        Args:
            tissues_to_train (list): List of GTEx tissues to be trained.
            requested_regions (list): List of desired geneset (list).
            desired_seq_len (int): Desired length of input DNA sequences. These sequences will be TSS-centered, and this controls how big of a region around the gene's TSS will be used.
            num_individuals_per_gene (int): Number of people assigned to a gene for one effective gradient-accumulated batch. This is distinct from batch size and there can be more than one effective batch per gene. For example, if train batch size is 8 and num_individuals_per_gene is 128, then the next 128 // 8 = 16 consecutive batches will include the same gene and 128 people will be assigned to that gene for training during those 16 batches. If there are 512 total people with data for a given tissue, then there are 512 / 128 = 4 gradient accumulated batches for each gene per epoch, and these batches need not be consecutive (they occur randomly)
            donor_list_path (str): Path to a line-separated txt file denoting the GTEx donor IDs for use during training. There are multiple train/validation/test files for different cross validation splits.
            gene_expression_df (pandas.DataFrame): DataFrame containing gene expression data for the same single tissue as in tissues_to_train.
            true_eQTLS (dict): dictionary of true eQTLs in different tissues.
            paths (dict): Resolved data paths from CREAM.config.dataset_paths().
        """

        assert type(requested_regions) == list
        assert type(tissues_to_train) == list
        #assert len(tissues_to_train) == 1, "Only single tissue training is currently supported"

        paths = resolve_dataset_paths(paths)
        self.paths = paths
        self.DATA_DIR = paths["data_dir"]
        self.consensus_seq_filename = paths["consensus_seq_filename"]
        self.desired_seq_len = desired_seq_len
        self.max_shift = shift
        self.shift_freq = shift_freq
        self.reverse_complement = rc
        self.rc_freq = rc_freq
        self.donor_list_path = donor_list_path
        self.gene_expression_df = gene_expression_df
        self.true_eQTLs = true_eQTLs
        self.requested_regions = []
        self.mask_dna_prop = mask_dna_prop
        self.discretize_expr = discretize_expr
        # self.expr_bins = np.array([0.2, 1, 2, 3]) # hard coded for now. [-1, -0.3, 0.3, 1]
        # self.expr_bins_means = np.array([0.1, 0.5, 1.5, 2.5, 4.2]) # hard coded for now.  [-1.5, -0.62, 0, 0.62, 1.5]
        self.stratify_expr = stratify_expr
        self.batch_size = batch_size
        self.repeat = repeat

        if type(tissues_to_train) == str:
            self.tissues_to_train = [tissues_to_train]
        else:
            self.tissues_to_train = tissues_to_train

        # define genomic regions to be used
        self.all_regions = pd.read_csv(
            paths["genomic_intervals_file"],
            sep=",",  # "\t"
        )
        self.all_regions['gene_name'] = self.all_regions['gene_id'].str.replace(r'\.\d+$', '', regex=True)
        # all_requested_regions = set().union(*requested_regions)
        # assert all(g in self.all_regions['gene_name'] for g in all_requested_regions).values, "not all requested genes are in genomic region file"

        # check if all requested regions (genes) are in all_regions
        # all genes in the expression files
        all_expressed_genes = set()
        for df in self.gene_expression_df.values():
            all_expressed_genes |= set(df['Description'])
        
        all_genes = set(self.all_regions["gene_name"]) & all_expressed_genes # also intersect with genes with expression values

        for gs in requested_regions:
            gss = list(all_genes.intersection(gs))
            if len(gss) > 0:
                self.requested_regions.append(gss)
        print("number of gene sets after filtering", len(self.requested_regions))
        all_requested_regions = set().union(*self.requested_regions)

        self.genomic_regions_df = self.all_regions[
            self.all_regions["gene_name"].isin(all_requested_regions)
        ]  # h: some gene_names in genomic_regions.csv have '/' for two genes in the same TSS bin, but none of the train/test genes have '/'.
        self.genomic_regions_df = self.genomic_regions_df.reset_index(drop=True)
        self.genes_in_dataset = list(self.genomic_regions_df["gene_name"].unique())

        # select donors that have WGS & RNA-seq in the desired tissue(s) and that are meant to be used in the current CV split
        self.individuals_in_split = self.select_donors()

        assert num_individuals_per_gene == -1 or repeat == 1

        if (
            num_individuals_per_gene == -1
        ):  # EvalGTExDataset overwrites this class and sets this to -1 to use all people. Whereas during training,
            self.num_individuals_per_gene = (
                len(self.individuals_in_split) // self.batch_size
            ) * self.batch_size  # len(self.individuals_in_split)
        else:
            self.num_individuals_per_gene = num_individuals_per_gene

        # an epoch is a pass not only through all genes with one gradient accumulated step per gene (equal to num_individuals_per_gene forward steps),
        # but multiple steps per gene will be taken. As many steps will be taken as possible, given the number of accumulations desired
        self.n_gene_replicate_batches_per_epoch = (
            len(self.individuals_in_split) // self.num_individuals_per_gene
        )  # how many times each gene will appear in a batch per epoch,

        if num_individuals_per_gene == -1:
            assert self.n_gene_replicate_batches_per_epoch == 1
        self.consensus_seq_dir = paths["consensus_seq_dir"]

        self.shuffle_and_define_epoch()

    def _get_gene_and_individual_from_idx(self, idx):
        # Determines which gene to use. If the number of individuals per gene is 10 and the idx is 0, pick the 0th gene
        # if it is 10 pick the 1st gene and so on
        region_idx = idx // (self.num_individuals_per_gene * self.repeat)
        region_row = self.region_rows_in_epoch[region_idx]

        # Determines which person to use. First indexes the list of donors that match the current gene by indexing the gene.
        # Then the modulus selects the correct individual. If there are 10 individuals per gene and the idx is 0, the modulus is 0 so pick the 0th individual
        # if the idx is 10, the gene idx is 1 and the modulus is 0, so pick the 0th individual fror the 1st gene. If the idx is 11, pick the 1st index individual for the 1st gene.
        individual_idx = idx % (self.num_individuals_per_gene * self.repeat)
        individual = self.indivs_per_epoch[region_idx][individual_idx]

        shift = (
            self.shifts_per_epoch[region_idx][individual_idx // self.batch_size]
            if self.max_shift > 0
            else None
        )
        rc = (
            self.rc_per_epoch[region_idx][individual_idx // self.batch_size]
            if self.reverse_complement
            else None
        )

        return region_row, individual, shift, rc

    def select_donors(self):
        """
        Returns donors from the desired Cross-Validation Split (defined by donor_list_path) that include gene expression data for the desired tissue.
        Currently this only supports single tissue training
        """
        #assert len(self.tissues_to_train) == 1, "Multi Tissue Training is not supported yet."

        #gene_expression_df_cols = list(self.gene_expression_df[self.tissues_to_train[0]].columns) # need to change to either union or intersect if training with real data
        gene_expression_df_cols = list(set.intersection(*[set(df.columns) for df in self.gene_expression_df.values()]))
        with open(self.donor_list_path, "r") as f:
            file = f.read()
            GTEx_data_split_IDs = file.split("\n")
            GTEx_data_split_IDs = [
                ID for ID in GTEx_data_split_IDs if ID != ""
            ]  # remove any empty strings, if they exist

        donors_for_tissue = sorted(
            list(set([x for x in gene_expression_df_cols if x in GTEx_data_split_IDs]))
        )  # expr columns in this gene expression matrix are donor ids. Select desired donors with expression data
        return donors_for_tissue

    @staticmethod
    def _one_hot_encode(sequence):
        ## one hot encodes DNA using the same code from the original Enformer paper. Ensures one-hot encoding is consistent with representations Enformer has already learned for more efficient transfer learning
        return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

    @staticmethod
    def _reverse_complement_seq(seq):
        """
        Reverse complements the sequence while reversing the location of the gene expression signal.
        Citation: Geeksforgeeks.com
        """
        # First complement the sequence.
        seq = seq.replace("A", "t").replace("C", "g").replace("T", "a").replace("G", "c")
        seq = seq.upper()
        # reverse the strand
        rc_seq = seq[::-1]
        return rc_seq

    def _one_hot_encode_diploid(self, seq1, seq2):
        """
        Returns a single one hot encoded sequence from an unphased diploid genome in order to pass in as input into the model.
        It does this by taking two haplotypes from a diploid genome, one hot encoding each, and taking the average.
        Heterozygous positions are therefore encoded as 0.5s.
        This is only appropriate for unphased sequences without indels. It should be changed for phased genomes since heterozygous SNPs can be mapped to one haplotype or the other in that case.
        """
        one_hot_seq1 = self._one_hot_encode(seq1)
        one_hot_seq2 = self._one_hot_encode(seq2)
        return (one_hot_seq1 + one_hot_seq2) / 2

    def get_path_to_consensus_seq(self, donor_id, haplotype_num):
        return os.path.join(
            self.consensus_seq_dir,
            self.consensus_seq_filename.format(donor_id=donor_id, haplotype=haplotype_num),
        )

    def _get_single_GTEx_donor_sequence(self, gtex_id, region_chr, region_start, region_end, rc=0):
        consensus1_open = pysam.Fastafile(self.get_path_to_consensus_seq(gtex_id, 1))
        consensus2_open = pysam.Fastafile(self.get_path_to_consensus_seq(gtex_id, 2))
        seq1 = consensus1_open.fetch(region_chr, region_start, region_end).upper()
        assert (
            len(seq1) == self.desired_seq_len
        ), f"Seq1 should be length {self.desired_seq_len}. Offending region: {region_chr}:{region_start}-{region_end}"
        seq2 = consensus2_open.fetch(region_chr, region_start, region_end).upper()
        assert (
            len(seq2) == self.desired_seq_len
        ), f"Seq2 should be length {self.desired_seq_len}. Offending region: {region_chr}:{region_start}-{region_end}"
        consensus1_open.close()
        consensus2_open.close()

        if rc == 1:  # get rc for both alleles, so the two alleles are similar except SNPs
            seq1 = self._reverse_complement_seq(seq1)
            seq2 = self._reverse_complement_seq(seq2)

        diploid_one_hot = self._one_hot_encode_diploid(
            seq1, seq2
        )  # one hot encode a diploid DNA sequence. Heterozygosity ==> 0.5/0.5
        return diploid_one_hot

    def _get_single_GTEx_donor_expression(self, gtex_id, gene_name):
        """
        gtex_id (str): GTEx donor ID
        gene_name (list): Gene to get expression from, in the form of a list of length 1. In the scenario where multiple genes align to the same bin, multiple genes are passed in and their summed expression is used
        """
        if self.discretize_expr > 0:
            gene_expression_vector = np.zeros(
                (len(self.tissues_to_train)), dtype=np.float32
            )  # long
        else:
            gene_expression_vector = np.zeros((len(self.tissues_to_train)), dtype=np.float32)
        # indiv_tpm = self.gene_expression_df[['#chr', 'start', 'gene_id','Description', gtex_id]]
       
        for tissue_idx, tissue in enumerate(self.tissues_to_train):
            indiv_tpm = self.gene_expression_df[tissue][["Description", gtex_id]]
            indiv_tpm = indiv_tpm[indiv_tpm["Description"].isin(gene_name)]
            # assert (
            #     indiv_tpm.shape[0] > 0
            # ), f"There is no gene expression data available for gene(s) at position {gene_name}"
            gene_expression_value = indiv_tpm[gtex_id].sum(min_count=0)
            if self.discretize_expr > 0:
                # gene_expression_vector[tissue_idx] = np.digitize(gene_expression_value, self.expr_bins)
                # offsets = gene_expression_value - self.expr_bins_means[bins]
                # gene_expression_vector[tissue_idx] = np.concatenate([bins, offsets])
                gene_expression_vector[tissue_idx] = gene_expression_value
            else:
                gene_expression_vector[tissue_idx] = gene_expression_value
            
        # if gene_name[0] == 'ENSG00000234997':
        #     print (1)

        # assert (
        #     not np.all(gene_expression_vector == 0)
        #     ), f"There is no gene expression data available for gene(s) at position {gene_name}"
        return gene_expression_vector

    def _get_multiple_GTEx_donor_expression(self, gtex_ids, gene_name):  # only single tissue
        """
        gtex_ids (list): GTEx donor ID
        gene_name (list): Gene to get expression from, in the form of a list of length 1. In the scenario where multiple genes align to the same bin, multiple genes are passed in and their summed expression is used
        """
        # if self.discretize_expr > 0:
        #     gene_expression_vector = np.zeros((len(gtex_ids)), dtype=np.long)
        # else:
        gene_expression_vector = np.zeros((len(gtex_ids)), dtype=np.float32)
        # indiv_tpm = self.gene_expression_df[['#chr', 'start', 'gene_id','Description']+ gtex_ids]
        indiv_tpm = self.gene_expression_df[["Description"] + gtex_ids]
        indiv_tpm = indiv_tpm[indiv_tpm["Description"].isin(gene_name)]
        assert (
            indiv_tpm.shape[0] > 0
        ), f"There is no gene expression data available for gene(s) at position {gene_name}"
        gene_expression_value = indiv_tpm[gtex_ids].sum(min_count=1)
        # if self.discretize_expr > 0:
        #     gene_expression_vector = np.digitize(gene_expression_value, self.expr_bins)
        # else:
        gene_expression_vector = gene_expression_value.tolist()
        return gene_expression_vector
    
    def _get_gene_eQTL(self, gene_name, region_start, region_chr):
        
        eQTL_vector = np.zeros((len(self.tissues_to_train), self.desired_seq_len), dtype=np.float32)
        # indiv_tpm = self.gene_expression_df[['#chr', 'start', 'gene_id','Description', gtex_id]]
        for tissue_idx, tissue in enumerate(self.tissues_to_train):
            eQTL = self.true_eQTLs[tissue]
            eQTL = eQTL[eQTL['gene_id'].isin(gene_name)]
            eQTL = eQTL[eQTL['chrom'] == region_chr]

            # a gene with no credible set in this tissue keeps an all-zero track; the TSS
            # bump in LitModel.training_step then makes the target TSS-only attention
            # compute index into the sequence window from the 0-based genomic position
            indices = (eQTL['pos0'] - region_start).astype(int)
            # keep only variants that fall within the sequence window
            in_window = (indices >= 0) & (indices < self.desired_seq_len)
            # group by window index and take max pip to handle multiple variants at the same position
            max_pip = (
                eQTL.loc[in_window, 'pip']
                .groupby(indices[in_window])
                .max()
            )
            eQTL_vector[tissue_idx, max_pip.index.values] = max_pip.values

        return eQTL_vector

    def _get_gene_embedding(self, gene_name):
        gene_embedding = self.gene_embedding_df.loc[
            self.gene_embedding_df["Description"].isin(gene_name)
        ].iloc[
            :, 0:256
        ]  # remove desciprtion, or use drop
        if gene_embedding.shape[0] < 1:
            # print(gene_name)
            gene_embedding = np.zeros(256)
        else:
            gene_embedding = gene_embedding.mean().values
        return gene_embedding.astype(np.float32)

    def generate_train_batch_one_gene(
        self,
        sampled_individual,
        region_chr,
        region_center,
        region_start,
        region_end,
        gene_name,
        snps_loc,
        shift,
        rc,
    ):
        # If the desired sequence length is not Enformer's original 196kb, it resets the start and end positions by trimming the ends as necessary
        # if self.desired_seq_len != 196608:
        if self.desired_seq_len != 49152:
            #region_center = region_end - (49152 // 2) # potentially wrong if hit boundary of the chromsome
            region_start = region_center - (self.desired_seq_len // 2)
            region_end = region_center + (self.desired_seq_len // 2)

        assert type(gene_name) == str, f"Not a string: {gene_name}"

        gene_name = gene_name.split(
            "/"
        )  # gene_name can be a list of multiple genes overlap TSS bin in the embedding. Splitting converts to a list of 1 gene if none overlap, or multiple genes
        if self.max_shift > 0:
            # shift = np.random.randint(-self.shift, self.shift+1)
            region_start = region_start + shift
            region_end = region_end + shift
        if region_start < 0:
            region_start = 0
        dna_seq = self._get_single_GTEx_donor_sequence(
            sampled_individual, region_chr, region_start, region_end, rc
        )
        tss = (region_center - region_start) / self.desired_seq_len # tss proportion, can be not 1/2 if either shifting or boundary

        eQTL_pips = None
        if self.true_eQTLs:
            eQTL_pips = self._get_gene_eQTL(gene_name, region_start, region_chr)

        if self.mask_dna_prop > 0:
            snps_loc = [int(x) for x in snps_loc.split(",")]
            snps_loc = snps_loc - region_start
            snps_loc = snps_loc[snps_loc >= 0]
            snps_loc = snps_loc[snps_loc < self.desired_seq_len]
            masks = np.random.binomial(1, self.mask_dna_prop, self.desired_seq_len)  # to mask
            # extends snps -15 to 15
            for i in snps_loc:
                for j in np.arange(-14, 15):
                    if i + j >= 0 and i + j < len(masks):
                        masks[i + j] = 0
            # masks[snps_loc] = 0 # snps position not mask
            dna_seq[masks == 1, :] = 0
        gene_expression = self._get_single_GTEx_donor_expression(sampled_individual, gene_name)
        # gene_embd = self._get_gene_embedding(gene_name)
        return dna_seq, gene_expression, eQTL_pips, tss  # gene_embd,

    def shuffle_and_define_epoch(self):
        """
        This method shuffles the dataset in a way that ensures each batch contains only one gene, and the same gene appears in consecutive batches until the number of desired batches for gradient accumulation has been satisfied.
        If there are enough individuals for multiple full gradient-accumulated batches, it repeats this multiple times, while allowing fo those gradient-accumulated batches to be separated

        For example, if num_individuals_per_gene is 128 and there are 550 people with data for the desired tissue, each gradient accumulated batch will include 128 people for the same gene. Thus, data from 128 different random people and the same gene will appear consecutively. Then this will continue for another random gene.
        Each gene will appear in 3 more gradient accumuated batches, appearing with 512 total random individuals, and the remaining individuals will be dropped this epoch.
        """

        # self.genomic_regions_df = self.genomic_regions_df.sample(frac=1).reset_index(drop=True) #shuffle dataset. Keep this here so the dataset shuffling within litmodel.on_train_epoch_end persists  # q:?
        random.shuffle(
            self.requested_regions
        )  # q: why do we need this? since genes are gonna to be shuffled in the last
        self.shifts_per_epoch = []
        self.rc_per_epoch = []
        self.indivs_per_epoch = (
            []
        )  # This will be a list of lists. Each inner list will contain list of donors paired to each gene in a gradient accumulated effective batch. There will be as many inner lists as gradient accumulated batches per epoch
        self.region_rows_in_epoch = (
            []
        )  # will contain order of gene sets. True batches will yield the same gene until the accumulated batch ends, then the next gene will be the next element in this list

        for i in range(0, len(self.requested_regions), 1):
            indivs_for_gene_all = []
            for r in range(
                self.repeat
            ):  # either self.repeat == 1 or self.n_gene_replicate_batches_per_epoch == 1
                indivs_for_gene = random.sample(
                    self.individuals_in_split,
                    self.num_individuals_per_gene * self.n_gene_replicate_batches_per_epoch,
                )  # randomly sample ppl for this gene. If n_gene_replicate_batches_per_epoch == 1, you get enough people for one gradient accumulated batch. If this value is 4, the # of people will be equal to 4x the length of one gradient accumulated batch
                if (
                    self.stratify_expr > 1
                ):  # only works for single gene (not gene set), single tissue
                    gene_name = self.requested_regions[i]
                    gene_expression_values = self._get_multiple_GTEx_donor_expression(
                        indivs_for_gene, gene_name
                    )
                    indivs_for_gene = np.array(indivs_for_gene)
                    indivs_for_gene = indivs_for_gene[np.argsort(gene_expression_values)]
                    y = len(indivs_for_gene) // self.stratify_expr  # need to be dividable
                    shuffle_indivs_in_split = [
                        np.random.permutation(y) for x in range(self.stratify_expr)
                    ]
                    indivs_for_gene = indivs_for_gene[
                        np.argsort(np.concatenate(shuffle_indivs_in_split))
                    ]
                    # shuffles samples within batch, for contrast learing each consecutive pairs
                    split_indivs_for_gene = np.array_split(indivs_for_gene, y)
                    split_indivs_for_gene = [
                        np.random.permutation(x) for x in split_indivs_for_gene
                    ]
                    indivs_for_gene = np.concatenate(split_indivs_for_gene, axis=0)
                indivs_for_gene_all.extend(indivs_for_gene)

            split_indivs_for_gene = np.array_split(
                indivs_for_gene_all, self.n_gene_replicate_batches_per_epoch
            )  # split indivs_for_gene into different arrays. They are split so the same gene doesn't need to appear in consecutive accumulated batches. The arrays will be equal length (equal to the length of a gradient accumulated batch, self.num_individuals_per_gene)
            for j in range(
                self.n_gene_replicate_batches_per_epoch
            ):  # Create a new accumulated batch for each gene if n_gene_replicate_batches_per_epoch >1. Else if n_gene_replicate_batches_per_epoch == 1, you get just one accumulated batch per gene
                self.region_rows_in_epoch.append(self.requested_regions[i])
                self.indivs_per_epoch.append(list(split_indivs_for_gene[j]))

        # shuffle order of genes (and match the assigned people), so each accumulated batch doesn't have the same genes back to back
        # Since region_rows_in_epoch will only iterate to the next element (likewise for indivs_per_epoch since it is a nested list) after an accumulated batch is complete
        # this won't mix genes per accumualted batch or mini batch
        shuffled_idxs = np.random.permutation(len(self.region_rows_in_epoch))
        self.region_rows_in_epoch = [self.region_rows_in_epoch[i] for i in shuffled_idxs]
        self.indivs_per_epoch = [self.indivs_per_epoch[i] for i in shuffled_idxs]

        if self.max_shift > 0:
            for i in range(len(self.region_rows_in_epoch)):
                ng = len(self.region_rows_in_epoch[i])  # number of genes in gene set?? ng == 1?
                self.shifts_per_epoch.append(list())
                for j in range(len(self.indivs_per_epoch[i])):
                    # to_shift = np.random.binomial(n=1, p=self.shift_freq, size = ng)
                    shifts = np.random.randint(-self.max_shift, self.max_shift + 1, ng)
                    self.shifts_per_epoch[i].append(
                        list(shifts ) #* to_shift
                    )

        if self.reverse_complement:
            for i in range(len(self.region_rows_in_epoch)):
                ng = len(self.region_rows_in_epoch[i])  # number of genes
                self.rc_per_epoch.append(list())
                for j in range(len(self.indivs_per_epoch[i])):
                    self.rc_per_epoch[i].append(list(np.random.randint(0, 2, ng)))
                    # self.rc_per_epoch[i].append(list(np.random.binomial(n=1, p = self.rc_freq, size = ng)))

    def __getitem__(self, idx):

        batch = {}
        region_row, individual, shift, rc = self._get_gene_and_individual_from_idx(
            idx
        )  # region_row is a gene set
        dna_vector_list = []
        expression_vector_list = []
        eQTL_vector_list = []
        tss_list = []
        # if self.max_shift > 0 and idx % self.train_batch_size == 0:
        #     self.current_shift = np.random.randint(-self.max_shift, self.max_shift+1)
        for ix, gn in enumerate(region_row):
            other_region_row = self.genomic_regions_df.loc[
                self.genomic_regions_df["gene_name"] == gn
            ].squeeze()

            assert not other_region_row.empty, f"Empty row: {gn}"

            dna_vector, expression_vector, eQTL_array, tss = (  # embedding_vector,
                self.generate_train_batch_one_gene(
                    individual,
                    other_region_row["seqnames"],
                    other_region_row["gene_start"],
                    other_region_row["starts"],
                    other_region_row["ends"],
                    other_region_row["gene_name"],
                    "",  # other_region_row["SNPs"],
                    shift[ix] if shift != None else None,
                    rc[ix] if rc != None else 0,
                )
            )
            dna_vector_list.append(dna_vector)
            expression_vector_list.append(expression_vector)  # (n_tissue,), keep the axis when there is one tissue
            if eQTL_array is not None:  # provided only when true_eQTLs are enabled
                eQTL_vector_list.append(eQTL_array) # seq_len * number of tissue
            tss_list.append(tss)
        batch["seq_array"] = np.stack(dna_vector_list, axis=0)
        batch["expr_array"] = np.stack(expression_vector_list, axis=0)
        if self.true_eQTLs:
            batch["eQTL_array"] = np.stack(eQTL_vector_list, axis=0)
            
        batch["gene_names"] = region_row
        batch["individual"] = individual
        batch["idx"] = idx
        batch["tss"] = np.stack(tss_list, axis=0)
        return batch

    def __len__(self):
        """ """
        return (
            len(self.requested_regions)
            * self.num_individuals_per_gene
            * self.n_gene_replicate_batches_per_epoch
            * self.repeat
        )


if __name__ == "__main__":
    pass

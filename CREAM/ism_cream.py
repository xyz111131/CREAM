# adapted from Performer: https://github.com/shirondru/enformer_fine_tuning/tree/master/code 

import gc
import os

#os.environ["CUDA_VISIBLE_DEVICES"] ="1"

import vcfpy
import pysam
import torch
import argparse
import kipoiseq
import pandas as pd
import numpy as np
import random
import lightning.pytorch as pl
from torch.utils.data import IterableDataset, DataLoader
from CREAM.models.contrast_wrapper_attention_multiheads import ContrastWrapperAttention
from CREAM import config as cream_config
import yaml


torch.use_deterministic_algorithms(True)


def get_window_around_TSS(window, gene_info):
    """
    Returns a string denoting the start and end position of a region centered around the TSS +/- the window on each side.
    start and end position in gene_info already have TSS centered
    """
    gene_start = int(gene_info["gene_start"].item())
    region_start = gene_start - window
    region_end = gene_start + window
    region_chr = gene_info["seqnames"].item()

    region = region_chr + ":" + str(region_start) + "-" + str(region_end)
    assert region_end - region_start == (
        window * 2
    )  # checks the returned region is double the window size. It should be if it includes the window on each end of the TSS

    return region


def get_all_gtex_snps(variant_row, window=10000, config=None):
    """
    Gets all observed GTEx SNPs within +/- window distance from gene TSS and puts into a df with position, ref and alt for ISM
    """
    config = config if config is not None else cream_config.load_config()
    enformer_regions = pd.read_csv(cream_config.enformer_intervals_file(config))
    gene = variant_row["gene_name"]
    samples = []
    if not pd.isna(variant_row["het_donors"]):
        samples.extend(variant_row["het_donors"].split(","))
    if not pd.isna(variant_row["hom_donors"]):
        samples.extend(variant_row["hom_donors"].split(","))
    samples.append("ref")
    gene_info = enformer_regions[enformer_regions["gene_name"] == gene]
    variant_dict = {
        "region": [],
        "chrom": [],
        "pos0": [],
        "pos1": [],
        "ref": [],
        "alt": [],
        "donor": [],
    }
    if gene_info.shape[0] > 0:
        # get all observed SNPs within +/- window of TSS
        region = get_window_around_TSS(window, gene_info)  # define region +/- window of TSS
        for _, sample in enumerate(samples):
            variant_dict["region"].append(region)
            variant_dict["chrom"].append(variant_row["chrom"])
            variant_dict["pos1"].append(variant_row["pos1"])
            variant_dict["pos0"].append(variant_row["pos1"] - 1)
            variant_dict["ref"].append(variant_row["ref"])
            variant_dict["alt"].append(variant_row["alt"])
            variant_dict["gene_name"] = gene
            variant_dict["donor"].append(sample)
        return pd.DataFrame(variant_dict)
    else:
        return pd.DataFrame(variant_dict)  # if gene not in enformer regions return empty dataframe


def get_ckpt(save_dir):
    ckpt = os.listdir(f"{save_dir}/checkpoints")
    if len(ckpt) > 0:
        assert len(ckpt) == 1
        ckpt = ckpt[0]
        return ckpt
    else:
        return None


def get_single_GTEx_donor_sequence(
    gtex_id, region_chr, region_start, region_end, desired_seq_len, config=None
):
    # return both sequence
    # ind = random.randint(0, 1) + 1
    consensus1_open = pysam.Fastafile(get_path_to_consensus_seq(gtex_id, 1, config))
    consensus2_open = pysam.Fastafile(get_path_to_consensus_seq(gtex_id, 2, config))
    seq1 = consensus1_open.fetch(region_chr, region_start, region_end).upper()
    assert (
        len(seq1) == desired_seq_len
    ), f"Seq1 should be length {desired_seq_len}. Offending region: {region_chr}:{region_start}-{region_end}"
    seq2 = consensus2_open.fetch(region_chr, region_start, region_end).upper()
    assert (
        len(seq2) == desired_seq_len
    ), f"Seq2 should be length {desired_seq_len}. Offending region: {region_chr}:{region_start}-{region_end}"
    consensus1_open.close()
    consensus2_open.close()

    # diploid_one_hot = one_hot_encode_diploid(seq1, seq2)  # one hot encode a diploid DNA sequence. Heterozygosity ==> 0.5/0.5
    return seq1, seq2


def get_path_to_consensus_seq(donor_id, haplotype_num, config=None):
    config = config if config is not None else cream_config.load_config()
    return os.path.join(
        cream_config.consensus_seq_dir(config),
        cream_config.consensus_seq_filename(config).format(
            donor_id=donor_id, haplotype=haplotype_num
        ),
    )


def one_hot_encode_diploid(seq1, seq2):
    """
    Returns a single one hot encoded sequence from an unphased diploid genome in order to pass in as input into the model.
    It does this by taking two haplotypes from a diploid genome, one hot encoding each, and taking the average.
    Heterozygous positions are therefore encoded as 0.5s.
    This is only appropriate for unphased sequences without indels. It should be changed for phased genomes since heterozygous SNPs can be mapped to one haplotype or the other in that case.
    """
    one_hot_seq1 = one_hot_encode(seq1)
    one_hot_seq2 = one_hot_encode(seq2)
    return (one_hot_seq1 + one_hot_seq2) / 2


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


# def tss_centered_sequences(variant_df, desired_seq_len):
#     cwd = os.getcwd()
#     DATA_DIR = os.path.join(cwd, "data")
#     gene = variant_df["gene_name"].unique().item()

#     ref_seq_open = pysam.Fastafile(os.path.join(DATA_DIR, "hg38_genome.fa"))
#     enformer_regions = pd.read_csv(
#         os.path.join(DATA_DIR, "Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv")
#     )
#     gene_info = enformer_regions[enformer_regions["gene_name"] == gene]
#     seq_window = (
#         desired_seq_len // 2
#     )  # the sequence used for ism has length seq_window * 2 because it is before and after the TSS. Divide by 2 to get a sequence whose length is desired_seq_len
#     region = get_window_around_TSS(seq_window, gene_info)
#     region_chr = region.split(":")[0]
#     region_start = int(region.split(":")[1].split("-")[0])
#     region_end = int(region.split(":")[1].split("-")[1])
#     assert region_end - region_start == desired_seq_len
#     ref_seq = ref_seq_open.fetch(region_chr, region_start, region_end).upper()

#     for _, variant_info in variant_df.iterrows():
#         index = variant_info["pos0"] - region_start
#         assert (
#             ref_seq[index] == variant_info["ref"]
#         ), "nucleotide in reference genome should be identical to the reference allele of the current SNP at the current position"
#         alt_seq = (
#             ref_seq[:index] + variant_info["alt"] + ref_seq[index + 1 :]
#         )  # put alt allele in its correct position and ref seq around it
#         yield {
#             "inputs": {"ref": one_hot_encode(ref_seq), "alt": one_hot_encode(alt_seq)},
#             "metadata": {
#                 "chrom": variant_info.chrom,
#                 "pos": variant_info.pos0,
#                 "ref": variant_info.ref,
#                 "alt": variant_info.alt,
#                 "gene_name": gene,
#                 "region_chr": region_chr,
#                 "region_start": region_start,
#                 "region_end": region_end,
#             },
#         }
#         # yield one hot encoded sequences and metadata


def tss_centered_sample_sequences(
    variant_df, desired_seq_len, config=None
):  # variant_df contains one variants but many samples to score
    config = config if config is not None else cream_config.load_config()
    gene = variant_df["gene_name"].unique().item()

    enformer_regions = pd.read_csv(cream_config.enformer_intervals_file(config))
    gene_info = enformer_regions[enformer_regions["gene_name"] == gene]
    seq_window = (
        desired_seq_len // 2
    )  # the sequence used for ism has length seq_window * 2 because it is before and after the TSS. Divide by 2 to get a sequence whose length is desired_seq_len
    region = get_window_around_TSS(seq_window, gene_info)
    region_chr = region.split(":")[0]
    region_start = int(region.split(":")[1].split("-")[0])
    region_end = int(region.split(":")[1].split("-")[1])
    assert region_end - region_start == desired_seq_len

    for _, variant_info in variant_df.iterrows():  # iterate over samples
        index = variant_info["pos0"] - region_start
        sample = variant_info["donor"]
        if sample == "ref":
            ref_seq_open = pysam.Fastafile(cream_config.genome_fasta(config))
            ref_seq = ref_seq_open.fetch(region_chr, region_start, region_end).upper()
            assert (
                ref_seq[index] == variant_info["ref"]
            ), "nucleotide in reference genome should be identical to the reference allele of the current SNP at the current position"
            alt_seq = (
                ref_seq[:index] + variant_info["alt"] + ref_seq[index + 1 :]
            )  # put alt allele in its correct position and ref seq around it
            ref_seq = one_hot_encode(ref_seq)
            alt_seq = one_hot_encode(alt_seq)
            ref_seq_open.close()
        else:
            donor_seq1, donor_seq2 = get_single_GTEx_donor_sequence(
                sample, region_chr, region_start, region_end, desired_seq_len, config
            )
            ref_seq = one_hot_encode_diploid(donor_seq1, donor_seq2)
            count = 0
            if donor_seq1[index] == variant_info["alt"]:
                donor_seq1 = donor_seq1[:index] + variant_info["ref"] + donor_seq1[index + 1 :]
                count += 1
            else:
                assert (
                    donor_seq1[index] == variant_info["ref"]
                ), "nucleotide in donor genome should be identical to the reference allele of the current SNP at the current position"
            # put ref allele in its correct position and donor seq around it
            if donor_seq2[index] == variant_info["alt"]:
                donor_seq2 = donor_seq2[:index] + variant_info["ref"] + donor_seq2[index + 1 :]
                count += 1
            else:
                assert (
                    donor_seq2[index] == variant_info["ref"]
                ), "nucleotide in donor genome should be identical to the reference allele of the current SNP at the current position"
            assert count > 0, "no rare variants"
            alt_seq = one_hot_encode_diploid(donor_seq1, donor_seq2)

        yield {
            "inputs": {"ref": ref_seq, "alt": alt_seq},
            "metadata": {
                "chrom": variant_info.chrom,
                "pos": variant_info.pos0,
                "ref": variant_info.ref,
                "alt": variant_info.alt,
                "donor": sample,
                "gene_name": gene,
                "region_chr": region_chr,
                "region_start": region_start,
                "region_end": region_end,
            },
        }
        # yield one hot encoded sequences and metadata


def parse_gene_files(filepath):
    gene_list = []
    with open(filepath, "r") as file:
        for gene in file:
            gene_list.append(gene.strip())
    return gene_list


class LitModelCREAM_ISM_discrete(pl.LightningModule):  ## repeat, need delete
    """To wrap Model within LightningModule to form ISM predictions within Lightning Trainer object, for easy handling of precision among other things"""

    def __init__(self, model, run_id, ckpt):
        super().__init__()
        self.model = model
        self.results_df = pd.DataFrame(
            columns=[
                "chrom",
                "pos0",
                "ref",
                "alt",
                "ref_pred",
                "alt_pred",
                "run_id",
                "gene_name",
                "region_chr",
                "region_start",
                "region_end",
                "model_ckpt",
            ]
        )
        self.run_id = run_id
        self.ckpt = ckpt

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Implement the logic for a single prediction step
        # Extract inputs from batch, perform model forward pass, and return predictions
        inputs = batch["inputs"]
        ref_seq = inputs["ref"]
        alt_seq = inputs["alt"]
        if batch_idx == 0:
            temp_inputs = {}
            temp_inputs["seq_array"] = ref_seq.unsqueeze(1)
            self.ref_pred = self.model(temp_inputs)
            self.ref_pred = self.ref_pred.squeeze()
            exp_tensor = torch.exp(self.ref_pred)
            probabilities = exp_tensor / exp_tensor.sum()
            # self.ref_pred =  self.ref_pred[:,self.ref_pred.shape[1]//2,:].cpu().item() #turn into an attribute to avoid forming same prediction over and over. Get ref_pred once and store it, then ignore it. Because
            self.ref_pred = probabilities.cpu().numpy()
            self.region_start_in_first_batch = batch["metadata"]["region_start"][0].cpu().item()
            self.region_end_in_first_batch = batch["metadata"]["region_end"][0].cpu().item()
            self.region_chr_in_first_batch = batch["metadata"]["region_chr"][0]

        temp_inputs = {}
        temp_inputs["seq_array"] = alt_seq.unsqueeze(1)
        alt_pred = self.model(temp_inputs).squeeze()
        exp_tensor = torch.exp(alt_pred)
        probabilities = exp_tensor / exp_tensor.sum()
        alt_pred = probabilities.cpu().numpy()
        # alt_pred =  alt_pred[:,alt_pred.shape[1]//2,:] #turn into an attribute to avoid forming same prediction over and over. Get ref_pred once and store it, then ignore it. Because
        # assert alt_pred.shape[0] == 1, "Larger batch sizes not implemented yet"
        # alt_pred = alt_pred.cpu().item()

        chrom = batch["metadata"]["chrom"]
        assert len(chrom) == 1, "Larger batch sizes not implemented yet"

        region_start = batch["metadata"]["region_start"][0].cpu().item()
        region_end = batch["metadata"]["region_end"][0].cpu().item()
        region_chr = batch["metadata"]["region_chr"][0]
        assert (
            region_start == self.region_start_in_first_batch
        ), "ISM should be occuring in just one region because you are re-using ref_pred!"
        assert region_end == self.region_end_in_first_batch
        assert region_chr == self.region_chr_in_first_batch

        gene_name = batch["metadata"]["gene_name"][0]
        chrom = chrom[0]
        pos = batch["metadata"]["pos"][0].cpu().item()
        ref = batch["metadata"]["ref"][0]
        alt = batch["metadata"]["alt"][0]

        self.results_df.loc[self.results_df.shape[0], :] = [
            chrom,
            pos,
            ref,
            alt,
            self.ref_pred,
            alt_pred,
            self.run_id,
            gene_name,
            region_chr,
            region_start,
            region_end,
            self.ckpt,
        ]
        return None


class LitModelCREAM_ISM(pl.LightningModule):
    """To wrap Model within LightningModule to form ISM predictions within Lightning Trainer object, for easy handling of precision among other things"""

    def __init__(self, model, run_id, ckpt):
        super().__init__()
        self.model = model
        self.results_df = pd.DataFrame(
            columns=[
                "chrom",
                "pos0",
                "ref",
                "alt",
                "diff_pred",
                #"diff_pred_rev",
                "donor",
                "run_id",
                "gene_name",
                "region_chr",
                "region_start",
                "region_end",
                "model_ckpt",
            ]
        )
        self.run_id = run_id
        self.ckpt = ckpt

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Implement the logic for a single prediction step
        # Extract inputs from batch, perform model forward pass, and return predictions
        inputs = batch["inputs"]
        ref_seq = inputs["ref"]
        alt_seq = inputs["alt"]

        temp_inputs = {}
        temp_inputs["seq_array"] = torch.cat((ref_seq.unsqueeze(1), alt_seq.unsqueeze(1)))
        diff_pred = self.model(temp_inputs).squeeze().cpu()  # .item()

        # reverse the order of two seqs
        # temp_inputs["seq_array"] = torch.cat((alt_seq.unsqueeze(1), ref_seq.unsqueeze(1)))
        # diff_pred1 = self.model(temp_inputs).squeeze().cpu().item()

        chrom = batch["metadata"]["chrom"]
        assert len(chrom) == 1, "Larger batch sizes not implemented yet"

        region_start = batch["metadata"]["region_start"][0].cpu().item()
        region_end = batch["metadata"]["region_end"][0].cpu().item()
        region_chr = batch["metadata"]["region_chr"][0]

        gene_name = batch["metadata"]["gene_name"][0]
        chrom = chrom[0]
        pos = batch["metadata"]["pos"][0].cpu().item()
        ref = batch["metadata"]["ref"][0]
        alt = batch["metadata"]["alt"][0]
        donor = batch["metadata"]["donor"][0]

        self.results_df.loc[self.results_df.shape[0], :] = [
            chrom,
            pos,
            ref,
            alt,
            diff_pred.item(),
            #diff_pred1.item(),
            donor,
            self.run_id,
            gene_name,
            region_chr,
            region_start,
            region_end,
            self.ckpt,
        ]
        return None


class IsmDataset(IterableDataset):
    """To wrap ISM data generator as a pytorch dataset"""

    def __init__(self, it, length):
        self.it = it
        self.length = length

    def __iter__(self):
        return self.it

    def __len__(self):
        return self.length


def load_model(ckpt, save_dir):  # ,run_id
    path_to_ckpt = os.path.join(save_dir, f"checkpoints/{ckpt}")
    loaded_model = ContrastWrapperAttention.load_from_checkpoint(path_to_ckpt)
    return loaded_model


def main():
    parser = argparse.ArgumentParser(description="For ISM")
    parser.add_argument(
        "--path_to_metadata",
        type=str,
        help="Metadata from Wandb run to ensure correct specifications are used",
    )
    parser.add_argument(
        "--path_to_variants_file",
        type=str,
        nargs="?",
        help="txt file containing one variant per line. Each line has donors for which ISM is performed",
    )
    parser.add_argument("--model_type", type=str, help="One of SingleGene or MultiGene")
    print(torch.cuda.device_count())
    args = parser.parse_args()
    # metadata = pd.read_csv(args.path_to_metadata)
    # metadata = metadata.rename(columns = {'ID':'run_id'})

    with open(args.path_to_metadata, "r") as yf:
        metadata = yaml.safe_load(yf)

    # the run's own config, filled in with anything it predates from defaults.yaml
    config = cream_config.from_wandb_metadata(metadata)

    model_type = args.model_type
    assert model_type in ["SingleGene", "MultiGene"]

    path_to_variants_file = args.path_to_variants_file
    variants_to_score = pd.read_table(
        path_to_variants_file
    )  # parse_gene_files(path_to_variants_file)

    pl.seed_everything(0, workers=True)
    # for idx, row in metadata.iterrows():
    # run_id = row['run_id']
    tissues_to_train = metadata["tissues_to_train"][
        "value"
    ]  # .strip('"[]"').split(',') #configure as a list of strings
    assert len(tissues_to_train) == 1, "ISM on only 1 output tissue is supported"
    desired_seq_len = int(metadata["seq_length"]["value"])
    window = (
        desired_seq_len // 2
    )  # window is the amount of bp ahead and behind the TSS to score. Half the sequence length to score the entire sequence
    precision = metadata["precision"]["value"]
    save_dir = metadata["save_dir"]["value"]
    ckpt = get_ckpt(save_dir)
    if (
        ckpt is None
    ):  # some models won't have checkpoints saved. Namely, if the gene they were meant to be trained with is incompatible and there are no other train genes available to train them, training exits and there is no ckpt
        # continue
        print("cannot find ckpt")

    loaded_model = load_model(ckpt, save_dir)  # ,run_id
    run_id = metadata["save_dir"]["value"].split("/")[-1]
    fold = metadata["save_dir"]["value"].split("/")[-2]  # Fold-0
    model = LitModelCREAM_ISM(
        model=loaded_model, ckpt=ckpt, run_id=run_id
    )  # Loaded model will be used for predictions, but predict_step overwritten to perform ISM
    model.eval()
    model.cuda()

    tissue_str = cream_config.tissue_dirname(tissues_to_train[0])
    outpath = cream_config.output_dir(
        config, "ism_rare_variants_subdir", f"{tissue_str}Models", model_type, run_id
    )
    if not os.path.exists(os.path.join(outpath)):
        os.makedirs(os.path.join(outpath))

    filename = os.path.join(outpath, f"{run_id}_ContrastMultiAttention_{window * 2}bp_{fold}.csv")

    for (
        variant_idx,
        variant_sample,
    ) in variants_to_score.iterrows():  # enumerate(variants_to_score):
        variant_id = variant_sample["variant_id"]
        # sample = gene_sample[1]

        variant_df = get_all_gtex_snps(variant_sample, window, config)
        if variant_df.shape[0] > 0:  # and (
            # not os.path.exists(filename)
            # ):  # gene must be in enformer regions, must have SNPs nearby, and must not already have been evaluated
            print(f"Performing ISM on variant {variant_id} {variant_idx}/{len(variants_to_score)}")
            it = tss_centered_sample_sequences(variant_df, desired_seq_len, config)
            dataset = IsmDataset(it, length=variant_df.shape[0])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            lit_model = LitModelCREAM_ISM(model, run_id, ckpt)
            trainer = pl.Trainer(
                precision=precision,
                num_sanity_val_steps=0,  # check all validation data before starting to train
                deterministic=True,
            )
            trainer.predict(lit_model, dataloader)  # perform ISM
            if variant_idx == 0:
                lit_model.results_df.to_csv(filename, index=False)
            else:
                lit_model.results_df.to_csv(filename, mode="a", header=False, index=False)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

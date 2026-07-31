# adapted from Performer: https://github.com/shirondru/enformer_fine_tuning/tree/master/code 

import argparse
from CREAM.metrics.metric_logger import MetricLogger
from CREAM.metrics.metric_logger_cat import MetricLogger_cat
from CREAM.models.head_adapter_attention import HeadAdapterWrapper_Attention
from CREAM.models.head_adapter import MultiHeadWrapper
from CREAM.models.contrast_wrapper import ContrastWrapper
from CREAM.models.contrast_wrapper_attention_multiheads_rev2 import ContrastWrapperAttention  #contrast_wrapper_attention_multihead0
from CREAM.models.head_adapter_gene_embedding import HeadAdapterGeneEmbeddingWrapper
from CREAM.datasets.gtex_dataset0 import *
from CREAM import config as cream_config
import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
import warnings
import os
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path


warnings.filterwarnings("ignore", ".*does not have many workers.*")


def parse_gene_files(filepath):
    """
    For parsing txt files containing one gene name per row
    """
    gene_list = []
    with open(filepath, "r") as file:
        for gene in file:
            gene = gene.strip().split(",")
            gene_list.append([g for g in gene if g != ""])
    return gene_list


def prepare_genes(config):
    model_type = config["model_type"]
    tissue = config["tissues_to_train"].split(",")[0]

    train_gene_filedir = cream_config.gene_set_dir(config, model_type)
    if model_type == "SingleGene":
        train_gene_filenames = os.listdir(
            train_gene_filedir
        )  # if its a single gene model, there will be 1 txt file per train gene, if multi-gene it will 1 total txt file containing all genes
        valid_genes = []
        test_genes = []
    elif model_type == "MultiGene":
        # train_gene_filenames = os.listdir(train_gene_filedir) #if each gene set as multiple genes as a single model
        train_gene_filenames = [config["train_gene_file"]]

        valid_gene_path = cream_config.gene_set_path(config, config["val_gene_file"], model_type)
        valid_genes = parse_gene_files(valid_gene_path)
       #valid_genes = []
        test_gene_path = cream_config.gene_set_path(config, config["test_gene_file"], model_type)
        test_genes = parse_gene_files(test_gene_path)
        #test_genes = []

    return train_gene_filedir, train_gene_filenames, valid_genes, test_genes


def ensure_no_gene_overlap(train_genes, val_genes, test_genes):
    train_gene_set = set().union(*train_genes)
    valid_gene_set = set().union(*val_genes)
    test_gene_set = set().union(*test_genes)

    train_valid_overlap = train_gene_set & valid_gene_set
    train_test_overlap = train_gene_set & test_gene_set
    valid_test_overlap = valid_gene_set & test_gene_set

    assert (
        len(list(train_valid_overlap)) == 0
    ), f"There is overlap between genes in the train and valid set via the following genes {train_valid_overlap}"
    assert (
        len(list(train_test_overlap)) == 0
    ), f"There is overlap between genes in the train and test set via the following genes {train_test_overlap}"
    assert (
        len(list(valid_test_overlap)) == 0
    ), f"There is overlap between genes in the valid and test set via the following genes {valid_test_overlap}"
    assert len(train_genes) > 0, "You have no genes to train on!"


def ensure_no_donor_overlap(train_ds, val_ds, test_ds):
    train_set = set(train_ds.individuals_in_split)
    val_set = set(val_ds.individuals_in_split)
    test_set = set(test_ds.individuals_in_split)

    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0


def define_donor_paths(config, dataset):
    if dataset == "gtex":
        rare_variants = config.rare_variants
        for split, key in (
            ("train", "train_donor_path"),
            ("val", "valid_donor_path"),
            ("test", "test_donor_path"),
        ):
            config.update(
                {
                    key: cream_config.donor_list_path(
                        config, split, config.fold, dataset, rare_variants
                    )
                }
            )
    else:
        raise Exception(f"Dataset: {dataset} not supported!")


def load_gtex_datasets(config, train_genes, valid_genes, test_genes):
    def instantiate_dataset(
        config,
        gene_list,
        donor_path,
        tissues_to_train,
        gene_expression_df,
        true_eQTLs,
        num_individuals_per_gene,
        mask_dna_prop,
        shift,
        shift_freq,
        rc, 
        rc_freq,
        stratify_expr,
        repeat,
    ):
        ds = GTExDataset(
            tissues_to_train,
            gene_list,
            config.seq_length,
            shift,
            shift_freq,
            rc,
            rc_freq,
            mask_dna_prop,
            config.discretize_bins,
            stratify_expr,
            config.train_batch_size,
            num_individuals_per_gene,
            donor_path,
            repeat,
            gene_expression_df,
            true_eQTLs,
            cream_config.dataset_paths(config),
        )
        return ds

    tissues_to_train = config.tissues_to_train.split(",")  # ex: 'Whole Blood' -> ['Whole Blood]
    #assert len(tissues_to_train) == 1, "Multi-tissue training not yet supported"

    # load one gene expression matrix per tissue. Which directory they live in and
    # how they are named comes from the config (expression_filepath, data.*_glob)
    expression_files = cream_config.expression_files(config, predicted=bool(config.predict_expr))

    gene_expression_df = {}
    for p in expression_files:
        if config.predict_expr:
            ts = p.name.split('_', 1)[0]
        else:
            ts = p.name.split('.', 1)[0]
        if ts in tissues_to_train:
            df = pd.read_csv(p, sep="\t")

            if (not config.raw_expr) and (not config.predict_expr):
                df['Description'] =  df['gene_id'].str.replace(r'\.\d+$', '', regex=True)

            gene_expression_df[ts] = df

    # load eQTL metadata and files for selected tissues
    true_eQTLs = {}
    if config.eQTL_guided:
        eqtl_dir = Path(cream_config.eqtl_dir(config))
        tissue_id = pd.read_csv(cream_config.eqtl_tissue_label_file(config))

        for tissue in tissues_to_train:
            tissue1 = cream_config.tissue_alias(config, tissue)
            matched = tissue_id.loc[tissue_id['data_tissue'] == tissue1]
            if matched.empty:
                raise ValueError(f"No eQTL metadata found for tissue: {tissue}")

            eqtl_filename = matched.iloc[0]['path']
            #tissue_name = matched.iloc[0]['tabix_tissue']
            eqtl_df = pd.read_csv(eqtl_dir / eqtl_filename, sep='\t')
            variant_parts = eqtl_df['variant'].astype(str).str.split('_')
            eqtl_df['chrom'] = variant_parts.str[0]
            eqtl_df['pos'] = pd.to_numeric(variant_parts.str[1], errors='coerce')
            eqtl_df['pos0'] = eqtl_df['pos'] - 1
            true_eQTLs[tissue] = eqtl_df

    # train on just train genes. When validating w/ set of validation ppl, evaluate on train genes and valid donors (valid donors can be empty)
    # when evaluating on test set of donors, evaluate on all genes. The lightning module keeps track of which is which and early stopping and checkpoint callbacks
    # can monitor different groups of genes
    genes_to_train = train_genes
    # if len(train_genes) > 500:
    #     sample_genes = random.sample(train_genes, 500) 
    #     genes_to_validate = sample_genes + valid_genes
    # else:
    genes_to_validate = train_genes + valid_genes
    genes_to_test = train_genes + valid_genes + test_genes

    train_ds = instantiate_dataset(
        config,
        genes_to_train,
        config.train_donor_path,
        tissues_to_train,
        gene_expression_df,
        true_eQTLs,  # gene_embedding_df,
        config.num_individuals_per_gene,
        config.mask_dna_prop,
        config.shift,
        config.shift_freq, 
        config.reverse_complement,
        config.rc_freq, 
        config.stratify_expr,
        1,  # repeat
    )
    # instantiate eval datasets using different donor paths, lists of genes, and set num_individuals_per_gene to -1 so all people are used. Otherwise those that don't fit a multiple of gradient accumulated batch size will be dropped
    valid_ds = instantiate_dataset(
        config,
        genes_to_validate,
        config.valid_donor_path,
        tissues_to_train,
        gene_expression_df,
        [],  # gene_embedding_df,
        -1,  # config.train_batch_size
        0,  # mask
        0,  # shift
        0,
        0,  # reverse_complement
        0,
        1,  # stratify
        8 if bool(config.contrast_embed) else 1,  # repeat
    )
    test_ds = instantiate_dataset(
        config,
        genes_to_test,
        config.test_donor_path,
        tissues_to_train,
        gene_expression_df,
        [],  # gene_embedding_df,
        -1,  #config.train_batch_size
        0,  # mask
        0,  # shift
        0,
        0,  # reverse_complement
        0,
        1,  # stratify
        10 if bool(config.contrast_embed) else 1,  # repeat
    )

    if not config.use_test_data:
        ensure_no_donor_overlap(train_ds, valid_ds, test_ds)

    return train_ds, valid_ds, test_ds


def load_trainer(config):
    metric_logger, checkpoint_callback, lr_monitor = load_callbacks(config)  # early_stopper,

    if config.num_gpus > 1:
        # Multi gpu
        devices = config.num_gpus
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        # Single gpu
        devices = [0]
        strategy = "auto"

    trainer = Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        accumulate_grad_batches=config.num_individuals_per_gene
        // config.train_batch_size,  # accumulate as many batches as necessary to achieve num_individuals_per_gene effective samples per gradient accumulated step
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[checkpoint_callback, metric_logger, lr_monitor],  # ,early_stopper
        logger=WandbLogger(),
        num_sanity_val_steps=0,  # don't do any validation before training, as all sorts of R2 metrics will be computed during callbacks. Could lead to error with small sample size
        log_every_n_steps=1,
        devices=devices,
        strategy=strategy,  # DDPStrategy or None
    )
    return trainer


def load_callbacks(config):
    checkpoint_dir = os.path.join(config.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    monitor = "mean_r2_across_train_genes_across_valid_donors"  # all
    mode = "max"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, save_top_k=1, monitor=monitor, mode=mode
    )
    # early_stopper = EarlyStopping(monitor = monitor, mode = mode, min_delta = 0, patience = 20)
    if config.discretize_bins > 0:
        metric_logger = MetricLogger_cat()
    else:
        metric_logger = MetricLogger()
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return metric_logger, checkpoint_callback, lr_monitor  # early_stopper,


def train_gtex(
    config: wandb.config, train_genes: list, valid_genes: list, test_genes: list, ckpt: str
) -> None:
    ensure_no_gene_overlap(train_genes, valid_genes, test_genes)
    define_donor_paths(config, "gtex")

    assert config.shift < (
        config.seq_length // 2
    ), f"shift longer than half of the desired seq length"

    train_ds, valid_ds, test_ds = load_gtex_datasets(config, train_genes, valid_genes, test_genes)
    data_module = CustomDataModule(train_ds, valid_ds, test_ds, config)
    
    if config.model_type == "MultiGene" or config.model_type == "SingleGene":
        if config.contrast_embed and config.atten_pool:
            model = ContrastWrapperAttention(
                config.tissues_to_train.split(","),
                config.save_dir,
                train_ds,
                float(config.learning_rate),
                config.alpha,
                config.discretize_bins,
                config.raw_expr,
                config.contrast_embed,
                config.eQTL_guided,
                config.max_epochs,
                config.train_batch_size * len(train_genes),  # not used
                train_genes,
                valid_genes,
                test_genes,
                config.attn_dim,
                config.attn_heads,
                config.final_div_factor,
                config.pct_start,
            )
        elif config.contrast_embed:
            model = ContrastWrapper(
                config.tissues_to_train.split(","),
                config.save_dir,
                train_ds,
                float(config.learning_rate),
                config.alpha,
                config.discretize_bins,
                config.raw_expr,
                config.contrast_embed,
                config.max_epochs,
                config.train_batch_size * len(train_genes),
                train_genes,
                valid_genes,
                test_genes,
            )
        elif config.gene_embed:
            model = HeadAdapterGeneEmbeddingWrapper(
                config.tissues_to_train.split(","),
                config.save_dir,
                train_ds,
                float(config.learning_rate),
                config.alpha,
                config.discretize_bins,
                config.contrast_embed,
                config.max_epochs,
                config.train_batch_size * len(train_genes),
                train_genes,
                valid_genes,
                test_genes,
            )
        elif config.atten_pool:
            model = HeadAdapterWrapper_Attention(
                config.tissues_to_train.split(","),
                config.save_dir,
                train_ds,
                float(config.learning_rate),
                config.alpha,
                config.discretize_bins,
                config.raw_expr,
                config.contrast_embed,
                config.max_epochs,
                config.train_batch_size * len(train_genes),
                train_genes,
                valid_genes,
                test_genes,
            )
        else:
            model = MultiHeadWrapper(
                config.tissues_to_train.split(","),
                config.save_dir,
                train_ds,
                float(config.learning_rate),
                config.alpha,
                config.discretize_bins,
                config.raw_expr,
                config.contrast_embed,
                config.max_epochs,
                config.train_batch_size * len(train_genes),
                train_genes,
                valid_genes,
                test_genes,
            )

    # wandb.watch(model, log="all", log_freq = 50)
    trainer = load_trainer(config)
    # trainer.fit(
    #     model=model,
    #     train_dataloaders=DataLoader(train_ds, batch_size=config.train_batch_size, num_workers=1),
    #     val_dataloaders=DataLoader(
    #         valid_ds,
    #         batch_size=2,
    #         num_workers=1,
    #     ),  # code for logging and storing validation/test results expects batch size of 1 for these
    # )

    # trainer.test(
    #     model, DataLoader(test_ds, batch_size=2, num_workers=1), ckpt_path="best"
    # )  
    trainer.fit(model = model, datamodule = data_module) 
    #trainer.test(model, datamodule=data_module, ckpt_path = 'best')


def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--runid", type=str)
    parser.add_argument("--rare_variants", action="store_true", default=False)
    parser.add_argument("--use_test_data", action="store_true", default=False)
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()
    config_path = args.config_path
    fold = int(args.fold)
    model_type = args.model_type
    runid = args.runid
    use_test_data = args.use_test_data
    rare_variants = args.rare_variants
    num_gpus = args.num_gpus

    config = cream_config.load_config(
        config_path, use_test_data=use_test_data, model_type=model_type
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cream_config.setting(
        config, "env", "cublas_workspace_config"
    )

    if runid != None:
        ckpt_dir = cream_config.checkpoint_dir(
            config,
            cream_config.run_dir(
                config,
                model_type=model_type,
                train_gene_set=cream_config.gene_set_name(config["train_gene_file"]),
                fold=fold,
                run_id=runid,
                rare_variants=rare_variants,
            ),
        )
        ckpt = os.listdir(ckpt_dir)
        assert len(ckpt) == 1, "no or more than one checkpoint files"
        ckpt = os.path.join(ckpt_dir, ckpt[0])
    else:
        ckpt = ""

    train_gene_filedir, train_gene_filenames, valid_genes, test_genes = prepare_genes(config)

    # if training a single gene model, loop through all single gene files in the dir. If its a multi gene model, there is only 1 train gene file and loop will exit after 1 iteration
    for train_gene_filename in train_gene_filenames:
        train_gene_set = cream_config.gene_set_name(train_gene_filename)
        wandb_filename = f"{config['model_type']}_{train_gene_set}"
        train_gene_path = os.path.join(os.path.join(train_gene_filedir, train_gene_filename))
        train_genes = parse_gene_files(
            train_gene_path
        )  # will contain 1 geneset if this is a single gene set model, else it will contain ~100 genesets

        #run_id = datetime.now().strftime("%m%d_%H_%M_%S%f")[:-3]
        wandb.init(
            #id = run_id,
            project=cream_config.wandb_project(
                config,
                use_test_data=use_test_data,
                rare_variants=rare_variants,
                raw_expr=config["raw_expr"],
            ),
            entity=cream_config.wandb_entity(config),
            name=config["experiment_name"] + f"_Fold-{fold}_" + wandb_filename,
            group=config["experiment_name"],
            config=config.to_dict(),
        )  # hhh: might return "run" and input "run" as logger in Trainer
        wandb.config.update({"fold": fold})
        wandb.config.update({"train_genes": train_genes})
        wandb.config.update({"valid_genes": valid_genes})
        wandb.config.update({"test_genes": test_genes})

        wandb.config.update(
            {
                "save_dir": cream_config.run_dir(
                    config,
                    model_type=model_type,
                    train_gene_set=train_gene_set,
                    fold=fold,
                    run_id=wandb.run.id,
                    rare_variants=rare_variants,
                )
            }
        )
        wandb.config.update({"use_test_data": use_test_data})
        wandb.config.update({"num_gpus": num_gpus})

        wandb.config.update({"rare_variants": rare_variants})

        pl.seed_everything(int(wandb.config.seed), workers=True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        train_gtex(wandb.config, train_genes, valid_genes, test_genes, ckpt)
        wandb.finish()


if __name__ == "__main__":
    main()

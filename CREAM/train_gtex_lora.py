import argparse
from CREAM.models.multi_genes_attention0 import MultiGeneAttentionWrapper0
from CREAM.metrics.metric_logger import MetricLogger
from CREAM.metrics.metric_logger_cat import MetricLogger_cat
from CREAM.models.head_adapter_attention import HeadAdapterWrapper_Attention
from CREAM.models.head_adapter import HeadAdapterWrapper
from CREAM.models.contrast_wrapper_lora import ContrastWrapperLoRA
from CREAM.models.multi_genes_attention import MultiGeneAttentionWrapper
from CREAM.models.head_adapter_gene_embedding import HeadAdapterGeneEmbeddingWrapper
from CREAM.datasets.gtex_dataset0 import *
import yaml
import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from peft import LoraConfig
import warnings
import os
import torch

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
    tissue = (
        config["tissues_to_train"]
        .replace(" -", "")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    data_dir = config["DATA_DIR"]

    train_gene_filedir = os.path.join(data_dir, "genes", tissue, model_type)  # genesets
    if model_type == "MultiGeneSet":
        valid_genes = []
        train_gene_filenames = ["120_train_genesets.txt"]
        test_gene_path = os.path.join(data_dir, "genesets", tissue, model_type, "test_genesets.txt")
        test_genes = parse_gene_files(test_gene_path)
    elif model_type == "SingleGeneSet" or model_type == "SingleGene":
        train_gene_filenames = os.listdir(
            train_gene_filedir
        )  # if its a single gene model, there will be 1 txt file per train gene, if multi-gene it will 1 total txt file containing all genes
        valid_genes = []
        test_genes = []
    elif model_type == "MultiGene":
        # train_gene_filenames = os.listdir(train_gene_filedir) #if each gene set as multiple genes as a single model
        valid_genes = []
        # test_genes = []
        train_gene_filenames = ["1000_train_genes.txt"]
        test_gene_path = os.path.join(data_dir, "genes", tissue, model_type, "test_genes.txt")
        test_genes = parse_gene_files(test_gene_path)

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
        donor_dir = os.path.join(config.DATA_DIR, "cross_validation_folds", dataset, "cv_folds")
        config.update(
            {"train_donor_path": os.path.join(donor_dir, f"person_ids-train-fold{config.fold}.txt")}
        )
        config.update(
            {"valid_donor_path": os.path.join(donor_dir, f"person_ids-val-fold{config.fold}.txt")}
        )
        config.update(
            {"test_donor_path": os.path.join(donor_dir, f"person_ids-test-fold{config.fold}.txt")}
        )
    elif dataset == "rosmap":
        # train and validation set are from rosmap. Test set will be individuals from gtex to enable cross-cohort evaluation
        rosmap_dir = os.path.join(config.DATA_DIR, "cross_validation_folds", dataset)
        config.update(
            {
                "train_donor_path": os.path.join(
                    rosmap_dir, f"person_ids-train-fold{config.fold}.txt"
                )
            }
        )
        config.update(
            {"valid_donor_path": os.path.join(rosmap_dir, f"person_ids-val-fold{config.fold}.txt")}
        )

        # using all individuals from gtex as test set. Dataset will keep only those with brain cortex data.
        all_gtex_donor_path = os.path.join(
            config.DATA_DIR, "cross_validation_folds", "gtex", "All_GTEx_ID_list.txt"
        )
        config.update({"test_donor_path": all_gtex_donor_path})
    else:
        raise Exception(f"Dataset: {dataset} not supported!")


def load_gtex_datasets(config, train_genes, valid_genes, test_genes):
    def instantiate_dataset(
        config,
        gene_list,
        donor_path,
        tissues_to_train,
        gene_expression_df,
        gene_embedding_df,
        num_individuals_per_gene,
        mask_dna_prop,
        shift,
        rc,
        stratify_expr,
        repeat,
    ):
        ds = GTExDataset(
            tissues_to_train,
            gene_list,
            config.seq_length,
            shift,
            rc,
            mask_dna_prop,
            config.discretize_bins,
            stratify_expr,
            config.train_batch_size,
            num_individuals_per_gene,
            donor_path,
            repeat,
            gene_expression_df,
            gene_embedding_df,
            config.DATA_DIR,
        )
        return ds

    tissues_to_train = config.tissues_to_train.split(",")  # ex: 'Whole Blood' -> ['Whole Blood]
    assert len(tissues_to_train) == 1, "Multi-tissue training not yet supported"
    tissue_str = (
        tissues_to_train[0].replace(" -", "").replace(" ", "_").replace("(", "").replace(")", "")
    )

    # load gene expression df, merge in gene names onto gene ids
    expression_dir = os.path.join(config.DATA_DIR, "gtex_eqtl_expression_matrix")
    gene_id_mapping = pd.read_csv(os.path.join(expression_dir, "gene_id_mapping.csv"))
    if config.residual_expr:
        df_path = os.path.join(
            expression_dir, f"{tissue_str}.v8.normalized_expression_remove_top40.bed.gz"
        )
    elif config.raw_expr:
        df_path = os.path.join(expression_dir, f"gene_log_tpm_2017-06-05_v8_{tissue_str}.gct")
    else:
        df_path = os.path.join(expression_dir, f"{tissue_str}.v8.normalized_expression.bed.gz")
    gene_expression_df = pd.read_csv(df_path, sep="\t")
    if not config.raw_expr:
        gene_expression_df = gene_expression_df.merge(
            gene_id_mapping, left_on="gene_id", right_on="Name"
        )

    # load gene embedding, merge in gene names onto gene ids
    gene_embedding_df = pd.read_csv(
        os.path.join(config.DATA_DIR, "Geneformer_gene_embedding.csv.gz"), index_col=0
    )  # TODO: add tissue here
    gene_id_mapping["Name"] = [x.split(".")[0] for x in gene_id_mapping["Name"]]
    gene_embedding_df = gene_embedding_df.merge(gene_id_mapping, left_index=True, right_on="Name")

    # train on just train genes. When validating w/ set of validation ppl, evaluate on train genes and valid donors (valid donors can be empty)
    # when evaluating on test set of donors, evaluate on all genes. The lightning module keeps track of which is which and early stopping and checkpoint callbacks
    # can monitor different groups of genes
    genes_to_train = train_genes
    genes_to_validate = train_genes + valid_genes
    genes_to_test = train_genes + valid_genes + test_genes

    train_ds = instantiate_dataset(
        config,
        genes_to_train,
        config.train_donor_path,
        tissues_to_train,
        gene_expression_df,
        gene_embedding_df,
        config.num_individuals_per_gene,
        config.mask_dna_prop,
        config.shift,
        config.reverse_complement,
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
        gene_embedding_df,
        2,  # num_individuals_per_gene
        0,  # mask
        0,  # shift
        0,  # reverse_complement
        1,  # stratify
        10,  # repeat
    )
    test_ds = instantiate_dataset(
        config,
        genes_to_test,
        config.test_donor_path,
        tissues_to_train,
        gene_expression_df,
        gene_embedding_df,
        2,  # num_individuals_per_gene
        0,  # mask
        0,  # shift
        0,  # reverse_complement
        1,  # stratify
        10,  # repeat
    )

    ensure_no_donor_overlap(train_ds, valid_ds, test_ds)

    return train_ds, valid_ds, test_ds


def load_trainer(config):
    metric_logger, checkpoint_callback, lr_monitor = load_callbacks(config)  # early_stopper,
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
        # strategy=DDPStrategy(find_unused_parameters=True),
        devices=[0],
    )
    return trainer


def load_callbacks(config):
    checkpoint_dir = os.path.join(config.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    monitor = "mean_r2_across_train_genes_across_valid_donors"  # q: where does it define?
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


def train_gtex_lora(
    config: wandb.config,
    config_lora: LoraConfig,
    train_genes: list,
    valid_genes: list,
    test_genes: list,
    target_module_name: str | None = None,
    config_lora_conv1D: dict | None = None,
    config_lora_conv2D: dict | None = None,
) -> None:
    ensure_no_gene_overlap(train_genes, valid_genes, test_genes)
    define_donor_paths(config, "gtex")

    assert config.shift < (
        config.seq_length // 2
    ), f"shift longer than half of the desired seq length"

    train_ds, valid_ds, test_ds = load_gtex_datasets(config, train_genes, valid_genes, test_genes)
    if config.contrast_embed:
        model = ContrastWrapperLoRA(
            config_lora,
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
            target_module_name,
            config_lora_conv1D,
            config_lora_conv2D,
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
        model = HeadAdapterWrapper(
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
    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(train_ds, batch_size=config.train_batch_size),
        val_dataloaders=DataLoader(
            valid_ds, batch_size=2
        ),  # code for logging and storing validation/test results expects batch size of 1 for these
    )

    trainer.test(
        model, DataLoader(test_ds, batch_size=2), ckpt_path="best"
    )  # "results/FinalPaperWholeBlood/SingleGeneSet/es/Fold-0/nw3s5f9n/checkpoints/epoch=56-step=228.ckpt")


def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--config_path_lora", type=str)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--runid", type=str)
    args = parser.parse_args()
    config_path = args.config_path
    config_path_lora = args.config_path_lora
    fold = int(args.fold)
    model_type = args.model_type
    runid = args.runid
    assert model_type in ["SingleGene", "MultiGene"]

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir, "../data")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    with open(config_path_lora, "r") as lora_file:
        config_lora_dict = yaml.safe_load(lora_file)
    config_lora = LoraConfig(**config_lora_dict["peft"])

    if ("conv1D" in config_lora_dict) or ("conv2D" in config_lora_dict):
        target_module_name = "base_model.model.enformer"
    if "conv1D" in config_lora_dict:
        config_lora_conv1D_dict = config_lora_dict["conv1D"]
    if "conv2D" in config_lora_dict:
        config_lora_conv2D_dict = config_lora_dict["conv2D"]

    config["model_type"] = model_type
    config["DATA_DIR"] = DATA_DIR

    if runid != None:
        ckpt_dir = os.path.join(
            current_dir,
            f"../results/{config['experiment_name']}/{model_type}/120_train_genesets/Fold-{fold}/{runid}/checkpoints/",
        )
        ckpt = os.listdir(ckpt_dir)
        assert len(ckpt) == 1, "no or more than one checkpoint files"
        ckpt = os.path.join(ckpt_dir, ckpt[0])
    else:
        ckpt = ""

    train_gene_filedir, train_gene_filenames, valid_genes, test_genes = prepare_genes(config)

    # if training a single gene model, loop through all single gene files in the dir. If its a multi gene model, there is only 1 train gene file and loop will exit after 1 iteration
    for train_gene_filename in train_gene_filenames:
        wandb_filename = f"{config['model_type']}_{train_gene_filename.strip('.txt')}"
        train_gene_path = os.path.join(os.path.join(train_gene_filedir, train_gene_filename))
        train_genes = parse_gene_files(
            train_gene_path
        )  # will contain 1 geneset if this is a single gene set model, else it will contain ~100 genesets
        project_name = (
            "fine_tune_enformer_multigene_raw_expr"
            if config["raw_expr"]
            else "fine_tune_enformer_multigene_diff"
        )
        wandb.init(
            project=project_name,  # multigenes_test raw_expr mae_loss
            name=config["experiment_name"] + f"_Fold-{fold}_" + wandb_filename,
            group=config["experiment_name"],
            config=config,
        )  # hhh: might return "run" and input "run" as logger in Trainer
        wandb.config.update({"fold": fold})
        wandb.config.update({"train_genes": train_genes})
        wandb.config.update({"valid_genes": valid_genes})
        wandb.config.update({"test_genes": test_genes})
        wandb.config.update(
            {
                "save_dir": os.path.join(
                    current_dir,
                    f"../results/{config['experiment_name']}/{model_type}/{train_gene_filename.strip('.txt')}/Fold-{fold}/{wandb.run.id}",
                )
            }
        )
        pl.seed_everything(int(wandb.config.seed), workers=True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if ("conv1D" in config_lora_dict) or ("conv2D" in config_lora_dict):
            train_gtex_lora(
                wandb.config,
                config_lora,
                train_genes,
                valid_genes,
                test_genes,
                target_module_name,
                config_lora_conv1D_dict,
                config_lora_conv2D_dict,
            )
        else:
            train_gtex_lora(wandb.config, config_lora, train_genes, valid_genes, test_genes)
        wandb.finish()


if __name__ == "__main__":
    main()

# adapted from Performer: https://github.com/shirondru/enformer_fine_tuning/tree/master/code

"""Discretized counterpart of train_gtex2.py.

Trains the contrast + attention-pooling model with a binned expression head
(ContrastWrapperAttentionCat) instead of a single continuous output. Everything
about gene lists, donor folds, datasets and the trainer is shared with
train_gtex2.py; only model construction differs.
"""

import argparse
import os

import lightning.pytorch as pl
import torch
import wandb

from CREAM import config as cream_config
from CREAM.datasets.gtex_dataset0 import CustomDataModule
from CREAM.models.contrast_wrapper_attention_multiheads_rev2_cat import (
    ContrastWrapperAttentionCat,
)
from CREAM.train_gtex2 import (
    define_donor_paths,
    ensure_no_gene_overlap,
    load_gtex_datasets,
    load_trainer,
    parse_gene_files,
    prepare_genes,
)


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

    if config.model_type not in ("MultiGene", "SingleGene"):
        raise ValueError(f"model_type {config.model_type} is not supported for discrete training")
    if not (config.contrast_embed and config.atten_pool):
        # The other wrappers emit (batch, gene, feature) predictions, which the
        # binned eval path in LitModelCat cannot read.
        raise ValueError(
            "Discrete training here covers contrast_embed + atten_pool only. "
            "Set both in the config, or use train_gtex2.py for the other model variants."
        )

    model = ContrastWrapperAttentionCat(
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
        bin_offset_alpha=config.get("bin_offset_alpha", 0.9),
    )

    trainer = load_trainer(config)
    trainer.fit(model=model, datamodule=data_module)
    #trainer.test(model, datamodule=data_module, ckpt_path = 'best')


def main():
    parser = argparse.ArgumentParser(description="Run a discretized training experiment")
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

    if config["discretize_bins"] <= 0:
        raise ValueError(
            f"{config_path} sets discretize_bins to {config['discretize_bins']}. "
            "Use train_gtex2.py for continuous training."
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

        wandb.init(
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
        )
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

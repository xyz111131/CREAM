# adapted from Performer: https://github.com/shirondru/enformer_fine_tuning/tree/master/code

"""Discretized counterpart of test_gtex.py.

Evaluates a checkpoint trained by train_gtex2_cat.py. Gene lists, donor folds and
datasets are shared with test_gtex.py; only the model class differs, and
load_callbacks already selects MetricLogger_cat because discretize_bins > 0.
"""

import argparse
import os

import lightning.pytorch as pl
import torch
import wandb
from torch.utils.data import DataLoader

from CREAM import config as cream_config
from CREAM.ism_cream import get_ckpt
from CREAM.models.contrast_wrapper_attention_multiheads_rev2_cat import (
    ContrastWrapperAttentionCat,
)
from CREAM.test_gtex import prepare_genes
from CREAM.train_gtex2 import define_donor_paths, load_gtex_datasets, load_trainer

torch.use_deterministic_algorithms(True)


def eval_test_genes(config: wandb.config, test_genes: list, checkpath) -> None:
    define_donor_paths(config, "gtex")

    train_ds, valid_ds, test_ds = load_gtex_datasets(config, [], [], test_genes)

    ckpt = get_ckpt(checkpath)
    path_to_ckpt = os.path.join(checkpath, f"checkpoints/{ckpt}")
    if not config.atten_pool:
        # Only the contrast + attention-pooling model has a discretized variant.
        raise ValueError(
            "Discrete evaluation covers atten_pool models only; use test_gtex.py otherwise."
        )
    model = ContrastWrapperAttentionCat.load_from_checkpoint(path_to_ckpt)
    model.eval()
    model.genes_for_test = set().union(*test_genes)
    model.save_dir = config.save_dir
    trainer = load_trainer(config)
    trainer.test(model, DataLoader(test_ds, batch_size=int(config.train_batch_size)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a discretized checkpoint")
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
    rare_variants = args.rare_variants
    use_test_data = args.use_test_data
    num_gpus = args.num_gpus

    config = cream_config.load_config(
        config_path,
        use_test_data=use_test_data,
        model_type=model_type,
        fold=fold,
        num_gpus=num_gpus,
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cream_config.setting(
        config, "env", "cublas_workspace_config"
    )

    if config["discretize_bins"] <= 0:
        raise ValueError(
            f"{config_path} sets discretize_bins to {config['discretize_bins']}. "
            "Use test_gtex.py to evaluate a continuous checkpoint."
        )

    train_gene_filedir, train_gene_filenames, train_genes, valid_genes, test_genes = prepare_genes(config)

    checkpath = cream_config.run_dir(
        config,
        model_type=model_type,
        train_gene_set=cream_config.gene_set_name(train_gene_filenames[0]),
        fold=fold,
        run_id=runid,
        rare_variants=rare_variants,
    )

    config.update({"valid_genes": valid_genes})
    config.update({"test_genes": test_genes})
    config.update(
        {
            "save_dir": os.path.join(
                checkpath,
                cream_config.setting(config, "outputs", "test_gene_subdir"),  # need to change
            )
        }
    )
    config.update({"use_test_data": use_test_data})
    config.update({"rare_variants": rare_variants})
    # no attention supervision at eval time, and it keeps the eQTL files out of
    # dataset construction
    config['eQTL_guided'] = False

    pl.seed_everything(int(config.seed), workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    eval_test_genes(config, test_genes, checkpath)
    #eval_test_genes(config, valid_genes, checkpath)
    #eval_test_genes(config, train_genes[::3], checkpath)


if __name__ == "__main__":
    main()

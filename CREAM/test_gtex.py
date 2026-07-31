# adapted from Performer: https://github.com/shirondru/enformer_fine_tuning/tree/master/code 

import os
#os.environ["CUDA_VISIBLE_DEVICES"] ="1"
from CREAM.train_gtex2 import *
from CREAM.models.contrast_wrapper_attention_multiheads_rev2 import ContrastWrapperAttention
from CREAM.models.contrast_wrapper import ContrastWrapper
from CREAM.ism_cream import get_ckpt
import torch

torch.use_deterministic_algorithms(True)


def prepare_genes(config):
    model_type = config["model_type"]
    tissue = config["tissues_to_train"].split(",")[0]

    train_gene_filedir = cream_config.gene_set_dir(config, model_type)
    if model_type == "MultiGeneSet":
        valid_genes = []
        train_gene_filenames = [config.get("train_gene_file", "120_train_genesets.txt")]
        train_genes = []
        test_gene_path = cream_config.multi_geneset_path(
            config, config.get("test_gene_file", "test_genesets.txt"), model_type, tissue
        )
        test_genes = parse_gene_files(test_gene_path)
    elif model_type == "SingleGeneSet" or model_type == "SingleGene":
        train_gene_filenames = os.listdir(
            train_gene_filedir
        )  # if its a single gene model, there will be 1 txt file per train gene, if multi-gene it will 1 total txt file containing all genes
        train_genes = []
        valid_genes = []
        test_genes = []
    elif model_type == "MultiGene":
        # train_gene_filenames = os.listdir(train_gene_filedir) #if each gene set as multiple genes as a single model
        train_gene_filenames = [config["train_gene_file"]]

        train_gene_path = cream_config.gene_set_path(config, train_gene_filenames[0], model_type)
        train_genes = parse_gene_files(train_gene_path)
        valid_gene_path = cream_config.gene_set_path(config, config["val_gene_file"], model_type)
        valid_genes = parse_gene_files(valid_gene_path)
        test_gene_path = cream_config.gene_set_path(config, config["test_gene_file"], model_type)
        test_genes = parse_gene_files(test_gene_path)

    return train_gene_filedir, train_gene_filenames, train_genes, valid_genes, test_genes


def eval_test_genes(config: wandb.config, test_genes: list, checkpath) -> None:
    define_donor_paths(config, "gtex")

    train_ds, valid_ds, test_ds = load_gtex_datasets(config, [], [], test_genes)
    # model = LitModelContrastWrapper(
    #             config.tissues_to_train.split(','),
    #             config.save_dir,
    #             train_ds,
    #             float(config.learning_rate),
    #             config.alpha,
    #             config.discretize_bins,
    #             config.raw_expr,
    #             config.contrast_embed,
    #             config.max_epochs,
    #             config.train_batch_size, # not used
    #             [],
    #             [],
    #             test_genes
    #         )
    ckpt = get_ckpt(checkpath)
    path_to_ckpt = os.path.join(checkpath, f"checkpoints/{ckpt}")
    if config.atten_pool: 
        model = ContrastWrapperAttention.load_from_checkpoint(
            path_to_ckpt
        )  # map_location=torch.device('cuda:1')
    else:
        model = ContrastWrapper.load_from_checkpoint(
                path_to_ckpt
            )  # map_location=torch.device('cuda:1')
    model.eval()
    # optimizer = torch.optim.AdamW(model.parameters())
    # checkpoint = torch.load('checkpoint.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.genes_for_test = set().union(*test_genes)
    model.save_dir = config.save_dir
    trainer = load_trainer(config)
    trainer.test(model, DataLoader(test_ds, batch_size=int(config.train_batch_size)))


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

    train_gene_filedir, train_gene_filenames, train_genes, valid_genes, test_genes = prepare_genes(config)
    # test_genes = ['NDUFB6', 'LMNA', 'TRAPPC13', 'BDH1', 'SCO2', 'DNAJC27']

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
    config['eQTL_guided'] = False

    pl.seed_everything(int(config.seed), workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    eval_test_genes(config, train_genes[::3], checkpath)
    #eval_test_genes(config, valid_genes, checkpath)
    #eval_test_genes(config, test_genes, checkpath)


if __name__ == "__main__":
    main()

import os
#os.environ["CUDA_VISIBLE_DEVICES"] ="1"
from performer.train_gtex2 import *
from performer.models.contrast_wrapper_attention_multiheads_rev2 import ContrastWrapperAttention
from performer.models.contrast_wrapper import ContrastWrapper
from performer.ism_performer import get_ckpt
from box import Box
import torch

torch.use_deterministic_algorithms(True)


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
        #valid_genes = []
        # test_genes = []
        #train_gene_filenames = ["1000_train_genes.txt"]
        train_gene_filenames = [config["train_gene_file"]] 
        val_gene_filename = config["val_gene_file"]
        test_gene_filename = config["test_gene_file"]

        train_gene_path = os.path.join(data_dir, "genes", model_type, "egenes", train_gene_filenames[0])
        train_genes = parse_gene_files(train_gene_path)
        valid_gene_path = os.path.join(data_dir, "genes", model_type, "egenes", val_gene_filename)
        valid_genes = parse_gene_files(valid_gene_path)
        test_gene_path = os.path.join(data_dir, "genes", model_type, "egenes", test_gene_filename) 
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

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    current_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(current_dir, "../data")
    config_path = os.path.join(current_dir, config_path)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config["model_type"] = model_type
    config["DATA_DIR"] = DATA_DIR
    config = Box(config)
    config.update({"fold": fold})
    config.update({"num_gpus": num_gpus})
    #config.update({"save_dir": fold})

    train_gene_filedir, train_gene_filenames, train_genes, valid_genes, test_genes = prepare_genes(config)
    # test_genes = ['NDUFB6', 'LMNA', 'TRAPPC13', 'BDH1', 'SCO2', 'DNAJC27']

    checkpath = os.path.join(
        current_dir,
        f"../results/{config['experiment_name']}/{model_type}/{train_gene_filenames[0].strip('.txt')}/Fold-{fold}/{runid}",
    )

    config.update({"valid_genes": valid_genes})
    config.update({"test_genes": test_genes})
    config.update(
        {
            "save_dir": os.path.join(
                current_dir,
                f"../results/{config['experiment_name']}/{model_type}/{train_gene_filenames[0].strip('.txt')}/Fold-{fold}/{runid}/train_genes",  # need to change
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

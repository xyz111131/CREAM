import lightning.pytorch as pl
import pandas as pd
import torch
from torchmetrics.regression import PearsonCorrCoef, R2Score, SpearmanCorrCoef
import psutil


import os


class MetricLogger(pl.Callback):
    """
    A callback intended to log and save raw predictions and summary statistics (R2, PCC) from validation/test loops
    """

    def __init__(self):
        super().__init__()
        self.metrics_history = {
            "epoch": [],
            "pearsonr": [],
            "r2": [],
            "gene_name": [],
            "tissue": [],
            "per_gene_tissue_val_loss": [],
            "per_gene_tissue_mse_loss": [],
            "per_gene_tissue_contrast_loss": [],
            "gene_split": [],
            "donor_split": [],
        }  #'spearmanr':[],
        self.sync_dist = True
        self.rank_zero = False

    def log_predictions(self, trainer, pl_module, donor_split):
        # pred_df = pd.DataFrame(pl_module.pred_dict).apply(pd.Series.explode)
        pred_df = pd.DataFrame(pl_module.pred_dict)
        pred_df = pred_df.reset_index(names="tissue").melt(
            id_vars=["tissue"], var_name="gene", value_name="y_pred"
        )
        pred_df["y_pred"] = pred_df["y_pred"].apply(
            lambda y_pred: [x.item() for x in y_pred]
        )  # y_pred is a list of torch tensors. convert it to a list of floats
        pred_df = pred_df.explode(
            "y_pred"
        )  # explode the list of floats so they each get their own row and the list is removed
        # target_df = pd.DataFrame(pl_module.target_dict).apply(pd.Series.explode)
        target_df = pd.DataFrame(pl_module.target_dict)
        target_df = target_df.reset_index(names="tissue").melt(
            id_vars=["tissue"], var_name="gene", value_name="y_true"
        )
        target_df["y_true"] = target_df["y_true"].apply(
            lambda y_true: [x.item() for x in y_true]
        )  # y_pred is a list of torch tensors. convert it to a list of floats
        target_df = target_df.explode("y_true")
        # donor_df = pd.DataFrame(pl_module.donor_dict).apply(pd.Series.explode)
        donor_df = pd.DataFrame(pl_module.donor_dict)
        donor_df = donor_df.reset_index(names="tissue").melt(
            id_vars=["tissue"], var_name="gene", value_name="donor"
        )
        donor_df = donor_df.explode("donor")

        rank_df = pd.DataFrame(pl_module.rank_dict)
        rank_df = rank_df.reset_index(names="tissue").melt(
            id_vars=["tissue"], var_name="gene", value_name="rank"
        )
        rank_df = rank_df.explode("rank")

        if donor_split == 'test' and (pl_module.attn_dict): # only output attention weights during test time
            attn_weights_df = pd.DataFrame(pl_module.attn_dict)
            attn_weights_df = attn_weights_df.reset_index(names="tissue").melt(
                id_vars=["tissue"], var_name="gene", value_name="attn_weights"
            )
            attn_weights_df["attn_weights"] = attn_weights_df["attn_weights"].apply(
                lambda attn_weights: [[x.item() for x in ls ]for ls in attn_weights]
            ) 
            attn_weights_df = attn_weights_df.explode("attn_weights")
            attn_weights_df['attn_weights'] = attn_weights_df['attn_weights'].apply(lambda x: ','.join(map(str,x)))

            attn_inds_df = pd.DataFrame(pl_module.attn_inds_dict)
            attn_inds_df = attn_inds_df.reset_index(names="tissue").melt(
                id_vars=["tissue"], var_name="gene", value_name="attn_inds"
            )
            attn_inds_df["attn_inds"] = attn_inds_df["attn_inds"].apply(
                lambda attn_inds: [[x.item() for x in ls ]for ls in attn_inds]
            ) 
            attn_inds_df = attn_inds_df.explode("attn_inds")
            attn_inds_df['attn_inds'] = attn_inds_df['attn_inds'].apply(lambda x: ','.join(map(str,x)))

            df = pd.concat(
                [pred_df, target_df[["y_true"]], donor_df[["donor"]], rank_df[["rank"]],attn_inds_df[['attn_inds']], attn_weights_df[['attn_weights']]], axis=1
            )
        else:
            df = pd.concat(
                [pred_df, target_df[["y_true"]], donor_df[["donor"]], rank_df[["rank"]]], axis=1
            )

        df["end_of_epoch"] = self.epoch
        # trainer.logger.experiment.log({'PredictionResults':wandb.Table(dataframe = df)})

        memory_info = psutil.virtual_memory()
        print(f"Total Memory: {memory_info.total / (1024**3):.2f} GB")
        print(f"Used Memory: {memory_info.used / (1024**3):.2f} GB")

        print(f"Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

        
        df.to_csv(
            os.path.join(
                pl_module.save_dir, f"Prediction_Results_{self.epoch}_in_{donor_split}_donors.csv"
            ),
            index=False,
        )

    def add_to_metrics_history(
        self, gene_name, tissue, pl_module, gene_split, pearson, spearman, r2_score, donor_split
    ):
        """
        Indexes dictionaries of predictions and true values for a given gene and records summary statistics
        """
        # loss_fn = pl_module.loss_fn #if the train loss func is a multi gene one, it doesn't work here because this code runs one gene at a time. The valid loss func returns a single gene loss func
        all_predictions = torch.cat(pl_module.pred_dict[gene_name][tissue]).detach().cpu()
        all_targets = torch.cat(pl_module.target_dict[gene_name][tissue]).detach().cpu()

        gene_pearsonr = pearson(all_predictions, all_targets)
        # gene_spearmanr = spearman(all_predictions, all_targets)
        gene_r2 = r2_score(all_predictions, all_targets)

        loss_mse = (
            pl_module.mse_loss(all_predictions.unsqueeze(1), all_targets.unsqueeze(1)).cpu().numpy()
        )  # unsqueeze because the values per tissue were returned and have shape [batch_size]. Add a tissue dimension, which the loss function expects
        loss_contrast = (
            pl_module.constr_loss(all_predictions.unsqueeze(1), all_targets.unsqueeze(1))
            .cpu()
            .numpy()
        )
        loss_val = loss_mse + loss_contrast
        self.metrics_history["pearsonr"].append(gene_pearsonr.detach().cpu().numpy())
        # self.metrics_history['spearmanr'].append(gene_spearmanr.detach().cpu().numpy())
        self.metrics_history["r2"].append(gene_r2.detach().cpu().numpy())
        self.metrics_history["tissue"].append(tissue)
        self.metrics_history["gene_name"].append(gene_name)
        self.metrics_history["epoch"].append(self.epoch)
        self.metrics_history["per_gene_tissue_val_loss"].append(loss_val)
        self.metrics_history["per_gene_tissue_mse_loss"].append(loss_mse)
        self.metrics_history["per_gene_tissue_contrast_loss"].append(loss_contrast)
        self.metrics_history["gene_split"].append(gene_split)
        self.metrics_history["donor_split"].append(donor_split)

    def log_per_gene_per_tissue_metrics(self, trainer, pl_module, donor_split):
        metrics_history = pd.DataFrame(self.metrics_history)
        df = metrics_history[metrics_history["donor_split"] == donor_split].copy()
        name = f"CrossIndivMetrics_{donor_split}_donors_Epoch{self.epoch}_rank{trainer.global_rank}.csv"
        df.to_csv(os.path.join(pl_module.save_dir, name), index=False)

        assert len(df["donor_split"].unique()) == 1 & (
            df["donor_split"].unique().item() == donor_split
        ), f"You have data from {df['donor_split'].unique()} donors but should only have {donor_split}!"
        for gene_split in list(df["gene_split"].unique()):
            # save results across all tissues for this epoch
            mean_epoch_corr = df[df["gene_split"] == gene_split]["pearsonr"].mean()
            # mean_epoch_corr2 = df[df['gene_split'] == gene_split]['spearmanr'].mean()
            mean_epoch_r2 = df[df["gene_split"] == gene_split]["r2"].mean()
            mean_epoch_mse_loss = df[df["gene_split"] == gene_split][
                "per_gene_tissue_mse_loss"
            ].mean()
            mean_epoch_contrast_loss = df[df["gene_split"] == gene_split][
                "per_gene_tissue_contrast_loss"
            ].mean()
            mean_epoch_loss = df[df["gene_split"] == gene_split]["per_gene_tissue_val_loss"].mean()
            epoch_dict = {
                f"mean_pearsonr_across_{gene_split}_genes_across_{donor_split}_donors": mean_epoch_corr.item(),
                # f'mean_spearmanr_across_{gene_split}_genes_across_{donor_split}_donors':mean_epoch_corr2.item(),
                f"mean_r2_across_{gene_split}_genes_across_{donor_split}_donors": mean_epoch_r2.item(),
                f"mean_loss_{gene_split}_genes_across_{donor_split}_donors": mean_epoch_loss.item(),
                f"mean_mse_loss_{gene_split}_genes_across_{donor_split}_donors": mean_epoch_mse_loss.item(),
                f"mean_contrast_loss_{gene_split}_genes_across_{donor_split}_donors": mean_epoch_contrast_loss.item(),
                #f"mean_r2_across_all_genes_across_{donor_split}_donors": df["r2"].mean().item(),
            }
            #print(f"mean_r2_across_{gene_split}_genes_across_{donor_split}_donors: {mean_epoch_r2.item()}")
            if trainer.num_devices > 1:  # sync across multiple GPUs, if applicable
                pl_module.log_dict(epoch_dict, sync_dist=self.sync_dist, rank_zero_only=self.rank_zero, on_epoch = True)
            else:
                pl_module.log_dict(epoch_dict)
        

    def add_all_genes_to_metrics_history(self, pl_module, donor_split):
        pearson = PearsonCorrCoef().to("cpu")
        spearman = []  # SpearmanCorrCoef().to("cpu")
        r2_score = R2Score().to("cpu")
        for gene_name in pl_module.pred_dict.keys():
            if gene_name in pl_module.genes_for_training:
                gene_split = "train"
            elif gene_name in pl_module.genes_for_valid:
                gene_split = "valid"
            elif gene_name in pl_module.genes_for_test:
                gene_split = "test"
            else:
                raise Exception(
                    f"Gene {gene_name} not in desired train set, valid set, or test set"
                )
            for tissue_idx, tissue in enumerate(
                pl_module.tissues_to_train
            ):  # loop through each tissue and calculate pearsonr
                self.add_to_metrics_history(
                    gene_name,
                    tissue,
                    pl_module,
                    gene_split,
                    pearson,
                    spearman,
                    r2_score,
                    donor_split,
                )

    def get_epoch(self, trainer, pl_module):
        if (
            pl_module.global_step == 0
        ):  # when performing validation loop before training, no steps will have occured but epoch is defined as 0. It will be overwritte after the 0th training epoch. Define as -1 to save these values.
            epoch = -1
        else:
            epoch = pl_module.current_epoch
        self.epoch = epoch

    def log_and_save_eval(self, trainer, pl_module, donor_split):
        self.get_epoch(trainer, pl_module)  # define self.epoch
        self.add_all_genes_to_metrics_history(pl_module, donor_split)
        self.log_predictions(
            trainer, pl_module, donor_split
        )  # log y_pred and y_true for each donor and gene being evaluated
        self.log_per_gene_per_tissue_metrics(trainer, pl_module, donor_split)

        # Clear predictions and targets for the next epoch
        pl_module.pred_dict = {}
        pl_module.target_dict = {}
        pl_module.donor_dict = {}
        pl_module.rank_dict = {}
        self.metrics_history = {
            "epoch": [],
            "pearsonr": [],
            "r2": [],
            "gene_name": [],
            "tissue": [],
            "per_gene_tissue_val_loss": [],
            "per_gene_tissue_mse_loss": [],
            "per_gene_tissue_contrast_loss": [],
            "gene_split": [],
            "donor_split": [],
        }  #'spearmanr':[],

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_and_save_eval(trainer, pl_module, "valid")
        # print(f"[{trainer.global_rank}] callback_metrics: {trainer.callback_metrics}")
        torch.cuda.empty_cache()

    def on_test_epoch_end(self, trainer, pl_module):
        self.sync_dist = False
        self.rank_zero = True
        if trainer.global_rank == 0:
            self.log_and_save_eval(trainer, pl_module, "test")
        torch.cuda.empty_cache()

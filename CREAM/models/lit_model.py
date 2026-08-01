# adapted from Performer: https://github.com/shirondru/enformer_fine_tuning/tree/master/code 

import lightning.pytorch as pl
import torch
import torch.nn as nn
import wandb
from contextlib import contextmanager


# def exists(val):
#     return val is not None

# def Sequential(*modules):
#     return nn.Sequential(*filter(exists, modules))


def masked_mse(y_hat, y):
    """
    removes NaNs from y (for example, when someone is missing data from a particular tissue)
    """
    mask = torch.isnan(y)
    mse = torch.mean((y[~mask] - y_hat[~mask]) ** 2)
    return mse


def masked_mae(y_hat, y):
    """
    removes NaNs from y (for example, when someone is missing data from a particular tissue)
    """
    mask = torch.isnan(y)
    # mse = torch.mean(torch.abs(y[~mask]-y_hat[~mask]))
    loss_fn = nn.SmoothL1Loss()
    mse = loss_fn(y[~mask], y_hat[~mask])
    return mse


def masked_cross_entropy(y_hat, y):
    """
    removes NaNs from y (for example, when someone is missing data from a particular tissue)
    """
    mask = torch.isnan(y)
    # mask2 = mask.unsqueeze(1).repeat(1,y_hat.shape[1],1)
    # mse = torch.mean(torch.abs(y[~mask]-y_hat[~mask]))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=1e-4)  # 1e-2
    mse = loss_fn(y_hat[~mask, :], y[~mask])  # reduce one dimension after masking, (N, C) vs (N)
    return mse


def get_diff_one_gene(y, y_hat):
    """
    Calculates pairwise differences between observed expression values among different people, as well as pairwise differences between predicted expression values among different people. Returns the difference between these matrices
    """
    true_differences = y.unsqueeze(1) - y
    predicted_differences = y_hat.unsqueeze(1) - y_hat
    diff = predicted_differences - true_differences
    return diff


def remove_l_tri_flatten_3d(diff):
    """
    Removes the lower triangle of `diff`, the difference between pairwise observed differences and pairwise predicted differences. Thus, it keeps only pairwise comparisons between different, unique pairs of people. This handles the case when training on multiple tissues at the same time
    and the matrix is 3D. Returns a flattened array.
    """
    # Create a mask for the upper triangle
    mask_2d = torch.triu(
        torch.ones(diff.shape[0:2], dtype=torch.bool), diagonal=1
    )  # everything above diagonal is True, everything else is False. So things are True when it is the paired differences between a person and someone else

    mask_3d = mask_2d.unsqueeze(-1).repeat(
        1, 1, diff.shape[-1]
    )  # Make this mask 3D. So the same 2D matrix is repeated along the third dimension, so the same mask is applied to each tissue in the tissue dimension
    upper_tri_3d = torch.where(
        mask_3d.to(diff.device), diff, torch.tensor(float("nan")).to(diff.device)
    )  # convert False values to NaN and returns the diff values where True. Uses the 3d mask as the condition

    # Flatten and remove NaN values
    flat = upper_tri_3d.flatten()
    return flat[~torch.isnan(flat)]


class LitModel(pl.LightningModule):
    def __init__(
        self,
        tissues_to_train,
        save_dir,
        train_dataset,
        learning_rate,
        alpha,
        discretize_expr,
        raw_expr,
        contrast_embed,
        eQTL_guided=False,
        max_epoch=None,
        batch_size=None,
        genes_for_training=None,
        genes_for_valid=None,
        genes_for_test=None,
        final_div_factor = 1e4,
        pct_start = 0.3,
    ):
        super().__init__()  # q: LitModel?
        required_args = {
            "max_epoch": max_epoch,
            "batch_size": batch_size,
            "genes_for_training": genes_for_training,
            "genes_for_valid": genes_for_valid,
            "genes_for_test": genes_for_test,
        }
        missing_args = [name for name, value in required_args.items() if value is None]
        if missing_args:
            raise TypeError(f"Missing required arguments: {', '.join(missing_args)}")

        self.save_hyperparameters()
        self.attn_dict = {}
        self.attn_inds_dict = {}
        self.pred_dict = {}
        self.target_dict = {}
        self.donor_dict = {}
        self.rank_dict = {}
        self.tissues_to_train = tissues_to_train
        self.save_dir = save_dir
        self.train_dataset = train_dataset
        self.genes_for_training = set().union(*genes_for_training)
        self.genes_for_valid = set().union(*genes_for_valid)
        self.genes_for_test = set().union(*genes_for_test)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.final_div_factor = final_div_factor
        self.pct_start = pct_start
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.ensure_no_gene_overlap()
        self.discretize_expr = discretize_expr
        self.raw_expr = raw_expr
        self.contrast_embed = contrast_embed
        self.eQTL_guided = bool(eQTL_guided)
        if self.raw_expr:
            self.expr_bins = torch.tensor(
                [0.2, 1, 2, 3]
            ).cuda()  # hard coded for now. [-1, -0.3, 0.3, 1]
            self.expr_bins_means = torch.tensor(
                [0.1, 0.5, 1.5, 2.5, 4.2]
            ).cuda()  # hard coded for now.  [-1.5, -0.62, 0, 0.62, 1.5]
        else:
            self.expr_bins = torch.tensor([-1, -0.3, 0.3, 1]).cuda()  # hard coded for now.
            self.expr_bins_means = torch.tensor(
                [-1.5, -0.62, 0, 0.62, 1.5]
            ).cuda()  # hard coded for now.

    @staticmethod
    def wandb_hook(run, step):
        """Weights & Biases histogram hook."""

        def hook(module, input, output):
            if hasattr(module, "name"):
                if module.name == "attn":
                    run.experiment.log(
                        {module.name: wandb.Histogram(output.detach().cpu())}, step=step
                    )

        return hook

    def ensure_no_gene_overlap(self):
        """
        Since it can be valuable to evaluate train genes on held out people -- to understand the extent to which the model has learned to predict loci it has seen in unseen people with unseen variants --
        It complicates the train/valid/test split of genes, because validation and test datalaoders may contain train genes (in principle).

        To deal with this, I ensure here that the genes in the train dataset are only the desired train genes. During the validation and test loops, since evaluations
        may be performed on train/valid/test genes, I denote which are which using class attributes. These are accessed within the ValidMetricLogger Callback, and when this object calculates performance on these genes,
        it labels them as train,valid, or test genes accordingly. Therefore, I triple check here, once again, that there is no overlap between these genes
        Within the `train_full_enformer function`, I also call `final_check_ensure_no_gene_or_person_leakage_in_datasets` to ensure the genes being used in each of these datasets follow these conventions.

        Ensuring that there is no overlap between genes for training and genes for valid and genes for test ensures genes cannot have more than 1 label, or have conflicting labels, during evaluation in ValidMetricLogger
        `final_check_ensure_no_gene_or_person_leakage_in_datasets` Will check to ensure the genes coming from these datalaoders align with these labels (while being flexible to the fact that, for example, the valid datalaoder can yield train *And* valid genes)
        """
        train_gene_set = self.genes_for_training
        valid_gene_set = self.genes_for_valid
        test_gene_set = self.genes_for_test

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

        assert len(self.genes_for_training) > 0, "You have no genes to train on!"

        # assert len([x for x in self.train_dataset.genes_in_dataset if x not in self.genes_for_training]) == 0, "There are genes in the train set besides those desired for training!"

    # def loss_fn(self,y_hat, y, alpha = 0.5): #does not need to inherit nn.Module because no trainable variables within
    #     mse = masked_mse(y_hat,y)
    #     diff = get_diff_one_gene(y,y_hat)
    #     flat = remove_l_tri_flatten_3d(diff)
    #     contrastive_term = torch.mean(flat**2)

    #     loss = (alpha * mse) + ((1 - alpha) * contrastive_term)
    #     return loss

    def mse_loss(
        self, y_hat, y
    ):  # does not need to inherit nn.Module because no trainable variables within
        return masked_mse(y_hat, y)

    def mae_loss(
        self, y_hat, y
    ):  # does not need to inherit nn.Module because no trainable variables within
        return masked_mae(y_hat, y)

    def cross_entropy_loss(
        self, y_hat, y
    ):  # does not need to inherit nn.Module because no trainable variables within
        return masked_cross_entropy(y_hat, y)

    def constr_loss(
        self, y_hat, y
    ):  # does not need to inherit nn.Module because no trainable variables within
        diff = get_diff_one_gene(y, y_hat)
        flat = remove_l_tri_flatten_3d(diff)
        contrastive_term = torch.mean(flat**2)
        return contrastive_term

    def constr_loss_abs(
        self, y_hat, y
    ):  # does not need to inherit nn.Module because no trainable variables within
        diff = get_diff_one_gene(y, y_hat)
        flat = remove_l_tri_flatten_3d(diff)
        loss_fn = nn.SmoothL1Loss()
        # contrastive_term = torch.mean(torch.abs(flat))
        contrastive_term = loss_fn(flat, torch.zeros_like(flat))
        return contrastive_term

    # def predict_step(self, batch, batch_idx,dataloader_idx = 0):

    #     # get em
    #     x = batch['seq']
    #     y_hat = self(x) #q: forward
    #     y_hat = y_hat[:,y_hat.shape[1]//2,:] #keep value at center of sequence. The sequence axis is removed

    #     #
    #     return y_hat

    def training_step(self, batch, batch_idx, alpha=0.5):
        # x, y,genes,donor,dataloader_idx = batch
        y = batch["expr_array"]
        # if self.contrast_embed:
        #     y_hat1, y_hat2 = self(batch)
        #     loss, contrastive, mse = self.loss_fn(y_hat1, y_hat2, y, alpha)
        # else:
        res = self(batch)
        y_hat = res["y"]
        if self.eQTL_guided:
            weights = batch["eQTL_array"].squeeze(1) # B * 1 * T * L -> B * T * L, only the gene axis
            tss = batch["tss"].squeeze(1) * weights.shape[2]
            tss = tss[0].int()
            attn_weights = res.get("attn_weights")
            attn_mask = res.get("attn_mask")
            attn_inds = res.get("attn_inds")
            if attn_weights is None or attn_mask is None:
                raise ValueError(
                    "eQTL_guided=True requires `attn_weights` and `attn_mask` in model output"
                )

            attn_weights = attn_weights.squeeze(-1)   # B * T * L * 1 -> B * T * L
            #l = weights.shape[2]
            #weights[:, :, ((l //2) -1) : ((l //2) + 1) ] = weights[:, :, ((l //2) -1) : ((l //2) + 1) ] + 0.5 # more attention to TSS
            weights[:, :, tss ] = weights[:, :, tss ] + 0.001 # more attention to TSS, was 0.01
            
            weights = weights[:-1, :, attn_inds]
            attn_mask_expanded = attn_mask.unsqueeze(1)  # tissue
            masked_weights = weights.masked_fill_(attn_mask_expanded, 0)
           
            masked_weights = masked_weights / masked_weights.sum(dim=2, keepdim=True)

            masked_weights = masked_weights.masked_fill(attn_mask_expanded, float('nan'))
            attn_weights = attn_weights.masked_fill(attn_mask_expanded, float('nan'))

            loss, contrastive, mse = self.loss_fn(y_hat, y, attn_weights, masked_weights, alpha)
        else:
            loss, contrastive, mse = self.loss_fn(y_hat, y, alpha)

        # if self.discretize_expr > 0:
        #     loss = self.cross_entropy_loss(y_hat, y)
        #     contrastive = 0
        #     mse = 0
        # elif self.contrast_embed:
        #     mse = self.mae_loss(y_hat1, y)
        #     contrastive = self.constr_loss2(y_hat2, y)
        #     loss = (alpha * mse) + ((1 - alpha) * contrastive)
        # else:
        #     mse = self.mae_loss(y_hat, y)
        #     contrastive = self.constr_loss_abs(y_hat, y)
        #     loss = (alpha * mse) + ((1 - alpha) * contrastive)

        freq = int(len(self.genes_for_training) * 16 / y.shape[1])
        if batch_idx % freq == 0:
            #     with torch.nn.modules.module.register_module_forward_hook(
            #         LitModel.wandb_hook(self.logger, self.global_step)):
            #         self(batch)  # call forward method
            wandb.log({"train/loss": loss, "train/contrastive": contrastive, "train/mse": mse})
            # optimizer = self.trainer.optimizers[0]
            # lr = optimizer.param_groups[0]['lr']
            # wandb.log({"learning_rate": lr}) #"step": self.global_step
        # print(f"{batch_idx=}, {self.global_step=}, {self.current_epoch=}, {self.trainer.optimizers[0].param_groups[0]['lr']}, {self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()}")

        # self.log('mse',mse,batch_size = y.shape[0], logger = True, on_step = True, on_epoch = False) #accumulates loss over the epoch and only logs the average at the end, to reduce logging overhead
        # self.log('contrastive',contrastive,batch_size = y.shape[0], logger = True, on_step = True, on_epoch = False) #if averaging, batch_size should be n_indiv * n_gene so the weight for each step is correct
        self.log("train_loss", loss, batch_size=y.shape[0], on_step=False, on_epoch=True)
        return {"loss": loss}

    def save_eval_results(self, y_hat, y, donor, rank, gene_names, attn_weights=None, attn_inds=None):
        """
        To log predictions and true values for each donor during validation/test loop, separately in each tissue (if training multiple tissues)
        split all the genes in a gene set assuming no overlaps of genes among gene set
        """
        for gene_name in gene_names:
            gene_name = gene_name[0]
            if gene_name not in self.pred_dict:
                self.pred_dict[gene_name] = {tissue: [] for tissue in self.tissues_to_train}
                self.target_dict[gene_name] = {tissue: [] for tissue in self.tissues_to_train}
                self.donor_dict[gene_name] = {tissue: [] for tissue in self.tissues_to_train}
                self.rank_dict[gene_name] = {tissue: [] for tissue in self.tissues_to_train}
                if attn_weights is not None:
                    self.attn_dict[gene_name] = {tissue: [] for tissue in self.tissues_to_train}
                    self.attn_inds_dict[gene_name] = {tissue: [] for tissue in self.tissues_to_train}

        # tissue_indices_without_data = torch.isnan(y).nonzero(as_tuple=True)[1] #NaNs exist in tissues where donor is missing data. Expected validation batch size is 1, so this finds tissues where this individual is missing data, and results for this person won't be saved for this tissue

        for tissue_idx, tissue in enumerate(self.tissues_to_train):
            # if tissue_idx not in tissue_indices_without_data:
            for gene_idx, gene_name in enumerate(gene_names):
                gene_name = gene_name[0]
                if not torch.isnan(y[:, gene_idx]).any():
                    if self.discretize_expr > 0:
                        exp_tensor = torch.exp(y_hat[:, gene_idx, : self.discretize_expr])
                        probabilities = exp_tensor / exp_tensor.sum(dim=1, keepdim=True)
                        offsets = y_hat[:, gene_idx, self.discretize_expr :]
                        self.pred_dict[gene_name][tissue].append(
                            torch.concat((probabilities, offsets), dim=1)
                        )
                    else:
                        self.pred_dict[gene_name][tissue].append(y_hat[:, gene_idx, tissue_idx])
                    self.target_dict[gene_name][tissue].append(y[:, gene_idx, tissue_idx])
                    self.donor_dict[gene_name][tissue].append(donor)
                    self.rank_dict[gene_name][tissue].append(rank)  # q: ?
                    if attn_weights is not None:
                        if attn_weights.ndim == 3: 
                            self.attn_dict[gene_name][tissue].append(attn_weights.squeeze()) ## only work for single gene, for model 3
                        else:
                            self.attn_dict[gene_name][tissue].append(attn_weights[:, tissue_idx, :].squeeze()) ## only work for single gene for models 1 & 2
                        self.attn_inds_dict[gene_name][tissue].append(attn_inds.squeeze()) 

    @contextmanager
    def precision_context(self, precision=None):
        """Context manager to temporarily change precision."""
        if precision == "fp32":
            # Save original dtype
            original_dtypes = {}
            for name, param in self.named_parameters():
                original_dtypes[name] = param.dtype
                param.data = param.data.to(torch.float32)

            # Use float precision for forward pass
            with torch.autocast(device_type="cuda", enabled=False):
                try:
                    yield
                finally:
                    # Restore original dtypes
                    for name, param in self.named_parameters():
                        if name in original_dtypes:
                            param.data = param.data.to(original_dtypes[name])
        else:
            # Use default precision (will use trainer's global setting)
            yield

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # x, y,gene_name,donor,_ = batch
        ##gene_names = ','.join(gene_names)
        rank = self.trainer.global_rank
        # if self.contrast_embed:
        #     y_hat,_ = self(batch)#self.predict_step(batch,batch_idx)
        # else:

        with self.precision_context(precision="fp32"):
            res = self(batch)
            attn_weights = None
            attn_inds = None
            y_hat = res['y']
            if 'attn_weights' in res:
                attn_weights = res['attn_weights']
                attn_inds = res['attn_inds']

        if self.contrast_embed:
            for i  in range(len(batch["expr_array"])-1):
                y = batch["expr_array"][[i]] - batch["expr_array"][[i+1]]
                donor = batch["individual"][i] + ":" + batch["individual"][i+1]
                gene_names = batch["gene_names"]  # will be [(g, g, g, g)]
                if attn_weights is not None:
                    self.save_eval_results(
                        y_hat[[i]], y, donor, rank, gene_names, attn_weights[[i]], attn_inds,
                    )  # results for each person are stored and then loss/r2/pcc will be computed at the end of the epoch using all individuals via MetricLogger Callback
                else:
                    self.save_eval_results(
                        y_hat[[i]], y, donor, rank, gene_names,
                    ) 
        else:
            for i  in range(len(batch["expr_array"])):
                y = batch["expr_array"][[i]] 
                donor = batch["individual"][i]
                gene_names = batch["gene_names"] 
                self.save_eval_results(
                        y_hat[[i]], y, donor, rank, gene_names,
                    ) 


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # x, y,gene_name,donor,_ = batch
        ##gene_names = ','.join(gene_names)
        rank = self.trainer.global_rank
        if rank != 0:
            return None  # Skip test on other ranks
        # if self.contrast_embed:
        #     y_hat,_ = self(batch)#self.predict_step(batch,batch_idx)
        # else:

        with self.precision_context(precision="fp32"):
            attn_weights = None
            attn_inds = None
            res = self(batch)
            y_hat = res['y']
            if 'attn_weights' in res:
                attn_weights = res['attn_weights']
                attn_inds = res['attn_inds']
        
        if self.contrast_embed:
            for i  in range(len(batch["expr_array"])-1):
                y = batch["expr_array"][[i]] - batch["expr_array"][[i+1]]
                donor = batch["individual"][i] + ":" + batch["individual"][i+1]
                gene_names = batch["gene_names"]  # will be [(g, g, g, g)]
                if attn_weights is not None:
                    self.save_eval_results(
                        y_hat[[i]], y, donor, rank, gene_names, attn_weights[[i]], attn_inds,
                    )  # results for each person are stored and then loss/r2/pcc will be computed at the end of the epoch using all individuals via MetricLogger Callback
                else:
                    self.save_eval_results(
                        y_hat[[i]], y, donor, rank, gene_names,
                    )                
        else:
            for i  in range(len(batch["expr_array"])):
                y = batch["expr_array"][[i]] 
                donor = batch["individual"][i]
                gene_names = batch["gene_names"] 
                self.save_eval_results(
                        y_hat[[i]], y, donor, rank, gene_names,
                    ) 

    def on_train_epoch_end(self):
        self.train_dataset.shuffle_and_define_epoch()  # shuffle dataset. Ensures this occurs on the main process even if num_workers > 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=0.1, #lr=self.learning_rate, 
        )  # , weight_decay=0.1 # default
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            final_div_factor= self.final_div_factor, ## 1
            pct_start  = self.pct_start
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }

        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': scheduler
        # }
        return [optimizer], [lr_scheduler_config]

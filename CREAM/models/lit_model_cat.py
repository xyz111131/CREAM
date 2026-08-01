# adapted from Performer: https://github.com/shirondru/enformer_fine_tuning/tree/master/code

import torch
import wandb

from CREAM.models.lit_model import LitModel


# Cut points for the discretized expression head, and the representative value
# of each of the len(edges) + 1 resulting bins. The head predicts a distribution
# over the bins plus an offset from each bin's representative value, so
# discretize_bins in the config must equal len(means).
RAW_EXPR_BIN_EDGES = [0.2, 1.0, 2.0, 3.0]
RAW_EXPR_BIN_MEANS = [0.1, 0.5, 1.5, 2.5, 4.2]

NORM_EXPR_BIN_EDGES = [
    -1.4657382, -0.9538726, -0.5449254, -0.1777120,
    0.1777120, 0.5449254, 0.9538726, 1.4657382,
]
NORM_EXPR_BIN_MEANS = [
    -2.1982420, -1.1966764, -0.7441936, -0.3592933, 0.0000000,
    0.3592933, 0.7441936, 1.1966764, 2.1982420,
]


class LitModelCat(LitModel):
    """LitModel for heads that predict a bin distribution plus a per-bin offset.

    Predictions are shaped (batch, gene, tissue, 2 * discretize_expr): the first
    discretize_expr entries along the last axis are bin logits, the remainder are
    offsets from the corresponding bin mean. This differs from the continuous
    LitModel, whose predictions are (batch, gene, tissue).
    """

    def __init__(
        self,
        *args,
        expr_bin_edges=None,
        expr_bin_means=None,
        bin_offset_alpha=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if expr_bin_edges is None:
            expr_bin_edges = RAW_EXPR_BIN_EDGES if self.raw_expr else NORM_EXPR_BIN_EDGES
        if expr_bin_means is None:
            expr_bin_means = RAW_EXPR_BIN_MEANS if self.raw_expr else NORM_EXPR_BIN_MEANS

        if len(expr_bin_means) != len(expr_bin_edges) + 1:
            raise ValueError(
                f"Got {len(expr_bin_edges)} bin edges but {len(expr_bin_means)} bin means; "
                f"expected {len(expr_bin_edges) + 1} means."
            )
        if self.discretize_expr != len(expr_bin_means):
            raise ValueError(
                f"discretize_bins is {self.discretize_expr} but the bins define "
                f"{len(expr_bin_means)} categories. Set discretize_bins to "
                f"{len(expr_bin_means)} in the config."
            )

        # Registered as buffers so they follow the module across devices under DDP.
        self.register_buffer(
            "expr_bins", torch.tensor(expr_bin_edges, dtype=torch.float32)
        )
        self.register_buffer(
            "expr_bins_means", torch.tensor(expr_bin_means, dtype=torch.float32)
        )

        self.bin_offset_alpha = bin_offset_alpha

    def bin_targets(self, y):
        """Bin `y` and return (bin index, offset of y from its bin's mean, valid mask).

        The mask comes from `y` rather than the bin indices because bucketize maps
        NaN onto a real bin, so donors missing a tissue would otherwise train the
        classifier towards whichever bin NaN lands in.
        """
        y = y.float()
        valid = ~torch.isnan(y)

        y_bins = torch.bucketize(y, self.expr_bins, right=True)  # each bin is (-inf,), [,)
        offsets = torch.gather(
            y.unsqueeze(-1) - self.expr_bins_means, dim=-1, index=y_bins.unsqueeze(-1)
        ).squeeze(-1)
        return y_bins, offsets, valid

    def split_prediction(self, y_hat):
        """Split a head output into (bin logits, per-bin offsets)."""
        return y_hat[..., : self.discretize_expr], y_hat[..., self.discretize_expr :]

    def training_step(self, batch, batch_idx):
        y = batch["expr_array"]
        res = self(batch)
        y_hat = res["y"]
        if self.eQTL_guided:
            weights = batch["eQTL_array"].squeeze(1)  # B * 1 * T * L -> B * T * L, only the gene axis
            tss = batch["tss"].squeeze(1) * weights.shape[2]
            tss = tss[0].int()
            attn_weights = res.get("attn_weights")
            attn_mask = res.get("attn_mask")
            attn_inds = res.get("attn_inds")
            if attn_weights is None or attn_mask is None:
                raise ValueError(
                    "eQTL_guided=True requires `attn_weights` and `attn_mask` in model output"
                )

            attn_weights = attn_weights.squeeze(-1)  # B * T * L * 1 -> B * T * L
            weights[:, :, tss] = weights[:, :, tss] + 0.001  # more attention to TSS, was 0.01

            weights = weights[:-1, :, attn_inds]
            attn_mask_expanded = attn_mask.unsqueeze(1)  # tissue
            masked_weights = weights.masked_fill_(attn_mask_expanded, 0)

            masked_weights = masked_weights / masked_weights.sum(dim=2, keepdim=True)

            masked_weights = masked_weights.masked_fill(attn_mask_expanded, float("nan"))
            attn_weights = attn_weights.masked_fill(attn_mask_expanded, float("nan"))

            loss, contrastive, mse = self.loss_fn(y_hat, y, attn_weights, masked_weights)
        else:
            loss, contrastive, mse = self.loss_fn(y_hat, y)

        freq = int(len(self.genes_for_training) * 16 / y.shape[1])
        if batch_idx % freq == 0:
            wandb.log({"train/loss": loss, "train/contrastive": contrastive, "train/mse": mse})

        self.log("train_loss", loss, batch_size=y.shape[0], on_step=False, on_epoch=True)
        return {"loss": loss}

    def save_eval_results(
        self, y_hat, y, donor, rank, gene_names, attn_weights=None, attn_inds=None
    ):
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

        for tissue_idx, tissue in enumerate(self.tissues_to_train):
            for gene_idx, gene_name in enumerate(gene_names):
                gene_name = gene_name[0]
                if not torch.isnan(y[:, gene_idx]).any():
                    logits, offsets = self.split_prediction(y_hat[:, gene_idx, tissue_idx])
                    probabilities = torch.softmax(logits.float(), dim=-1)
                    # MetricLogger_cat reads back the first ncat columns as
                    # probabilities and the rest as offsets.
                    self.pred_dict[gene_name][tissue].append(
                        torch.concat((probabilities, offsets.float()), dim=1)
                    )
                    self.target_dict[gene_name][tissue].append(y[:, gene_idx, tissue_idx])
                    self.donor_dict[gene_name][tissue].append(donor)
                    self.rank_dict[gene_name][tissue].append(rank)
                    if attn_weights is not None:
                        if attn_weights.ndim == 3:
                            self.attn_dict[gene_name][tissue].append(attn_weights.squeeze())  ## only work for single gene, for model 3
                        else:
                            self.attn_dict[gene_name][tissue].append(attn_weights[:, tissue_idx, :].squeeze())  ## only work for single gene for models 1 & 2
                        self.attn_inds_dict[gene_name][tissue].append(attn_inds.squeeze())

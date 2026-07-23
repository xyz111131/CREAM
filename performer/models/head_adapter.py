from performer.models.lit_model import LitModel


import torch
import torch.nn as nn
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper


class HeadAdapterWrapper(LitModel):
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
        max_epoch,
        batch_size,
        genes_for_training,
        genes_for_valid,
        genes_for_test,
    ):
        super().__init__(
            tissues_to_train,
            save_dir,
            train_dataset,
            learning_rate,
            alpha,
            discretize_expr,
            raw_expr,
            contrast_embed,
            max_epoch,
            batch_size,
            genes_for_training,
            genes_for_valid,
            genes_for_test,
        )
        enformer = Enformer.from_pretrained(
            "EleutherAI/enformer-official-rough",
            target_length=-1,  # disable cropping for use with shorter sequences
        )
        # enformer = Enformer.from_hparams(
        #     dim = 1536,
        #     depth = 11,
        #     heads = 8,
        #     # output_heads = dict(human = 5313, mouse = 1643),
        #     target_length = -1,
        # )
        if self.discretize_expr > 0:
            self.model = HeadAdapterWrapper(
                enformer=enformer,
                num_tracks=2 * self.discretize_expr * len(self.tissues_to_train),
                post_transformer_embed=False,  # important to keep False
                output_activation=nn.Identity(),
            )
        else:
            self.model = HeadAdapterWrapper(
                enformer=enformer,
                num_tracks=len(self.tissues_to_train),
                post_transformer_embed=False,  # important to keep False
                output_activation=nn.Identity(),
            )

    def forward(self, x):
        y_hat = self.model(x["seq_array"].squeeze(1), freeze_enformer=False)
        # get the bin with tss
        tss_bin = y_hat.shape[1] * (x["tss"].squeeze(1))
        tss_bin_floor = torch.floor(tss_bin)
        tss_bin_ceil = torch.ceil(tss_bin) - 1
        y_hat = (
            y_hat[torch.arange(y_hat.shape[0]), tss_bin_floor.int(), :]
            + y_hat[torch.arange(y_hat.shape[0]), tss_bin_ceil.int(), :]
        ) / 2  # y_hat.shape[1]//2
        if self.discretize_expr > 0:
            y_hat = torch.reshape(
                y_hat,
                (
                    y_hat.shape[0],
                    y_hat.shape[1] // (2 * self.discretize_expr),
                    (2 * self.discretize_expr),
                ),
            )
        return y_hat

    def loss_fn(self, y_hat, y, alpha=0.5):
        if self.discretize_expr > 0:
            y_bins = torch.bucketize(y, self.expr_bins, right=True)  # each bin is (-inf,), [,)
            y_offets = y.unsqueeze(2) - self.expr_bins_means

            # gather element for correct bins
            bins = y_bins.unsqueeze(2)
            y_offets = torch.gather(y_offets, dim=2, index=bins).squeeze()

            y_pred = y_hat[:, :, self.discretize_expr :]
            y_pred = torch.gather(y_pred, dim=2, index=bins).squeeze()

            contrastive = self.cross_entropy_loss(y_hat[:, :, : self.discretize_expr], y_bins)
            mse = self.mae_loss(y_pred, y_offets)
            loss = (alpha * mse) + ((1 - alpha) * contrastive)
        else:
            mse = self.mae_loss(y_hat, y)
            contrastive = self.constr_loss_abs(y_hat, y)
            loss = (alpha * mse) + ((1 - alpha) * contrastive)
        return loss, contrastive, mse

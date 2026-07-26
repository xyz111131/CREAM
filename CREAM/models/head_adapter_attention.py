from CREAM.models.lit_model import LitModel


import torch
import torch.nn as nn
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper


class HeadAdapterWrapper_Attention(LitModel):
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

        self.enformer = HeadAdapterWrapper(
            enformer=enformer,
            num_tracks=enformer.dim,
            post_transformer_embed=False,  # important to keep False
            output_activation=nn.Identity(),
        )

        self.dim_head = 256
        self.scale = self.dim_head**-0.5
        self.to_query = nn.Linear(enformer.dim, self.dim_head, bias=False)
        self.to_key = nn.Linear(enformer.dim, self.dim_head, bias=False)
        ntracks = (
            len(self.tissues_to_train)
            if self.discretize_expr == 0
            else 2 * self.discretize_expr * len(self.tissues_to_train)
        )

        self.to_pred = nn.Sequential(
            nn.Linear(enformer.dim, enformer.dim // 2),
            nn.GELU(),
            nn.Linear(enformer.dim // 2, ntracks),
            nn.Identity(),
        )

    def forward(self, x):
        y_hat = self.enformer(x["seq_array"].squeeze(1), freeze_enformer=False)
        # attention pooling
        q2 = self.to_query(y_hat[:, y_hat.shape[1] // 2, :])  # b,1,d
        k2 = self.to_key(y_hat)  # b,l, d
        scores = torch.einsum("b d, b l d-> b l", q2, k2) * self.scale  # add scale
        weights = scores.softmax(dim=-1)

        output = torch.einsum("b l, b l d-> b d", weights, y_hat)

        # to prediction
        y_hat = self.to_pred(output)
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

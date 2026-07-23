from performer.models.lit_model import LitModel, masked_mae

from typing import Optional
import torch
import torch.nn as nn
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper


class ContrastWrapperAttention(LitModel):
    """
    Attention pooling to combine all the output bins
    """

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
        dim_head=512,
        output_bins=1,
        output_activation: Optional[nn.Module] = nn.Identity(),
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

        # assert isinstance(enformer, Enformer)
        # self.enformer = enformer
        enformer_hidden_dim = enformer.dim  # *2

        self.dim_head = dim_head
        self.scale = self.dim_head**-0.5

        self.enformer = HeadAdapterWrapper(
            enformer=enformer,
            num_tracks=enformer_hidden_dim,
            post_transformer_embed=False,  # important to keep False
            output_activation=nn.Identity(),
        )

        self.output_bins = output_bins
        self.seq_query_norm = nn.LayerNorm(enformer_hidden_dim)

        self.to_query = nn.Linear(enformer_hidden_dim * 2, self.dim_head, bias=False)
        self.to_key = nn.Linear(enformer_hidden_dim * 2, self.dim_head, bias=False)

        # self.to_pred  = nn.Sequential(
        #     nn.Linear(enformer_hidden_dim, enformer_hidden_dim//2),
        #     nn.GELU(),
        #     nn.Linear(enformer_hidden_dim//2, enformer_hidden_dim//4),
        #     nn.GELU(),
        #     nn.Linear(enformer_hidden_dim//4, 1),
        #     output_activation
        # )

        self.to_pred_contrast = nn.Sequential(
            nn.Linear(enformer_hidden_dim * 2, enformer_hidden_dim),
            nn.GELU(),
            nn.Linear(enformer_hidden_dim, enformer_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(enformer_hidden_dim // 2, 1),
            output_activation,
        )

        for name, module in self.named_modules():
            module.name = name

    def constr_loss2(
        self, y_hat, y
    ):  # does not need to inherit nn.Module because no trainable variables within
        n = y.shape[0]
        y_diff = y - torch.cat((y[1:n, :], y[0:1, :]))  # [0 : (n - 1), :]
        return masked_mae(y_hat, y_diff)

    def loss_fn(self, y_hat2, y, alpha=0.5):
        mse = 0  # self.mae_loss(y_hat1, y)
        contrastive = self.constr_loss2(y_hat2, y)
        loss = contrastive
        return loss, contrastive, mse

    def forward(self, x):
        """
        b - batch
        l - sequence length
        d - dimension
        h - attention heads
        """
        n = x["seq_array"].shape[1]
        b = x["seq_array"].shape[0]

        enformer_kwargs = dict()

        # sequence embeddings for all genes
        # seq = rearrange(x['seq_array'], 'b n l d -> (b n) l d')
        seq = x["seq_array"].squeeze(1)  # only one gene
        # seq_embeddings = get_enformer_embeddings(self.enformer, seq, freeze = False, train_layernorms_only = False, train_last_n_layers_only = None, enformer_kwargs = enformer_kwargs)
        seq_embeddings = self.enformer(seq, freeze_enformer=False)

        mid = seq_embeddings.shape[1] // 2
        nbin = (self.output_bins + 1) // 2
        # seq_embeddings = seq_embeddings[:, mid - nbin : mid + nbin, :]  # only keep the middle bins
        # average over all bins
        # seq_embeddings = torch.mean(seq_embeddings, dim=1)
        seq_embeddings = self.seq_query_norm(seq_embeddings)  # b,l,d

        # constrast layer
        seq_embeddings1 = seq_embeddings  # [0 : (b - 1), :, :]
        seq_embeddings2 = torch.cat((seq_embeddings[1:b, :, :], seq_embeddings[0:1, :, :]), 0)
        # seq_constrast_embeddings1 = seq_embeddings1 - seq_embeddings2
        # seq_constrast_embeddings2 = seq_embeddings1 * seq_embeddings2
        seq_contrast_embeddings = torch.cat((seq_embeddings1, seq_embeddings2), 2)  # b-1, l, 2d

        # attention pooling
        q2 = self.to_query(seq_contrast_embeddings[:, mid - nbin : mid + nbin, :])  # b-1,2,2d
        k2 = self.to_key(seq_contrast_embeddings)  # b-1,2,2d
        scores = torch.einsum("b n d, b l d-> b n l", q2, k2) * self.scale  # add scale
        weights = scores.softmax(dim=-1)

        seq_contrast_embeddings = torch.einsum(
            "b n l, b l d-> b n d", weights, seq_contrast_embeddings
        )
        seq_contrast_embeddings = torch.mean(seq_contrast_embeddings, dim=1)

        # to prediction
        # seq_embeddings = seq_embeddings.squeeze(1) # only one bin
        # seq_contrast_embeddings = seq_contrast_embeddings.squeeze(1)
        # pred = self.to_pred(seq_embeddings)
        pred_contrast = self.to_pred_contrast(seq_contrast_embeddings)

        return pred_contrast

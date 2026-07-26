from CREAM.models.lit_model import LitModel

from typing import Optional
import torch.nn as nn
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper


class HeadAdapterGeneEmbeddingWrapper(LitModel):
    def __init__(
        self,
        tissues_to_train,
        save_dir,
        train_dataset,
        learning_rate,
        alpha,
        discretize_expr,
        contrast_embed,
        max_epoch,
        batch_size,
        genes_for_training,
        genes_for_valid,
        genes_for_test,
        gene_embedding_dim=256,
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

        self.enformer = HeadAdapterWrapper(
            enformer=enformer,
            num_tracks=gene_embedding_dim,  # enformer_hidden_dim,
            post_transformer_embed=False,  # important to keep False
            output_activation=nn.Identity(),
        )

        self.output_bins = output_bins
        self.seq_query_norm = nn.LayerNorm(gene_embedding_dim)  # enformer_hidden_dim
        self.gene_query_norm = nn.LayerNorm(gene_embedding_dim)

        self.to_pred = nn.Sequential(
            nn.Linear(gene_embedding_dim, gene_embedding_dim // 2),
            nn.GELU(),
            nn.Linear(gene_embedding_dim // 2, 1),
            output_activation,
        )

        # self.to_pred  = nn.Sequential(
        #     nn.Linear(enformer_hidden_dim + gene_embedding_dim, (enformer_hidden_dim + gene_embedding_dim)//2),
        #     nn.GELU(),
        #     nn.Linear((enformer_hidden_dim + gene_embedding_dim)//2, 1),
        #     output_activation
        # )

        for name, module in self.named_modules():
            module.name = name

    def loss_fn(self, y_hat, y, alpha=0.5):
        mse = self.mae_loss(y_hat, y)
        contrastive = self.constr_loss_abs(y_hat, y)
        loss = (alpha * mse) + ((1 - alpha) * contrastive)
        return loss, contrastive, mse

    def forward(self, x):
        """
        b - batch
        l - sequence length
        d - dimension
        h - attention heads
        """
        # n = x['seq_array'].shape[1]
        # b = x['seq_array'].shape[0]

        # enformer_kwargs = dict()

        # sequence embeddings for all genes
        # seq = rearrange(x['seq_array'], 'b n l d -> (b n) l d')
        seq = x["seq_array"].squeeze(1)  # only one gene
        # seq_embeddings = get_enformer_embeddings(self.enformer, seq, freeze = False, train_layernorms_only = False, train_last_n_layers_only = None, enformer_kwargs = enformer_kwargs)
        seq_embeddings = self.enformer(seq, freeze_enformer=False)

        mid = seq_embeddings.shape[1] // 2
        nbin = self.output_bins // 2
        seq_embeddings = seq_embeddings[
            :, mid - nbin : mid + nbin + 1, :
        ]  # only keep the middle bin
        seq_embeddings = self.seq_query_norm(seq_embeddings)  # b,l,d

        gene_embeddings = self.gene_query_norm(x["embd_array"])  # b,n,d

        # seq + gene embeddings
        # seq_gene_embeddings = torch.cat((seq_embeddings,  gene_embeddings ),2)
        seq_gene_embeddings = seq_embeddings + gene_embeddings

        # to prediction
        seq_gene_embeddings = seq_gene_embeddings.squeeze(1)  # only one bin
        pred = self.to_pred(seq_gene_embeddings)

        return pred

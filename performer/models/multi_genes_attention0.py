from performer.models.Attention import Attention
from performer.models.lit_model import LitModel

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper


class MultiGeneAttentionWrapper0(LitModel):
    def __init__(
        self,
        tissues_to_train,
        save_dir,
        train_dataset,
        learning_rate,
        alpha,
        max_epoch,
        batch_size,
        genes_for_training,
        genes_for_valid,
        genes_for_test,
        gene_embedding_dim=256,
        heads=8,
        dim_head=64,
        output_bins=5,
        output_activation: Optional[nn.Module] = nn.Identity(),
    ):
        super().__init__(
            tissues_to_train,
            save_dir,
            train_dataset,
            learning_rate,
            alpha,
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
            num_tracks=enformer_hidden_dim,
            post_transformer_embed=False,  # important to keep False
            output_activation=nn.Identity(),
        )

        self.output_bins = output_bins

        self.seq_query_norm = nn.LayerNorm(enformer_hidden_dim)
        # self.gene_query_norm = nn.LayerNorm(gene_embedding_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = heads * dim_head
        self.seq_to_queries = nn.Linear(enformer_hidden_dim, inner_dim, bias=False)
        # self.gene_to_queries = nn.Linear(gene_embedding_dim, inner_dim, bias = False)

        # self.null_key = nn.Parameter(torch.randn(inner_dim))
        # self.null_value = nn.Parameter(torch.randn(inner_dim))

        # self.null_key = torch.zeros(inner_dim)
        # self.null_value = torch.zeros(inner_dim)

        self.seq_to_key_values = nn.Linear(enformer_hidden_dim, inner_dim * 2, bias=False)
        # self.gene_to_key_values = nn.Linear(gene_embedding_dim, inner_dim * 2, bias = False)

        self.attn = Attention(heads, self.scale)

        self.to_out = nn.Sequential(
            nn.Linear(
                inner_dim, inner_dim
            ),  # might change it to more linear layers and different activation function
            nn.GELU(),
            nn.Linear(inner_dim, enformer_hidden_dim),
            # nn.GELU(),
        )

        # attention pooling layer between of output_bins of a gene
        self.to_query = nn.Linear(enformer_hidden_dim, dim_head, bias=False)
        self.to_key = nn.Linear(enformer_hidden_dim, dim_head, bias=False)

        self.to_pred = nn.Sequential(
            nn.Linear(enformer_hidden_dim, 1), Rearrange("b n ... 1 -> b ... n"), output_activation
        )

        for name, module in self.named_modules():
            module.name = name

    def forward(self, x):
        """
        b - batch
        n - number of genes
        l - sequence length
        d - dimension
        h - attention heads
        """
        h = self.heads
        n = x["seq_array"].shape[1]
        b = x["seq_array"].shape[0]

        enformer_kwargs = dict()

        # sequence embeddings for all genes
        seq = rearrange(x["seq_array"], "b n l d -> (b n) l d")
        # seq_embeddings = get_enformer_embeddings(self.enformer, seq, freeze = False, train_layernorms_only = False, train_last_n_layers_only = None, enformer_kwargs = enformer_kwargs)
        seq_embeddings = self.enformer(seq, freeze_enformer=False)

        mid = seq_embeddings.shape[1] // 2
        nbin = self.output_bins // 2
        seq_embeddings = seq_embeddings[
            :, mid - nbin : mid + nbin + 1, :
        ]  # only keep the middle 5 bins
        seq_embeddings = rearrange(seq_embeddings, "(b n) l d -> b n l d", n=n)

        # attention layer
        seq_embeddings = self.seq_query_norm(seq_embeddings)  # b,n, l,d
        # gene_embeddings = self.gene_query_norm(x['embd_array']) #b,n,d

        # gene_embd_query = self.gene_to_queries(gene_embeddings)
        # gene_embd_query = rearrange(gene_embd_query, 'b n d -> b n 1 d')
        q = self.seq_to_queries(seq_embeddings)  # + gene_embd_query #b, n, l, d
        k, v = self.seq_to_key_values(seq_embeddings[:, :, nbin, :]).chunk(
            2, dim=-1
        )  # + self.gene_to_key_values(gene_embeddings)).chunk(2, dim = -1) #b, n, d

        # null_k, null_v = map(lambda t: repeat(t, 'd -> b 1 d', b = b), (self.null_key, self.null_value))

        # k = torch.cat((k, null_k), dim = 1)
        # v = torch.cat((v, null_v), dim = 1)

        k = F.pad(k, (0, 0, 0, 1))
        v = F.pad(v, (0, 0, 0, 1))

        # split out head
        q = rearrange(q, "b n l (h d) -> b h n l d", h=h)
        k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (k, v))

        # sim = torch.einsum('b h n l d, b h m d -> b h n l m', q, k) * self.scale

        # # masking, avoiding attention within same gene
        # #context_mask = F.pad(context_mask, (1, 0), value = True)
        # context_mask = torch.ones(n, n+1).bool().cuda()
        # context_mask.fill_diagonal_(False)

        # context_mask =rearrange(context_mask, 'n m -> 1 1 n 1 m')
        # sim = sim.masked_fill(~context_mask, -torch.finfo(sim.dtype).max)

        # # attention

        # attn = sim.softmax(dim = -1)
        attn = self.attn(q, k, n)

        # aggregate

        out = torch.einsum("b h n l m, b h m d -> b h n l d", attn, v)
        out = rearrange(out, "b h n l d -> b n l (h d)", h=h)

        # combine heads

        branch_out = self.to_out(out)  # might add a linear norm before this

        # residual

        seq_embeddings = seq_embeddings + branch_out

        # attention pooling
        q2 = self.to_query(seq_embeddings[:, :, nbin, :])  # b,n,d
        k2 = self.to_key(seq_embeddings)  # b,n,l, d
        scores = torch.einsum("b n d, b n l d-> b n l", q2, k2) * self.scale  # add scale
        weights = scores.softmax(dim=-1)

        output = torch.einsum("b n l, b n l d-> b n d", weights, seq_embeddings)

        # to prediction

        pred = self.to_pred(output)

        return pred  # q: means forward? yes

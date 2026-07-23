#from performer.models.attention_pooling_wi import AttentionPooling
from performer.models.lit_model import LitModel, masked_mae

from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from enformer_pytorch import Enformer
# from enformer_pytorch.finetune import HeadAdapterWrapper
from performer.models.head_adapter_wrapper.custom_head_adapter_wrapper import CustomHeadAdapterWrapper


class ContrastWrapperAttention(LitModel):
    """
    Attention pooling to combine all the output bins
    Separate output layers for each tissue
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
        dim_head= 1024,  # need to reduce for multi-tissue
        heads = 4,
        final_div_factor = 1e4,
        pct_start = 0.3,
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
            final_div_factor,
            pct_start
        )

        enformer = Enformer.from_pretrained(
            "EleutherAI/enformer-official-rough",
            target_length=-1,  # disable cropping for use with shorter sequences
        )

        # assert isinstance(enformer, Enformer)
        # self.enformer = enformer
        enformer_hidden_dim = enformer.dim *2

        self.dim_head = dim_head # total attention dim including all heads
        self.scale = (self.dim_head // heads) **-0.5
        self.heads = heads
        self.ntissue = len(tissues_to_train)

        self.enformer = CustomHeadAdapterWrapper(
            enformer=enformer,
            num_tracks= 1, #enformer_hidden_dim, not used
            post_transformer_embed=False,  # important to keep False
            output_activation=nn.Identity(),
        )

        self.output_bins = output_bins
        self.seq_query_norm = nn.LayerNorm(enformer_hidden_dim)

        self.conv_squash = nn.ModuleList()
        _channel_dim = 768
        for i in range(6):
            self.conv_squash.append(nn.Linear(_channel_dim + i * 128, out_features=64))
        
        squash_dim = 64 * 5
        self.squash_norm = nn.LayerNorm(squash_dim)

        self.to_query = nn.Linear(squash_dim * 2, self.dim_head * self.ntissue, bias=False) #squash_dim 
        self.to_key = nn.Linear(enformer_hidden_dim * 2, self.dim_head * self.ntissue, bias=False)
        self.to_value = nn.Linear(enformer_hidden_dim * 2, enformer_hidden_dim // 2 * self.ntissue, bias=False)

         # RoPE
        self.rope_length = 3072
        self.rope_freqs = self.register_buffer(
            "rope_frequencies", precompute_freqs(dim_head // heads, self.rope_length),  # 1536: arbitrary rope length
            persistent=False,
        )
        # self.freqs_k = get_rope_frequencies(1536, dim_head // heads)


        # self.to_pred  = nn.Sequential(
        #     nn.Linear(enformer_hidden_dim, enformer_hidden_dim//2),
        #     nn.GELU(),
        #     nn.Linear(enformer_hidden_dim//2, enformer_hidden_dim//4),
        #     nn.GELU(),
        #     nn.Linear(enformer_hidden_dim//4, 1),
        #     output_activation
        # )

        self.attention_score = nn.Linear(enformer_hidden_dim // 2, 1)


        if self.discretize_expr > 0:
            self.to_pred_per_tissue = nn.ModuleList([
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(enformer_hidden_dim // 2, enformer_hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(enformer_hidden_dim // 4, 2 * self.discretize_expr),
                    output_activation,
                )
                for _ in range(self.ntissue)
            ])

        else: 
            self.to_pred_per_tissue = nn.ModuleList([
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(enformer_hidden_dim // 2, enformer_hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(enformer_hidden_dim // 4, 1),
                    output_activation,
                )
                for _ in range(self.ntissue)
            ])

        for name, module in self.named_modules():
            module.name = name

            #module.register_forward_hook(self._save_features)

    def constr_loss2(
        self, y_hat, y
    ):  # does not need to inherit nn.Module because no trainable variables within
        n = y.shape[0]
        y_diff = y[0 : (n - 1), :] - y[1:n, :]
        return masked_mae(y_hat, y_diff)

    def loss_fn(self, y_hat2, y, alpha=0.5):
        mse = 0  # self.mae_loss(y_hat1, y)
        if self.discretize_expr > 0:
            y_bins = torch.bucketize(y, self.expr_bins, right=True)  # each bin is (-inf,), [,)
            y_offets = y.unsqueeze(3) - self.expr_bins_means

            # gather element for correct bins
            bins = y_bins.unsqueeze(3)
            y_offets = torch.gather(y_offets, dim=3, index=bins).squeeze()

            y_pred = y_hat2[:, :,:, self.discretize_expr :]
            y_pred = torch.gather(y_pred, dim=2, index=bins).squeeze()

            contrastive = self.cross_entropy_loss(y_hat2[:, :,:, : self.discretize_expr], y_bins)

        else:
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

        h = self.heads
        b, n, l, _ = x["seq_array"].shape
        nt = self.ntissue

        #assert b == 18, 'batch size not equal'

        enformer_kwargs = dict()

        # sequence embeddings for all genes
        # seq = rearrange(x['seq_array'], 'b n l d -> (b n) l d')
        seq = x["seq_array"].squeeze(1)  # only one gene
        # seq_embeddings = get_enformer_embeddings(self.enformer, seq, freeze = False, train_layernorms_only = False, train_last_n_layers_only = None, enformer_kwargs = enformer_kwargs)
        # seq_embeddings = self.enformer(seq, freeze_enformer=False)
        enformer_features_dict = self.enformer(seq, freeze_enformer=False)
        seq_embeddings = enformer_features_dict["seq_embeddings"]

        # Find diffs
        seq_diffs = (seq[:-1] != seq[1:]).any(dim=2)  # (B-1, L)
        seq_diffs[:, (l//2 - 1) : (l//2 + 1 )] = True  # Force middle bin to be True
        # seq_diff_inds = seq_diffs.nonzero(as_tuple=False)  # To create masks
        # seq_diff_bool = seq_diffs.any(dim=0)  # (Nsnp)
        # seq_diff_bool[l // 2] = True 
        # seq_diff_all_inds = seq_diffs.any(dim=0, keepdims=True).nonzero(as_tuple=False).squeeze()
        seq_diff_all_inds = seq_diffs.any(dim=0).nonzero(as_tuple=False).squeeze(dim=1)  # (Nsnp)

        #features_stem = enformer_features_dict["enformer.enformer.stem.2"].transpose(2, 1)
        all_squashed_features = []
        for i, k in enumerate(["enformer.conv_tower.0", "enformer.conv_tower.1", "enformer.conv_tower.2", "enformer.conv_tower.3", "enformer.conv_tower.4"]):
            v = enformer_features_dict[f"enformer.{k}"]
            features_conv_tower = v.transpose(2, 1)
            shrink = l // features_conv_tower.shape[1]
            stem_diff_inds = seq_diff_all_inds // shrink 

            sampled_conv_tower_features = features_conv_tower[:, stem_diff_inds]

            all_squashed_features.append(self.conv_squash[i](sampled_conv_tower_features))  # (?, 32)

        squashed_features = torch.cat(all_squashed_features, dim=2)
        

        # mid = seq_embeddings.shape[1] // 2
        # nbin = (self.output_bins + 1) // 2
        # seq_embeddings = seq_embeddings[:, mid - nbin : mid + nbin, :]  # only keep the middle bins
        # average over all bins
        # seq_embeddings = torch.mean(seq_embeddings, dim=1)
        seq_embeddings = self.seq_query_norm(seq_embeddings)  # b,l,d

        squashed_features = self.squash_norm(squashed_features)
        squashed_features1 = squashed_features[0 : (b - 1), :, :]
        squashed_features2 = squashed_features[1:b, :, :]
        squashed_contrast_features = torch.cat((squashed_features1, squashed_features2), 2)  # b-1, l, 2d

        # constrast layer
        seq_embeddings1 = seq_embeddings[0 : (b - 1), :, :]
        seq_embeddings2 = seq_embeddings[1:b, :, :]
        # seq_constrast_embeddings1 = seq_embeddings1 - seq_embeddings2
        # seq_constrast_embeddings2 = seq_embeddings1 * seq_embeddings2
        seq_contrast_embeddings = torch.cat((seq_embeddings1, seq_embeddings2), 2)  # b-1, l, 2d

        # attention pooling
        q2 = self.to_query(squashed_contrast_features)  # b-1,l,2d
        k2 = self.to_key(seq_contrast_embeddings)  # b-1,l,2d
        v2 = self.to_value(seq_contrast_embeddings)

        # split out head
        q2 = rearrange(q2, "b l (h t d) -> b h t l d", h=h, t = nt)
        k2, v2 = map(lambda t: rearrange(t, "b l (h t d) -> b h t l d", h=h, t = nt), (k2, v2))

        # RoPE
        freqs_k = self.rope_frequencies[::(self.rope_length//k2.shape[3])]  # 1536/96 = 16
        freqs_q = self.rope_frequencies[seq_diff_all_inds // (l // self.rope_length)]  # 12,288 / 1536(seq_len)

        q2_rope = apply_rope(q2, freqs_q)
        k2_rope = apply_rope(k2, freqs_k)
       
        # dot
        scores = torch.einsum("b h t n d, b h t l d-> b h t n l", q2_rope, k2_rope) * self.scale  # add scale

        # Apply weights
        weights = scores.softmax(dim=-1)
        v2_weighted = torch.einsum(
            "b h t n l, b h t l d-> b h t n d", weights, v2  # b, h, 2, d // h
        )
        v2_reshaped = rearrange(v2_weighted, "b h t n d -> b t n (h d)", h=h)

      
        # attention pooling
        attn_weights = self.attention_score(v2_reshaped)         # (B, T, L, 1), L is number of SNPs + TSS
        # mask 
        attn_mask = ~seq_diffs[:, seq_diff_all_inds]
        attn_mask_expanded = attn_mask.unsqueeze(2).unsqueeze(1)
        masked_attn_weights = attn_weights.masked_fill_(attn_mask_expanded, float('-inf'))
        attn_weights = masked_attn_weights.softmax(dim=2)  # (B, T, L, 1), softmax over L

        # Weighted sum of features across L
        v2_updated = torch.sum(attn_weights * v2_reshaped, dim=2)    # (B,T, C)
        #v2_updated = torch.mean(v2_reshaped, dim=1)

        # to prediction
        # seq_embeddings = seq_embeddings.squeeze(1) # only one bin
        # seq_contrast_embeddings = seq_contrast_embeddings.squeeze(1)
        # pred = self.to_pred(seq_embeddings)
        # pred_contrast = self.to_pred_contrast(v2_updated)

        preds = []
        for t, mod in enumerate(self.to_pred_per_tissue):
            out = mod(v2_updated[:, t, :])          # shape (B, 1)
            preds.append(out)

        # stack into (B, T, 1)
        pred_contrast = torch.stack(preds, dim=1)

        # switch axes for pred_contrast between tissue and gene idx (which is one-dim)
        pred_contrast = pred_contrast.transpose(1,2)

        return { 'y': pred_contrast, 'attn_weights': attn_weights, 'attn_inds':  seq_diff_all_inds}


def apply_rope(x, freqs):
    """Apply RoPE to query/key: x shape [batch, seq, n_heads, head_dim]"""
    x = x.float()
    x1 = x[..., ::2]  # even dims
    x2 = x[..., 1::2]  # odd dims
    cos = freqs[..., 0]  # [seq, dim//2]
    sin = freqs[..., 1]
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated.flatten(-2)  # merge last dim back to head_dim


# def get_rope_frequencies(seq_len, dim):
#     theta = 10000 ** (-torch.arange(0, dim, 2).float() / dim)
#     positions = torch.arange(seq_len).float()
#     return torch.outer(positions, theta)


def precompute_freqs(dim, seq_len, base=10000):
    """Precompute frequency matrix for RoPE."""
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.einsum('i,j->ij', positions, theta)  # [seq_len, dim//2]
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # [seq_len, dim//2, 2]

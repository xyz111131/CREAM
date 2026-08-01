#from CREAM.models.attention_pooling_wi import AttentionPooling
from CREAM.models.lit_model import masked_mae
from CREAM.models.lit_model_cat import LitModelCat

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from enformer_pytorch import Enformer
# from enformer_pytorch.finetune import HeadAdapterWrapper
from CREAM.models.head_adapter_wrapper.custom_head_adapter_wrapper import CustomHeadAdapterWrapper


class ContrastWrapperAttentionCat(LitModelCat):
    """
    Discretized counterpart of ContrastWrapperAttention in
    contrast_wrapper_attention_multiheads_rev2.py.

    Attention pooling to combine all the output bins, separate output layers for
    each tissue. Each tissue head emits 2 * discretize_expr values -- a logit per
    expression bin followed by an offset per bin -- instead of a single value.
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
        eQTL_guided=False,
        max_epoch=None,
        batch_size=None,
        genes_for_training=None,
        genes_for_valid=None,
        genes_for_test=None,
        dim_head= 1024,  # need to reduce for multi-tissue
        heads = 4,
        final_div_factor = 1e4,
        pct_start = 0.3,
        output_bins=1,
        output_activation: Optional[nn.Module] = nn.Identity(),
        expr_bin_edges=None,
        expr_bin_means=None,
        bin_offset_alpha=0.9,
    ):
        if not discretize_expr or discretize_expr <= 0:
            raise ValueError(
                "ContrastWrapperAttentionCat needs discretize_bins > 0; use "
                "contrast_wrapper_attention_multiheads_rev2.ContrastWrapperAttention "
                "for continuous training."
            )

        super().__init__(
            tissues_to_train,
            save_dir,
            train_dataset,
            learning_rate,
            alpha,
            discretize_expr,
            raw_expr,
            contrast_embed,
            eQTL_guided,
            max_epoch,
            batch_size,
            genes_for_training,
            genes_for_valid,
            genes_for_test,
            final_div_factor,
            pct_start,
            expr_bin_edges=expr_bin_edges,
            expr_bin_means=expr_bin_means,
            bin_offset_alpha=bin_offset_alpha,
        )

        enformer = Enformer.from_pretrained(
            "EleutherAI/enformer-official-rough",
            target_length=-1,  # disable cropping for use with shorter sequences
        )

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

        self.attention_score = nn.Linear(enformer_hidden_dim // 2, 1)

        # bin logits followed by per-bin offsets
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

        for name, module in self.named_modules():
            module.name = name

            #module.register_forward_hook(self._save_features)

    def constr_loss2(self, y_hat, y):
        """Cross entropy over expression bins, and MAE on the offset predicted
        for the bin each donor pair actually falls in."""
        n = y.shape[0]
        y_diff = y[0 : (n - 1), :] - y[1:n, :]

        y_bins, y_offsets, valid = self.bin_targets(y_diff)
        logits, offsets = self.split_prediction(y_hat)

        offsets = torch.gather(offsets, dim=-1, index=y_bins.unsqueeze(-1)).squeeze(-1)

        mae = masked_mae(offsets, y_offsets)
        cross_entropy = F.cross_entropy(
            logits[valid], y_bins[valid], label_smoothing=1e-4
        )
        return mae, cross_entropy

    def loss_fn(self, y_hat2, y, atten_weights=0, weights=0, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        mae, cross_entropy = self.constr_loss2(y_hat2, y)
        expr_loss = (
            self.bin_offset_alpha * mae + (1 - self.bin_offset_alpha) * cross_entropy
        )

        if self.eQTL_guided:
            attn_mae = self.mae_loss(atten_weights, weights)
            loss = alpha * expr_loss + (1 - alpha) * attn_mae
        else:
            attn_mae = 0
            loss = expr_loss
        return loss, cross_entropy, mae

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

        enformer_kwargs = dict()

        # sequence embeddings for all genes
        seq = x["seq_array"].squeeze(1)  # only one gene
        enformer_features_dict = self.enformer(seq, freeze_enformer=False)
        seq_embeddings = enformer_features_dict["seq_embeddings"]

        # Find diffs
        seq_diffs = (seq[:-1] != seq[1:]).any(dim=2)  # (B-1, L)
        tss_ind = x["tss"].squeeze(1) * l
        tss_ind = tss_ind[0].int()
        seq_diffs[:,tss_ind] = True #seq_diffs[:, (l//2 - 1) : (l//2 + 1 )] = True  # Force middle bin to be True
        seq_diff_all_inds = seq_diffs.any(dim=0).nonzero(as_tuple=False).squeeze(dim=1)  # (Nsnp)

        all_squashed_features = []
        for i, k in enumerate(["enformer.conv_tower.0", "enformer.conv_tower.1", "enformer.conv_tower.2", "enformer.conv_tower.3", "enformer.conv_tower.4"]):
            v = enformer_features_dict[f"enformer.{k}"]
            features_conv_tower = v.transpose(2, 1)
            shrink = l // features_conv_tower.shape[1]
            stem_diff_inds = seq_diff_all_inds // shrink

            sampled_conv_tower_features = features_conv_tower[:, stem_diff_inds]

            all_squashed_features.append(self.conv_squash[i](sampled_conv_tower_features))  # (?, 32)

        squashed_features = torch.cat(all_squashed_features, dim=2)

        seq_embeddings = self.seq_query_norm(seq_embeddings)  # b,l,d

        squashed_features = self.squash_norm(squashed_features)
        squashed_features1 = squashed_features[0 : (b - 1), :, :]
        squashed_features2 = squashed_features[1:b, :, :]
        squashed_contrast_features = torch.cat((squashed_features1, squashed_features2), 2)  # b-1, l, 2d

        # constrast layer
        seq_embeddings1 = seq_embeddings[0 : (b - 1), :, :]
        seq_embeddings2 = seq_embeddings[1:b, :, :]
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

        preds = []
        for t, mod in enumerate(self.to_pred_per_tissue):
            out = mod(v2_updated[:, t, :])          # shape (B, 2 * discretize_expr)
            preds.append(out)

        # stack into (B, T, 2 * discretize_expr)
        pred_contrast = torch.stack(preds, dim=1)

        # add the (single) gene axis, giving B, G, T, 2 * discretize_expr. The
        # continuous model transposes here instead, which only works because its
        # head is one unit wide.
        pred_contrast = pred_contrast.unsqueeze(1)

        return { 'y': pred_contrast, 'attn_weights': attn_weights, 'attn_inds':  seq_diff_all_inds, 'attn_mask': attn_mask}


def apply_rope(x, freqs):
    """Apply RoPE to query/key: x shape [batch, seq, n_heads, head_dim]"""
    x = x.float()
    x1 = x[..., ::2]  # even dims
    x2 = x[..., 1::2]  # odd dims
    cos = freqs[..., 0]  # [seq, dim//2]
    sin = freqs[..., 1]
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated.flatten(-2)  # merge last dim back to head_dim


def precompute_freqs(dim, seq_len, base=10000):
    """Precompute frequency matrix for RoPE."""
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.einsum('i,j->ij', positions, theta)  # [seq_len, dim//2]
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # [seq_len, dim//2, 2]

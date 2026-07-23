#!/usr/bin/env python
r"""
demo_variant_washout.py

Demonstration
-------------
"Sparse polymorphisms between individuals are washed out by the successive
downsampling convolution layers in Enformer."

The same phenomenon holds for Basenji2 -- Enformer inherits Basenji2's
convolutional *stem + tower* that halves the sequence resolution at every step,
so the argument below transfers directly to any of these "traditional" models.

What the script does
--------------------
  1. Build a pair of "personalized" haplotypes that are identical everywhere
     except a single sparse causal variant (a stand-in for a fine-mapped eQTL).
  2. One-hot encode the reference and the alternative allele exactly the way the
     GTEx dataloader / enformer_pytorch do (A,C,G,T order).
  3. Push both through the *frozen* pretrained Enformer trunk and tap the
     feature-activation maps at consecutive layers of the downsampling stack:

         input (1 bp)                    <- raw one-hot
         enformer.stem.0  (conv, 1 bp)   <- first convolution, before any pooling
         enformer.stem    (pool, 2 bp)
         enformer.conv_tower.0   (4 bp)
         enformer.conv_tower.1   (8 bp)
         enformer.conv_tower.2  (16 bp)
         enformer.conv_tower.3  (32 bp)
         enformer.conv_tower.4  (64 bp)
         enformer.conv_tower.5 (128 bp)
         seq_embeddings (trunk output, 128 bp, post-transformer)

  4. Compute the absolute difference in activations between the two alleles,
     |act(alt) - act(ref)|, at each layer and quantify how it decays with depth.

Why this matters for this repo
------------------------------
The model in
    performer/models/contrast_wrapper_attention_multiheads_rev2.py
is built precisely to *avoid* this failure mode: it taps
``enformer.conv_tower.0..4`` and samples the feature columns at the SNP
positions

    sampled_conv_tower_features = features_conv_tower[:, stem_diff_inds]

i.e. it reads the allele-specific signal out of the *early* conv layers, before
the downsampling pools it away.  This script quantifies exactly why that early
tap is necessary.

The Enformer model is instantiated the same way as in that file:
    Enformer.from_pretrained("EleutherAI/enformer-official-rough", target_length=-1)
    wrapped in CustomHeadAdapterWrapper (which registers the conv_tower hooks).

Usage
-----
    conda activate enformer-pytorch-dev
    cd /pollard/data/projects/zhhu/enformer_fine_tuning_dev
    python performer/demo_variant_washout.py                    # synthetic reference
    python performer/demo_variant_washout.py --n-trials 16      # smoother curves
    python performer/demo_variant_washout.py \                  # real genomic window
        --fasta /path/hg38.fa --chrom chr10 --start 100000000 --variant-offset 0

Outputs (written to --outdir): a metrics CSV, a JSON summary, and two figures
(the per-layer difference profiles and the aggregate wash-out curves).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from enformer_pytorch import Enformer
from performer.models.head_adapter_wrapper.custom_head_adapter_wrapper import (
    CustomHeadAdapterWrapper,
)

# one-hot base order used by kipoiseq.one_hot_dna and enformer_pytorch.str_to_one_hot
BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}

# consecutive layers of the downsampling stack, ordered from shallow to deep.
# each entry is (feature_dict_key, human_label).  conv_tower.0..5 are hooked by
# CustomHeadAdapterWrapper; the two stem taps are added in load_model().
CONV_LAYERS = [
    ("enformer.stem.0", "stem.conv (pre-pool)"),
    ("enformer.stem", "stem (pool x2)"),
    ("enformer.conv_tower.0", "conv_tower.0"),
    ("enformer.conv_tower.1", "conv_tower.1"),
    ("enformer.conv_tower.2", "conv_tower.2"),
    ("enformer.conv_tower.3", "conv_tower.3"),
    ("enformer.conv_tower.4", "conv_tower.4"),
    ("enformer.conv_tower.5", "conv_tower.5"),
]


# --------------------------------------------------------------------------- #
# sequence construction
# --------------------------------------------------------------------------- #
def build_reference(seq_len, rng, fasta=None, chrom=None, start=None):
    """Return an int-coded reference sequence (L,) with values in {0,1,2,3}."""
    if fasta is not None:
        import pysam

        fa = pysam.Fastafile(fasta)
        s = fa.fetch(chrom, int(start), int(start) + seq_len).upper()
        fa.close()
        idx = np.array([BASE_TO_IDX.get(c, 0) for c in s], dtype=np.int64)
        if idx.shape[0] != seq_len:
            raise ValueError(
                f"fetched {idx.shape[0]} bp for {chrom}:{start}, expected {seq_len}"
            )
        return idx
    # The wash-out is an architectural property of the downsampling stack, so the
    # exact bases are irrelevant; a real reference window can be supplied via
    # --fasta. Use a deterministic pseudo-random sequence otherwise.
    return rng.integers(0, 4, size=seq_len, dtype=np.int64)


def load_intervals(csv_path, seq_len):
    """Load a TSS-centred interval table and keep only the intervals whose full
    Enformer window fits inside the chromosome.  Returns a DataFrame with an
    added integer `center` column (the TSS, = midpoint of starts/ends)."""
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["center"] = (df["starts"].astype(int) + df["ends"].astype(int)) // 2
    half = seq_len // 2
    fits = (df["center"] - half >= 0) & (df["center"] + half <= df["chr_length"].astype(int))
    df = df[fits].reset_index(drop=True)
    return df


def sample_intervals(df, n, rng):
    """Randomly pick `n` intervals (without replacement) -> list of dicts."""
    m = min(n, len(df))
    sel = rng.choice(len(df), size=m, replace=False)
    return [
        {"chrom": str(df.iloc[i]["seqnames"]), "center": int(df.iloc[i]["center"]), "gene": str(df.iloc[i]["gene_name"])}
        for i in sel
    ]


def fetch_ref_window(fa, chrom, center, seq_len):
    """Fetch a seq_len window centred on `center` from an open pysam.Fastafile.
    Returns int codes (L,) in {0,1,2,3}, or None if the window is out of bounds."""
    half = seq_len // 2
    s = fa.fetch(chrom, center - half, center + half).upper()
    if len(s) != seq_len:
        return None
    return np.array([BASE_TO_IDX.get(c, 0) for c in s], dtype=np.int64)


def make_alt(ref_idx, pos, rng):
    """Single homozygous SNP: flip the base at `pos` to a different nucleotide."""
    alt_idx = ref_idx.copy()
    choices = [b for b in range(4) if b != ref_idx[pos]]
    alt_idx[pos] = int(rng.choice(choices))
    return alt_idx


def one_hot(seq_idx, het_pos=None, het_ref=None):
    """int codes (L,) -> float32 one-hot (L,4).

    If het_pos/het_ref are given, that position is encoded as a heterozygote
    (0.5 on the reference base, 0.5 on the alternative base) to mimic the
    unphased-diploid encoding used by the GTEx dataloader.
    """
    oh = np.zeros((seq_idx.shape[0], 4), dtype=np.float32)
    oh[np.arange(seq_idx.shape[0]), seq_idx] = 1.0
    if het_pos is not None:
        oh[het_pos] = 0.0
        oh[het_pos, seq_idx[het_pos]] = 0.5
        oh[het_pos, het_ref] = 0.5
    return oh


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(device):
    enformer = Enformer.from_pretrained(
        "EleutherAI/enformer-official-rough",
        target_length=-1,  # disable cropping, as in contrast_wrapper_*_rev2.py
    )
    model = CustomHeadAdapterWrapper(
        enformer=enformer,
        num_tracks=1,
        post_transformer_embed=False,
        output_activation=nn.Identity(),
    )
    # CustomHeadAdapterWrapper._save_features stores features by ``module.name``.
    # ContrastWrapperAttention sets that attribute in __init__; replicate it here.
    for name, module in model.named_modules():
        module.name = name
    # additionally tap the two stem layers so we can see the earliest activations
    # (conv_tower.0..5 are already hooked inside the wrapper).
    model.enformer.stem[0].register_forward_hook(model._save_features)  # 1 bp
    model.enformer.stem.register_forward_hook(model._save_features)  # 2 bp
    model.eval().to(device)
    return model


@torch.no_grad()
def capture(model, seq_oh, device):
    """seq_oh (L,4) float32 -> {layer_name: (C, L) cpu tensor}."""
    model.features_dict = {}  # reset the hook buffer
    x = torch.from_numpy(seq_oh).unsqueeze(0).to(device)  # (1, L, 4)
    out = model(x, freeze_enformer=False)
    feats = {}
    for k, v in out.items():
        if k == "seq_embeddings":
            feats[k] = v[0].transpose(0, 1).float().cpu().contiguous()  # (C, L)
        else:
            feats[k] = v[0].float().cpu().contiguous()  # (C, L)
    del out, x
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return feats


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def profile(ref, alt, channel_axis):
    """Per-position magnitude of the absolute-difference vector, and the ref
    activation magnitude.  Returns two 1-D tensors of length L (= n bins)."""
    d = (alt - ref).pow(2).sum(dim=channel_axis).sqrt()  # ||act(alt)-act(ref)||_2 per bin
    a = ref.pow(2).sum(dim=channel_axis).sqrt()  # ||act(ref)||_2 per bin
    return d, a


def metrics_from_profile(d, a, seq_len):
    L = d.shape[0]
    resolution = seq_len / L
    peak = d.max().item()
    peak_idx = int(d.argmax().item())
    background = a.mean().item()
    rel_peak = peak / (background + 1e-12)  # variant signal in units of a typical activation
    rel_energy = (torch.linalg.norm(d) / (torch.linalg.norm(a) + 1e-12)).item()
    # spatial footprint of the perturbation: bins carrying > 1% of the peak
    affected = d > 0.01 * peak
    n_aff = int(affected.sum().item())
    if n_aff > 0:
        idxs = torch.nonzero(affected, as_tuple=False).squeeze(-1)
        span_bins = int((idxs.max() - idxs.min()).item()) + 1
    else:
        span_bins = 0
    return dict(
        L=L,
        resolution=resolution,
        peak=peak,
        peak_idx=peak_idx,
        background=background,
        rel_peak=rel_peak,
        rel_energy=rel_energy,
        n_affected=n_aff,
        span_bp=span_bins * resolution,
        frac_affected=n_aff / L,
    )


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_profiles(profiles, variant_pos, seq_len, outpath):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [lab for (_, lab, _, _) in profiles]
    n = len(profiles)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), squeeze=False)

    # choose a common bp window: 1.3x the widest 1%-peak footprint across layers
    max_span = max((m["span_bp"] for (_, _, _, m) in profiles), default=1024.0)
    window = max(1024.0, 1.3 * max_span)

    for i, (_, lab, d, m) in enumerate(profiles):
        ax = axes[i // ncol][i % ncol]
        res = m["resolution"]
        # map bin centres to bp offset relative to the variant
        bins = np.arange(d.shape[0])
        rel_bp = (bins + 0.5) * res - variant_pos
        sel = np.abs(rel_bp) <= window
        # normalize by the layer's background activation -> "signal in units of a
        # typical activation", so the y-axis is comparable across layers.
        y = d.numpy()[sel] / (m["background"] + 1e-12)
        ax.fill_between(rel_bp[sel], 0, y, step="mid", color="#c62828", alpha=0.85)
        ax.axhline(1.0, ls="--", lw=0.9, color="0.4")  # background level
        ax.set_title(f"{lab}\n{res:.0f} bp/bin", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3.0)
        ax.set_xlim(-window, window)
        ax.tick_params(labelsize=7)
        ax.text(
            0.03,
            0.92,
            f"peak/bg={m['rel_peak']:.2g}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
        )
    # hide any unused axes
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(
        "Per-bin |activation(alt) - activation(ref)| / background\n"
        "single SNP washed out by successive downsampling (dashed = background)",
        fontsize=11,
    )
    fig.supxlabel("distance from variant (bp)", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_curves(agg, outpath):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [a["label"] for a in agg]
    x = np.arange(len(agg))
    rel_peak = np.array([a["rel_peak_mean"] for a in agg])
    rel_peak_sd = np.array([a["rel_peak_std"] for a in agg])
    span = np.array([a["span_bp_mean"] for a in agg])
    nbins = np.array([a["n_affected_mean"] for a in agg])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.errorbar(x, rel_peak, yerr=rel_peak_sd, marker="o", color="#c62828", capsize=3)
    ax1.axhline(1.0, ls="--", lw=0.9, color="0.4")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Maximum |Δ activation| / background activation")
    #ax1.set_title("Variant signal collapses with depth (washed out)")
    ax1.grid(True, which="both", alpha=0.25)

    ax2.plot(x, span, marker="s", color="#1565c0", label="footprint span (bp)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Variant footprint span (bp)", color="#1565c0")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax2b = ax2.twinx()
    ax2b.plot(x, nbins, marker="^", color="#2e7d32", label="# bins perturbed")
    ax2b.set_yscale("log")
    ax2b.set_ylabel("# of bins perturbed", color="#2e7d32")
    #ax2.set_title("...while the perturbation smears over a wider region")
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def washout_sample(ref_idx, ref_feats, alt_oh, alt_feats, L, per_layer):
    """Compute per-layer wash-out metrics for one (reference, single-SNP alt) pair
    and append them to `per_layer`.  Returns the per-layer (key,label,d,metrics)
    rows used for the representative profile figure."""
    rows = []
    # input layer (raw one-hot): channel axis = 1. The reference haplotype is
    # always the plain homozygous reference (matching how ref_feats were
    # captured); in --het mode only the alternative allele carries the 0.5/0.5
    # heterozygous encoding.
    ref_oh_t = torch.from_numpy(one_hot(ref_idx))
    d_in, a_in = profile(ref_oh_t, torch.from_numpy(alt_oh), channel_axis=1)
    m_in = metrics_from_profile(d_in, a_in, L)
    per_layer["input"].append(m_in)
    rows.append(("input", "input (one-hot)", d_in, m_in))

    for key, label in CONV_LAYERS:  # conv/stem layers: channel axis = 0
        d, a = profile(ref_feats[key], alt_feats[key], channel_axis=0)
        m = metrics_from_profile(d, a, L)
        per_layer[key].append(m)
        rows.append((key, label, d, m))

    d_e, a_e = profile(ref_feats["seq_embeddings"], alt_feats["seq_embeddings"], channel_axis=0)
    per_layer["seq_embeddings"].append(metrics_from_profile(d_e, a_e, L))
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-len", type=int, default=49152, help="Enformer input length (bp)")
    ap.add_argument("--n-trials", type=int, default=8, help="number of random variant placements to average over")
    ap.add_argument("--variant-offset", type=int, default=0, help="representative variant offset from centre (bp)")
    ap.add_argument("--het", action="store_true", help="encode the variant as a heterozygote (0.5/0.5) instead of a homozygous flip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fasta", type=str, default=None, help="optional reference FASTA for a single real genomic window")
    ap.add_argument("--chrom", type=str, default=None)
    ap.add_argument("--start", type=int, default=None, help="0-based start of the window in --fasta")
    ap.add_argument("--intervals", type=str, default=None,
                    help="CSV of TSS-centred intervals (e.g. data/Gencode.v46.TSSCentered_49K_Intervals.csv); "
                         "enables real-genome sampling: one SNP per interval, averaged over --n-intervals")
    ap.add_argument("--n-intervals", type=int, default=100, help="number of intervals to randomly sample in --intervals mode")
    ap.add_argument("--random-anchor", action="store_true",
                    help="in --intervals mode, place each interval's SNP at a random central position instead of the TSS")
    ap.add_argument("--genome-fasta", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "data", "hg38_genome.fa"),
                    help="reference genome FASTA used to fetch interval windows")
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(__file__), "results", "variant_washout_demo"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = args.device
    L = args.seq_len
    centre = L // 2

    print(f"[setup] device={device}  seq_len={L}  het={args.het}")
    print("[setup] loading pretrained Enformer ...")
    model = load_model(device)

    # accumulate metrics: layer_key -> list of per-sample metric dicts
    per_layer = {key: [] for (key, _) in CONV_LAYERS}
    per_layer["input"] = []
    per_layer["seq_embeddings"] = []
    rep_profiles = None  # for the profile figure (first sample)
    rep_variant_pos = None  # genomic coordinate of the first sample's variant (profile x-axis)
    sample_tags = []
    tss_pos = int(np.clip(centre + args.variant_offset, 0, L - 1))  # variant at the TSS + offset
    lo, hi = int(0.25 * L), int(0.75 * L)  # central region for randomly-placed anchors

    if args.intervals:
        # ---- real-genome mode: one SNP per sampled interval ---- #
        # By default the SNP sits at the TSS (window centre); with --random-anchor
        # it is drawn uniformly from the central region of each interval instead.
        import pysam

        df = load_intervals(args.intervals, L)
        picks = sample_intervals(df, args.n_intervals, rng)
        fa = pysam.Fastafile(args.genome_fasta)
        where = "random central position" if args.random_anchor else "the TSS"
        print(f"[setup] intervals mode: {len(picks)} of {len(df)} eligible intervals sampled "
              f"from {os.path.basename(args.intervals)}; SNP placed at {where}; genome={os.path.basename(args.genome_fasta)}")
        for i, iv in enumerate(picks):
            ref_idx = fetch_ref_window(fa, iv["chrom"], iv["center"], L)
            if ref_idx is None:
                continue
            pos = int(rng.integers(lo, hi)) if args.random_anchor else tss_pos
            het_ref = int(ref_idx[pos]) if args.het else None
            alt_idx = make_alt(ref_idx, pos, rng)
            alt_oh = one_hot(alt_idx, het_pos=pos if args.het else None, het_ref=het_ref)
            ref_feats = capture(model, one_hot(ref_idx), device)
            alt_feats = capture(model, alt_oh, device)
            rows = washout_sample(ref_idx, ref_feats, alt_oh, alt_feats, L, per_layer)
            sample_tags.append(f"{iv['gene']}@{iv['chrom']}:{iv['center']}+{pos - centre}")
            if rep_profiles is None:
                rep_profiles, rep_variant_pos = rows, pos
            del ref_feats, alt_feats
            if i == 0 or (i + 1) % 25 == 0 or (i + 1) == len(picks):
                print(f"[run] interval {i + 1}/{len(picks)}  {iv['gene']} {iv['chrom']}:{iv['center']}  SNP@offset {pos - centre}")
        fa.close()
    else:
        # ---- synthetic / single-window mode: many placements on one reference ---- #
        ref_idx = build_reference(L, rng, args.fasta, args.chrom, args.start)
        first_pos = int(np.clip(centre + args.variant_offset, lo, hi - 1))
        variant_positions = [first_pos] + rng.integers(lo, hi, size=max(0, args.n_trials - 1)).tolist()
        print(f"[setup] synthetic/single-window mode  n_trials={len(variant_positions)}")
        ref_feats = capture(model, one_hot(ref_idx), device)
        for t, pos in enumerate(variant_positions):
            het_ref = int(ref_idx[pos]) if args.het else None
            alt_idx = make_alt(ref_idx, pos, rng)
            alt_oh = one_hot(alt_idx, het_pos=pos if args.het else None, het_ref=het_ref)
            print(f"[run] trial {t + 1}/{len(variant_positions)}  variant@{pos}  base {BASES[ref_idx[pos]]}->{BASES[alt_idx[pos]]}")
            alt_feats = capture(model, alt_oh, device)
            rows = washout_sample(ref_idx, ref_feats, alt_oh, alt_feats, L, per_layer)
            sample_tags.append(f"variant@{pos}")
            if t == 0:
                rep_profiles, rep_variant_pos = rows, pos
            del alt_feats

    n_used = len(per_layer["input"])
    assert n_used > 0, "no samples were processed (check --intervals / --genome-fasta)"

    # ------------------------------------------------------------------ #
    # aggregate + report
    # ------------------------------------------------------------------ #
    ordered = ["input"] + [k for (k, _) in CONV_LAYERS] + ["seq_embeddings"]
    labels = {"input": "input (one-hot)", "seq_embeddings": "seq_embeddings (trunk)"}
    labels.update({k: lab for (k, lab) in CONV_LAYERS})

    def agg_of(key):
        ms = per_layer[key]
        arr = lambda f: np.array([m[f] for m in ms])
        return dict(
            key=key,
            label=labels[key],
            resolution=ms[0]["resolution"],
            n_bins=ms[0]["L"],
            rel_peak_mean=float(arr("rel_peak").mean()),
            rel_peak_std=float(arr("rel_peak").std()),
            rel_energy_mean=float(arr("rel_energy").mean()),
            span_bp_mean=float(arr("span_bp").mean()),
            frac_affected_mean=float(arr("frac_affected").mean()),
            n_affected_mean=float(arr("n_affected").mean()),
            peak_mean=float(arr("peak").mean()),
            background_mean=float(arr("background").mean()),
        )

    agg = [agg_of(k) for k in ordered]

    # console table
    hdr = f"{'layer':<24}{'bp/bin':>8}{'n_bins':>10}{'peak/bg':>12}{'relL2':>10}{'span_bp':>12}{'%bins':>8}"
    print("\n" + "=" * len(hdr))
    print("WASH-OUT OF A SINGLE SPARSE VARIANT THROUGH ENFORMER'S DOWNSAMPLING STACK")
    src = f"{n_used} intervals from {os.path.basename(args.intervals)}" if args.intervals else f"{n_used} variant placements"
    print(f"(mean over {src})")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for a in agg:
        print(
            f"{a['label']:<24}{a['resolution']:>8.0f}{a['n_bins']:>10d}"
            f"{a['rel_peak_mean']:>12.3g}{a['rel_energy_mean']:>10.3g}"
            f"{a['span_bp_mean']:>12.0f}{100 * a['frac_affected_mean']:>8.2f}"
        )
    print("-" * len(hdr))

    # headline number: collapse of peak/background from first conv to conv_tower.5
    first = next(a for a in agg if a["key"] == "enformer.stem.0")
    last = next(a for a in agg if a["key"] == "enformer.conv_tower.5")
    fold = first["rel_peak_mean"] / (last["rel_peak_mean"] + 1e-12)
    print(
        f"\nPeak variant signal (relative to background) drops {fold:.1f}x from the "
        f"first conv layer ({first['rel_peak_mean']:.2g}) to conv_tower.5 "
        f"({last['rel_peak_mean']:.2g}, 128 bp bins),"
    )
    print(
        f"while its spatial footprint spreads from {first['span_bp_mean']:.0f} bp to "
        f"{last['span_bp_mean']:.0f} bp -> the sparse polymorphism is washed out."
    )

    # ------------------------------------------------------------------ #
    # write outputs
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(args.outdir, "washout_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("layer,bp_per_bin,n_bins,peak_over_bg_mean,peak_over_bg_std,rel_L2_mean,span_bp_mean,pct_bins_perturbed,raw_peak_mean,background_mean\n")
        for a in agg:
            f.write(
                f"{a['label']},{a['resolution']:.0f},{a['n_bins']},{a['rel_peak_mean']:.6g},"
                f"{a['rel_peak_std']:.6g},{a['rel_energy_mean']:.6g},{a['span_bp_mean']:.3f},"
                f"{100 * a['frac_affected_mean']:.4f},{a['peak_mean']:.6g},{a['background_mean']:.6g}\n"
            )

    json_path = os.path.join(args.outdir, "washout_summary.json")
    with open(json_path, "w") as f:
        json.dump(
            {"config": vars(args), "n_samples": n_used, "samples": sample_tags, "layers": agg, "fold_reduction_stem_to_convtower5": fold},
            f,
            indent=2,
        )

    prof_pdf = os.path.join(args.outdir, "washout_profiles.pdf")
    curve_pdf = os.path.join(args.outdir, "washout_curves.pdf")
    # profile figure uses conv/stem layers + input (local perturbations)
    plot_rows = [r for r in rep_profiles]
    plot_profiles(plot_rows, rep_variant_pos, L, prof_pdf)
    plot_curves([a for a in agg], curve_pdf)

    print(f"\n[out] {csv_path}")
    print(f"[out] {json_path}")
    print(f"[out] {prof_pdf}")
    print(f"[out] {curve_pdf}")


if __name__ == "__main__":
    main()

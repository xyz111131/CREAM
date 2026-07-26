#!/usr/bin/env python
r"""
demo_variant_proximity.py

Demonstration (companion to demo_variant_washout.py)
----------------------------------------------------
"Sparse polymorphisms between individuals are washed out by the successive
downsampling convolution layers in Enformer" -- shown here from the angle of
*variant resolution*:

    If two variants are close to each other, their effect on the activations
    becomes indistinguishable at deeper layers, and the distance at which two
    variants "merge" grows with every downsampling step.

Idea
----
Each downsampling AttentionPool halves the sequence resolution, so two distinct
SNPs that fall into the same (increasingly large) bin -- or the same receptive
field -- produce the *same* change in the feature map.  We make this concrete:

  1. Pick an anchor SNP and a ladder of "probe" SNPs at increasing genomic
     distances d = 1, 2, 4, ... bp from it (each a single-base change).
  2. Feed the reference and every single-variant alternative through the frozen
     pretrained Enformer.
  3. At each layer extract the activation map and form the *effect* of a variant,
     Delta = |activation(alt) - activation(ref)|.
  4. Compare variants: the cosine similarity between the anchor's Delta map and a
     probe's Delta map measures how indistinguishable the two variants look at
     that layer.

At the input the two SNPs live at different single positions, so their Delta maps
are orthogonal (similarity ~ 0) no matter how close they are.  As the sequence is
pooled, nearby SNPs land in overlapping bins and their Delta maps converge
(similarity -> 1).  The "merge distance" (separation at which similarity crosses
0.5) grows in lock-step with the bin size -> the model can no longer tell nearby
polymorphisms apart: they are washed together.

Why this matters for this repo
------------------------------
CREAM/models/contrast_wrapper_attention_multiheads_rev2.py taps the *early*
conv layers (``enformer.conv_tower.0..4``) and samples features at the SNP
positions precisely so that nearby variants are still resolvable before the
downsampling merges them.  This script quantifies how quickly that resolution is
lost with depth.

Model instantiation and the layer taps are reused from demo_variant_washout.py
(same Enformer.from_pretrained + CustomHeadAdapterWrapper as the contrast model).

Usage
-----
    conda activate enformer-pytorch-dev
    cd /pollard/data/projects/zhhu/enformer_fine_tuning_dev
    python CREAM/demo_variant_proximity.py                     # synthetic reference
    python CREAM/demo_variant_proximity.py --n-anchors 6       # smoother curves
    python CREAM/demo_variant_proximity.py \                   # real genomic window
        --fasta /path/hg38.fa --chrom chr10 --start 100000000
"""

import argparse
import json
import math
import os

import numpy as np
import torch

# reuse the verified model / capture / encoding helpers from the washout demo
from CREAM.demo_variant_washout import (
    BASES,
    CONV_LAYERS,
    build_reference,
    capture,
    fetch_ref_window,
    load_intervals,
    load_model,
    make_alt,
    one_hot,
    sample_intervals,
)

# consecutive layers analysed, shallow -> deep
LAYERS = ["input"] + [k for (k, _) in CONV_LAYERS] + ["seq_embeddings"]
LABELS = {"input": "input (one-hot)", "seq_embeddings": "seq_embeddings (trunk)"}
LABELS.update({k: lab for (k, lab) in CONV_LAYERS})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def full_map(layer, oh_np, feats):
    """Return the (C, L_layer) activation map for `layer`.

    For the raw one-hot input the 4 nucleotide channels play the role of C.
    """
    if layer == "input":
        return torch.from_numpy(oh_np).transpose(0, 1)  # (4, L)
    return feats[layer]  # (C, L_layer), already (C, L) for seq_embeddings too


def window_vec(cmap, p0, half, seq_len):
    """Flatten the activation map inside a bp window centred on p0 (absolute
    coordinates), so maps from different alleles are sliced identically."""
    Lf = cmap.shape[1]
    res = seq_len / Lf
    b0 = max(0, int((p0 - half) // res))
    b1 = min(Lf, int((p0 + half) // res) + 1)
    return cmap[:, b0:b1].reshape(-1).float()


def cosine(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def merge_distance(dists, cosv, thr=0.5):
    """Separation (bp) at which the similarity curve crosses `thr`, log-interp.
    Curve is high at small d and decays; return the crossing to low similarity."""
    prev_d = prev_c = None
    for d, c in zip(dists, cosv):
        if c < thr:
            if prev_d is None:
                return float(d)  # already below threshold at the smallest probe
            if c == prev_c:
                return float(d)
            t = (thr - prev_c) / (c - prev_c)
            return math.exp(math.log(prev_d) + t * (math.log(d) - math.log(prev_d)))
        prev_d, prev_c = d, c
    return float(dists[-1])  # never dropped below threshold within the ladder


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_similarity_curves(dists, mean_cos, sem_cos, outpath):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6))
    cm = plt.cm.viridis
    n = len(LAYERS)
    for i, layer in enumerate(LAYERS):
        y = np.array([mean_cos[layer][d] for d in dists])
        e = np.array([sem_cos[layer][d] for d in dists])
        c = cm(i / max(1, n - 1))
        ax.plot(dists, y, marker="o", ms=4, color=c, label=LABELS[layer])
        ax.fill_between(dists, y - e, y + e, color=c, alpha=0.15)
    #ax.axhline(0.5, ls="--", lw=0.9, color="0.4")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("distance between the two variants (bp)")
    ax.set_ylabel("cosine similarity of the two variants' |Δactivation| maps")
    #ax.set_title(
    #    "Nearby variants become indistinguishable at deeper layers\n"
    #    "(the 'merge distance' grows with each downsampling step)"
    #)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_similarity_heatmaps(offsets, heat, seq_len, layer_len, outpath):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(LAYERS)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.1 * nrow), squeeze=False, layout="constrained")
    ticks = list(range(len(offsets)))
    ticklabels = [str(o) for o in offsets]
    im = None
    for i, layer in enumerate(LAYERS):
        ax = axes[i // ncol][i % ncol]
        im = ax.imshow(heat[layer], vmin=0, vmax=1, cmap="magma", origin="lower")
        res = seq_len / layer_len[layer]
        ax.set_title(f"{LABELS[layer]}\n{res:.0f} bp/bin", fontsize=8)
        ax.set_xticks(ticks[::2])
        ax.set_xticklabels(ticklabels[::2], fontsize=6, rotation=90)
        ax.set_yticks(ticks[::2])
        ax.set_yticklabels(ticklabels[::2], fontsize=6)
        ax.tick_params(length=0)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(
        "Pairwise cosine similarity of variants' |Δactivation| maps\n"
        "(axes = variant offset in bp; bright block = variants that have merged)",
        fontsize=11,
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-len", type=int, default=49152)
    ap.add_argument("--n-anchors", type=int, default=4, help="anchor SNPs to average the curves over")
    ap.add_argument("--max-distance", type=int, default=2048, help="largest probe separation (bp)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fasta", type=str, default=None, help="single real genomic window (chrom/start below)")
    ap.add_argument("--chrom", type=str, default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--intervals", type=str, default=None,
                    help="CSV of TSS-centred intervals (e.g. data/Gencode.v46.TSSCentered_49K_Intervals.csv); "
                         "each sampled interval is one anchor context (anchor SNP at the TSS)")
    ap.add_argument("--n-intervals", type=int, default=100, help="number of intervals to randomly sample in --intervals mode")
    ap.add_argument("--random-anchor", action="store_true",
                    help="in --intervals mode, place each interval's anchor SNP at a random central position instead of the TSS")
    ap.add_argument("--genome-fasta", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "data", "hg38_genome.fa"),
                    help="reference genome FASTA used to fetch interval windows")
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(__file__), "results", "variant_proximity_demo"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = args.device
    L = args.seq_len

    # geometric ladder of probe separations up to --max-distance
    distances = [d for d in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192) if d <= args.max_distance]
    offsets = [0] + distances  # offset 0 == the anchor variant itself
    half = args.max_distance + 1024  # window padding for the receptive field

    print(f"[setup] device={device}  seq_len={L}  distances={distances}")
    print("[setup] loading pretrained Enformer ...")
    model = load_model(device)
    centre = L // 2
    lo, hi = int(0.30 * L), int(0.60 * L)  # central region for anchors (leaves room for +max_distance)

    # build the list of anchors.  Each anchor is one reference context whose SNP
    # sits at the anchor position; probes are placed d bp away.
    intervals_mode = bool(args.intervals)
    fa = None
    if intervals_mode:
        import pysam

        df = load_intervals(args.intervals, L)
        anchor_items = sample_intervals(df, args.n_intervals, rng)  # list of dicts, anchor SNP at the TSS
        fa = pysam.Fastafile(args.genome_fasta)
        where = "random central position" if args.random_anchor else "the TSS"
        print(f"[setup] intervals mode: {len(anchor_items)} of {len(df)} eligible intervals sampled "
              f"from {os.path.basename(args.intervals)}; anchor SNP at {where}; genome={os.path.basename(args.genome_fasta)}")
    else:
        ref_idx_g = build_reference(L, rng, args.fasta, args.chrom, args.start)
        ref_oh_g = one_hot(ref_idx_g)
        ref_feats_g = capture(model, ref_oh_g, device)
        anchor_items = [centre] + rng.integers(lo, hi, size=max(0, args.n_anchors - 1)).tolist()
        print(f"[setup] synthetic/single-window mode  n_anchors={len(anchor_items)}")

    # accumulate cosine(anchor, probe) over anchors: sims[layer][d] -> list
    sims = {layer: {d: [] for d in distances} for layer in LAYERS}
    heat = None  # pairwise cosine matrices for the first anchor (for the heatmap)
    layer_len = None
    sample_tags = []

    for ai, anchor in enumerate(anchor_items):
        # resolve this anchor's reference context and anchor position p0
        if intervals_mode:
            ref_idx = fetch_ref_window(fa, anchor["chrom"], anchor["center"], L)
            if ref_idx is None:
                continue
            ref_oh = one_hot(ref_idx)
            ref_feats = capture(model, ref_oh, device)
            # anchor SNP at the TSS (window centre) by default, or a random central
            # position with --random-anchor
            p0 = int(rng.integers(lo, hi)) if args.random_anchor else centre
            tag = f"{anchor['gene']} {anchor['chrom']}:{anchor['center']}+{p0 - centre}"
        else:
            ref_idx, ref_oh, ref_feats = ref_idx_g, ref_oh_g, ref_feats_g
            p0 = anchor
            tag = f"@{p0}"
        if layer_len is None:
            layer_len = {layer: (L if layer == "input" else ref_feats[layer].shape[1]) for layer in LAYERS}

        # reference window vector per layer (sliced once per anchor)
        ref_vec = {layer: window_vec(full_map(layer, ref_oh, ref_feats), p0, half, L) for layer in LAYERS}

        # Delta window vector for every variant offset in this anchor's cluster
        delta = {layer: {} for layer in LAYERS}
        for off in offsets:
            alt_idx = make_alt(ref_idx, p0 + off, rng)
            alt_oh = one_hot(alt_idx)
            alt_feats = capture(model, alt_oh, device)
            for layer in LAYERS:
                alt_vec = window_vec(full_map(layer, alt_oh, alt_feats), p0, half, L)
                delta[layer][off] = (alt_vec - ref_vec[layer]).abs()
            del alt_feats

        # cosine(anchor Delta, probe Delta) at each separation
        for layer in LAYERS:
            a0 = delta[layer][0]
            for d in distances:
                sims[layer][d].append(cosine(a0, delta[layer][d]))

        # keep the first anchor's full pairwise matrices for the heatmap figure
        if heat is None:
            heat = {}
            for layer in LAYERS:
                vecs = [delta[layer][o] for o in offsets]
                M = np.zeros((len(offsets), len(offsets)), dtype=np.float32)
                for r in range(len(offsets)):
                    for c in range(len(offsets)):
                        M[r, c] = cosine(vecs[r], vecs[c])
                heat[layer] = M

        sample_tags.append(tag)
        if intervals_mode:
            del ref_feats
        del delta
        if ai == 0 or (ai + 1) % 10 == 0 or (ai + 1) == len(anchor_items):
            print(f"[run] anchor {ai + 1}/{len(anchor_items)}  {tag}")

    if fa is not None:
        fa.close()
    n_used = len(sims[LAYERS[0]][distances[0]])
    assert n_used > 0, "no anchors were processed (check --intervals / --genome-fasta)"

    # ------------------------------------------------------------------ #
    # aggregate
    # ------------------------------------------------------------------ #
    mean_cos = {layer: {d: float(np.mean(sims[layer][d])) for d in distances} for layer in LAYERS}
    sem_cos = {
        layer: {d: float(np.std(sims[layer][d]) / max(1, math.sqrt(len(sims[layer][d])))) for d in distances}
        for layer in LAYERS
    }
    merge = {layer: merge_distance(distances, [mean_cos[layer][d] for d in distances]) for layer in LAYERS}

    # ------------------------------------------------------------------ #
    # report
    # ------------------------------------------------------------------ #
    hdr = f"{'layer':<24}{'bp/bin':>8}{'merge_dist(bp)':>16}   cosine(anchor, probe) at d = ..."
    print("\n" + "=" * 100)
    print("HOW CLOSE MUST TWO VARIANTS BE TO LOOK IDENTICAL? (similarity of their |Δactivation| maps)")
    src = f"{n_used} intervals from {os.path.basename(args.intervals)}" if intervals_mode else f"{n_used} anchors"
    print(f"(mean over {src}; merge_dist = separation where similarity crosses 0.5)")
    print("=" * 100)
    show_d = [d for d in (1, 8, 64, 512, 2048) if d in distances]
    print(hdr + "  [" + ", ".join(str(d) for d in show_d) + "]")
    print("-" * 100)
    for layer in LAYERS:
        res = L / layer_len[layer]
        cs = "  ".join(f"{mean_cos[layer][d]:.2f}" for d in show_d)
        print(f"{LABELS[layer]:<24}{res:>8.0f}{merge[layer]:>16.0f}   [{cs}]")
    print("-" * 100)

    conv5 = "enformer.conv_tower.5"
    stem0 = "enformer.stem.0"
    print(
        f"\nMerge distance grows from ~{merge[stem0]:.0f} bp at the first conv layer to "
        f"~{merge[conv5]:.0f} bp at conv_tower.5 (128 bp bins):"
    )
    print(
        "two SNPs that are distinct at the input become an indistinguishable single "
        "perturbation once they fall within a pooled bin -> sparse polymorphisms are washed together."
    )

    # ------------------------------------------------------------------ #
    # outputs
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(args.outdir, "proximity_merge_distance.csv")
    with open(csv_path, "w") as f:
        f.write("layer,bp_per_bin,merge_distance_bp," + ",".join(f"cos_d{d}" for d in distances) + "\n")
        for layer in LAYERS:
            res = L / layer_len[layer]
            row = ",".join(f"{mean_cos[layer][d]:.4f}" for d in distances)
            f.write(f"{LABELS[layer]},{res:.0f},{merge[layer]:.2f},{row}\n")

    json_path = os.path.join(args.outdir, "proximity_summary.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "distances": distances,
                "n_samples": n_used,
                "anchors": sample_tags,
                "mean_cosine": {layer: mean_cos[layer] for layer in LAYERS},
                "merge_distance_bp": {layer: merge[layer] for layer in LAYERS},
            },
            f,
            indent=2,
        )

    curve_pdf = os.path.join(args.outdir, "proximity_similarity_curves.pdf")
    heat_pdf = os.path.join(args.outdir, "proximity_similarity_heatmaps.pdf")
    plot_similarity_curves(distances, mean_cos, sem_cos, curve_pdf)
    plot_similarity_heatmaps(offsets, heat, L, layer_len, heat_pdf)

    print(f"\n[out] {csv_path}")
    print(f"[out] {json_path}")
    print(f"[out] {curve_pdf}")
    print(f"[out] {heat_pdf}")


if __name__ == "__main__":
    main()

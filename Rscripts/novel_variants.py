#!/usr/bin/env python
"""
Identify de novo / novel SNVs in the held-out (valid) donors for each test gene.

A variant in a gene's model-input window (49,152 bp, TSS-centered) is "novel" if it is
carried (het or hom) by >=1 valid donor but by 0 training donors of the same CV fold.

Outputs a per-(gene, valid-donor) table plus a per-gene summary.
"""
import argparse, csv, sys, time
import pysam

FOLD = 0
DATA = "data"
VCF = f"{DATA}/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz"
CVDIR = "data0/cross_validation_folds/gtex/cv_folds"
INTERVALS = "data/Gencode.v46.TSSCentered_49K_Intervals.csv"

def load_ids(path):
    return [l.strip() for l in open(path) if l.strip()]

def load_intervals():
    iv = {}
    with open(INTERVALS) as f:
        for row in csv.DictReader(f):
            gid = row["gene_id"].split(".")[0]
            iv[gid] = (row["seqnames"], int(row["starts"]), int(row["ends"]))
    return iv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", required=True, help="file with one test gene id per line")
    ap.add_argument("--limit", type=int, default=0, help="only process first N genes (0=all)")
    ap.add_argument("--out_pair", required=True)
    ap.add_argument("--out_gene", required=True)
    ap.add_argument("--out_variant", required=True)
    ap.add_argument("--out_carrier", required=True)
    args = ap.parse_args()

    train = load_ids(f"{CVDIR}/person_ids-train-fold{FOLD}.txt")
    val   = load_ids(f"{CVDIR}/person_ids-test-fold{FOLD}.txt") #val
    genes = load_ids(args.genes)
    if args.limit:
        genes = genes[:args.limit]
    iv = load_intervals()

    vcf = pysam.VariantFile(VCF)
    all_samples = set(vcf.header.samples)
    train = [d for d in train if d in all_samples]
    val   = [d for d in val   if d in all_samples]
    vcf.subset_samples(train + val)   # parse only needed genotype columns
    train_set = set(train)
    val_set = set(val)

    pair_rows = []   # gene, donor, n_novel
    gene_rows = []   # gene, chrom, n_variants, n_novel_variants, n_val_donors_with_novel
    variant_rows = []  # variant_id, gene_name, n_val_carriers  (unique novel variants)
    carrier_rows = []  # gene_name, variant_id, donor  (one row per novel variant x val carrier)
    seen_variants = set()  # dedup variant_id across (overlapping) gene windows
    # per-donor novel counts accumulate across genes as well
    donor_novel_total = {d: 0 for d in val}

    t0 = time.time()
    missing_iv = 0
    for gi, g in enumerate(genes):
        if g not in iv:
            missing_iv += 1
            continue
        chrom, s, e = iv[g]
        novel_carriers = {d: 0 for d in val}   # per val donor: count of novel variants carried in this window
        n_var = 0
        n_novel = 0
        try:
            recs = vcf.fetch(chrom, s, e)
        except ValueError:
            recs = []
        for rec in recs:
            n_var += 1
            train_has = False
            vcarriers = []
            for d in train:
                gt = rec.samples[d].get("GT")
                if gt and any(a is not None and a > 0 for a in gt):
                    train_has = True
                    break
            if train_has:
                continue  # seen in training -> not novel
            for d in val:
                gt = rec.samples[d].get("GT")
                if gt and any(a is not None and a > 0 for a in gt):
                    vcarriers.append(d)
            if vcarriers:      # >=1 val carrier and 0 train carriers -> novel
                n_novel += 1
                for d in vcarriers:
                    novel_carriers[d] += 1
                vid = f"{rec.chrom}_{rec.pos}_{rec.ref}_{','.join(rec.alts or [])}"
                for d in vcarriers:
                    carrier_rows.append((g, vid, d))  # per-gene (eQTL match is gene-specific)
                if vid not in seen_variants:
                    seen_variants.add(vid)
                    variant_rows.append((vid, g, len(vcarriers)))
        ndonor_with_novel = 0
        for d in val:
            c = novel_carriers[d]
            pair_rows.append((g, d, c))
            donor_novel_total[d] += c
            if c > 0:
                ndonor_with_novel += 1
        gene_rows.append((g, chrom, n_var, n_novel, ndonor_with_novel))
        if (gi + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"[{gi+1}/{len(genes)}] {dt:.0f}s  ({dt/(gi+1)*1000:.0f} ms/gene)", file=sys.stderr)

    with open(args.out_pair, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene_name", "donor", "n_novel"])
        w.writerows(pair_rows)
    with open(args.out_gene, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene_name", "chrom", "n_variants", "n_novel_variants", "n_val_donors_with_novel"])
        w.writerows(gene_rows)
    with open(args.out_variant, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant_id", "gene_name", "n_val_carriers"])
        w.writerows(variant_rows)
    with open(args.out_carrier, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene_name", "variant_id", "donor"])
        w.writerows(carrier_rows)

    # summary
    tot_novel = sum(r[3] for r in gene_rows)
    genes_with_novel = sum(1 for r in gene_rows if r[3] > 0)
    donors_ever = sum(1 for d in val if donor_novel_total[d] > 0)
    print("==== SUMMARY ====")
    print(f"genes processed: {len(gene_rows)} (missing interval: {missing_iv})")
    print(f"valid donors: {len(val)}   training donors: {len(train)}")
    print(f"total novel variants (summed over gene windows): {tot_novel}")
    print(f"unique novel variants (dedup across windows): {len(seen_variants)}")
    print(f"test genes with >=1 novel variant: {genes_with_novel}")
    print(f"valid donors carrying >=1 novel variant (any gene): {donors_ever}/{len(val)}")
    import statistics as st
    per_donor = [donor_novel_total[d] for d in val]
    print(f"per-donor novel-variant count: min={min(per_donor)} median={st.median(per_donor)} max={max(per_donor)} mean={sum(per_donor)/len(per_donor):.1f}")
    print(f"elapsed: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

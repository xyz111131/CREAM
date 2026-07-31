"""Substitute simulated SNPs into the test-data consensus sequences.

The four consensus FASTAs under ``testdata/ConsensusSeqs_SNPsOnlyUnphased`` are
byte-identical copies of the chr22 reference: the real GTEx haplotypes are
protected data and cannot be shipped, so both donors currently see exactly the
same input sequence and nothing in the test pipeline depends on genotype. This
script draws random SNPs in a window around the TSS of every gene in the
test-data gene lists and writes them into the donor haplotypes, so a test run
exercises the donor- and haplotype-dependent code paths.

Sites are drawn once and shared between donors, then genotyped per donor under
Hardy-Weinberg from a random allele frequency, conditioned on at least one donor
carrying the alternate allele (a site no one carries would be invisible with
only two donors). Heterozygous sites put the alternate allele on H1 or H2 with
equal probability: the source data is unphased, so which haplotype carries it is
arbitrary. Alternate alleles follow a 2:1 transition/transversion ratio and keep
the soft-masking case of the reference base.

Every substitution is written to a manifest TSV beside the FASTAs. The manifest
doubles as the undo log: a re-run first reverts the sites it recorded, so
repeated runs replace the simulated variation instead of compounding it.

Usage::

    .venv/bin/python scripts/simulate_testdata_snps.py
    .venv/bin/python scripts/simulate_testdata_snps.py --window 10000 --snps-per-gene 20
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CREAM import config as cream_config

MANIFEST_NAME = "simulated_snps.tsv"
LINE_WIDTH = 60
SPLITS = ("train", "val", "test")
TRANSITION = {"A": "G", "G": "A", "C": "T", "T": "C"}
BASES = ("A", "C", "G", "T")


def read_single_contig_fasta(path):
    """Return (header line, sequence as a bytearray) for a one-contig FASTA."""
    with open(path) as handle:
        header = handle.readline().rstrip("\n")
        if not header.startswith(">"):
            raise ValueError(f"{path} does not start with a FASTA header")
        body = handle.read()
    if ">" in body:
        raise ValueError(f"{path} holds more than one contig; this script expects one")
    return header, bytearray(body.replace("\n", ""), "ascii")


def write_single_contig_fasta(path, header, sequence):
    view = memoryview(sequence)
    with open(path, "w") as handle:
        handle.write(header + "\n")
        for start in range(0, len(sequence), LINE_WIDTH):
            handle.write(view[start : start + LINE_WIDTH].tobytes().decode("ascii") + "\n")
    pysam.faidx(str(path))


def gene_lists(config, model_type):
    """{split: [gene_id, ...]} read from the gene list files named in the config."""
    lists = {}
    for split in SPLITS:
        filename = config.get(f"{split}_gene_file")
        if filename is None:
            raise KeyError(f"config has no '{split}_gene_file' entry")
        path = cream_config.gene_set_path(config, filename, model_type)
        with open(path) as handle:
            lists[split] = [line.strip() for line in handle if line.strip()]
    return lists


def donor_ids(config, fold):
    ids = []
    for split in SPLITS:
        with open(cream_config.donor_list_path(config, split, fold)) as handle:
            for line in handle:
                donor = line.strip()
                if donor and donor not in ids:
                    ids.append(donor)
    return ids


def gene_windows(config, lists, window, chrom_length):
    """One row per gene: the TSS and the [start, end) interval to place SNPs in."""
    intervals = pd.read_csv(cream_config.genomic_intervals_file(config))
    intervals["gene"] = intervals["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    by_gene = intervals.drop_duplicates("gene").set_index("gene")

    rows = []
    for split, genes in lists.items():
        for gene in genes:
            if gene not in by_gene.index:
                print(f"  skipping {gene} ({split}): not in the genomic intervals file")
                continue
            record = by_gene.loc[gene]
            tss = int(record["gene_start"])
            rows.append(
                {
                    "split": split,
                    "gene_id": gene,
                    "gene_name": record["gene_name"],
                    "chrom": record["seqnames"],
                    "tss": tss,
                    "start": max(0, tss - window),
                    "end": min(chrom_length, tss + window),
                }
            )
    rows.sort(key=lambda row: (row["chrom"], row["tss"]))
    return rows


def draw_alt(ref_base, rng):
    """An alternate allele with a 2:1 transition/transversion ratio."""
    if rng.random() < 2 / 3:
        return TRANSITION[ref_base]
    return rng.choice([b for b in BASES if b != ref_base and b != TRANSITION[ref_base]])


def draw_genotypes(donors, allele_freq, rng):
    """Per-donor (H1, H2) alt-carrier flags, conditioned on someone carrying."""
    for _ in range(20):
        genotypes = {}
        for donor in donors:
            n_alt = rng.binomial(2, allele_freq)
            if n_alt == 0:
                genotypes[donor] = (False, False)
            elif n_alt == 2:
                genotypes[donor] = (True, True)
            else:
                on_h1 = bool(rng.random() < 0.5)
                genotypes[donor] = (on_h1, not on_h1)
        if any(any(hap) for hap in genotypes.values()):
            return genotypes
    carrier = donors[rng.integers(len(donors))]
    genotypes = {donor: (False, False) for donor in donors}
    on_h1 = bool(rng.random() < 0.5)
    genotypes[carrier] = (on_h1, not on_h1)
    return genotypes


def sample_sites(reference, windows, donors, snps_per_gene, rng):
    """Draw SNP sites gene by gene, skipping N bases and positions already used."""
    sites = []
    taken = set()
    for window in windows:
        candidates = np.arange(window["start"], window["end"])
        rng.shuffle(candidates)
        drawn = 0
        for position in candidates:
            if drawn == snps_per_gene:
                break
            position = int(position)
            if position in taken:
                continue
            ref_base = chr(reference[position])
            if ref_base.upper() not in TRANSITION:
                continue  # N or any other ambiguity code
            alt_base = draw_alt(ref_base.upper(), rng)
            if ref_base.islower():
                alt_base = alt_base.lower()
            allele_freq = float(rng.uniform(0.05, 0.5))
            sites.append(
                {
                    "chrom": window["chrom"],
                    "pos0": position,
                    "ref": ref_base,
                    "alt": alt_base,
                    "gene_id": window["gene_id"],
                    "gene_name": window["gene_name"],
                    "split": window["split"],
                    "tss": window["tss"],
                    "offset_from_tss": position - window["tss"],
                    "allele_freq": round(allele_freq, 4),
                    "genotypes": draw_genotypes(donors, allele_freq, rng),
                }
            )
            taken.add(position)
            drawn += 1
        if drawn < snps_per_gene:
            print(
                f"  only placed {drawn}/{snps_per_gene} SNPs for {window['gene_name']}"
                " (window ran out of usable bases)"
            )
    sites.sort(key=lambda site: (site["chrom"], site["pos0"]))
    return sites


def revert_manifest(reference, manifest_path):
    """Undo a previous run so repeated runs replace rather than accumulate SNPs."""
    with open(manifest_path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        reference[int(row["pos0"])] = ord(row["ref"])
    print(f"reverted {len(rows)} SNPs from the previous run ({manifest_path})")


def write_manifest(path, sites, donors):
    columns = [
        "chrom",
        "pos",
        "pos0",
        "ref",
        "alt",
        "gene_id",
        "gene_name",
        "split",
        "tss",
        "offset_from_tss",
        "allele_freq",
    ]
    for donor in donors:
        columns += [f"{donor}_GT", f"{donor}_H1", f"{donor}_H2"]

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for site in sites:
            row = {key: site[key] for key in columns if key in site}
            row["pos"] = site["pos0"] + 1
            for donor in donors:
                h1_alt, h2_alt = site["genotypes"][donor]
                row[f"{donor}_GT"] = f"{int(h1_alt)}/{int(h2_alt)}"
                row[f"{donor}_H1"] = site["alt"] if h1_alt else site["ref"]
                row[f"{donor}_H2"] = site["alt"] if h2_alt else site["ref"]
            writer.writerow(row)


def summarise(sites, donors, seq_length):
    half = seq_length // 2
    print(f"\nplaced {len(sites)} SNP sites")
    for split in SPLITS:
        in_split = [s for s in sites if s["split"] == split]
        genes = len({s["gene_id"] for s in in_split})
        print(f"  {split:<5} {len(in_split):>4} sites across {genes} genes")
    in_window = sum(1 for s in sites if abs(s["offset_from_tss"]) < half)
    print(
        f"  {in_window} of them fall inside the +/-{half} bp model input window"
        f" (seq_length {seq_length})"
    )
    for donor in donors:
        het = sum(1 for s in sites if sum(s["genotypes"][donor]) == 1)
        hom = sum(1 for s in sites if sum(s["genotypes"][donor]) == 2)
        print(f"  {donor}: {het} heterozygous, {hom} homozygous alternate")
    # The dataset averages the two haplotypes into one diploid encoding, so what
    # the model can actually tell apart is the alternate-allele dosage.
    differing = sum(
        1 for s in sites if len({sum(s["genotypes"][d]) for d in donors}) > 1
    )
    print(f"  {differing} sites where the donors differ in alternate-allele dosage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/blood_config_test_run2.yaml",
        help="experiment config naming the gene list files (default: %(default)s)",
    )
    parser.add_argument("--model_type", default="MultiGene")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--window",
        type=int,
        default=10000,
        help="place SNPs within this many bp either side of the TSS (default: %(default)s)",
    )
    parser.add_argument(
        "--snps-per-gene",
        type=int,
        default=20,
        help="SNP sites drawn per gene window (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without touching the FASTAs",
    )
    args = parser.parse_args()

    config = cream_config.load_config(
        args.config, use_test_data=True, model_type=args.model_type
    )
    consensus_dir = cream_config.consensus_seq_dir(config)
    filename_template = cream_config.consensus_seq_filename(config)
    donors = donor_ids(config, args.fold)
    manifest_path = os.path.join(consensus_dir, MANIFEST_NAME)

    print(f"consensus sequences: {consensus_dir}")
    print(f"donors: {', '.join(donors)}")

    source = os.path.join(consensus_dir, filename_template.format(donor_id=donors[0], haplotype=1))
    header, reference = read_single_contig_fasta(source)
    if os.path.exists(manifest_path):
        revert_manifest(reference, manifest_path)

    lists = gene_lists(config, args.model_type)
    windows = gene_windows(config, lists, args.window, len(reference))
    print(f"{len(windows)} gene windows of +/-{args.window} bp around the TSS")

    rng = np.random.default_rng(args.seed)
    sites = sample_sites(reference, windows, donors, args.snps_per_gene, rng)
    summarise(sites, donors, int(config["seq_length"]))

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    for donor in donors:
        for haplotype in (1, 2):
            sequence = bytearray(reference)
            for site in sites:
                carries = site["genotypes"][donor][haplotype - 1]
                sequence[site["pos0"]] = ord(site["alt"] if carries else site["ref"])
            path = os.path.join(
                consensus_dir, filename_template.format(donor_id=donor, haplotype=haplotype)
            )
            write_single_contig_fasta(path, header, sequence)
            print(f"wrote {path}")

    write_manifest(manifest_path, sites, donors)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

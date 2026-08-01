#!/bin/bash
#SBATCH -p gidbkb,gidb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=80G
#SBATCH --time=12:00:00
# stdout/stderr are written relative to the directory you run sbatch from;
# override with `sbatch -o <dir>/%j.out -e <dir>/%j.err` (see CREAM_TRAIN_LOG_DIR /
# CREAM_TEST_LOG_DIR from scripts/cream_env.sh)
#SBATCH -o ../logs/test/slurm_stdout/%j.out
#SBATCH -e ../logs/test/slurm_stdout/%j.err
#SBATCH --job-name=variant_washout_demos

# Runs the two "sparse variants are washed out by Enformer's downsampling" demos
# over 1,000 randomly-sampled TSS-centred intervals (real hg38 sequence).
#   - demo_variant_washout.py   : one SNP per interval washed out through depth (~1 h)
#   - demo_variant_proximity.py : nearby SNPs merge with depth (several hours)
# Edit N_INTERVALS / EXTRA_ARGS below to taste; add --random-anchor to EXTRA_ARGS
# to place each interval's SNP at a random central position instead of the TSS.

# project paths, virtualenv and CUBLAS_WORKSPACE_CONFIG come from the YAML config
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/cream_env.sh"
source "$CREAM_VENV/bin/activate"

cd "$CREAM_PROJECT_ROOT"

# --intervals without a value uses data.genomic_intervals_file from the config
N_INTERVALS=100
SEED=0
EXTRA_ARGS= "--random-anchor"  #or  "--random-anchor --het"

python -m CREAM.demo_variant_washout \
    --intervals \
    --n-intervals "$N_INTERVALS" \
    --seed "$SEED" \
    $EXTRA_ARGS

python -m CREAM.demo_variant_proximity \
    --intervals \
    --n-intervals "$N_INTERVALS" \
    --seed "$SEED" \
    $EXTRA_ARGS

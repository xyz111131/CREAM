#!/bin/bash
#SBATCH -p gidbkb,gidb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH -o /pollard/data/projects/zhhu/enformer_fine_tuning_dev/logs/test/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/zhhu/enformer_fine_tuning_dev/logs/test/slurm_stdout/%j.err
#SBATCH --job-name=variant_washout_demos

# Runs the two "sparse variants are washed out by Enformer's downsampling" demos
# over 1,000 randomly-sampled TSS-centred intervals (real hg38 sequence).
#   - demo_variant_washout.py   : one SNP per interval washed out through depth (~1 h)
#   - demo_variant_proximity.py : nearby SNPs merge with depth (several hours)
# Edit N_INTERVALS / EXTRA_ARGS below to taste; add --random-anchor to EXTRA_ARGS
# to place each interval's SNP at a random central position instead of the TSS.

source activate enformer-pytorch-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd /pollard/data/projects/zhhu/enformer_fine_tuning_dev

INTERVALS=data/Gencode.v46.TSSCentered_49K_Intervals.csv
N_INTERVALS=100
SEED=0
EXTRA_ARGS= "--random-anchor"  #or  "--random-anchor --het"

python CREAM/demo_variant_washout.py \
    --intervals "$INTERVALS" \
    --n-intervals "$N_INTERVALS" \
    --seed "$SEED" \
    $EXTRA_ARGS

python CREAM/demo_variant_proximity.py \
    --intervals "$INTERVALS" \
    --n-intervals "$N_INTERVALS" \
    --seed "$SEED" \
    $EXTRA_ARGS

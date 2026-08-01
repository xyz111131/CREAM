#!/bin/bash
#SBATCH -p gidbkb,gidb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=60G
#SBATCH --time=72:00:00
# stdout/stderr are written relative to the directory you run sbatch from;
# override with `sbatch -o <dir>/%j.out -e <dir>/%j.err` (see CREAM_TRAIN_LOG_DIR /
# CREAM_TEST_LOG_DIR from scripts/cream_env.sh)
#SBATCH -o ../logs/test/slurm_stdout/%j.out
#SBATCH -e ../logs/test/slurm_stdout/%j.err
#SBATCH --job-name=gtex_dev_shift_diff_test
###SBATCH --nodelist=arrietty-h100-gpu03

# project paths, virtualenv and CUBLAS_WORKSPACE_CONFIG come from the YAML config
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/cream_env.sh"
source "$CREAM_VENV/bin/activate"

# run from the project root so "python -m CREAM.<module>" always uses this checkout
cd "$CREAM_PROJECT_ROOT"

config_path=configs/blood_config0-4-shift-diff_m12.yaml
fold=0
model_type=MultiGene
python -m CREAM.test_gtex --config_path $config_path --fold $fold --model_type $model_type --runid icb2whi4







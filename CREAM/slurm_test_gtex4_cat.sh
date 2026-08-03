#!/bin/bash
#SBATCH -p gidbkb,gidb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=30G
#SBATCH --time=72:00:00
# stdout/stderr are written relative to the directory you run sbatch from;
# override with `sbatch -o <dir>/%j.out -e <dir>/%j.err` (see CREAM_TRAIN_LOG_DIR /
# CREAM_TEST_LOG_DIR from scripts/cream_env.sh)
#SBATCH -o ../logs/test/slurm_stdout/%j.out
#SBATCH -e ../logs/test/slurm_stdout/%j.err
#SBATCH --job-name=discrete_real_test
###SBATCH --nodelist=arrietty-h100-gpu03

# project paths, virtualenv and CUBLAS_WORKSPACE_CONFIG come from the YAML config
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/cream_env.sh"
source "$CREAM_VENV/bin/activate"

# run from the project root so "python -m CREAM.<module>" always uses this checkout
cd "$CREAM_PROJECT_ROOT"

# runid carried over from the discrete branch; replace with the run you want to
# evaluate, it must be a checkpoint trained by CREAM.train_gtex2_cat
config_path=configs/blood_config0-5-shift-diff_cat.yaml
fold=0
model_type=MultiGene
python -m CREAM.test_gtex_cat --config_path $config_path --fold $fold --model_type $model_type --runid iczglgc5

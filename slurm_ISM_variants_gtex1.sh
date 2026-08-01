#!/bin/bash
#SBATCH -p gidbkb,gidb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=30G
#SBATCH --time=12:00:00
# stdout/stderr are written relative to the directory you run sbatch from;
# override with `sbatch -o <dir>/%j.out -e <dir>/%j.err` (see CREAM_TRAIN_LOG_DIR /
# CREAM_TEST_LOG_DIR from scripts/cream_env.sh)
#SBATCH -o ./logs/test/slurm_stdout/%j.out
#SBATCH -e ./logs/test/slurm_stdout/%j.err
#SBATCH --job-name=gtex_dev_shift_diff_test
###SBATCH --nodelist=arrietty-h100-gpu03

# project paths, virtualenv and CUBLAS_WORKSPACE_CONFIG come from the YAML config
source "$(dirname "${BASH_SOURCE[0]}")/scripts/cream_env.sh"
source "$CREAM_VENV/bin/activate"

# run from the project root so "python -m CREAM.<module>" always uses this checkout
cd "$CREAM_PROJECT_ROOT"

config_path=CREAM/wandb/run-20250419_144530-1zw1vgdj/files/config.yaml
variants_path="$CREAM_DATA_DIR"/rare_variants_folds/select_genotype-fold2.txt
model_type=MultiGene
python -m CREAM.ism_cream --path_to_metadata $config_path --model_type $model_type --path_to_variants_file $variants_path


config_path=CREAM/wandb/run-20250419_144530-b4ihfjbn/files/config.yaml
variants_path="$CREAM_DATA_DIR"/rare_variants_folds/select_genotype-fold3.txt
model_type=MultiGene
python -m CREAM.ism_cream --path_to_metadata $config_path --model_type $model_type --path_to_variants_file $variants_path

config_path=CREAM/wandb/run-20250419_144530-aaipwoio/files/config.yaml
variants_path="$CREAM_DATA_DIR"/rare_variants_folds/select_genotype-fold4.txt
model_type=MultiGene
python -m CREAM.ism_cream --path_to_metadata $config_path --model_type $model_type --path_to_variants_file $variants_path

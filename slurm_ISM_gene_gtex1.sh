#!/bin/bash
#SBATCH -p gidbkb,gidb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=20G
#SBATCH --time=24:00:00
# stdout/stderr are written relative to the directory you run sbatch from;
# override with `sbatch -o <dir>/%j.out -e <dir>/%j.err` (see CREAM_TRAIN_LOG_DIR /
# CREAM_TEST_LOG_DIR from scripts/cream_env.sh)
#SBATCH -o ./logs/test/slurm_stdout/%j.out
#SBATCH -e ./logs/test/slurm_stdout/%j.err
#SBATCH --job-name=gtex_ism
###SBATCH --nodelist=arrietty-h100-gpu03

# project paths, conda env and CUBLAS_WORKSPACE_CONFIG come from the YAML config
source "$(dirname "${BASH_SOURCE[0]}")/scripts/cream_env.sh"
source activate "$CREAM_CONDA_ENV"

# run from the project root so "python -m CREAM.<module>" always uses this checkout
cd "$CREAM_PROJECT_ROOT"

#config_path=CREAM/wandb/run-20241019_193151-4fplk4yy/files/config.yaml
#variants_path="$CREAM_DATA_DIR"/genes/Whole_Blood/MultiGene/300_train_genes.txt
#model_type=MultiGene
#python -m CREAM.ism_cream0 --path_to_metadata $config_path --model_type $model_type --path_to_genes_file $variants_path


config_path=CREAM/wandb/run-20250525_094514-gl9h617j/files/config.yaml
variants_path="$CREAM_DATA_DIR"/genes/Whole_Blood/MultiGene/test_genes.txt
model_type=MultiGene
python -m CREAM.ism_cream0 --path_to_metadata $config_path --model_type $model_type --path_to_genes_file $variants_path

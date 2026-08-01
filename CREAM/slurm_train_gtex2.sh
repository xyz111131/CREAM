#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=300G
#SBATCH --time=168:00:00
# stdout/stderr are written relative to the directory you run sbatch from;
# override with `sbatch -o <dir>/%j.out -e <dir>/%j.err` (see CREAM_TRAIN_LOG_DIR /
# CREAM_TEST_LOG_DIR from scripts/cream_env.sh)
#SBATCH -o ../logs/train/slurm_stdout/%j.out
#SBATCH -e ../logs/train/slurm_stdout/%j.err
#SBATCH --job-name=simu_attn3
####SBATCH --array=0
####SBATCH --nodelist=arrietty-gpu01

# project paths, virtualenv and CUBLAS_WORKSPACE_CONFIG come from the YAML config
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/cream_env.sh"
source "$CREAM_VENV/bin/activate"

# run from the project root so "python -m CREAM.<module>" always uses this checkout
cd "$CREAM_PROJECT_ROOT"

echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURM_LOCALID=$SLURM_LOCALID"


config_path=configs/blood_config0-7-shift-diff_m12.yaml  #4
fold=0  ##${SLURM_ARRAY_TASK_ID}
model_type=MultiGene
srun python -m CREAM.train_gtex2 --config_path $config_path --fold $fold --model_type $model_type --num_gpus 4

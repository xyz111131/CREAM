#!/bin/bash
#SBATCH -p gidbkb,gidb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH -o /pollard/data/projects/zhhu/enformer_fine_tuning_dev/logs/test/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/zhhu/enformer_fine_tuning_dev/logs/test/slurm_stdout/%j.err
#SBATCH --job-name=gtex_dev_shift_diff_test
###SBATCH --nodelist=arrietty-h100-gpu03

#eval "$(/pollard/home/sdrusinsky/miniforge3/bin/conda shell.bash hook)"
source activate enformer-pytorch-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

config_path=performer/wandb/run-20250419_144530-1zw1vgdj/files/config.yaml
variants_path=data/rare_variants_folds/select_genotype-fold2.txt
model_type=MultiGene
python performer/ism_performer.py --path_to_metadata $config_path --model_type $model_type --path_to_variants_file $variants_path


config_path=performer/wandb/run-20250419_144530-b4ihfjbn/files/config.yaml
variants_path=data/rare_variants_folds/select_genotype-fold3.txt
model_type=MultiGene
python performer/ism_performer.py --path_to_metadata $config_path --model_type $model_type --path_to_variants_file $variants_path

config_path=performer/wandb/run-20250419_144530-aaipwoio/files/config.yaml
variants_path=data/rare_variants_folds/select_genotype-fold4.txt
model_type=MultiGene
python performer/ism_performer.py --path_to_metadata $config_path --model_type $model_type --path_to_variants_file $variants_path

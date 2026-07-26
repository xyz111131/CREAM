#!/bin/bash
#SBATCH -p gidbkb,gidb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH -o /pollard/data/projects/zhhu/enformer_fine_tuning_dev/logs/test/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/zhhu/enformer_fine_tuning_dev/logs/test/slurm_stdout/%j.err
#SBATCH --job-name=gtex_ism
###SBATCH --nodelist=arrietty-h100-gpu03

#eval "$(/pollard/home/sdrusinsky/miniforge3/bin/conda shell.bash hook)"
source activate enformer-pytorch-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

#config_path=CREAM/wandb/run-20241019_193151-4fplk4yy/files/config.yaml
#variants_path=data/genes/Whole_Blood/MultiGene/300_train_genes.txt
#model_type=MultiGene
#python CREAM/ism_cream0.py --path_to_metadata $config_path --model_type $model_type --path_to_genes_file $variants_path


config_path=CREAM/wandb/run-20250525_094514-gl9h617j/files/config.yaml
variants_path=data/genes/Whole_Blood/MultiGene/test_genes.txt
model_type=MultiGene
python CREAM/ism_cream0.py --path_to_metadata $config_path --model_type $model_type --path_to_genes_file $variants_path

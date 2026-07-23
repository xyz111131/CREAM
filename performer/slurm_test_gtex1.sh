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

config_path=configs/blood_config0-4-shift-diff.yaml
fold=0
model_type=MultiGene
python ./test_gtex.py --config_path $config_path --fold $fold --model_type $model_type --runid 9itew05z







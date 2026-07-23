#!/bin/bash
#SBATCH -p gidbkb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=400G
#SBATCH --time=162:00:00
#SBATCH -o /pollard/data/projects/zhhu/enformer_fine_tuning_dev/logs/train/slurm_stdout/%j.out
#SBATCH -e /pollard/data/projects/zhhu/enformer_fine_tuning_dev/logs/train/slurm_stdout/%j.err
#SBATCH --job-name=multi_tissue_simu_attn0
####SBATCH --array=0
####SBATCH --nodelist=arrietty-gpu01

#eval "$(/pollard/home/sdrusinsky/miniforge3/bin/conda shell.bash hook)"
source activate enformer-pytorch-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURM_LOCALID=$SLURM_LOCALID"


config_path=configs/blood_config0-4-shift-diff.yaml  #4
fold=0  ##${SLURM_ARRAY_TASK_ID}
model_type=MultiGene
srun python ./train_gtex2.py --config_path $config_path --fold $fold --model_type $model_type --num_gpus 4

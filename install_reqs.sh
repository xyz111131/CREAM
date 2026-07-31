# the environment name comes from env.conda_env in CREAM/configs/defaults.yaml
source "$(dirname "${BASH_SOURCE[0]}")/scripts/cream_env.sh"
eval "$(conda shell.bash hook)"
conda activate "$CREAM_CONDA_ENV"
conda update --all

conda \
    install -y \
    -c pytorch \
    -c nvidia \
    pytorch=2.3.1 \
    pytorch-cuda=12.1

conda \
  install -y \
  -c conda-forge \
  cython \
  ipython \
  matplotlib \
  pandas \
  scikit-learn \
  seaborn \
  pyarrow \
  tqdm \
  lightning \
  mkl \ 
  rich \
  torchmetrics \ 


pip install enformer_pytorch kipoiseq pysam wandb vcfpy

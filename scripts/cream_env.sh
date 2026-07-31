#!/bin/bash
# Exports the CREAM paths and settings from the YAML config, so that no shell
# script has to hard-code a project directory or a conda environment name.
#
#   source /path/to/scripts/cream_env.sh [config_path]
#
# Sets: CREAM_PROJECT_ROOT, CREAM_DATA_DIR, CREAM_RESULTS_DIR, CREAM_LOG_DIR,
#       CREAM_TRAIN_LOG_DIR, CREAM_TEST_LOG_DIR, CREAM_CONDA_ENV,
#       CREAM_WANDB_PROJECT, CUBLAS_WORKSPACE_CONFIG.
#
# Values come from CREAM/configs/defaults.yaml merged with the config passed in.
# Export CREAM_PROJECT_ROOT beforehand to point at a different checkout.

_cream_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_cream_config=${1:-}

if _cream_exports=$(cd "$_cream_repo_root" && python -m CREAM.config --shell \
        ${_cream_config:+--config_path "$_cream_config"} 2>/dev/null) \
        && [ -n "$_cream_exports" ]; then
    eval "$_cream_exports"
else
    # No usable python yet (this is often sourced before activating the conda
    # environment), so read the handful of settings we need out of the defaults.
    _cream_yaml=$_cream_repo_root/CREAM/configs/defaults.yaml
    _cream_get() { sed -n "s/^  $1: *//p" "$_cream_yaml" | head -1 | tr -d '"'; }

    export CREAM_PROJECT_ROOT=${CREAM_PROJECT_ROOT:-$_cream_repo_root}
    export CREAM_DATA_DIR=$CREAM_PROJECT_ROOT/$(_cream_get data_dir)
    export CREAM_RESULTS_DIR=$CREAM_PROJECT_ROOT/$(_cream_get results_dir)
    export CREAM_LOG_DIR=$CREAM_PROJECT_ROOT/$(_cream_get log_dir)
    export CREAM_TRAIN_LOG_DIR=$CREAM_LOG_DIR/$(_cream_get train_log_subdir)
    export CREAM_TEST_LOG_DIR=$CREAM_LOG_DIR/$(_cream_get test_log_subdir)
    export CREAM_CONDA_ENV=$(_cream_get conda_env)
    export CUBLAS_WORKSPACE_CONFIG=$(_cream_get cublas_workspace_config)
    unset _cream_get _cream_yaml
fi

# slurm will not create these itself, and a missing directory kills the job
mkdir -p "$CREAM_TRAIN_LOG_DIR" "$CREAM_TEST_LOG_DIR"

unset _cream_repo_root _cream_config _cream_exports

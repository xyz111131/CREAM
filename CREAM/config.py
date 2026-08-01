"""Central configuration for CREAM.

Paths, project names and data file names live in YAML rather than in the code.
``CREAM/configs/defaults.yaml`` holds the defaults and the experiment config
passed on the command line is merged on top of it, so an experiment config only
lists what it changes::

    from CREAM import config as cream_config

    config = cream_config.load_config("configs/blood_config.yaml", fold=0)
    intervals = cream_config.genomic_intervals_file(config)

Relative paths are resolved against the project root: ``$CREAM_PROJECT_ROOT``
if set, else ``paths.project_root`` from the config, else the repository root.
No other module should build a path from ``__file__``, ``os.getcwd()`` or a
literal directory name.

The helpers below read a config with plain ``config["section"]["key"]`` access
so they work with a :class:`Config`, a plain dict, or a ``wandb.config``.
"""

import argparse
import copy
import os
import shlex
import sys
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "configs"
DEFAULTS_PATH = CONFIG_DIR / "defaults.yaml"
REPO_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT_ENV_VAR = "CREAM_PROJECT_ROOT"


class Config(dict):
    """A dict whose keys are also readable as attributes, nested dicts included."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            super().__setitem__(key, _wrap(value))

    def __setitem__(self, key, value):
        super().__setitem__(key, _wrap(value))

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def to_dict(self):
        """Plain nested dicts, for wandb.init(config=...) and yaml.safe_dump()."""
        return _unwrap(self)


def _wrap(value):
    if isinstance(value, Config):
        return value
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value):
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base, override):
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_config_path(config_path):
    """Find an experiment config given as an absolute path, a path relative to
    the working directory, or just its name in ``CREAM/configs``."""
    candidates = [Path(config_path)]
    if not Path(config_path).is_absolute():
        candidates += [CONFIG_DIR / config_path, CONFIG_DIR / Path(config_path).name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Config {config_path!r} not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def load_config(config_path=None, use_test_data=False, **overrides):
    """Load ``defaults.yaml``, merge ``config_path`` and ``overrides`` on top of
    it, and resolve every path in the ``paths`` section to an absolute path.

    Set ``use_test_data`` to point ``data_dir``/``results_dir`` at the small
    test dataset (``paths.test_data_dir`` / ``paths.test_results_dir``).
    """
    config = _read_yaml(DEFAULTS_PATH)
    if config_path is not None:
        config = _deep_merge(config, _read_yaml(resolve_config_path(config_path)))
        config["config_path"] = str(resolve_config_path(config_path))
    if overrides:
        config = _deep_merge(config, overrides)
    return _resolve_paths(Config(config), use_test_data=use_test_data)


def from_wandb_metadata(metadata, use_test_data=False, **overrides):
    """Load a config from a wandb run's ``config.yaml``, where every entry is
    stored as ``{"value": ...}``. Missing sections fall back to the defaults, so
    runs logged before a setting existed still resolve."""
    flattened = {
        key: value["value"]
        for key, value in metadata.items()
        if isinstance(value, dict) and "value" in value
    }
    config = _deep_merge(_read_yaml(DEFAULTS_PATH), flattened)
    if overrides:
        config = _deep_merge(config, overrides)
    return _resolve_paths(Config(config), use_test_data=use_test_data)


def _project_root(configured_root):
    root = os.environ.get(PROJECT_ROOT_ENV_VAR) or configured_root
    if not root:
        return REPO_ROOT
    return Path(os.path.expandvars(os.path.expanduser(str(root)))).resolve()


def _absolute(value, root):
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return os.path.normpath(expanded if expanded.is_absolute() else (Path(root) / expanded))


def _resolve_paths(config, use_test_data=False):
    paths = config.setdefault("paths", Config())
    root = _project_root(paths.get("project_root"))
    paths["project_root"] = str(root)
    for key, value in list(paths.items()):
        if key == "project_root" or value is None:
            continue
        paths[key] = _absolute(value, root)

    if use_test_data:
        paths["data_dir"] = paths.get("test_data_dir") or paths["data_dir"]
        paths["results_dir"] = paths.get("test_results_dir") or paths["results_dir"]
    config["use_test_data"] = bool(use_test_data)

    # Flat aliases: the datasets take DATA_DIR, and wandb run metadata written by
    # earlier versions of the pipeline refers to it by that name.
    config["DATA_DIR"] = paths["data_dir"]
    config["RESULTS_DIR"] = paths["results_dir"]
    return config


# --------------------------------------------------------------------------- #
# reading settings (works with Config, dict and wandb.config)
# --------------------------------------------------------------------------- #
def _section(config, name):
    try:
        section = config[name]
    except (KeyError, TypeError, AttributeError):
        return {}
    return section if isinstance(section, dict) else {}


def setting(config, section, key, default=None):
    """One value out of a config section, falling back to ``defaults.yaml``."""
    value = _section(config, section).get(key, None)
    if value is None:
        value = _section(_DEFAULTS, section).get(key, default)
    return value


_DEFAULTS = _read_yaml(DEFAULTS_PATH)


def path_setting(config, key):
    """A resolved path from the ``paths`` section."""
    value = _section(config, "paths").get(key) or _section(_DEFAULTS, "paths").get(key)
    if value is None:
        return None
    return _absolute(value, project_root(config))


def project_root(config=None):
    if config is not None:
        configured = _section(config, "paths").get("project_root")
    else:
        configured = None
    return str(_project_root(configured))


def data_dir(config):
    try:
        return config["DATA_DIR"]
    except (KeyError, TypeError):
        return path_setting(config, "data_dir")


def results_dir(config):
    try:
        return config["RESULTS_DIR"]
    except (KeyError, TypeError):
        return path_setting(config, "results_dir")


def log_dir(config=None):
    return path_setting(config or {}, "log_dir")


# --------------------------------------------------------------------------- #
# data files
# --------------------------------------------------------------------------- #
def data_path(config, *parts):
    """A path inside the data directory."""
    return os.path.join(data_dir(config), *[str(p) for p in parts])


def data_file(config, key):
    """A data file named by ``data.<key>`` in the config."""
    return data_path(config, setting(config, "data", key))


def genomic_intervals_file(config):
    return data_file(config, "genomic_intervals_file")


def enformer_intervals_file(config):
    return data_file(config, "enformer_intervals_file")


def genome_fasta(config):
    return data_file(config, "genome_fasta_file")


def gtex_vcf(config):
    return data_file(config, "gtex_vcf_file")


def gene_id_mapping_file(config):
    return os.path.join(expression_dir(config), setting(config, "data", "gene_id_mapping_file"))


def gene_embedding_file(config):
    return data_file(config, "gene_embedding_file")


def empty_gene_file(config):
    return data_file(config, "empty_gene_file")


def expression_dir(config):
    """Directory holding the expression matrices (``expression_filepath``)."""
    try:
        subdir = config["expression_filepath"]
    except (KeyError, TypeError):
        subdir = _DEFAULTS["expression_filepath"]
    return data_path(config, subdir)


def expression_files(config, predicted=False):
    """Expression matrices in ``expression_dir``, sorted by name."""
    pattern = setting(
        config, "data", "predicted_expression_glob" if predicted else "expression_glob"
    )
    return sorted(Path(expression_dir(config)).glob(pattern))


def expression_file(config, tissue, kind="normalized"):
    """A single expression matrix for one tissue.

    ``kind`` is one of ``normalized``, ``residual`` or ``raw``.
    """
    keys = {
        "normalized": "expression_filename",
        "residual": "residual_expression_filename",
        "raw": "raw_expression_filename",
    }
    template = setting(config, "data", keys[kind])
    return os.path.join(expression_dir(config), template.format(tissue=tissue))


def consensus_seq_dir(config, cohort="gtex"):
    key = "consensus_seq_subdir" 
    return data_path(config, setting(config, "data", key))


def consensus_seq_filename(config):
    return setting(config, "data", "consensus_seq_filename")


def tissue_dirname(tissue):
    """The on-disk form of a GTEx tissue name: 'Brain - Cortex' -> 'Brain_Cortex'."""
    return tissue.replace(" -", "").replace(" ", "_").replace("(", "").replace(")", "")


def tissue_alias(config, tissue):
    """The full GTEx tissue name for a short name used in a config ('blood')."""
    aliases = _section(config, "tissue_aliases") or _section(_DEFAULTS, "tissue_aliases")
    return aliases.get(tissue, tissue_dirname(tissue))


def gene_set_dir(config, model_type, tissue=None, by_tissue=False):
    """Directory holding the train/valid/test gene list files."""
    key = "gene_set_dir_template_by_tissue" if by_tissue else "gene_set_dir_template"
    template = setting(config, "data", key)
    return data_path(
        config,
        template.format(
            genes_subdir=setting(config, "data", "genes_subdir"),
            gene_set_subdir=setting(config, "data", "gene_set_subdir"),
            model_type=model_type,
            tissue=tissue_dirname(tissue) if tissue else "",
        ),
    )


def gene_set_path(config, filename, model_type, tissue=None, by_tissue=False):
    return os.path.join(gene_set_dir(config, model_type, tissue, by_tissue), filename)


def multi_geneset_path(config, filename, model_type, tissue):
    template = setting(config, "data", "multi_geneset_dir_template")
    return data_path(
        config,
        template.format(model_type=model_type, tissue=tissue_dirname(tissue)),
        filename,
    )


def donor_dir(config, dataset="gtex", rare_variants=False):
    if rare_variants:
        return data_path(config, setting(config, "data", "rare_variants_subdir"))
    cv_dir = data_path(config, setting(config, "data", "cross_validation_subdir"), dataset)
    if dataset == "gtex":
        cv_dir = os.path.join(cv_dir, "cv_folds")
    return cv_dir


def donor_list_path(config, split, fold, dataset="gtex", rare_variants=False):
    """Donor IDs for one cross-validation split: ``split`` is train/val/test."""
    filename = setting(config, "data", "donor_ids_filename").format(split=split, fold=fold)
    return os.path.join(donor_dir(config, dataset, rare_variants), filename)



def eqtl_dir(config):
    configured = path_setting(config, "eqtl_dir")
    return configured or data_path(config, "eQTL_susie")


def eqtl_tissue_label_file(config):
    return os.path.join(eqtl_dir(config), setting(config, "data", "eqtl_tissue_label_file"))


def dataset_paths(config, cohort="gtex"):
    """The paths GTExDataset needs, as a plain dict (it is pickled into
    checkpoints along with the dataset, so keep it free of live objects)."""
    return {
        "data_dir": data_dir(config),
        "genomic_intervals_file": genomic_intervals_file(config),
        "enformer_intervals_file": enformer_intervals_file(config),
        "consensus_seq_dir": consensus_seq_dir(config, cohort),
        "consensus_seq_filename": consensus_seq_filename(config),
        "genome_fasta": genome_fasta(config),
    }


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def fold_dirname(config, fold, rare_variants=False):
    key = "rare_variants_fold_dirname" if rare_variants else "fold_dirname"
    return setting(config, "outputs", key).format(fold=fold)


def run_dir(config, model_type, train_gene_set, fold, run_id, rare_variants=False):
    """Directory for one training run's checkpoints and predictions."""
    template = setting(config, "outputs", "run_dir_template")
    return os.path.join(
        results_dir(config),
        template.format(
            experiment_name=config["experiment_name"],
            model_type=model_type,
            train_gene_set=train_gene_set,
            fold_dir=fold_dirname(config, fold, rare_variants),
            run_id=run_id,
        ),
    )


def gene_set_name(train_gene_filename):
    """Directory name a run is stored under, derived from its train gene file.

    This is ``str.strip('.txt')``, which strips any leading/trailing '.', 't'
    and 'x' rather than just the extension: 'train_filter_egenes_5K.txt' becomes
    'rain_filter_egenes_5K'. Existing result directories are named that way, so
    the behaviour is kept.
    """
    return train_gene_filename.strip(".txt")


def checkpoint_dir(config, run_directory):
    return os.path.join(run_directory, setting(config, "outputs", "checkpoint_subdir"))


def output_dir(config, key, *parts):
    """A directory under the results directory named by ``outputs.<key>``."""
    return os.path.join(results_dir(config), setting(config, "outputs", key), *[str(p) for p in parts])


# --------------------------------------------------------------------------- #
# experiment tracking and model
# --------------------------------------------------------------------------- #
def wandb_project(config, use_test_data=False, rare_variants=False, raw_expr=False):
    if use_test_data:
        return setting(config, "wandb", "project_test_data")
    if rare_variants:
        return setting(config, "wandb", "project_rare_variants")
    if raw_expr:
        return setting(config, "wandb", "project_raw_expr")
    return setting(config, "wandb", "project")


def wandb_entity(config):
    return setting(config, "wandb", "entity")


# --------------------------------------------------------------------------- #
# command line: let shell scripts read the same config
# --------------------------------------------------------------------------- #
def _shell_exports(config):
    venv = setting(config, "env", "venv")
    return {
        "CREAM_PROJECT_ROOT": project_root(config),
        "CREAM_DATA_DIR": data_dir(config),
        "CREAM_RESULTS_DIR": results_dir(config),
        "CREAM_LOG_DIR": path_setting(config, "log_dir"),
        "CREAM_TRAIN_LOG_DIR": os.path.join(
            path_setting(config, "log_dir"), setting(config, "env", "train_log_subdir")
        ),
        "CREAM_TEST_LOG_DIR": os.path.join(
            path_setting(config, "log_dir"), setting(config, "env", "test_log_subdir")
        ),
        "CREAM_VENV": _absolute(venv, project_root(config)) if venv else None,
        "CREAM_CONDA_ENV": setting(config, "env", "conda_env"),
        "CREAM_WANDB_PROJECT": wandb_project(config),
        "CUBLAS_WORKSPACE_CONFIG": setting(config, "env", "cublas_workspace_config"),
    }


def _lookup(config, dotted_key):
    value = config
    for part in dotted_key.split("."):
        value = value[part]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Read CREAM configuration from the shell, e.g. "
        'eval "$(python -m CREAM.config --shell)"'
    )
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--use_test_data", action="store_true", default=False)
    parser.add_argument("--shell", action="store_true", help="print export statements")
    parser.add_argument("--get", type=str, default=None, help="print one value, e.g. paths.data_dir")
    args = parser.parse_args()

    config = load_config(args.config_path, use_test_data=args.use_test_data)

    if args.get:
        print(_lookup(config, args.get))
    elif args.shell:
        for name, value in _shell_exports(config).items():
            if value is not None:
                print(f"export {name}={shlex.quote(str(value))}")
    else:
        yaml.safe_dump(config.to_dict(), sys.stdout, sort_keys=False)


if __name__ == "__main__":
    main()

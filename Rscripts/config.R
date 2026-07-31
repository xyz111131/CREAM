# Reads the CREAM YAML configuration so the R analyses use the same paths as the
# Python pipeline instead of hard-coding them.
#
#   source("Rscripts/config.R")
#   cream <- cream_config()          # or cream_config("CREAM/configs/blood_config.yaml")
#   fread(file.path(cream$eqtl_dir, "dataset_tissue_label.csv"))
#
# Relative paths in the config are resolved against the project root: the
# CREAM_PROJECT_ROOT environment variable if set, else paths.project_root from
# the config, else the repository root.

cream_repo_root <- function() {
  # this file lives in <repo>/Rscripts
  for (f in sys.frames()) {
    path <- f$ofile
    if (!is.null(path)) return(normalizePath(file.path(dirname(path), "..")))
  }
  normalizePath(".")
}

.cream_read_yaml <- function(path) {
  if (requireNamespace("yaml", quietly = TRUE)) {
    return(yaml::read_yaml(path))
  }
  # Minimal fallback for the two-level "key:" / "  key: value" structure of the
  # CREAM configs, so the analyses run without the yaml package installed.
  lines <- readLines(path, warn = FALSE)
  lines <- lines[!grepl("^\\s*(#|$)", lines)]
  out <- list()
  section <- NULL
  for (line in lines) {
    nested <- grepl("^  \\S", line)
    kv <- sub("\\s+#.*$", "", trimws(line))
    key <- sub(":.*$", "", kv)
    value <- trimws(sub("^[^:]*:", "", kv))
    value <- gsub('^"|"$', "", gsub("^'|'$", "", value))
    if (value == "") {
      section <- key
      out[[key]] <- list()
    } else if (nested && !is.null(section)) {
      out[[section]][[key]] <- if (value == "null") NULL else value
    } else {
      section <- NULL
      out[[key]] <- if (value == "null") NULL else value
    }
  }
  out
}

.cream_merge <- function(base, override) {
  for (key in names(override)) {
    if (is.list(base[[key]]) && is.list(override[[key]])) {
      base[[key]] <- .cream_merge(base[[key]], override[[key]])
    } else {
      base[[key]] <- override[[key]]
    }
  }
  base
}

# lexical cleanup of "." and ".." segments, matching what the Python loader does
# (normalizePath() would resolve symlinks and can rewrite a path to a different
# mount point)
.cream_normalize <- function(path) {
  absolute <- substr(path, 1, 1) == "/"
  parts <- strsplit(path, "/", fixed = TRUE)[[1]]
  out <- character(0)
  for (part in parts) {
    if (part == "" || part == ".") next
    if (part == ".." && length(out) > 0 && out[length(out)] != "..") {
      out <- out[-length(out)]
    } else {
      out <- c(out, part)
    }
  }
  paste0(if (absolute) "/" else "", paste(out, collapse = "/"))
}

.cream_absolute <- function(value, root) {
  if (is.null(value)) return(NULL)
  value <- path.expand(value)
  if (substr(value, 1, 1) == "/") return(.cream_normalize(value))
  .cream_normalize(file.path(root, value))
}

cream_config <- function(config_path = NULL, root = NULL) {
  if (is.null(root)) root <- cream_repo_root()
  defaults_path <- file.path(root, "CREAM", "configs", "defaults.yaml")
  cfg <- .cream_read_yaml(defaults_path)
  if (!is.null(config_path)) {
    cfg <- .cream_merge(cfg, .cream_read_yaml(config_path))
  }

  env_root <- Sys.getenv("CREAM_PROJECT_ROOT", unset = "")
  project_root <- if (nzchar(env_root)) {
    normalizePath(env_root, mustWork = FALSE)
  } else if (!is.null(cfg$paths$project_root)) {
    .cream_absolute(cfg$paths$project_root, root)
  } else {
    root
  }

  paths <- lapply(cfg$paths, .cream_absolute, root = project_root)
  paths$project_root <- project_root

  data_file <- function(key) file.path(paths$data_dir, cfg$data[[key]])
  c(
    paths,
    list(
      config = cfg,
      genomic_intervals_file = data_file("genomic_intervals_file"),
      enformer_intervals_file = data_file("enformer_intervals_file"),
      genome_fasta = data_file("genome_fasta_file"),
      eqtl_tissue_label_file = file.path(paths$eqtl_dir, cfg$data$eqtl_tissue_label_file),
      data_path = function(...) file.path(paths$data_dir, ...),
      results_path = function(...) file.path(paths$results_dir, ...)
    )
  )
}

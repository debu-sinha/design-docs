# MLflow UV Package Manager Support - Design Document v2

|                    |                                                           |
| ------------------ | --------------------------------------------------------- |
| **Author(s)**      | Debu Sinha ([@debu-sinha](https://github.com/debu-sinha)) |
| **Organization**   | MLflow Community                                          |
| **Status**         | IMPLEMENTED                                               |
| **GitHub Issue**   | [#12478](https://github.com/mlflow/mlflow/issues/12478)   |
| **Pull Request**   | [#20344](https://github.com/mlflow/mlflow/pull/20344)     |

**Change Log:**

- 2025-12-05: Initial design version (v1)
- 2025-01-26: Updated to reflect Phase 1 + Phase 2 implementation
- 2025-01-30: Renamed `uv_lock` to `uv_project_path`; added `MLFLOW_UV_AUTO_DETECT`; added `_create_virtualenv` integration
- 2026-02-10: Full rewrite with Excalidraw diagrams, `uv_groups`/`uv_extras` API params, complete implementation state (v2 final)

---

## Executive Summary

This document describes the design and implementation of native UV package manager support in MLflow for automatic dependency inference during model logging. UV is 8-10x faster than pip (cold cache) and 80-115x faster (warm cache), with growing adoption in the ML/data science community.

**What's Implemented:**

- Automatic UV project detection (`uv.lock` + `pyproject.toml` in CWD)
- Dependency export via `uv export` with PEP 508 marker filtering
- UV artifact logging (`uv.lock`, `pyproject.toml`, `.python-version`)
- `uv_project_path` parameter for monorepo support
- `uv_groups` and `uv_extras` parameters on `save_model()` / `log_model()` for selective dependency export
- `MLFLOW_UV_AUTO_DETECT` environment variable to disable auto-detection
- `MLFLOW_LOG_UV_FILES` environment variable to disable artifact logging
- UV-based environment restoration functions (`setup_uv_sync_environment`, `run_uv_sync`)
- Private index URL extraction utility

---

# Part I: Design Overview

## Motivation

**The Problem (from GitHub Issue #12478):**

> "When using the uv package manager to create virtual environments and install packages, MLflow is unable to infer requirements.txt when auto-logging a model."

Users managing Python environments with UV must manually specify dependencies when logging models -- defeating the purpose of MLflow's automatic dependency inference.

**Why UV matters:**

- 8-10x faster than pip (cold cache), 80-115x faster (warm cache)
- Growing adoption in ML community (25+ upvotes, 22+ comments on issue)
- Combines pip, virtualenv, and pyenv functionality
- Better reproducibility via deterministic lock files

**Current workaround (painful):**

```python
# Users must manually export and specify dependencies
import subprocess
result = subprocess.run(
    ["uv", "export", "--no-dev", "--no-hashes", "--frozen"],
    capture_output=True, text=True
)
requirements = result.stdout.strip().split("\n")
mlflow.sklearn.log_model(model, "model", pip_requirements=requirements)
```

**Goal:** Make UV projects "just work" with MLflow's existing dependency inference.

---

## Architecture Overview

### Model Logging Flow

The diagram below shows the complete model logging flow with UV integration.

![UV Model Logging Flow](images/uv-logging-flow.svg)

When `save_model()` or `log_model()` is called:

1. Check `MLFLOW_UV_AUTO_DETECT` (default: `true`). If disabled, skip UV detection entirely.
2. Call `detect_uv_project()` to look for `uv.lock` + `pyproject.toml` in CWD (or `uv_project_path` if provided).
3. If found and UV >= 0.5.0 is installed, run `uv export` with the specified groups/extras.
4. Filter the output by PEP 508 environment markers for the current platform.
5. Copy UV artifacts (`uv.lock`, `pyproject.toml`, `.python-version`) unless `MLFLOW_LOG_UV_FILES=false`.
6. If any step fails, fall back gracefully to standard pip-based inference (capture imported packages).

### Parameter Threading Call Chain

The `uv_project_path`, `uv_groups`, and `uv_extras` parameters flow through the full call chain from user API to subprocess invocation.

![Parameter Threading Call Chain](images/uv-call-chain.svg)

```
log_model(uv_project_path, uv_groups, uv_extras)
  -> save_model(uv_project_path, uv_groups, uv_extras)
    -> _save_model_with_loader_module_and_data_path(uv_groups, uv_extras)
       OR _save_model_with_class_artifacts_params(uv_groups, uv_extras)
      -> infer_pip_requirements(uv_groups, uv_extras)
        -> export_uv_requirements(groups=uv_groups, extras=uv_extras)
          -> subprocess: uv export --group X --extra Y
```

### Environment Restoration Flow

When loading a model that was logged from a UV project, the restoration path depends on whether `uv.lock` is in the artifacts and whether `env_manager="uv"` is specified.

![Environment Restoration Flow](images/uv-restore-flow.svg)

---

## Implementation Summary

### Core Features

| Feature                      | Status | Description                                                             |
| ---------------------------- | ------ | ----------------------------------------------------------------------- |
| UV Project Detection         | Done   | Check for `uv.lock` + `pyproject.toml` in CWD                           |
| Dependency Export             | Done   | `uv export --no-dev --no-hashes --frozen --no-header --no-emit-project` |
| PEP 508 Marker Filtering     | Done   | Filter requirements by Python version/platform                          |
| UV Artifact Logging          | Done   | Log `uv.lock`, `pyproject.toml`, `.python-version`                      |
| `uv_project_path` parameter  | Done   | Explicit UV project path for monorepo support                           |
| `uv_groups` parameter        | Done   | Selective dependency group export on `save_model`/`log_model`           |
| `uv_extras` parameter        | Done   | Optional extras export on `save_model`/`log_model`                      |
| `MLFLOW_UV_AUTO_DETECT`      | Done   | Disable UV auto-detection (default: `true`)                             |
| `MLFLOW_LOG_UV_FILES`        | Done   | Disable UV file logging (default: `true`)                               |
| Dependency groups (env vars) | Done   | `MLFLOW_UV_GROUPS`, `MLFLOW_UV_ONLY_GROUPS`, `MLFLOW_UV_EXTRAS`         |
| UV Sync Functions            | Done   | `setup_uv_sync_environment()`, `run_uv_sync()`                          |
| Private Index Extraction     | Done   | Utility function (not auto-injected into requirements.txt)              |
| Graceful Fallback            | Done   | Falls back to pip inference on any UV failure                           |

### NOT Implemented (Design Decisions)

| Feature                      | Reason                                              |
| ---------------------------- | --------------------------------------------------- |
| `log_uv_files` API parameter | Environment variable approach is less error-prone   |
| Auto-prepend index URLs      | Error-prone, may inject wrong/stale URLs            |
| Parent directory search      | CWD-only detection is deterministic and predictable |

---

# Part II: Detailed Implementation

## Core Module: `mlflow/utils/uv_utils.py`

### Public Functions

| Function                          | Purpose                                              |
| --------------------------------- | ---------------------------------------------------- |
| `get_uv_version()`                | Get installed UV version as `packaging.version.Version` |
| `is_uv_available()`               | Check UV >= 0.5.0 is installed                       |
| `detect_uv_project(directory)`    | Find `uv.lock` + `pyproject.toml` in given dir      |
| `export_uv_requirements(...)`     | Run `uv export` and parse output                    |
| `get_python_version_from_uv_project(directory)` | Extract Python version from `.python-version` or `pyproject.toml` |
| `copy_uv_project_files(dest, src)` | Copy UV artifacts to model dir                      |
| `extract_index_urls_from_uv_lock(path)` | Extract private index URLs from `uv.lock`      |
| `setup_uv_sync_environment(env_dir, model_path, python_version)` | Prepare for `uv sync --frozen` |
| `run_uv_sync(project_dir)`        | Execute `uv sync` for environment restoration       |
| `has_uv_lock_artifact(model_path)` | Check if model has `uv.lock` artifact               |

### UV Detection

```python
_MIN_UV_VERSION = Version("0.5.0")

def detect_uv_project(directory: str | Path | None = None) -> dict[str, Path] | None:
    """
    Detect UV project by checking for BOTH uv.lock and pyproject.toml.
    CWD-only detection (no parent directory search).

    Returns {"uv_lock": Path, "pyproject": Path} or None.
    """
```

### Dependency Export

```python
def export_uv_requirements(
    directory: str | Path | None = None,
    no_dev: bool = True,
    no_hashes: bool = True,
    frozen: bool = True,
    groups: list[str] | None = None,
    only_groups: list[str] | None = None,
    extras: list[str] | None = None,
) -> list[str] | None:
```

The export function:
1. Builds the `uv export` command with appropriate flags
2. Handles group/extras parameters (`--group`, `--only-group`, `--extra`)
3. Parses the output and filters by PEP 508 environment markers
4. Deduplicates packages (UV may emit multiple entries with different markers)
5. Returns `None` on any failure (graceful fallback)

### PEP 508 Marker Evaluation

The `_evaluate_marker()` function handles environment markers that UV attaches to requirements:

- `python_version`, `python_full_version`
- `platform_python_implementation`, `sys_platform`, `platform_system`, `platform_machine`, `os_name`
- Supports `and`/`or` compound markers
- Conservative: unknown markers default to match

---

## Public API Changes

### `save_model()` / `log_model()` Parameters

Three new parameters added to `mlflow.pyfunc.save_model()` and `mlflow.pyfunc.log_model()`:

```python
mlflow.pyfunc.save_model(
    python_model=model,
    name="model",
    # UV-specific parameters (all optional, all experimental)
    uv_project_path="/path/to/monorepo/package",  # Explicit UV project dir
    uv_groups=["serving"],                          # Dependency groups to include
    uv_extras=["gpu"],                              # Optional extras to include
)
```

| Parameter         | Type                       | Default | Description                                                    |
| ----------------- | -------------------------- | ------- | -------------------------------------------------------------- |
| `uv_project_path` | `str \| Path \| None`     | `None`  | Explicit path to UV project (for monorepos)                    |
| `uv_groups`       | `list[str] \| None`       | `None`  | Dependency groups to include (`uv export --group`)             |
| `uv_extras`       | `list[str] \| None`       | `None`  | Optional extras to include (`uv export --extra`)               |

All three parameters are marked `@experimental` and may change in a future release without warning.

---

## Environment Variables

| Variable                | Default | Description                                               |
| ----------------------- | ------- | --------------------------------------------------------- |
| `MLFLOW_UV_AUTO_DETECT` | `true`  | Set to `false` to disable UV project auto-detection       |
| `MLFLOW_LOG_UV_FILES`   | `true`  | Set to `false`/`0`/`no` to disable UV file logging        |
| `MLFLOW_UV_GROUPS`      | (none)  | Comma-separated dependency groups to include              |
| `MLFLOW_UV_ONLY_GROUPS` | (none)  | Comma-separated groups (exclusive mode, takes precedence) |
| `MLFLOW_UV_EXTRAS`      | (none)  | Comma-separated optional extras to include                |

`MLFLOW_UV_AUTO_DETECT` is a standard MLflow `_BooleanEnvironmentVariable` defined in `mlflow/environment_variables.py`. The remaining env vars are parsed in `mlflow/utils/uv_utils.py`.

**Usage Examples:**

```bash
# Disable UV auto-detection entirely
MLFLOW_UV_AUTO_DETECT=false python train.py

# Disable UV file logging (for large projects)
MLFLOW_LOG_UV_FILES=false python train.py

# Include serving group dependencies
MLFLOW_UV_GROUPS="serving" python train.py

# Export ONLY serving group (minimal)
MLFLOW_UV_ONLY_GROUPS="serving" python train.py

# Combine groups and extras
MLFLOW_UV_GROUPS="serving" MLFLOW_UV_EXTRAS="gpu" python train.py
```

Note: When `uv_groups` or `uv_extras` are passed as API parameters, they take precedence over the environment variables.

---

## Integration with `infer_pip_requirements()`

The `infer_pip_requirements()` function in `mlflow/utils/environment.py` is the bridge between model saving and UV export:

```python
def infer_pip_requirements(
    model_uri,
    flavor,
    fallback=None,
    timeout=None,
    extra_env_vars=None,
    uv_groups=None,       # NEW
    uv_extras=None,       # NEW
):
```

When called:
1. Check `MLFLOW_UV_AUTO_DETECT.get()` -- if `false`, skip UV entirely
2. Call `detect_uv_project()` -- if no UV project found, use standard inference
3. Call `export_uv_requirements(directory, groups=uv_groups, extras=uv_extras)`
4. If UV export returns requirements, use them directly (skip model-based inference)
5. If UV export fails or returns `None`, fall back to standard package-capture inference

---

## UV-Based Environment Restoration

### Restoration Functions

```python
def setup_uv_sync_environment(
    env_dir: str | Path,
    model_path: str | Path,
    python_version: str,
) -> bool:
    """Prepare UV project structure for uv sync --frozen."""

def run_uv_sync(
    project_dir: str | Path,
    frozen: bool = True,
    no_dev: bool = True,
    capture_output: bool = False,
) -> bool:
    """Execute uv sync for environment restoration."""

def has_uv_lock_artifact(model_path: str | Path) -> bool:
    """Check if model has uv.lock artifact."""
```

The restoration flow:
1. Check if model artifacts contain `uv.lock`
2. Copy `uv.lock` and `pyproject.toml` (or create minimal one) to environment directory
3. Copy `.python-version` if available
4. Run `uv sync --frozen --no-dev`
5. Fall back to pip if any step fails

---

## Private Index Handling

**Problem:** `uv export` does not emit `--index-url` / `--extra-index-url`, which can cause pip-based restores to fail for private packages.

**Approach (implemented):**

- Do NOT auto-inject index URLs into requirements.txt (error-prone)
- Provide utility function `extract_index_urls_from_uv_lock()` for manual use/debugging
- Log WARNING when private indexes detected
- Recommend `env_manager="uv"` for private index scenarios

**Why we don't auto-inject:**

- Index URLs may be environment-specific (dev vs prod)
- Credentials still required -- auto-injection gives false sense of security
- Per-package index pinning (`[tool.uv.sources]`) not representable in requirements.txt

| Restore Method                    | Configuration Required                              |
| --------------------------------- | --------------------------------------------------- |
| `env_manager="uv"` (recommended) | Credentials via `UV_INDEX_*` env vars or `.netrc`   |
| pip-based restore                 | Manual config via `pip.conf`, env vars, or `.netrc` |

---

## Graceful Degradation

| Scenario                      | Behavior                                  |
| ----------------------------- | ----------------------------------------- |
| UV not installed              | Falls back to pip inference, logs warning |
| UV version < 0.5.0            | Falls back to pip inference, logs warning |
| `uv export` fails             | Falls back to pip inference, logs warning |
| No UV project detected        | Uses standard pip inference (no warning)  |
| `MLFLOW_UV_AUTO_DETECT=false` | Skips UV detection entirely               |
| `MLFLOW_LOG_UV_FILES=false`   | Skips UV file logging, logs info          |

---

## Model Artifacts Structure

When a UV project is detected and logged:

```
model/
+-- MLmodel
+-- model.pkl
+-- requirements.txt    # Generated via uv export (pip-compatible)
+-- uv.lock             # Original lock file (for uv sync restore)
+-- pyproject.toml      # Project definition
+-- .python-version     # Python version (if exists)
+-- conda.yaml
```

---

## Modified Files

| File                               | Changes                                              |
| ---------------------------------- | ---------------------------------------------------- |
| `mlflow/utils/uv_utils.py`        | NEW: All UV detection, export, and sync functions    |
| `mlflow/utils/environment.py`     | `infer_pip_requirements()`: UV detection + export    |
| `mlflow/pyfunc/__init__.py`       | `save_model`/`log_model`: new UV params + threading  |
| `mlflow/pyfunc/model.py`          | `_save_model_with_class_artifacts_params`: UV params |
| `mlflow/environment_variables.py` | `MLFLOW_UV_AUTO_DETECT` boolean env var              |
| `tests/utils/test_uv_utils.py`    | Unit tests (85 tests)                                |
| `tests/pyfunc/test_uv_model_logging.py` | Integration tests (29 tests)                   |

---

## Test Coverage

**Total: 114 tests + 56-check integration validation**

### Unit Tests (`tests/utils/test_uv_utils.py`) - 85 tests

- UV version detection and availability
- Project detection (CWD-only)
- Dependency export with various flags
- PEP 508 marker evaluation (python_version, platform, os, compound markers)
- UV file copying and artifact management
- Environment variable parsing
- Private index extraction (regex-based)
- UV sync setup functions

### Integration Tests (`tests/pyfunc/test_uv_model_logging.py`) - 29 tests

- Real UV project creation and locking
- Real `uv export` with dependency groups and extras
- Real `uv sync` environment setup
- End-to-end model logging with UV
- Environment variable behavior

### Manual Integration Validation - 56 checks

All 56 checks passed with real UV projects (no mocks):

| Category                      | Tests | Status |
| ----------------------------- | ----- | ------ |
| UV availability               | 3     | PASS   |
| detect_uv_project             | 2     | PASS   |
| export_uv_requirements        | 9     | PASS   |
| get_python_version            | 2     | PASS   |
| copy_uv_project_files         | 3     | PASS   |
| infer_pip_requirements (UV)   | 6     | PASS   |
| MLFLOW_UV_AUTO_DETECT toggle  | 4     | PASS   |
| save_model with UV            | 10    | PASS   |
| save_model with uv_groups     | 4     | PASS   |
| save_model with uv_extras     | 3     | PASS   |
| log_model with UV + uv_groups | 3     | PASS   |
| setup_uv_sync_environment     | 4     | PASS   |
| save_model with auto-detect off | 3   | PASS   |

```bash
# Run all UV tests
uv run pytest tests/utils/test_uv_utils.py tests/pyfunc/test_uv_model_logging.py -v
```

---

## Example Usage

### Basic Usage (Just Works)

```python
# User has UV project with uv.lock + pyproject.toml
# No code changes needed!

import mlflow
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

with mlflow.start_run():
    # UV detected automatically, dependencies exported via uv export
    mlflow.sklearn.log_model(model, "model")
```

### Monorepo Support

```python
mlflow.sklearn.log_model(
    model, "model",
    uv_project_path="/path/to/monorepo/package"
)
```

### Selective Dependency Groups (API Parameters)

```python
# Include serving group + GPU extras
mlflow.pyfunc.log_model(
    python_model=model,
    name="model",
    uv_groups=["serving"],
    uv_extras=["gpu"],
)
```

### Selective Dependency Groups (Environment Variables)

```bash
# Include only serving dependencies
MLFLOW_UV_ONLY_GROUPS="serving" python train.py

# Include serving group + gpu extras
MLFLOW_UV_GROUPS="serving" MLFLOW_UV_EXTRAS="gpu" python train.py
```

### Disable UV Auto-Detection

```bash
# Via environment variable
MLFLOW_UV_AUTO_DETECT=false python train.py
```

### Disable UV File Logging

```bash
# Via environment variable (for large projects)
MLFLOW_LOG_UV_FILES=false python train.py
```

---

## Design Decisions

### Decision 1: CWD-only vs parent directory search

| Option                  | Chosen | Rationale                                |
| ----------------------- | ------ | ---------------------------------------- |
| CWD-only detection      | Yes    | Deterministic, predictable, no surprises |
| Parent directory search | No     | May find wrong uv.lock in monorepos      |

### Decision 2: `MLFLOW_UV_AUTO_DETECT` as standard `_BooleanEnvironmentVariable`

| Option                              | Chosen | Rationale                                     |
| ----------------------------------- | ------ | --------------------------------------------- |
| `_BooleanEnvironmentVariable`       | Yes    | Consistent with MLflow patterns, centralized  |
| Custom env var parsing in uv_utils  | No     | Scattered, not discoverable                   |

### Decision 3: `uv_groups`/`uv_extras` on API vs env vars only

| Option                 | Chosen | Rationale                                           |
| ---------------------- | ------ | --------------------------------------------------- |
| Both API + env vars    | Yes    | API params for explicit control, env vars for CI/CD |
| Env vars only          | No     | Less discoverable, not as composable                |
| API params only        | No     | Less convenient for automation                      |

API parameters take precedence over environment variables when both are specified.

### Decision 4: Environment variable vs API parameter for disabling UV file logging

| Option                         | Chosen | Rationale                               |
| ------------------------------ | ------ | --------------------------------------- |
| `MLFLOW_LOG_UV_FILES` env var  | Yes    | No code changes needed, CI/CD friendly  |
| `log_uv_files=False` parameter | No     | Requires API changes across all flavors |

### Decision 5: Auto-inject private index URLs

| Option                           | Chosen | Rationale                      |
| -------------------------------- | ------ | ------------------------------ |
| Extract as utility only          | Yes    | Safe, no risk of wrong URLs    |
| Auto-prepend to requirements.txt | No     | Error-prone, env-specific URLs |

### Decision 6: TOML parsing for uv.lock

| Option                 | Chosen | Rationale                                       |
| ---------------------- | ------ | ----------------------------------------------- |
| Regex-based extraction | Yes    | No additional dependency                        |
| tomllib/tomli          | No     | Adds dependency, overkill for simple extraction |

---

## References

1. **GitHub Issue:** [#12478 - Support `uv` Package Installer](https://github.com/mlflow/mlflow/issues/12478)
2. **Pull Request:** [#20344 - Add UV package manager support](https://github.com/mlflow/mlflow/pull/20344)
3. **UV Documentation:** [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
4. **UV Benchmarks:** [https://github.com/astral-sh/uv#benchmarks](https://github.com/astral-sh/uv#benchmarks)
5. **PEP 508:** [https://peps.python.org/pep-0508/](https://peps.python.org/pep-0508/)

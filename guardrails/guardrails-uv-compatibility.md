# Guardrails AI uv Package Manager Compatibility

|                  |                                                           |
| ---------------- | --------------------------------------------------------- |
| **Author(s)**    | Debu Sinha ([@debu-sinha](https://github.com/debu-sinha)) |
| **Organization** | Guardrails AI Community                                   |
| **Status**       | PROPOSED                                                  |
| **GitHub Issue** | [#1392](https://github.com/guardrails-ai/guardrails/issues/1392) |
| **Pull Request** | TBD                                                       |

**Change Log:**

- 2026-02-19: Initial design version

---

## Executive Summary

Guardrails AI's Hub validator install system is fundamentally incompatible with the uv package manager. The hub install pipeline hardcodes pip as a subprocess, discovers `site-packages` by importing pip as a module, and mutates a barrel file (`guardrails/hub/__init__.py`) directly inside `site-packages`. When users run `uv sync`, it removes the "foreign" validator packages (invisible to `uv.lock`) and leaves behind a stale `__init__.py` that crashes the entire guardrails import chain.

This document proposes a phased fix addressing three root causes:

**What's Proposed:**

- Replace `importlib.import_module('pip')` with `sysconfig.get_paths()['purelib']` for site-packages discovery
- Add `uv pip install` as preferred installer with pip fallback via `shutil.which('uv')`
- Replace the fragile `__init__.py` barrel file mutation with a JSON registry at `~/.guardrails/hub_registry.json` and a dynamic `__getattr__` loader in `guardrails/hub/__init__.py`
- Add `GUARDRAILS_INSTALLER` environment variable for explicit installer override
- Add `guardrails hub repair` CLI command to rebuild the registry from installed packages

---

# Part I: Design Overview

## Motivation

**The Problem (from GitHub Issue #1392):**

> "When using uv as the package manager, `guardrails hub install` validators are silently removed by `uv sync` and the entire guardrails package becomes unimportable."

The root cause is architectural: Guardrails Hub treats validators as "foreign" pip packages installed into the same virtual environment but invisible to uv's dependency resolver. When `uv sync` reconciles the environment with `uv.lock`, it removes every package not declared in the lock file -- including all Hub validators.

**Why this matters:**

- uv adoption is accelerating (100M+ downloads/month, 50K+ GitHub stars)
- Guardrails users who adopt uv face a brick-wall failure: `import guardrails` crashes
- The crash is unrecoverable via the CLI -- `guardrails hub install` itself crashes because it imports guardrails
- Projects using both MLflow (which now has native uv support via PR #20344) and Guardrails cannot use uv at all

**Current failure mode (catastrophic):**

```python
# After: pip install guardrails-ai && guardrails hub install hub://guardrails/detect_pii
# Then:  uv sync

>>> import guardrails
Traceback (most recent call last):
  File "guardrails/hub/__init__.py", line 1, in <module>
    from guardrails_grhub_detect_pii import DetectPII
ModuleNotFoundError: No module named 'guardrails_grhub_detect_pii'
```

The entire `guardrails` namespace becomes unimportable. Even `guardrails --version` crashes.

---

## Architecture Overview

### Current Architecture (What Breaks)

The hub install pipeline has 6 steps, 3 of which are fragile:

![Current Hub Install Architecture](images/guardrails-hub-install-architecture.svg)

**Source files:** `guardrails/hub/install.py`, `guardrails/hub/validator_package_service.py`, `guardrails/cli/hub/utils.py`

#### Call Chain (Verified Against Source)

```
install(package_uri)                          # hub/install.py:37
  -> get_validator_id(package_uri)            # validator_package_service.py:154
       hub://guardrails/detect_pii -> ('guardrails/detect_pii', version)
  -> get_manifest_and_site_packages(validator_id)  # :51
       -> get_validator_manifest(validator_id)       # hub_client.py
            GET hub.api.guardrailsai.com/.../manifest
       -> get_site_packages_location()               # :57 [FRAGILE]
            Path(get_module_path("pip")).parent
            importlib.import_module('pip')            # FAILS without pip
  -> install_hub_module(validator_id)         # :243 [FRAGILE]
       pip_process_with_custom_exception(
         "install", "guardrails-grhub-detect-pii[validators]",
         ["--index-url=https://__token__:{jwt}@pypi.guardrailsai.com/simple",
          "--extra-index-url=https://pypi.org/simple"]
       )
       # Runs: sys.executable -m pip install ...
  -> run_post_install(manifest, site_packages)  # :179
       subprocess: python post_install.py
  -> add_to_hub_inits(manifest, site_packages)  # :107 [FRAGILE]
       Opens site-packages/guardrails/hub/__init__.py
       Appends: "from guardrails_grhub_detect_pii import DetectPII"
  -> get_validator_from_manifest(manifest)    # :84
       reload_module(import_path) -> Validator ready
```

### Why uv sync Breaks Everything

![Why uv sync Breaks Guardrails](images/guardrails-uv-sync-breakage.svg)

The fundamental problem is a state desynchronization:

1. **Before `uv sync`:** `site-packages` contains `guardrails-ai`, `guardrails_grhub_detect_pii`, and a mutated `hub/__init__.py` that imports from the validator package
2. **`uv sync` runs:** Reconciles environment with `uv.lock`. The lock file has no knowledge of hub validators (they were installed via pip subprocess, not declared in `pyproject.toml`). uv removes the validator package as "foreign"
3. **After `uv sync`:** The validator package is gone, but `hub/__init__.py` still has `from guardrails_grhub_detect_pii import DetectPII`. This line executes at import time and crashes with `ModuleNotFoundError`
4. **Cascading failure:** Since `guardrails.hub` is imported by `guardrails.__init__`, the entire `guardrails` namespace fails to import. The CLI itself crashes, so users cannot even run `guardrails hub install` to fix it

### Five Root Causes

| # | Root Cause | File | Line | Impact |
|---|------------|------|------|--------|
| 1 | `get_site_packages_location()` imports pip as a module | `validator_package_service.py` | 57-61 | Fails if pip not installed (uv-only envs) |
| 2 | `install_hub_module()` hardcodes `sys.executable -m pip` | `cli/hub/utils.py` | 63 | Cannot use uv for package installation |
| 3 | `add_to_hub_inits()` mutates `__init__.py` in site-packages | `validator_package_service.py` | 107-130 | Stale imports crash entire namespace |
| 4 | Validators are "foreign" packages invisible to `uv.lock` | (architectural) | -- | `uv sync` removes them |
| 5 | No recovery mechanism when `__init__.py` becomes stale | (missing) | -- | CLI itself crashes, cannot self-repair |

### Proposed Architecture

![Proposed uv-Compatible Hub Install Architecture](images/guardrails-proposed-architecture.svg)

---

## Phase Implementation Summary

### Phase 1 (Quick Fixes) - Backwards Compatible

| Feature | Status | Description |
| --- | --- | --- |
| Site-packages via sysconfig | Proposed | `sysconfig.get_paths()['purelib']` replaces pip import |
| uv pip install support | Proposed | `detect_installer()` prefers uv, falls back to pip |
| GUARDRAILS_INSTALLER env var | Proposed | Force `uv` or `pip` explicitly |
| Graceful fallback | Proposed | If uv unavailable, use pip (no crash) |

### Phase 2 (Registry Refactor) - Breaking Change

| Feature | Status | Description |
| --- | --- | --- |
| JSON validator registry | Proposed | `~/.guardrails/hub_registry.json` replaces `__init__.py` mutation |
| Dynamic `__getattr__` | Proposed | `guardrails/hub/__init__.py` loads from registry at import time |
| Safe import failure | Proposed | Missing validators log warning instead of crashing |
| `guardrails hub repair` | Proposed | Rebuild registry from `importlib.metadata.distributions()` |

### Phase 3 (First-Class uv Integration) - Future

| Feature | Status | Description |
| --- | --- | --- |
| Validator pyproject.toml entries | Future | Write validators as optional deps in project |
| `uv add --extra` integration | Future | Track validators in `uv.lock` natively |
| Workspace-aware installs | Future | Support uv workspaces for multi-package projects |

### NOT Proposed (Design Decisions)

| Feature | Reason |
| --- | --- |
| Auto-add validators to pyproject.toml | Too invasive, modifies user's project config |
| Custom uv source for hub PyPI | uv sources don't support per-package auth tokens |
| Vendored pip fallback | Adds maintenance burden, sysconfig is sufficient |

---

# Part II: Detailed Implementation

## Phase 1: Package Manager Abstraction

### 1.1 Fix `get_site_packages_location()`

**Current code** (`validator_package_service.py:57-61`):
```python
@staticmethod
def get_site_packages_location():
    pip_package_location = Path(ValidatorPackageService.get_module_path("pip"))
    site_packages_path = str(pip_package_location.parent)
    return site_packages_path
```

**Proposed replacement:**
```python
import sysconfig

@staticmethod
def get_site_packages_location():
    return sysconfig.get_paths()["purelib"]
```

**Why `sysconfig`:**
- Part of Python's standard library since 3.2
- Does not require pip to be installed
- Returns the canonical `site-packages` path for the running interpreter
- Works identically in pip-managed and uv-managed environments
- MLflow uses the same approach (validated in PR #20344)

### 1.2 Add Installer Detection

**New function** in `validator_package_service.py`:

```python
import shutil
import os

_GUARDRAILS_INSTALLER_ENV = "GUARDRAILS_INSTALLER"

@staticmethod
def detect_installer() -> str:
    """Detect preferred package installer.

    Precedence:
    1. GUARDRAILS_INSTALLER env var (explicit override)
    2. uv (if available via shutil.which)
    3. pip (fallback)
    """
    env_installer = os.environ.get(_GUARDRAILS_INSTALLER_ENV, "").strip().lower()
    if env_installer in ("uv", "pip"):
        return env_installer
    if shutil.which("uv") is not None:
        return "uv"
    return "pip"
```

### 1.3 Replace Hardcoded pip Subprocess

**Current code** (`cli/hub/utils.py:60-65`):
```python
def pip_process_with_custom_exception(action, package, flags, ...):
    command = [sys.executable, "-m", "pip", action]
    command.extend(flags)
    if package:
        command.append(package)
    result = subprocess.run(command, ...)
```

**Proposed change** - new `installer_process()` function:
```python
def installer_process(action, package, flags, installer="pip", ...):
    if installer == "uv":
        command = ["uv", "pip", action]
    else:
        command = [sys.executable, "-m", "pip", action]

    command.extend(flags)
    if package:
        command.append(package)
    result = subprocess.run(command, ...)
```

**Key difference for uv:** `uv pip install` uses `--extra-index-url` (same as pip), and supports the same `--index-url` flag with token auth. The private PyPI URL format `https://__token__:{jwt}@pypi.guardrailsai.com/simple` works with both pip and uv.

### 1.4 Update `install_hub_module()`

**Current code** (`validator_package_service.py:243-274`):
```python
@staticmethod
def install_hub_module(validator_id, ...):
    pip_flags = [
        f"--index-url=https://__token__:{guardrails_token}@pypi.guardrailsai.com/simple",
        "--extra-index-url=https://pypi.org/simple",
    ]
    download_output = pip_process_with_custom_exception(
        "install", full_package_name, pip_flags, quiet=quiet
    )
```

**Proposed change:**
```python
@staticmethod
def install_hub_module(validator_id, ...):
    installer = ValidatorPackageService.detect_installer()

    install_flags = [
        f"--index-url=https://__token__:{guardrails_token}@pypi.guardrailsai.com/simple",
        "--extra-index-url=https://pypi.org/simple",
    ]
    download_output = installer_process(
        "install", full_package_name, install_flags,
        installer=installer, quiet=quiet
    )
```

---

## Phase 2: Registry Refactor

### 2.1 JSON Validator Registry

Replace the fragile `__init__.py` barrel file mutation with a JSON registry stored outside `site-packages`.

**Registry location:** `~/.guardrails/hub_registry.json`

**Registry format:**
```json
{
  "version": 1,
  "validators": {
    "guardrails/detect_pii": {
      "import_path": "guardrails_grhub_detect_pii",
      "exports": ["DetectPII"],
      "installed_at": "2026-02-19T18:00:00Z",
      "package_name": "guardrails-grhub-detect-pii"
    },
    "guardrails/toxic_language": {
      "import_path": "guardrails_grhub_toxic_language",
      "exports": ["ToxicLanguage"],
      "installed_at": "2026-02-19T18:05:00Z",
      "package_name": "guardrails-grhub-toxic-language"
    }
  }
}
```

**Why `~/.guardrails/` instead of `site-packages`:**

- Survives `uv sync`, `pip install --force-reinstall`, and virtualenv recreation
- User-scoped (follows XDG conventions on Linux, `~/` on macOS/Windows)
- Already used by Guardrails for `.guardrailsrc` config file
- Can be committed to version control (optional) for team consistency

### 2.2 Replace `add_to_hub_inits()` with `register_validator()`

**Current code** (`validator_package_service.py:107-130`):
```python
@staticmethod
def add_to_hub_inits(manifest, site_packages):
    import_line = f"from {import_path} import {', '.join(sorted_exports)}"
    hub_init_location = os.path.join(
        site_packages, "guardrails", "hub", "__init__.py"
    )
    with open(hub_init_location, "a+") as hub_init:
        # ... appends import line to __init__.py
```

**Proposed replacement:**
```python
import json
from pathlib import Path
from datetime import datetime, timezone

_REGISTRY_DIR = Path.home() / ".guardrails"
_REGISTRY_FILE = _REGISTRY_DIR / "hub_registry.json"

@staticmethod
def register_validator(manifest):
    """Register a validator in the JSON registry."""
    _REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    registry = {"version": 1, "validators": {}}
    if _REGISTRY_FILE.exists():
        registry = json.loads(_REGISTRY_FILE.read_text())

    validator_id = manifest.id
    import_path = ValidatorPackageService.get_import_path_from_validator_id(
        validator_id
    )
    package_name = ValidatorPackageService.get_normalized_package_name(
        validator_id
    )

    registry["validators"][validator_id] = {
        "import_path": import_path,
        "exports": manifest.exports or [],
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "package_name": package_name,
    }

    _REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
```

### 2.3 Dynamic `__getattr__` in `guardrails/hub/__init__.py`

Replace the static barrel file with a dynamic loader:

```python
"""Guardrails Hub - Dynamic Validator Loader.

Validators are loaded on-demand from the hub registry.
Import usage: from guardrails.hub import DetectPII
"""

import importlib
import json
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)
_REGISTRY_FILE = Path.home() / ".guardrails" / "hub_registry.json"

def _load_registry():
    """Load the validator registry, returning empty dict on any failure."""
    if not _REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(_REGISTRY_FILE.read_text())
        return data.get("validators", {})
    except (json.JSONDecodeError, OSError):
        _logger.warning("Failed to load hub registry at %s", _REGISTRY_FILE)
        return {}

def _build_export_map():
    """Build mapping from export name to (import_path, export_name)."""
    export_map = {}
    for validator_id, info in _load_registry().items():
        import_path = info.get("import_path", "")
        for export_name in info.get("exports", []):
            export_map[export_name] = (import_path, export_name)
    return export_map

_EXPORT_MAP = None

def __getattr__(name):
    global _EXPORT_MAP
    if _EXPORT_MAP is None:
        _EXPORT_MAP = _build_export_map()

    if name in _EXPORT_MAP:
        import_path, export_name = _EXPORT_MAP[name]
        try:
            module = importlib.import_module(import_path)
            attr = getattr(module, export_name)
            # Cache in module namespace for subsequent access
            globals()[name] = attr
            return attr
        except (ModuleNotFoundError, AttributeError):
            _logger.warning(
                "Validator '%s' is registered but not installed. "
                "Run: guardrails hub install hub://%s",
                name,
                next(
                    (vid for vid, info in _load_registry().items()
                     if name in info.get("exports", [])),
                    "unknown/validator",
                ),
            )
            raise

    raise AttributeError(f"module 'guardrails.hub' has no attribute {name!r}")
```

**Key properties:**
- Missing validators produce a warning + `AttributeError` instead of crashing the entire namespace
- Registry is loaded lazily (only on first `from guardrails.hub import X`)
- Successful lookups are cached in module globals (same performance as static imports)
- No `site-packages` mutation -- the `__init__.py` file is part of the guardrails source tree

### 2.4 `guardrails hub repair` Command

Recovery command to rebuild the registry from actually-installed packages:

```python
import importlib.metadata

def repair_registry():
    """Scan installed packages and rebuild hub_registry.json."""
    registry = {"version": 1, "validators": {}}

    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name and name.startswith("guardrails-grhub-"):
            # Reverse the naming convention:
            # guardrails-grhub-detect-pii -> guardrails/detect_pii
            suffix = name.replace("guardrails-grhub-", "")
            validator_id = f"guardrails/{suffix.replace('-', '_')}"
            import_path = name.replace("-", "_")

            # Try to get exports from the module
            exports = []
            try:
                module = importlib.import_module(import_path)
                exports = getattr(module, "__validator_exports__", [])
                if not exports:
                    exports = [
                        attr for attr in dir(module)
                        if not attr.startswith("_")
                        and hasattr(getattr(module, attr), "rail_alias")
                    ]
            except ImportError:
                pass

            if exports:
                registry["validators"][validator_id] = {
                    "import_path": import_path,
                    "exports": exports,
                    "package_name": name,
                    "installed_at": datetime.now(timezone.utc).isoformat(),
                }

    _REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
    return registry
```

---

## Environment Variables

| Variable | Default | Phase | Description |
| --- | --- | --- | --- |
| `GUARDRAILS_INSTALLER` | (auto-detect) | 1 | Force `uv` or `pip` as the installer |
| `GUARDRAILS_HUB_REGISTRY` | `~/.guardrails/hub_registry.json` | 2 | Override registry file location |

**Usage Examples:**

```bash
# Force pip even when uv is available
GUARDRAILS_INSTALLER=pip guardrails hub install hub://guardrails/detect_pii

# Force uv explicitly
GUARDRAILS_INSTALLER=uv guardrails hub install hub://guardrails/detect_pii

# Custom registry location (CI/CD, shared envs)
GUARDRAILS_HUB_REGISTRY=/shared/hub_registry.json guardrails hub install hub://guardrails/detect_pii
```

---

## Graceful Degradation

| Scenario | Current Behavior | Proposed Behavior |
| --- | --- | --- |
| uv not installed | N/A (hardcoded pip) | Auto-detect pip, install normally |
| pip not installed (uv-only env) | Crash: `ModuleNotFoundError: pip` | Use `sysconfig` for site-packages, `uv pip install` for packages |
| `uv sync` removes validators | Crash: entire guardrails unimportable | Warning per missing validator, rest of guardrails works |
| Corrupted registry JSON | N/A | Fallback to empty registry, log warning |
| Stale registry entries | N/A | `ModuleNotFoundError` caught, warning logged, `hub repair` suggested |

---

## Migration Plan

### Phase 1 (Non-Breaking)
- Drop-in replacement of `get_site_packages_location()` and `install_hub_module()`
- No user-visible changes (same CLI, same API)
- Existing `__init__.py` barrel file still works (Phase 1 keeps it)

### Phase 2 (Breaking)
1. Ship new `guardrails/hub/__init__.py` with dynamic `__getattr__`
2. On first run after upgrade, `guardrails hub install` writes to both old `__init__.py` AND new registry (dual-write transition period)
3. After transition period (1 minor version), remove `add_to_hub_inits()` entirely
4. Document migration: `guardrails hub repair` rebuilds registry from installed packages

### Rollback
- Phase 1: Revert `get_site_packages_location()` and `install_hub_module()` -- pure code changes, no state
- Phase 2: Remove `hub_registry.json`, restore old `guardrails/hub/__init__.py` from `site-packages` backup

---

## Test Coverage

### Phase 1 Tests

- `test_detect_installer_uv_available`: Verify uv detection via `shutil.which`
- `test_detect_installer_pip_fallback`: Verify pip fallback when uv absent
- `test_detect_installer_env_override`: Verify `GUARDRAILS_INSTALLER` precedence
- `test_get_site_packages_sysconfig`: Verify `sysconfig.get_paths()['purelib']` matches expected path
- `test_install_hub_module_with_uv`: Verify `uv pip install` subprocess command
- `test_install_hub_module_with_pip`: Verify pip subprocess command (backwards compat)
- `test_install_hub_module_uv_fallback_to_pip`: Verify graceful degradation

### Phase 2 Tests

- `test_register_validator_creates_registry`: Verify JSON file creation
- `test_register_validator_appends`: Verify second validator appended
- `test_register_validator_idempotent`: Verify duplicate registration is safe
- `test_hub_getattr_loads_validator`: Verify dynamic import via `__getattr__`
- `test_hub_getattr_missing_validator_warns`: Verify warning instead of crash
- `test_hub_getattr_caches_result`: Verify subsequent access uses globals cache
- `test_hub_repair_rebuilds_from_installed`: Verify registry rebuild from metadata
- `test_hub_init_survives_uv_sync`: End-to-end test simulating uv sync removing packages

```bash
# Run Phase 1 tests
pytest tests/unit/test_installer_detection.py -v

# Run Phase 2 tests
pytest tests/unit/test_hub_registry.py -v

# End-to-end (requires uv installed)
pytest tests/integration/test_uv_compatibility.py -v
```

---

## Example Usage

### Before (Broken with uv)

```bash
# 1. Set up project
uv init my-guardrails-project
cd my-guardrails-project
uv add guardrails-ai

# 2. Install a hub validator
guardrails hub install hub://guardrails/detect_pii   # Works: pip subprocess

# 3. Use it
python -c "from guardrails.hub import DetectPII"     # Works

# 4. Add another dependency
uv add pandas                                         # Runs uv sync

# 5. CRASH
python -c "import guardrails"
# ModuleNotFoundError: No module named 'guardrails_grhub_detect_pii'
```

### After (Fixed with Proposed Changes)

```bash
# 1. Set up project
uv init my-guardrails-project
cd my-guardrails-project
uv add guardrails-ai

# 2. Install a hub validator (auto-detects uv)
guardrails hub install hub://guardrails/detect_pii   # Uses: uv pip install

# 3. Use it
python -c "from guardrails.hub import DetectPII"     # Works

# 4. Add another dependency
uv add pandas                                         # Runs uv sync

# 5. Validator package removed by uv sync, but...
python -c "import guardrails"                          # Works! (no crash)
python -c "from guardrails.hub import DetectPII"
# WARNING: Validator 'DetectPII' is registered but not installed.
# Run: guardrails hub install hub://guardrails/detect_pii

# 6. Re-install the validator
guardrails hub install hub://guardrails/detect_pii   # Quick re-install

# 7. Everything works again
python -c "from guardrails.hub import DetectPII; print(DetectPII)"
```

---

## Design Decisions

### Decision 1: `sysconfig` vs `importlib.import_module('pip')` for site-packages

| Option | Chosen | Rationale |
| --- | --- | --- |
| `sysconfig.get_paths()['purelib']` | Yes | Standard library, no pip dependency, works in all envs |
| `importlib.import_module('pip')` | No | Breaks in pip-less environments (uv-only) |
| `site.getsitepackages()` | No | Returns list, behavior varies by platform |

### Decision 2: uv detection via `shutil.which` vs version check

| Option | Chosen | Rationale |
| --- | --- | --- |
| `shutil.which('uv')` | Yes | Fast, no subprocess, sufficient for presence detection |
| `uv --version` subprocess | No | Slower, unnecessary (uv pip install works across versions) |
| `importlib.metadata` | No | uv is a standalone binary, not a Python package |

### Decision 3: JSON registry vs entry_points vs importlib.metadata

| Option | Chosen | Rationale |
| --- | --- | --- |
| JSON file at `~/.guardrails/` | Yes | Survives uv sync, simple, human-readable, no build-time dep |
| `importlib.metadata.entry_points` | No | Requires package metadata (removed by uv sync) |
| SQLite database | No | Overkill, harder to debug/repair manually |
| TOML file | No | Requires tomllib (3.11+) or tomli dependency |

### Decision 4: Warning vs crash on missing validator

| Option | Chosen | Rationale |
| --- | --- | --- |
| Warning + `AttributeError` on access | Yes | Guardrails core still works, only missing validator fails |
| Silent skip (return None) | No | Hides bugs, confusing behavior |
| Crash entire import (current) | No | Catastrophic, unrecoverable via CLI |

### Decision 5: Dual-write transition vs hard cut

| Option | Chosen | Rationale |
| --- | --- | --- |
| Dual-write for 1 minor version | Yes | Backwards compat, users on old guardrails still work |
| Hard cut to registry-only | No | Breaks users who don't upgrade `guardrails/hub/__init__.py` |

---

## Comparison with MLflow uv Support

Having implemented MLflow's uv support (PR #20344), several lessons directly apply:

| Aspect | MLflow Approach | Guardrails Approach | Rationale |
| --- | --- | --- | --- |
| Package detection | `shutil.which('uv')` + version check | `shutil.which('uv')` (no version check) | uv pip interface is stable across versions |
| Site-packages | N/A (uses uv export) | `sysconfig.get_paths()['purelib']` | Same stdlib approach, no pip dependency |
| Env var override | `MLFLOW_LOG_UV_FILES` | `GUARDRAILS_INSTALLER` | Consistent pattern: env var for CI/CD control |
| Fallback strategy | Warning + pip fallback | Warning + pip fallback | Never crash on missing tool |
| Registry | N/A (uses requirements.txt) | JSON file at `~/.guardrails/` | Guardrails-specific: validator state outside site-packages |

---

## References

1. **GitHub Issue:** [#1392 - Support uv package manager](https://github.com/guardrails-ai/guardrails/issues/1392)
2. **MLflow uv Support PR:** [#20344 - Add UV package manager support](https://github.com/mlflow/mlflow/pull/20344)
3. **MLflow uv Design Doc:** [uv-support-design-doc-v2.md](../mlflow/uv-support-design-doc-v2.md)
4. **uv Documentation:** [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
5. **uv pip interface:** [https://docs.astral.sh/uv/pip/](https://docs.astral.sh/uv/pip/)
6. **Python sysconfig:** [https://docs.python.org/3/library/sysconfig.html](https://docs.python.org/3/library/sysconfig.html)
7. **PEP 503 (Package naming):** [https://peps.python.org/pep-0503/](https://peps.python.org/pep-0503/)
8. **Guardrails Hub Architecture:** `guardrails/hub/install.py`, `guardrails/hub/validator_package_service.py`, `guardrails/cli/hub/utils.py`

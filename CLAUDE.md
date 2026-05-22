# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable/dev mode, sklearn backend only):**
```bash
uv sync
```

**Install with PyTorch backend support:**
```bash
uv sync --extra torch
```

**Install all extras (torch + test/dev tools):**
```bash
uv sync --extra torch --extra test
```

**Run all tests:**
```bash
uv run pytest -vs tests
# or
./run_tests.sh
```

**Run a single test file or test:**
```bash
uv run pytest -vs tests/test_prepare.py
uv run pytest -vs tests/test_prepare.py::test_factorize_torch
```

**Download reference data required by reproducibility tests** (must be run before `test_reproducibility.py`):
```bash
uv run python download_pytest_data.py
```

## Architecture

The package lives in `src/cnmf/` and exposes two public classes: `cNMF` and `Preprocess`.

### `cNMF` class — [src/cnmf/cnmf.py](src/cnmf/cnmf.py)

A 5-step pipeline for inferring gene expression programs (GEPs) from scRNA-seq count matrices:

1. **`prepare()`** — Loads counts (`.h5ad`, `.mtx`/`.mtx.gz`, `.npz`, or tab-delimited `.txt`), selects high-variance genes via Fano factor, variance-normalizes, and writes intermediate files. The `use_torch` flag is written into the YAML run-parameters file here, so all downstream steps pick it up automatically.
2. **`factorize()`** — Runs NMF `n_iter` times for each K. Parallelizable via `worker_index` / `total_workers`. Each run saves a spectra `.npz` in `cnmf_tmp/`.
3. **`combine()`** — Merges per-iteration spectra `.npz` files into one merged file per K.
4. **`k_selection_plot()`** — Plots stability (silhouette) vs. reconstruction error across K values.
5. **`consensus()`** — Clusters merged solutions, re-fits usages via batched OLS (`efficient_ols_all_cols`), writes final outputs. Also calls `_nmf()` for the usage refit step (the `refit_usage` path), so the torch backend is exercised here too.
6. **`load_results()`** — Returns `usage`, `spectra_scores`, `spectra_tpm`, `top_genes`.

**File layout during a run:**
- `output_dir/<name>/cnmf_tmp/` — intermediate `.h5ad`, `.npz`, `.yaml` files
- `output_dir/<name>/` — final `.txt` result files and diagnostic plots

All paths are tracked in `self.paths` (dict populated by `_initialize_dirs()`).

### NMF backends — [src/cnmf/cnmf.py](src/cnmf/cnmf.py)

The dispatch point is `_nmf(X, nmf_kwargs)`. It pops `use_torch` from the kwargs dict and routes to one of two backends:

**sklearn backend (default, `use_torch=False`):**
- Calls `sklearn.decomposition.non_negative_factorization` directly.
- Uses `solver='cd'` (coordinate descent) for Frobenius loss, `solver='mu'` (multiplicative update) for other losses.
- Operates in float64.

**PyTorch backend (`use_torch=True`) — `_nmf_torch()`:**
- Requires `uv sync --extra torch` (or `pip install torchnmf`).
- Automatically uses CUDA when available, falls back to CPU.
- Operates in **float32** (not float64 like sklearn) — outputs are numerically comparable but not bit-for-bit identical between backends.
- Convention translation: torchnmf uses `V ≈ H @ W^T` (W is genes×K, H is cells×K); the method transposes W to match cNMF's sklearn convention (spectra K×genes, usages cells×K).
- The usage-refit path in `consensus()` passes `H=fixed_spectra, update_H=False` into `_nmf()`; `_nmf_torch` detects this (`refit_mode`) and initializes torchnmf with `W=fixed_spectra.T, trainable_W=False`.
- Known limitations vs sklearn: `init='nndsvd'` is unsupported (falls back to random with a warning); a single `alpha` is applied to both factors (warns if `alpha_usage != alpha_spectra`); only `solver='mu'` is supported (warns if `solver='cd'` is requested).
- **Performance note**: Only beneficial with a CUDA GPU. CPU torch is 1.2–7× slower than sklearn's BLAS-backed float64.

### `Preprocess` class — [src/cnmf/preprocess.py](src/cnmf/preprocess.py)

Used upstream of `cNMF` for batch correction. Key method: `preprocess_for_cnmf(adata, harmony_vars=..., ...)` performs gene-level (not PC-space) Harmony correction and returns corrected count/TPM AnnData objects plus HVGs. Optional dependency: `harmonypy` + `scikit-misc`.

### Utility functions — [src/cnmf/cnmf.py](src/cnmf/cnmf.py)

- `get_highvar_genes_sparse` / `get_highvar_genes` — Fano factor-based HVG selection
- `efficient_ols_all_cols` — batched OLS that avoids densifying sparse matrices; used in the consensus usage-refit step
- `save_df_to_npz` / `load_df_from_npz` — custom numpy-based DataFrame serialization used throughout

### Tests

- **[tests/test_prepare.py](tests/test_prepare.py)** — Unit tests for `prepare()` (parametrized across formats, dtypes, densify) plus a full suite of PyTorch backend tests covering: flag persistence in YAML, factorize output files, warning paths (nndsvd, alpha mismatch, solver), ImportError when torchnmf absent, KL-divergence loss, and the consensus refit-usage code path.
- **[tests/test_reproducibility.py](tests/test_reproducibility.py)** — End-to-end pipeline tests against reference data (downloaded via `download_pytest_data.py`), plus a torch-backend reproducibility test verifying that identical seeds produce bit-for-bit identical spectra.

### CLI

The `cnmf` entry point (`cnmf:main`) exposes the 5 pipeline steps as subcommands: `prepare`, `factorize`, `combine`, `k_selection_plot`, `consensus`. The `--use-torch` flag maps to `use_torch=True` in `prepare()`.

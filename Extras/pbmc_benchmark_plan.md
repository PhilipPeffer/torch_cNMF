# Plan: PBMC3k Backend Benchmark and Reproducibility Test

## Context

The PyTorch backend has been added but benchmarked only on synthetic data. The user wants to:
1. Benchmark sklearn vs torch backends on real PBMC3k data and add results to the README
2. Run the existing reproducibility test (`test_reproducibility.py`) against reference data to verify the pipeline is correct

The existing `Extras/prepare_unittest_pbmc.ipynb` notebook contains download + prep logic but runs the full NMF pipeline (slow). We only need `counts.h5ad` for the benchmark. The `Extras/benchmark_backends.py` and `download_pytest_data.py` scripts already exist.

## Steps

### 1. Update `.gitignore` to exclude data files

Add entries to exclude downloaded/generated data that should not be tracked:
```
# Large data files (downloaded/generated, not tracked)
example_PBMC/
tests/test_data/
*.tar.gz
*.h5ad
```

### 2. Create `Extras/prepare_pbmc_counts.py`

A new tracked script that contains only the download + filter + save portion of the notebook (not the full NMF pipeline). The notebook runs factorize/combine/consensus which is slow and unnecessary for the benchmark — we only need `counts.h5ad`.

The script should:
- Download `pbmc3k_filtered_gene_bc_matrices.tar.gz` from `http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz`
- Extract, load with scanpy
- Filter: cells with ≥200 genes, genes in ≥3 cells
- Save to `{output_dir}/counts.h5ad`
- Accept `--output-dir` argument (default `./example_PBMC`)
- Print cell/gene counts before and after filtering

### 3. Run the data preparation script

```bash
PYTHONPATH=src python Extras/prepare_pbmc_counts.py --output-dir ./example_PBMC
```

### 4. Run the benchmark

```bash
PYTHONPATH=src python Extras/benchmark_backends.py \
  --counts-fn ./example_PBMC/counts.h5ad \
  --k-values 5 7 10 15 \
  --n-timing-runs 3 \
  --seed 14 \
  --output-csv ./benchmark_results.csv
```

Capture the full console output (the formatted table) for use in the README.

### 5. Download reference data and run reproducibility tests

```bash
PYTHONPATH=src python download_pytest_data.py
PYTHONPATH=src python -m pytest tests/test_reproducibility.py -v
```

Note: `test_reproducibility.py` skips factorization by copying pre-computed merged spectra from the reference dir, so it runs in minutes not hours.

### 6. Update `README.md`

Replace the current vague benchmark claim in the PyTorch GPU backend Notes:
> "benchmarks show 1.2–7x slower depending on k"

With a real results table. Add a `### CPU benchmark (PBMC3k, 2638 cells × 1000 HVGs)` subsection inside the existing `## PyTorch GPU backend` section, showing the actual timing table from step 4 (k, sklearn time, torch CPU time, speedup, recon error).

### 7. Commit tracked files only

Files to commit:
- `Extras/prepare_pbmc_counts.py` (new script)
- `.gitignore` (updated)
- `README.md` (updated with benchmark table)

Files NOT to commit (covered by .gitignore):
- `example_PBMC/` directory and contents
- `tests/test_data/` directory and contents
- `benchmark_results.csv`

## Verification

- `Extras/prepare_pbmc_counts.py` runs without error and produces `example_PBMC/counts.h5ad`
- Benchmark script completes and prints a table with timing for all k values
- `test_reproducibility.py` passes for both `example_cNMF` and `pbmc_cNMF` datasets
- `README.md` contains a real benchmark table with PBMC3k numbers

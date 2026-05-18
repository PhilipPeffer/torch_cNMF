#!/usr/bin/env python3
"""
Benchmark the sklearn vs PyTorch NMF backends in cNMF.

Measures wall-clock time and reconstruction error for each backend across
one or more values of K (number of components).

Usage (from the repo root, in an environment with cnmf and torchnmf installed):

    python Extras/benchmark_backends.py [options]

Install torchnmf first if needed:
    pip install torchnmf
"""

import argparse
import os
import sys
import time
import shutil
import tempfile
import warnings

import numpy as np
import scipy.sparse as sp
import yaml

from cnmf import cNMF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_info():
    lines = []
    try:
        import sklearn
        lines.append(f"  sklearn  : {sklearn.__version__}")
    except ImportError:
        lines.append("  sklearn  : NOT INSTALLED")

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            dev_name = torch.cuda.get_device_name(0)
            lines.append(f"  torch    : {torch.__version__}  (device: {dev_name})")
        else:
            lines.append(f"  torch    : {torch.__version__}  (device: cpu — no CUDA)")
    except ImportError:
        lines.append("  torch    : NOT INSTALLED")

    try:
        import torchnmf
        lines.append(f"  torchnmf : {torchnmf.__version__}")
    except ImportError:
        lines.append("  torchnmf : NOT INSTALLED  (torch backend will be skipped)")

    return "\n".join(lines)


def _recon_error(X, spectra, usages):
    """Frobenius reconstruction error ||X - usages @ spectra||_F."""
    if sp.issparse(X):
        X_dense = X.toarray().astype(np.float64)
    else:
        X_dense = np.array(X, dtype=np.float64)
    approx = usages.astype(np.float64) @ spectra.astype(np.float64)
    diff = X_dense - approx
    return float(np.sqrt((diff ** 2).sum()))


def _build_nmf_kwargs(base_kwargs, k, seed, use_torch):
    """Clone base_kwargs, set n_components / random_state / use_torch."""
    kw = dict(base_kwargs)
    kw['n_components'] = k
    kw['random_state'] = seed
    kw['use_torch'] = use_torch
    return kw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark sklearn vs PyTorch cNMF backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--counts-fn", default=None,
                   help="Path to an existing counts file (.h5ad, .txt, .npz, .mtx). "
                        "If omitted, synthetic data is generated.")
    p.add_argument("--n-cells", type=int, default=2000,
                   help="Number of cells for synthetic data.")
    p.add_argument("--n-genes", type=int, default=5000,
                   help="Number of genes for synthetic data.")
    p.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 15, 20],
                   help="K (number of components) values to benchmark.")
    p.add_argument("--n-timing-runs", type=int, default=5,
                   help="NMF calls per (backend, K) combination for averaging.")
    p.add_argument("--seed", type=int, default=14,
                   help="Random seed.")
    p.add_argument("--beta-loss", default="frobenius",
                   choices=["frobenius", "kullback-leibler"],
                   help="NMF beta-loss function.")
    p.add_argument("--output-dir", default=None,
                   help="Working directory for cNMF intermediate files. "
                        "Defaults to a temporary directory that is deleted afterwards.")
    p.add_argument("--output-csv", default=None,
                   help="Optional path to write results as a CSV file.")
    return p.parse_args()


def main():
    args = parse_args()

    torch_available = True
    try:
        import torchnmf  # noqa: F401
        import torch     # noqa: F401
    except ImportError:
        torch_available = False

    print("=" * 66)
    print("cNMF backend benchmark")
    print("=" * 66)
    print(_env_info())
    print()

    # ------------------------------------------------------------------
    # Working directory
    # ------------------------------------------------------------------
    cleanup_tmpdir = False
    if args.output_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="cnmf_bench_")
        cleanup_tmpdir = True
    else:
        tmpdir = args.output_dir
        os.makedirs(tmpdir, exist_ok=True)

    try:
        # ------------------------------------------------------------------
        # Counts data
        # ------------------------------------------------------------------
        if args.counts_fn is not None:
            counts_fn = args.counts_fn
            print(f"Using counts file: {counts_fn}")
        else:
            import scanpy as sc
            print(f"Generating synthetic data: {args.n_cells} cells × {args.n_genes} genes")
            np.random.seed(args.seed)
            data = np.random.binomial(
                n=100, p=0.01, size=(args.n_cells, args.n_genes)
            ).astype(np.int64)
            import anndata
            adata = anndata.AnnData(X=sp.csr_matrix(data))
            counts_fn = os.path.join(tmpdir, "synthetic_counts.h5ad")
            adata.write_h5ad(counts_fn)

        # ------------------------------------------------------------------
        # Prepare (shared between both backends)
        # ------------------------------------------------------------------
        print("Running prepare()…", flush=True)
        cnmf_obj = cNMF(output_dir=tmpdir, name="bench")
        max_k = max(args.k_values)
        nhvg = min(2000, args.n_genes if args.counts_fn is None else 99999)
        cnmf_obj.prepare(
            counts_fn=counts_fn,
            components=[max_k],   # only need norm_counts, K doesn't matter here
            n_iter=1,
            seed=args.seed,
            beta_loss=args.beta_loss,
            num_highvar_genes=nhvg,
            use_torch=False,
        )

        import scanpy as sc
        norm_counts = sc.read(cnmf_obj.paths['normalized_counts'])
        X = norm_counts.X

        with open(cnmf_obj.paths['nmf_run_parameters']) as f:
            base_kwargs = yaml.safe_load(f)
        # Remove keys that will be set per-run
        for key in ('n_components', 'random_state', 'use_torch'):
            base_kwargs.pop(key, None)

        n_cells, n_genes = X.shape
        print(f"Norm counts shape: {n_cells} cells × {n_genes} HVGs")
        print()

        # ------------------------------------------------------------------
        # Benchmark loop
        # ------------------------------------------------------------------
        results = []
        col_w = [5, 20, 20, 10, 14, 14]

        header = (
            f"{'K':<{col_w[0]}} "
            f"{'sklearn time (s)':<{col_w[1]}} "
            f"{'torch time (s)':<{col_w[2]}} "
            f"{'speedup':<{col_w[3]}} "
            f"{'sklearn err':<{col_w[4]}} "
            f"{'torch err':<{col_w[5]}}"
        )
        separator = "-" * len(header)
        summary_label = (
            f"n_cells={n_cells}, n_genes={n_genes}, "
            f"n_timing_runs={args.n_timing_runs}, beta_loss={args.beta_loss}"
        )
        print(f"Benchmark  ({summary_label})")
        print()
        print(header)
        print(separator)

        # Warmup the GPU before timing to avoid first-call initialization overhead.
        if torch_available:
            kw_warmup = _build_nmf_kwargs(base_kwargs, sorted(args.k_values)[0], args.seed, use_torch=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cnmf_obj._nmf(X, kw_warmup)

        for k in sorted(args.k_values):
            row = {'k': k}

            # --- sklearn ---
            sk_times = []
            sk_err = float('nan')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(args.n_timing_runs):
                    kw = _build_nmf_kwargs(base_kwargs, k, args.seed + i, use_torch=False)
                    t0 = time.perf_counter()
                    spectra, usages = cnmf_obj._nmf(X, kw)
                    sk_times.append(time.perf_counter() - t0)
            sk_err = _recon_error(X, spectra, usages)
            row['sklearn_mean'] = float(np.mean(sk_times))
            row['sklearn_std'] = float(np.std(sk_times))
            row['sklearn_err'] = sk_err

            # --- torch ---
            if torch_available:
                tr_times = []
                tr_err = float('nan')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # suppress known solver/init warnings
                    for i in range(args.n_timing_runs):
                        kw = _build_nmf_kwargs(base_kwargs, k, args.seed + i, use_torch=True)
                        t0 = time.perf_counter()
                        spectra_t, usages_t = cnmf_obj._nmf(X, kw)
                        tr_times.append(time.perf_counter() - t0)
                tr_err = _recon_error(X, spectra_t, usages_t)
                row['torch_mean'] = float(np.mean(tr_times))
                row['torch_std'] = float(np.std(tr_times))
                row['torch_err'] = tr_err
                speedup = row['sklearn_mean'] / row['torch_mean']
                row['speedup'] = speedup
                torch_str = f"{row['torch_mean']:.3f} ± {row['torch_std']:.3f}"
                speedup_str = f"{speedup:.1f}x"
                torch_err_str = f"{tr_err:.3e}"
            else:
                row['torch_mean'] = row['torch_std'] = row['torch_err'] = row['speedup'] = float('nan')
                torch_str = "n/a (not installed)"
                speedup_str = "n/a"
                torch_err_str = "n/a"

            results.append(row)

            sk_str = f"{row['sklearn_mean']:.3f} ± {row['sklearn_std']:.3f}"
            sk_err_str = f"{sk_err:.3e}"

            print(
                f"{k:<{col_w[0]}} "
                f"{sk_str:<{col_w[1]}} "
                f"{torch_str:<{col_w[2]}} "
                f"{speedup_str:<{col_w[3]}} "
                f"{sk_err_str:<{col_w[4]}} "
                f"{torch_err_str:<{col_w[5]}}"
            )

        print()

        # ------------------------------------------------------------------
        # Optional CSV output
        # ------------------------------------------------------------------
        if args.output_csv:
            import csv
            fieldnames = ['k', 'sklearn_mean', 'sklearn_std', 'sklearn_err',
                          'torch_mean', 'torch_std', 'torch_err', 'speedup']
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"Results written to: {args.output_csv}")

    finally:
        if cleanup_tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

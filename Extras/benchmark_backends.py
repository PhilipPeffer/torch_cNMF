#!/usr/bin/env python3
"""
Benchmark the sklearn vs PyTorch NMF backends in cNMF.

Measures total wall-clock time and reconstruction error for each backend
across one or more values of K (number of components).

For sklearn, ``--n-timing-runs`` jobs are distributed across ``--n-workers``
processes (default: os.cpu_count()), each pinned to 1 BLAS thread via
threadpoolctl.  This mirrors how cNMF's ``factorize --total-workers N``
actually parallelises on a multi-core CPU host.

For torch/GPU, the same number of jobs run sequentially in one process; the
GPU provides internal parallelism.  Comparing total wall-clock time for the
same number of jobs gives a fair apples-to-apples throughput comparison.

Usage (from the repo root, in an environment with cnmf and torchnmf installed):

    python Extras/benchmark_backends.py [options]

Install torchnmf first if needed:
    pip install torchnmf
"""

import argparse
import os
import time
import shutil
import tempfile
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import yaml

from cnmf import cNMF


# ---------------------------------------------------------------------------
# Pool initializer + picklable sklearn worker
# Must be module-level so multiprocessing can pickle them on non-fork platforms.
# ---------------------------------------------------------------------------

_SHARED_X = None  # set once per worker process by _pool_init


def _pool_init(X):
    global _SHARED_X
    _SHARED_X = X


def _sklearn_nmf_worker(nmf_kwargs):
    """Run one sklearn NMF call in a pool worker, pinned to 1 BLAS thread."""
    import warnings as _w
    from sklearn.decomposition import non_negative_factorization as _nnmf
    kw = dict(nmf_kwargs)
    kw.pop('use_torch', None)
    try:
        from threadpoolctl import threadpool_limits
        blas_ctx = threadpool_limits(limits=1, user_api='blas')
    except ImportError:
        from contextlib import nullcontext
        blas_ctx = nullcontext()
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        with blas_ctx:
            (usages, spectra, _) = _nnmf(_SHARED_X, **kw)
    return (spectra, usages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_info(n_workers):
    lines = []
    try:
        import sklearn
        lines.append(f"  sklearn  : {sklearn.__version__}  ({n_workers} workers, 1 BLAS thread each)")
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
        description="Benchmark sklearn (parallel CPU) vs PyTorch (GPU) cNMF backends.",
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
    p.add_argument("--n-timing-runs", type=int, default=None,
                   help="Total NMF calls per (backend, K). For sklearn these are "
                        "distributed across --n-workers processes; for torch they "
                        "run sequentially on the GPU.  Defaults to --n-workers so "
                        "every worker runs exactly one job simultaneously.")
    p.add_argument("--n-workers", type=int, default=None,
                   help="CPU worker processes for sklearn (default: os.cpu_count()). "
                        "Set to 1 to benchmark serial sklearn instead.")
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
    n_workers = args.n_workers if args.n_workers is not None else os.cpu_count()
    n_timing_runs = args.n_timing_runs if args.n_timing_runs is not None else n_workers

    torch_available = True
    try:
        import torchnmf  # noqa: F401
        import torch     # noqa: F401
    except ImportError:
        torch_available = False

    print("=" * 66)
    print("cNMF backend benchmark")
    print("=" * 66)
    print(_env_info(n_workers))
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

    pool = None
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
        # Start the sklearn worker pool once (X is sent to workers via the
        # initializer, not re-pickled for every task).
        # ------------------------------------------------------------------
        if n_workers > 1:
            pool = Pool(n_workers, initializer=_pool_init, initargs=(X,))
        else:
            _pool_init(X)   # set global for the in-process fallback

        # ------------------------------------------------------------------
        # Benchmark loop
        # ------------------------------------------------------------------
        results = []
        col_w = [5, 22, 22, 10, 14, 14]

        sk_label = f"sklearn total ({n_workers}w)"
        tr_label = "torch total (GPU)"
        header = (
            f"{'K':<{col_w[0]}} "
            f"{sk_label:<{col_w[1]}} "
            f"{tr_label:<{col_w[2]}} "
            f"{'speedup':<{col_w[3]}} "
            f"{'sklearn err':<{col_w[4]}} "
            f"{'torch err':<{col_w[5]}}"
        )
        separator = "-" * len(header)
        summary_label = (
            f"n_cells={n_cells}, n_genes={n_genes}, "
            f"n_timing_runs={n_timing_runs}, n_workers={n_workers}, "
            f"beta_loss={args.beta_loss}"
        )
        print(f"Benchmark  ({summary_label})")
        print(f"  sklearn: {n_timing_runs} jobs across {n_workers} processes, "
              f"each pinned to 1 BLAS thread")
        if torch_available:
            print(f"  torch  : {n_timing_runs} jobs sequentially on GPU")
        print()
        print(header)
        print(separator)

        # Warm up the GPU before timing to avoid first-call initialization overhead.
        if torch_available:
            kw_warmup = _build_nmf_kwargs(base_kwargs, sorted(args.k_values)[0], args.seed, use_torch=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cnmf_obj._nmf(X, kw_warmup)

        for k in sorted(args.k_values):
            row = {'k': k}

            # --- sklearn (parallel pool or single-process fallback) ---
            job_kwargs = [
                _build_nmf_kwargs(base_kwargs, k, args.seed + i, use_torch=False)
                for i in range(n_timing_runs)
            ]
            if pool is not None:
                t0 = time.perf_counter()
                sk_results = pool.map(_sklearn_nmf_worker, job_kwargs)
                sk_total = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                sk_results = [_sklearn_nmf_worker(kw) for kw in job_kwargs]
                sk_total = time.perf_counter() - t0
            spectra, usages = sk_results[-1]
            sk_err = _recon_error(X, spectra, usages)
            row['sklearn_total'] = sk_total
            row['sklearn_err'] = sk_err

            # --- torch (sequential; GPU parallelises internally) ---
            if torch_available:
                tr_total = 0.0
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for i in range(n_timing_runs):
                        kw = _build_nmf_kwargs(base_kwargs, k, args.seed + i, use_torch=True)
                        t0 = time.perf_counter()
                        spectra_t, usages_t = cnmf_obj._nmf(X, kw)
                        tr_total += time.perf_counter() - t0
                tr_err = _recon_error(X, spectra_t, usages_t)
                row['torch_total'] = tr_total
                row['torch_err'] = tr_err
                speedup = sk_total / tr_total
                row['speedup'] = speedup
                torch_str = f"{tr_total:.3f}s"
                speedup_str = f"{speedup:.1f}x"
                torch_err_str = f"{tr_err:.3e}"
            else:
                row['torch_total'] = row['torch_err'] = row['speedup'] = float('nan')
                torch_str = "n/a (not installed)"
                speedup_str = "n/a"
                torch_err_str = "n/a"

            results.append(row)

            sk_str = f"{sk_total:.3f}s"
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
            fieldnames = ['k', 'sklearn_total', 'sklearn_err',
                          'torch_total', 'torch_err', 'speedup']
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            print(f"Results written to: {args.output_csv}")

    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        if cleanup_tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

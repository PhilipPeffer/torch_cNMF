#!/usr/bin/env python3
"""
Benchmark the sklearn vs PyTorch NMF backends in cNMF.

Measures total wall-clock time for the factorize() step using the package's
built-in parallelisation:

  sklearn: n_workers parallel processes, each calling
           factorize(worker_i=i, total_workers=n_workers)
           — identical to running
               cnmf factorize --total-workers N --worker-index i
           from the command line.

  torch/GPU: single process calling factorize(worker_i=0, total_workers=1),
             with the GPU providing internal parallelism per NMF call.

Both backends complete the same total number of NMF iterations (--n-iter),
so reported wall-clock times reflect end-to-end factorize throughput including
disk I/O, which is what users actually experience.

Usage (from the repo root):
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
from multiprocessing import Process

import numpy as np
import scipy.sparse as sp

from cnmf import cNMF


# ---------------------------------------------------------------------------
# Worker — must be module-level so multiprocessing can pickle it
# ---------------------------------------------------------------------------

def _factorize_worker(output_dir, name, worker_i, total_workers):
    """Run factorize() as one parallel cNMF worker, suppressing per-run logs."""
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            obj = cNMF(output_dir=output_dir, name=name)
            obj.factorize(worker_i=worker_i, total_workers=total_workers)
        finally:
            sys.stdout = sys.__stdout__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_info(n_workers):
    lines = []
    try:
        import sklearn
        lines.append(f"  sklearn  : {sklearn.__version__}  ({n_workers} parallel workers)")
    except ImportError:
        lines.append("  sklearn  : NOT INSTALLED")
    try:
        import torch
        if torch.cuda.is_available():
            lines.append(f"  torch    : {torch.__version__}  "
                         f"(device: {torch.cuda.get_device_name(0)})")
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


def _run_parallel_factorize(output_dir, name, n_workers):
    """
    Spawn n_workers processes that each call factorize(worker_i=i, total_workers=n_workers).
    Returns total wall-clock time in seconds.
    """
    procs = [
        Process(target=_factorize_worker, args=(output_dir, name, i, n_workers))
        for i in range(n_workers)
    ]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0
    if any(p.exitcode != 0 for p in procs):
        raise RuntimeError("One or more factorize workers exited with errors.")
    return elapsed


def _prepare(base_tmpdir, run_name, counts_fn, k, n_iter, seed, nhvg,
             use_torch, beta_loss):
    """Prepare a fresh cNMF run directory and return the cNMF object."""
    output_dir = os.path.join(base_tmpdir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    obj = cNMF(output_dir=output_dir, name=run_name)
    obj.prepare(
        counts_fn=counts_fn,
        components=[k],
        n_iter=n_iter,
        seed=seed,
        num_highvar_genes=nhvg,
        beta_loss=beta_loss,
        use_torch=use_torch,
    )
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Benchmark sklearn (parallel CPU) vs PyTorch (GPU) cNMF backends "
            "using the actual factorize() pipeline."
        ),
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
    p.add_argument("--n-iter", type=int, default=None,
                   help="Total NMF iterations per K run by each backend "
                        "(default: --n-workers, so each sklearn worker handles "
                        "exactly one iteration).")
    p.add_argument("--n-workers", type=int, default=None,
                   help="Parallel sklearn worker processes (default: os.cpu_count()).")
    p.add_argument("--seed", type=int, default=14, help="Random seed.")
    p.add_argument("--beta-loss", default="frobenius",
                   choices=["frobenius", "kullback-leibler"],
                   help="NMF beta-loss function.")
    p.add_argument("--output-dir", default=None,
                   help="Working directory for cNMF files. "
                        "Defaults to a temporary directory deleted afterwards.")
    p.add_argument("--output-csv", default=None,
                   help="Optional path to write results as a CSV file.")
    return p.parse_args()


def main():
    args = parse_args()
    n_workers = args.n_workers if args.n_workers is not None else os.cpu_count()
    n_iter = args.n_iter if args.n_iter is not None else n_workers

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
            import anndata
            print(f"Generating synthetic data: {args.n_cells} cells × {args.n_genes} genes")
            np.random.seed(args.seed)
            data = np.random.binomial(
                n=100, p=0.01, size=(args.n_cells, args.n_genes)
            ).astype(np.int64)
            adata = anndata.AnnData(X=sp.csr_matrix(data))
            counts_fn = os.path.join(tmpdir, "synthetic_counts.h5ad")
            adata.write_h5ad(counts_fn)

        nhvg = min(2000, args.n_genes if args.counts_fn is None else 99999)

        # ------------------------------------------------------------------
        # Determine norm counts shape by running one prepare() for the
        # smallest K; this result is reused in the benchmark loop below.
        # ------------------------------------------------------------------
        print("Running prepare()…", flush=True)
        first_k = sorted(args.k_values)[0]
        sk_obj_first = _prepare(
            tmpdir, f"sk_k{first_k}", counts_fn, first_k, n_iter,
            args.seed, nhvg, use_torch=False, beta_loss=args.beta_loss,
        )
        import scanpy as sc
        norm_counts = sc.read(sk_obj_first.paths['normalized_counts'])
        n_cells, n_genes = norm_counts.shape
        print(f"Norm counts shape: {n_cells} cells × {n_genes} HVGs")
        print()

        # ------------------------------------------------------------------
        # Benchmark loop
        # ------------------------------------------------------------------
        results = []
        col_w = [5, 22, 22, 10]
        sk_label = f"sklearn total ({n_workers}w)"
        tr_label = "torch total (GPU)"
        header = (
            f"{'K':<{col_w[0]}} "
            f"{sk_label:<{col_w[1]}} "
            f"{tr_label:<{col_w[2]}} "
            f"{'speedup':<{col_w[3]}}"
        )
        separator = "-" * len(header)

        print(f"Benchmark  (n_iter={n_iter} per K, n_workers={n_workers}, "
              f"beta_loss={args.beta_loss})")
        print(f"  sklearn: factorize() distributed across {n_workers} parallel workers")
        if torch_available:
            print(f"  torch  : factorize() in 1 process on GPU ({n_iter} sequential calls)")
        print()
        print(header)
        print(separator)

        for k in sorted(args.k_values):
            row = {'k': k}

            # --- sklearn ---
            # Reuse the already-prepared first_k directory; prepare() the rest.
            if k == first_k:
                sk_run_dir = os.path.join(tmpdir, f"sk_k{k}")
            else:
                sk_obj = _prepare(
                    tmpdir, f"sk_k{k}", counts_fn, k, n_iter,
                    args.seed, nhvg, use_torch=False, beta_loss=args.beta_loss,
                )
                sk_run_dir = os.path.join(tmpdir, f"sk_k{k}")

            sk_time = _run_parallel_factorize(sk_run_dir, f"sk_k{k}", n_workers)
            row['sklearn_total'] = sk_time

            # --- torch ---
            if torch_available:
                import warnings
                tr_obj = _prepare(
                    tmpdir, f"tr_k{k}", counts_fn, k, n_iter,
                    args.seed, nhvg, use_torch=True, beta_loss=args.beta_loss,
                )
                with warnings.catch_warnings(), open(os.devnull, "w") as devnull:
                    warnings.simplefilter("ignore")
                    sys.stdout = devnull
                    try:
                        t0 = time.perf_counter()
                        tr_obj.factorize()
                        tr_time = time.perf_counter() - t0
                    finally:
                        sys.stdout = sys.__stdout__
                row['torch_total'] = tr_time
                speedup = sk_time / tr_time
                row['speedup'] = speedup
                torch_str = f"{tr_time:.3f}s"
                speedup_str = f"{speedup:.1f}x"
            else:
                row['torch_total'] = row['speedup'] = float('nan')
                torch_str = "n/a (not installed)"
                speedup_str = "n/a"

            results.append(row)
            sk_str = f"{sk_time:.3f}s"
            print(
                f"{k:<{col_w[0]}} "
                f"{sk_str:<{col_w[1]}} "
                f"{torch_str:<{col_w[2]}} "
                f"{speedup_str:<{col_w[3]}}"
            )

        print()

        # ------------------------------------------------------------------
        # Optional CSV output
        # ------------------------------------------------------------------
        if args.output_csv:
            import csv
            fieldnames = ['k', 'sklearn_total', 'torch_total', 'speedup']
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            print(f"Results written to: {args.output_csv}")

    finally:
        if cleanup_tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

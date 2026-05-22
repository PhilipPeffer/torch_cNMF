#!/usr/bin/env python3
"""
Download and filter PBMC3k raw counts for use in benchmarks and tutorials.

Produces a single counts.h5ad file with basic QC filtering applied.

Usage:
    python Extras/prepare_pbmc_counts.py [--output-dir ./example_PBMC]
"""

import argparse
import os

import scanpy as sc


def parse_args():
    p = argparse.ArgumentParser(
        description="Download and filter PBMC3k counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="./example_PBMC",
                   help="Directory to write counts.h5ad into.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "counts.h5ad")

    print("Downloading PBMC3k data via scanpy…")
    adata = sc.datasets.pbmc3k()

    print(f"Before filtering: {adata.n_obs} cells × {adata.n_vars} genes")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"After filtering:  {adata.n_obs} cells × {adata.n_vars} genes")

    adata.write_h5ad(out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

# Consensus NMF (cNMF)

<img src="https://storage.googleapis.com/sabeti-public/dkotliar/elife-cNMF-fig1.jpg" style="height: 800px;" />

cNMF is a pipeline for inferring gene expression programs from scRNA-Seq. It takes a count matrix (N cells X G genes) as input and produces a (K x G) matrix of gene expression programs (GEPs) and a (N x K) matrix specifying the usage of each program for each cell in the data. Read more about the method in the [publication](https://elifesciences.org/articles/43803) and check out examples on [simulated data](Tutorials/analyze_simulated_example_data.ipynb) and [PBMCs](Tutorials/analyze_pbmc_example_data.ipynb).

We have also created a tutorial for running cNMF from R. See the [Rmd notebook](Tutorials/R_vignette.Rmd) or the [compiled html](Tutorials/R_vignette.nb.html) for this.

# Installation
cNMF requires Python 3.11+ and scikit-learn>=1.0, scanpy>=1.8, and AnnData>=0.9

You can install with [pip](https://pypi.org/):

```bash
pip install cnmf
```

If you want to use the batch correction preprocessing, you also need to install the [Python implementation of Harmony](https://github.com/slowkow/harmonypy) and scikit-misc

```bash
pip install harmonypy
pip install scikit-misc
```

## Optional: PyTorch GPU backend

An optional PyTorch-based NMF backend is available via the [torchnmf](https://github.com/yoyolicoris/torchnmf) library. It automatically uses a CUDA GPU when one is available, falling back to CPU otherwise.

First install PyTorch for your hardware by following the instructions at [pytorch.org/get-started](https://pytorch.org/get-started/locally/) (choose your OS, CUDA version, or ROCm as appropriate). Then install the torchnmf dependency:

```bash
pip install cnmf[torch]
# or equivalently:
pip install torchnmf
```

### Docker: adding GPU support

The Docker image (`Extras/Dockerfile`) ships only the sklearn backend. To add GPU support, install PyTorch and torchnmf on top of it — either at build time by extending the image, or at runtime inside the container.

**At runtime** (quickest, but not persistent across container restarts):
```bash
docker run --rm -it --gpus all -v /path/to/data:/data cnmf
# inside the container:
pip install torchnmf torch --index-url https://download.pytorch.org/whl/cu124
```

**At build time** (bake it into a derived image — replace `cu124` with your CUDA version, e.g. `cu118`, `cu121`, or `rocm6.2` for AMD):
```dockerfile
FROM cnmf
RUN pip install --no-cache-dir torchnmf torch \
      --index-url https://download.pytorch.org/whl/cu124
```

Find the correct index URL for your driver at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). The right CUDA version to use is determined by your host GPU driver, not the container — run `nvidia-smi` on the host to check.

# Running cNMF

cNMF can be run from the command line without any parallelization using the example commands below:

```bash
cnmf prepare --output-dir ./example_data --name example_cNMF -c ./example_data/counts_prefiltered.txt -k 5 6 7 8 9 10 11 12 13 --n-iter 100 --seed 14

cnmf factorize --output-dir ./example_data --name example_cNMF --worker-index 0 --total-workers 1

cnmf combine --output-dir ./example_data --name example_cNMF

cnmf k_selection_plot --output-dir ./example_data --name example_cNMF

cnmf consensus --output-dir ./example_data --name example_cNMF --components 10 --local-density-threshold 0.01 --show-clustering
```

Or alternatively, the same steps can be run from within a Python environment using the commands below:

```python
from cnmf import cNMF
cnmf_obj = cNMF(output_dir="./example_data", name="example_cNMF")
cnmf_obj.prepare(counts_fn="./example_data/counts_prefiltered.txt", components=np.arange(5,14), n_iter=100, seed=14)
cnmf_obj.factorize(worker_i=0, total_workers=1)
cnmf_obj.combine()
cnmf_obj.k_selection_plot()
cnmf_obj.consensus(k=10, density_threshold=0.01)
usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=10, density_threshold=0.01)
```

For the Python environment approach, `usage` will contain the usage matrix with each cell normalized to sum to 1. `spectra_scores` contains the gene_spectra_scores output (aka Z-score unit gene expression matrix), `spectra_tpm` contains the GEP spectra in units of TPM and `top_genes` contains an ordered list of the top 100 associated genes for each program.

## PyTorch GPU backend

Pass `use_torch=True` to `prepare()` to enable the PyTorch backend for all factorization steps. The flag is persisted in the run-parameters file, so `factorize()`, `combine()`, and `consensus()` all pick it up automatically without any further changes.

```python
from cnmf import cNMF
import numpy as np

cnmf_obj = cNMF(output_dir="./example_data", name="example_cNMF")
cnmf_obj.prepare(
    counts_fn="./example_data/counts_prefiltered.txt",
    components=np.arange(5, 14),
    n_iter=100,
    seed=14,
    use_torch=True,   # <-- enables PyTorch backend
)
cnmf_obj.factorize()   # runs on GPU if available, CPU otherwise
cnmf_obj.combine()
cnmf_obj.k_selection_plot()
cnmf_obj.consensus(k=10, density_threshold=0.01)
usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=10, density_threshold=0.01)
```

**Notes:**
- Requires `pip install torchnmf` (or `pip install cnmf[torch]`).
- **Recommended for large K (≥10) when a CUDA or ROCm GPU is available.** At small K, parallel sklearn workers outperform sequential GPU calls because each job converges quickly. At large K the GPU's per-call efficiency dominates: 3–4× faster than 24 parallel sklearn workers on PBMC3k (see benchmark below).
- GPU acceleration is automatic when a CUDA device is present; no code changes are needed.
- `beta_loss='frobenius'` (beta=2) and `beta_loss='kullback-leibler'` (beta=1) are both supported.
- Results are computed in **float32** (vs float64 for sklearn). Outputs are numerically comparable but not bit-for-bit identical between backends; this is acceptable because cNMF averages over many random restarts in the consensus step.
- `init='nndsvd'` is not supported by torchnmf and falls back to random initialization with a warning.
- If `alpha_usage` and `alpha_spectra` differ, `alpha_usage` is used for both factors and a warning is emitted (torchnmf applies a single regularization alpha to both W and H).

### GPU benchmark (PBMC3k, 2700 cells × 2000 HVGs, AMD Radeon RX 7800 XT)

Measured with `Extras/benchmark_backends.py`, frobenius loss.  Both backends complete 24 NMF iterations
per K: sklearn uses `factorize(worker_i=i, total_workers=24)` across 24 parallel processes (identical
to `cnmf factorize --total-workers 24 --worker-index i`); torch runs all 24 iterations sequentially
in one process on the GPU.  Speedup < 1 means sklearn is faster; > 1 means GPU wins.

| K  | sklearn total (24w) | torch total (GPU) | speedup |
|----|---------------------|-------------------|---------|
| 5  | 0.924s              | 1.806s            | 0.5×    |
| 7  | 0.842s              | 1.651s            | 0.5×    |
| 10 | 5.047s              | 1.604s            | 3.1×    |
| 15 | 6.739s              | 1.673s            | 4.0×    |

At small K, 24 parallel sklearn workers finish 24 short jobs in roughly one job's time and beat the GPU.
At large K, the GPU's per-call efficiency dominates: torch completes 24 sequential iterations faster
than sklearn workers finish their individually slower jobs.

Output data files will all be available in the ./example_data/example_cNMF directory including:

   - Z-score unit gene expression program matrix - `example_data/example_cNMF/example_cNMF.gene_spectra_score.k_10.dt_0_01.txt`
   - TPM unit gene expression program matrix - `example_data/example_cNMF/example_cNMF.gene_spectra_tpm.k_10.dt_0_01.txt`
   - Usage matrix - `example_data/example_cNMF/example_cNMF.usages.k_10.dt_0_01.consensus.txt`
   - K selection plot - `example_data/example_cNMF/example_cNMF.k_selection.png`
   - Clustergram diagnostic plot - `example_data/example_cNMF/example_cNMF.clustering.k_10.dt_0_01.pdf`

Some usage notes:
 - __Parallelization__: The factorize step can be parallelized with the --total-workers flag and then submitting multiple jobs, one per worker, indexed starting by 0. For example:
  ```
  cnmf factorize --output-dir ./example_data --name example_cNMF --worker-index 0 --total-workers 3 &
  cnmf factorize --output-dir ./example_data --name example_cNMF --worker-index 1 --total-workers 3 &
  cnmf factorize --output-dir ./example_data --name example_cNMF --worker-index 2 --total-workers 3 &
  ```
  would break the factorization jobs up into 3 batches and submit them independently. This can be used with compute clusters to run the factorizations in parallel (see tutorials for example).
 - __Input data__: Input data can be provided in 3 ways:
    - 1. as a scanpy file ending in .h5ad containg counts as the data feature. See the PBMC dataset tutorial for an example of how to generate the Scanpy object from the data provided by 10X. Because Scanpy uses sparse matrices by default, the .h5ad data structure can take up much less memory than the raw counts matrix and can be much faster to load. 
    - 2. as a raw tab-delimited text file containing row labels with cell IDs (barcodes) and column labels as gene IDs
    - 3. as a 10x-Genomics-formatted mtx directory. You provide the path to the counts.mtx file or counts.mtx.gz file to counts_fn. It expects there to be barcodes.tsv and genes.tsv in the directory as well

    
See the tutorials or Stepwise_Guide.md for more details

# Integration of technical variables and batches

We have implemented a pipeline to integrate batch variables prior to running cNMF and to handle ADTs in CITE-Seq. It uses an adaptation of [Harmony](https://github.com/slowkow/harmonypy) that corrects the underlying count matrix rather than principal components. We describe it in our [recent preprint](https://www.biorxiv.org/content/10.1101/2024.05.03.592310v1). See the [batch correction tutorial](Tutorials/analyze_batcheffectcorrect_BaronEtAl.ipynb) as well for an example.

We use a separate Preprocess class to run batch correction. You pass in an AnnData object, as well as harmony_vars, a list of the names of variables to correct correspond to columns in the AnnData obs attribute. You also specify an output file base name to save the results to like below:

```
from cnmf import cNMF, Preprocess
#Initialize the Preprocess object
p = Preprocess(random_seed=14)

#Batch correct the data and save the corrected high-variance gene data to adata_c, and the TPM normalized data to adata_tpm 
(adata_c, adata_tpm, hvgs) = p.preprocess_for_cnmf(adata, harmony_vars=['Sex', 'Sample'], n_top_rna_genes = 2000, librarysize_targetsum= 1e6,
                                                    save_output_base='./example_islets/batchcorrect_example_sex')

#Then run cNMF passing in the corrected counts file, tpm_fn, and HVGs as inputs
cnmf_obj_corrected = cNMF(output_dir='./example_islets', name='BatchCorrected')
cnmf_obj_corrected.prepare(counts_fn='./example_islets/batchcorrect_example.Corrected.HVG.Varnorm.h5ad',
                           tpm_fn='./example_islets/batchcorrect_example.TP10K.h5ad',
                           genes_file='./example_islets/batchcorrect_example.Corrected.HVGs.txt',
                           components=[15], n_iter=20, seed=14, num_highvar_genes=2000)

#Then proceed with the rest of cNMF as normal
```

# Change log

### New in version 1.8
- Optional PyTorch/GPU backend via torchnmf. Enable with `use_torch=True` in `prepare()`. Automatically uses CUDA when available.

### New in version 1.7
- Use scipy hierachical clsutering grather than fastcluster for compatibility with numpy>2.0
- More efficient sparse + batched OLS computation uses significantly less memory
- Implemented basic testing suite

### New in version 1.6
- Added option in consensus() to build spectra for annotating new datasets with GEPs using [starCAT](https://github.com/immunogenomics/starCAT).
- Added option to factorize() to skip tasks that have already completed. 

### New in version 1.5
- Fixed bug in detecting and printing cells with 0 counts of overdispersed genes
- Added option in load_results() to return normalized or unnormalized usage.
- Added a Preprocess class to batch correct data prior to cNMF. See the added Tutorial analyze_batcheffectcorrect_BaronEtAl.ipynb to illustrate its basic usage.
- Now accepts 10x formatted .mtx directories (containing counts.mtx, barcodes.tsv, and genes.tsv files)

### New in version 1.4
- Usage is re-fit a final time from gene_spectra_tpm which increases accuracy in simulations
- Use cnmf_obj.load_results(K=_, density_threshold=_) to obtain usage, spectra_scores, spectra_tpm, and top_genes matrices
- cnmf_obj.combine() now has a skip_missing_files=True/False option to skip incomplete factorize iterations
- GEPs are now ordered by maximum total usage
- Now detects and errors when 0 counts of HVGs with interpretable error message

### New in version 1.3
- Installation via pip
- Object oriented interface for Python users and command line script option via `cnmf`  

### New in version 1.2
 - Increased the threshold for ignoring genes with low mean expression for determining high-variance genes from a TPM of 0.01 to 0.5. Some users were identifying uninterpretable programs with very low usage except in a tiny number of cells. We suspect that this was due to including genes as high-variance that are detected in a small number of cells. This change in the default parameter will help offset that problem in most cases.
 - Updated import of NMF for compatibility with scikit-learn versions >22
 - Colorbar for heatmaps included with consensus matrix plot

### New in version 1.1
 - Now operates by default on sparse matrices. Use --densify option in prepare step if data is not sparse
 - Now takes Scanpy AnnData object files (.h5ad) as input
 - Now has option to use KL divergence beta_loss instead of Frobenius. Frobenius is the default because it is much faster.
 - Includes a Docker file for creating a Docker container to run cNMF in parallel with cloud compute
 - Includes a tutorial on a simple PBMC dataset
 - Other minor fixes

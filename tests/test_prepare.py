import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import os
import yaml
import warnings
import scipy.sparse as sp
from unittest.mock import patch
from cnmf import cNMF, save_df_to_npz, load_df_from_npz

# Global parameters for data simulation
NUM_CELLS = 100
NUM_GENES = 500
BINOM_N = 100
BINOM_P = 0.01
SEED = 42

@pytest.fixture
def mock_cnmf(tmp_path):
    return cNMF(output_dir=str(tmp_path), name="test")

def generate_counts_file(tmp_path, file_format, dtype=np.int64, zero_count=False):
    """
    Generates a synthetic single-cell RNA-seq counts file in various formats.
    
    Args:
        tmp_path (Path): Temporary path for storing the file.
        file_format (str): One of ['txt', 'npz', 'h5ad'].
        dtype (numpy dtype, optional): The data type to store (int64, float32, etc.).
        zero_count (bool, optional): If True, makes the first cell have zero counts.

    Returns:
        str: Path to the generated counts file.
    """
    np.random.seed(SEED)
    data = np.random.binomial(n=BINOM_N, p=BINOM_P, size=(NUM_CELLS, NUM_GENES)).astype(dtype)
    
    if zero_count:
        data[0, :] = 0  # Introduce zero-count cells

    if file_format == "txt":
        df = pd.DataFrame(data, columns=[f"gene{i}" for i in range(NUM_GENES)],
                          index=[f"cell{i}" for i in range(NUM_CELLS)])
        counts_fn = tmp_path / f"counts_{dtype.__name__}.txt"
        df.to_csv(counts_fn, sep='\t')

    elif file_format == "npz":
        df = pd.DataFrame(data, columns=[f"gene{i}" for i in range(NUM_GENES)],
                          index=[f"cell{i}" for i in range(NUM_CELLS)])
        counts_fn = tmp_path / f"counts_{dtype.__name__}.npz"
        save_df_to_npz(df, counts_fn)

    elif file_format == "h5ad":
        adata = sc.AnnData(X=sp.csr_matrix(data))
        counts_fn = tmp_path / f"counts_{dtype.__name__}.h5ad"
        adata.write_h5ad(counts_fn)

    else:
        raise ValueError("Unsupported file format. Choose from ['txt', 'npz', 'h5ad'].")

    return str(counts_fn)

@pytest.mark.parametrize("file_format", ["txt", "npz", "h5ad"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("densify", [True, False])
def test_prepare(mock_cnmf, file_format, dtype, densify, tmp_path):
    counts_fn = generate_counts_file(tmp_path, file_format, dtype)
    
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    mock_cnmf.prepare(counts_fn, components=[5, 10], n_iter=10, densify=densify)
    
    # Check if output files were created
    expected_files = [
        mock_cnmf.paths['normalized_counts'],
        mock_cnmf.paths['nmf_replicate_parameters'],
        mock_cnmf.paths['nmf_run_parameters'],
        mock_cnmf.paths['nmf_genes_list'],
        mock_cnmf.paths['tpm'],
        mock_cnmf.paths['tpm_stats']
    ]
    
    for file in expected_files:
        assert os.path.exists(file), f"Expected output file {file} not found."
    
    # Clean up after test
    for file in expected_files:
        os.remove(file)

@pytest.mark.parametrize("file_format", ["txt", "npz", "h5ad"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("densify", [True, False])
def test_prepare_raises_on_zero_count_cells(mock_cnmf, file_format, dtype, densify, tmp_path):
    counts_fn = generate_counts_file(tmp_path, file_format, dtype, zero_count=True)

    with pytest.raises(Exception, match="Error: .* cells have zero counts of overdispersed genes.*"):
        mock_cnmf.prepare(counts_fn, components=[5, 10], n_iter=10, densify=densify)


# ---------------------------------------------------------------------------
# PyTorch backend tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_torch", [True, False])
def test_prepare_torch_flag_persisted_in_yaml(mock_cnmf, tmp_path, use_torch):
    """prepare() must write use_torch into the YAML run-parameters file."""
    counts_fn = generate_counts_file(tmp_path, "h5ad", np.int64)
    mock_cnmf.prepare(counts_fn, components=[3], n_iter=2, use_torch=use_torch)

    with open(mock_cnmf.paths['nmf_run_parameters']) as f:
        run_params = yaml.safe_load(f)
    assert run_params['use_torch'] is use_torch


def test_factorize_torch(tmp_path):
    """prepare() + factorize() with use_torch=True must produce iteration spectra files."""
    counts_fn = generate_counts_file(tmp_path, "h5ad", np.int64)
    cnmf_obj = cNMF(output_dir=str(tmp_path), name="torch_test")
    cnmf_obj.prepare(counts_fn, components=[3], n_iter=2, use_torch=True)
    cnmf_obj.factorize()

    run_params = load_df_from_npz(cnmf_obj.paths['nmf_replicate_parameters'])
    for _, row in run_params.iterrows():
        spectra_path = cnmf_obj.paths['iter_spectra'] % (row['n_components'], row['iter'])
        assert os.path.exists(spectra_path), f"Missing iter spectra file: {spectra_path}"
        spectra = load_df_from_npz(spectra_path)
        assert spectra.shape == (int(row['n_components']), NUM_GENES), \
            f"Unexpected spectra shape {spectra.shape}"


def test_nmf_torch_warns_nndsvd(mock_cnmf, tmp_path):
    """_nmf_torch must warn when init='nndsvd' (unsupported, falls back to random)."""
    counts_fn = generate_counts_file(tmp_path, "h5ad", np.int64)
    mock_cnmf.prepare(counts_fn, components=[3], n_iter=1, init='nndsvd', use_torch=True)

    run_params = load_df_from_npz(mock_cnmf.paths['nmf_replicate_parameters'])
    with open(mock_cnmf.paths['nmf_run_parameters']) as f:
        nmf_kwargs = yaml.safe_load(f)
    nmf_kwargs['random_state'] = int(run_params.iloc[0]['nmf_seed'])
    nmf_kwargs['n_components'] = int(run_params.iloc[0]['n_components'])

    norm_counts = sc.read(mock_cnmf.paths['normalized_counts'])
    use_torch = nmf_kwargs.pop('use_torch')
    assert use_torch is True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mock_cnmf._nmf_torch(norm_counts.X, nmf_kwargs)

    messages = [str(w.message) for w in caught]
    assert any("nndsvd" in m for m in messages), \
        f"Expected nndsvd warning, got: {messages}"


def test_nmf_torch_warns_alpha_mismatch(mock_cnmf, tmp_path):
    """_nmf_torch must warn when alpha_W != alpha_H."""
    counts_fn = generate_counts_file(tmp_path, "h5ad", np.int64)
    mock_cnmf.prepare(counts_fn, components=[3], n_iter=1,
                      alpha_usage=0.1, alpha_spectra=0.5, use_torch=True)

    run_params = load_df_from_npz(mock_cnmf.paths['nmf_replicate_parameters'])
    with open(mock_cnmf.paths['nmf_run_parameters']) as f:
        nmf_kwargs = yaml.safe_load(f)
    nmf_kwargs['random_state'] = int(run_params.iloc[0]['nmf_seed'])
    nmf_kwargs['n_components'] = int(run_params.iloc[0]['n_components'])

    norm_counts = sc.read(mock_cnmf.paths['normalized_counts'])
    nmf_kwargs.pop('use_torch')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mock_cnmf._nmf_torch(norm_counts.X, nmf_kwargs)

    messages = [str(w.message) for w in caught]
    assert any("alpha" in m for m in messages), \
        f"Expected alpha mismatch warning, got: {messages}"


def test_nmf_torch_import_error(mock_cnmf, tmp_path):
    """_nmf_torch must raise ImportError when torchnmf is not installed."""
    counts_fn = generate_counts_file(tmp_path, "h5ad", np.int64)
    mock_cnmf.prepare(counts_fn, components=[3], n_iter=1, use_torch=True)

    run_params = load_df_from_npz(mock_cnmf.paths['nmf_replicate_parameters'])
    with open(mock_cnmf.paths['nmf_run_parameters']) as f:
        nmf_kwargs = yaml.safe_load(f)
    nmf_kwargs['random_state'] = int(run_params.iloc[0]['nmf_seed'])
    nmf_kwargs['n_components'] = int(run_params.iloc[0]['n_components'])
    nmf_kwargs.pop('use_torch')

    norm_counts = sc.read(mock_cnmf.paths['normalized_counts'])

    with patch.dict('sys.modules', {'torchnmf': None, 'torchnmf.nmf': None}):
        with pytest.raises(ImportError, match="torchnmf"):
            mock_cnmf._nmf_torch(norm_counts.X, nmf_kwargs)


def test_nmf_torch_warns_solver(mock_cnmf, tmp_path):
    """_nmf_torch must warn when solver != 'mu' (unsupported by torchnmf)."""
    counts_fn = generate_counts_file(tmp_path, "h5ad", np.int64)
    mock_cnmf.prepare(counts_fn, components=[3], n_iter=1, use_torch=True)

    run_params = load_df_from_npz(mock_cnmf.paths['nmf_replicate_parameters'])
    with open(mock_cnmf.paths['nmf_run_parameters']) as f:
        nmf_kwargs = yaml.safe_load(f)
    nmf_kwargs['random_state'] = int(run_params.iloc[0]['nmf_seed'])
    nmf_kwargs['n_components'] = int(run_params.iloc[0]['n_components'])
    nmf_kwargs['solver'] = 'cd'
    nmf_kwargs.pop('use_torch')

    norm_counts = sc.read(mock_cnmf.paths['normalized_counts'])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mock_cnmf._nmf_torch(norm_counts.X, nmf_kwargs)

    messages = [str(w.message) for w in caught]
    assert any("solver" in m for m in messages), \
        f"Expected solver warning, got: {messages}"


def test_nmf_torch_kl_divergence(tmp_path):
    """factorize() with use_torch=True and beta_loss='kullback-leibler' must complete."""
    counts_fn = generate_counts_file(tmp_path, "h5ad", np.int64)
    cnmf_obj = cNMF(output_dir=str(tmp_path), name="torch_kl_test")
    cnmf_obj.prepare(counts_fn, components=[3], n_iter=2,
                     use_torch=True, beta_loss='kullback-leibler')
    cnmf_obj.factorize()

    run_params = load_df_from_npz(cnmf_obj.paths['nmf_replicate_parameters'])
    for _, row in run_params.iterrows():
        spectra_path = cnmf_obj.paths['iter_spectra'] % (row['n_components'], row['iter'])
        assert os.path.exists(spectra_path), f"Missing iter spectra file: {spectra_path}"


def test_torch_consensus_refit_usage(tmp_path):
    """consensus() with use_torch=True must complete and produce output files,
    exercising the refit_usage fixed-spectra code path in _nmf_torch."""
    counts_fn = generate_counts_file(tmp_path, "h5ad", np.int64)
    cnmf_obj = cNMF(output_dir=str(tmp_path), name="torch_consensus_test")
    k = 3
    # n_iter=15 ensures n_neighbors=int(0.3*15/3)=1, avoiding division-by-zero
    # in the density calculation inside consensus()
    cnmf_obj.prepare(counts_fn, components=[k], n_iter=15, seed=42, use_torch=True)
    cnmf_obj.factorize()
    cnmf_obj.combine()
    density_threshold = 0.5
    cnmf_obj.consensus(k=k, density_threshold=density_threshold, show_clustering=False)

    ldthresh_str = str(density_threshold).replace('.', '_')
    for fn_key in ['consensus_spectra', 'consensus_usages', 'gene_spectra_score', 'gene_spectra_tpm']:
        path = cnmf_obj.paths[fn_key] % (k, ldthresh_str)
        assert os.path.exists(path), f"Missing consensus output: {path}"

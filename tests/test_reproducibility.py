import pytest
import os
import shutil
import numpy as np
from cnmf import cNMF, load_df_from_npz
import scanpy as sc
import scipy.sparse as sp
import yaml
import warnings

TOLERANCE = 1e-4

def dicts_equal(d1, d2):
    """
    Recursively check if two potentially nested dictionaries are equal.
    Returns True if they are exactly the same, False otherwise.
    """
    # If either is not a dict, do a simple equality check
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return d1 == d2
    
    # If keys differ, they're not equal
    if d1.keys() != d2.keys():
        return False

    # Compare each key
    for key in d1:
        val1 = d1[key]
        val2 = d2[key]

        # Recurse if both values are dicts, else check direct equality
        if isinstance(val1, dict) and isinstance(val2, dict):
            if not dicts_equal(val1, val2):
                return False
        else:
            if val1 != val2:
                return False
    return True

@pytest.fixture
def cnmf_instance(tmp_path):
    """
    Create a cNMF instance that writes into a temporary directory
    so tests don't clutter your real file system.
    """
    cnmf_obj = cNMF(output_dir=str(tmp_path), name="test_cNMF")
    return cnmf_obj

@pytest.mark.parametrize("dataset_config", [
    {
        "name": "pbmc_cNMF",
        "counts_file": "./tests/test_data/example_PBMC/counts.h5ad",
        "k_values": np.arange(7,10),
        "n_iter": 15,
        "nhvg": 1000,
        "seed": 14,
        "reference_dir": "./tests/test_data/example_PBMC",
        "consensus":[(7, 0.1), (8, 0.1)]
    },
])
def test_cnmf_end_to_end(cnmf_instance, dataset_config, tmp_path):
    """
    Single end-to-end test that runs the cNMF pipeline for multiple example datasets.
    """
   
    cnmf_instance.prepare(
        counts_fn=dataset_config["counts_file"],
        components=dataset_config["k_values"],
        n_iter=dataset_config["n_iter"],
        num_highvar_genes=dataset_config["nhvg"],
        seed=dataset_config["seed"]
    )

    #Rather than re-running factorization, we simply copy the combined files 
    for K in dataset_config["k_values"]:
        test_fn = cnmf_instance.paths['merged_spectra'] % K
        ref_fn = test_fn.replace(cnmf_instance.output_dir, dataset_config['reference_dir']).replace('test_cNMF', dataset_config['name'])
        shutil.copy(ref_fn, test_fn)

    for K, ldthresh in dataset_config["consensus"]:
        cnmf_instance.consensus(k=K, density_threshold=ldthresh, show_clustering=False)

    print('\n')
    # Test Consensus results
    for fn in ['consensus_spectra', 'consensus_usages', 'gene_spectra_score', 'gene_spectra_tpm', 'starcat_spectra']:
        test_base = cnmf_instance.paths[fn]
        ref_base = test_base.replace(cnmf_instance.output_dir, dataset_config['reference_dir']).replace('test_cNMF', dataset_config['name'])
        for K, ldthresh in dataset_config["consensus"]:
            ldthresh_str = str(ldthresh).replace('.', '_')
            test_fn = test_base % (K, ldthresh_str)
            ref_fn = ref_base % (K, ldthresh_str)
            assert os.path.exists(test_fn), (
                f"Missing output file {test_fn} for dataset {dataset_config['name']}"
            )
            assert os.path.exists(test_fn), (
                f"Missing reference comparison file {ref_fn} for dataset {dataset_config['name']}"
            )
            test_df = load_df_from_npz(test_fn)
            orig_df = load_df_from_npz(ref_fn)
            rms = ((test_df - orig_df)**2).sum().sum()
            assert (rms < TOLERANCE), (
                f"{ref_fn} does not match."
            )
            print(f'PASSES: {ref_fn}. RMS: {rms:.15}')         

    # Test Prepare results
    for fn in ['normalized_counts', 'nmf_replicate_parameters', 'nmf_run_parameters', 'nmf_genes_list', 'tpm' ,'tpm_stats']:
        test_fn = cnmf_instance.paths[fn]
        ref_fn = test_fn.replace(cnmf_instance.output_dir, dataset_config['reference_dir']).replace('test_cNMF', dataset_config['name'])
        assert os.path.exists(test_fn), (
            f"Missing output file {test_fn} for dataset {dataset_config['name']}"
        )
        assert os.path.exists(test_fn), (
            f"Missing reference comparison file {ref_fn} for dataset {dataset_config['name']}"
        )
        file_root, file_ext = os.path.splitext(test_fn)
        if file_ext == '.h5ad':
            test_adata = sc.read(test_fn)
            orig_adata = sc.read(ref_fn)
            assert test_adata.shape == orig_adata.shape, (
                f"{ref_fn} shape does not match"
            )

            diff = test_adata.X - orig_adata.X
            if sp.issparse(diff):
                rms = diff.power(2).sum()
            else:
                rms = (diff **2).sum()
            
            assert rms < TOLERANCE, (
                f"{ref_fn} does not match. RMS: {rms:.15}"
            )
            print(f'PASSES: {ref_fn}. RMS: {rms:.15}')         

        elif file_ext == '.txt':
            with open (test_fn) as F:
                test_genes = F.read().split('\n')
            with open (ref_fn) as F:
                orig_genes = F.read().split('\n')

            assert test_genes == orig_genes, (
                f"{ref_fn} does not match."
            )
            print(f'PASSES: {ref_fn}')
        elif file_ext == '.npz':
            test_df = load_df_from_npz(test_fn)
            orig_df = load_df_from_npz(ref_fn)

            if fn == 'nmf_replicate_parameters':
                cols = ['n_components', 'iter', 'nmf_seed']
                assert test_df[cols].equals(orig_df[cols]), (
                f"{ref_fn} does not match."
                )
                print(f'PASSES: {ref_fn}')

            elif fn == 'tpm_stats':
                rms = ((test_df - orig_df)**2).sum().sum()
                assert (rms < TOLERANCE), (
                f"{ref_fn} does not match."
                )
                print(f'PASSES: {ref_fn}. RMS: {rms:.15}')         
            else:
                assert test_df.equals(orig_df), (
                    f"{ref_fn} does not match."
                )
                print(f'PASSES: {ref_fn}')
        elif file_ext == '.yaml':
            with open(test_fn, "r") as f:
                test_yaml = yaml.safe_load(f)

            with open(ref_fn, "r") as f:
                orig_yaml = yaml.safe_load(f)

            # use_torch was added after the reference data was generated;
            # strip it before comparing so old references stay valid.
            test_yaml.pop('use_torch', None)
            orig_yaml.pop('use_torch', None)

            assert dicts_equal(test_yaml, orig_yaml), (
                f"{ref_fn} does not match."
            )
            print(f'PASSES: {ref_fn}')                
        else:
            warnings.warn('SKIPPING: {test_fn} as not in the tested output files', UserWarning)

            (f'SKIPPING: {test_fn}')


def test_torch_backend_reproducible(tmp_path):
    """Given the same seed, two torch-backend runs must produce bit-for-bit identical
    iter spectra. This verifies that torch.manual_seed(nmf_seed) makes each
    individual NMF run fully deterministic (guaranteed on CPU; may require
    torch.use_deterministic_algorithms(True) on GPU).
    """
    np.random.seed(42)
    data = np.random.binomial(n=100, p=0.01, size=(50, 200)).astype(np.int64)
    adata = sc.AnnData(X=sp.csr_matrix(data))
    counts_fn = str(tmp_path / "counts.h5ad")
    adata.write_h5ad(counts_fn)

    k, n_iter, seed = 3, 3, 14

    cnmf1 = cNMF(output_dir=str(tmp_path), name="run1")
    cnmf1.prepare(counts_fn, components=[k], n_iter=n_iter, seed=seed, use_torch=True)
    cnmf1.factorize()

    cnmf2 = cNMF(output_dir=str(tmp_path), name="run2")
    cnmf2.prepare(counts_fn, components=[k], n_iter=n_iter, seed=seed, use_torch=True)
    cnmf2.factorize()

    run_params = load_df_from_npz(cnmf1.paths['nmf_replicate_parameters'])
    for _, row in run_params.iterrows():
        spec1 = load_df_from_npz(cnmf1.paths['iter_spectra'] % (row['n_components'], row['iter']))
        spec2 = load_df_from_npz(cnmf2.paths['iter_spectra'] % (row['n_components'], row['iter']))
        assert spec1.equals(spec2), (
            f"Torch backend produced different spectra for k={row['n_components']}, "
            f"iter={row['iter']} with the same seed"
        )

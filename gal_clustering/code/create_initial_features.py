"""
Stage 2 of the pipeline: build the initial patch feature matrix.

Loads every ``samples_chunk_XXX.npy`` written by the patch extractor into one
array, z-scores each feature, and reduces the dimensionality with PCA, keeping
the components that explain 95% of the variance.

The patch matrix can have hundreds of millions of rows, so both the
standardisation and the PCA are implemented here as chunked, in-place / low
memory versions of the corresponding sklearn routines (which would otherwise
create full float64 copies of the data).
"""
import os
import glob
import numpy as np
import gc
from tqdm import tqdm
from sklearn.utils.extmath import svd_flip, stable_cumsum

# Rows processed at a time by the chunked statistics/transform passes.
# 2M rows x 50 features x float64 ~ 0.8 GB of temporary memory.
CHUNK_ROWS = 2_000_000


def standardize_inplace(X, chunk_rows=CHUNK_ROWS):
    """Z-score each column of X in place, chunk by chunk.

    Equivalent to sklearn's StandardScaler(copy=False).fit_transform(X)
    (population variance, zero-variance columns left unscaled), but never
    allocates more than one chunk of temporary memory. sklearn's version
    builds a full float64 copy of X while computing the variance.
    Statistics are accumulated in float64 with a two-pass algorithm.
    """
    n_samples, n_features = X.shape

    # Pass 1: column means.
    total = np.zeros(n_features, dtype=np.float64)
    for i in range(0, n_samples, chunk_rows):
        total += X[i:i + chunk_rows].sum(axis=0, dtype=np.float64)
    mean = total / n_samples

    # Pass 2: sum of squared deviations from the mean (numerically more stable
    # than E[x^2] - E[x]^2). einsum('ij,ij->j') is a column-wise dot product.
    sq_dev = np.zeros(n_features, dtype=np.float64)
    for i in range(0, n_samples, chunk_rows):
        d = X[i:i + chunk_rows].astype(np.float64) - mean
        sq_dev += np.einsum('ij,ij->j', d, d)
    var = sq_dev / n_samples

    # Same rule as sklearn's _handle_zeros_in_scale: don't divide by ~0.
    scale = np.sqrt(var)
    eps = 10 * np.finfo(scale.dtype).eps
    scale[scale < eps] = 1.0

    # Pass 3: transform in place, chunk by chunk.
    for i in range(0, n_samples, chunk_rows):
        X[i:i + chunk_rows] -= mean
        X[i:i + chunk_rows] /= scale

    return X, mean, scale


def pca_chunked(X, n_components, chunk_rows=CHUNK_ROWS):
    """PCA of a tall-skinny array without copying it.

    Reproduces sklearn PCA's 'covariance_eigh' solver, which sklearn already
    picks automatically for data with many more rows than columns: covariance
    matrix -> eigendecomposition -> same sign convention -> same rule for
    choosing the number of components from a variance fraction. The
    covariance is accumulated chunk by chunk in float64, and the projection
    is written chunk by chunk into a float32 output array.
    """
    n_samples, n_features = X.shape

    # One pass accumulating the column sums and the Gram matrix X^T X; the
    # covariance follows from these without ever centring a copy of X.
    total = np.zeros(n_features, dtype=np.float64)
    gram = np.zeros((n_features, n_features), dtype=np.float64)
    for i in range(0, n_samples, chunk_rows):
        c = X[i:i + chunk_rows].astype(np.float64)
        total += c.sum(axis=0)
        gram += c.T @ c
    mean = total / n_samples

    # Unbiased sample covariance: (X^T X - n * mean mean^T) / (n - 1).
    cov = (gram - n_samples * np.outer(mean, mean)) / (n_samples - 1)
    # eigh returns eigenvalues in ascending order; flip to descending so the
    # first component explains the most variance. Tiny negative eigenvalues
    # from round-off are clipped to zero.
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    eigenvals = eigenvals[::-1].copy()
    eigenvecs = eigenvecs[:, ::-1]
    eigenvals[eigenvals < 0.0] = 0.0

    # Deterministic sign convention, identical to sklearn's, so results match
    # sklearn PCA exactly (eigenvectors are only defined up to sign).
    _, components = svd_flip(None, eigenvecs.T.copy(), u_based_decision=False)

    # n_components in (0, 1) is a target variance fraction: keep the smallest
    # number of components whose cumulative explained variance exceeds it.
    # Otherwise it is the number of components to keep.
    explained_variance_ratio = eigenvals / eigenvals.sum()
    if 0 < n_components < 1.0:
        ratio_cumsum = stable_cumsum(explained_variance_ratio)
        n_keep = int(np.searchsorted(ratio_cumsum, n_components, side="right") + 1)
    else:
        n_keep = int(n_components)

    components = components[:n_keep]
    explained_variance_ratio = explained_variance_ratio[:n_keep]

    # Project (X - mean) onto the kept components, chunk by chunk, into a
    # float32 output to halve the memory of the reduced data.
    out = np.empty((n_samples, n_keep), dtype=np.float32)
    for i in range(0, n_samples, chunk_rows):
        c = X[i:i + chunk_rows].astype(np.float64)
        c -= mean
        out[i:i + chunk_rows] = c @ components.T

    return out, n_keep, explained_variance_ratio


def create_vect(run_number, path_model_dir):
    """Load all patch chunks, standardise them and reduce them with PCA.

    Parameters
    ----------
    run_number : int or str
        Run identifier, used in log messages only.
    path_model_dir : str
        Run folder; chunk files are read from ``<path_model_dir>/patches``.

    Returns
    -------
    samples_pca : ndarray of float32, shape (n_patches, n_components)
        PCA-projected patch features (95% of the variance retained).
    samples_ids : ndarray, shape (n_patches,)
        Object ID of each patch row.
    """
    print(f'model {run_number}: CREATION OF INITIAL FEATURE VECTOR ...')

    # Sorted globbing keeps samples_chunk_XXX and samples_object_id_chunk_XXX
    # aligned file-by-file (the extractor guarantees matching numbering).
    patch_dir = os.path.join(path_model_dir, 'patches')
    sample_files = sorted(glob.glob(os.path.join(patch_dir, 'samples_chunk_*.npy')))
    id_files = sorted(glob.glob(os.path.join(patch_dir, 'samples_object_id_chunk_*.npy')))

    if len(sample_files) == 0:
        raise FileNotFoundError(f"No chunk files found in {patch_dir}.")

    # Read only the headers first so the full array can be allocated once,
    # instead of holding every chunk plus a vstack copy at the same time.
    shapes = [np.load(f, mmap_mode='r').shape for f in sample_files]
    n_total = sum(s[0] for s in shapes)
    n_features = shapes[0][1]
    sample_dtype = np.load(sample_files[0], mmap_mode='r').dtype
    id_dtype = np.load(id_files[0], mmap_mode='r').dtype

    samples = np.empty((n_total, n_features), dtype=sample_dtype)
    samples_ids = np.empty(n_total, dtype=id_dtype)

    # Copy each chunk into its slice of the pre-allocated arrays.
    pos = 0
    for s_file, i_file in tqdm(zip(sample_files, id_files), total=len(sample_files), desc="Loading chunks"):
        chunk = np.load(s_file)
        n = chunk.shape[0]
        samples[pos:pos + n] = chunk
        samples_ids[pos:pos + n] = np.load(i_file)
        pos += n
        del chunk

    print(f'model {run_number}: The number of patches extracted is: {len(samples_ids)}')
    print(f'Original feature vector dimensionality: {samples.shape[1]}')

    # Standardisation works in place, so the data must be floating point.
    # Then make sure there are no NaN/inf values (checked chunk by chunk to
    # avoid allocating a full boolean array).
    if not np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float64)
    for i in range(0, n_total, CHUNK_ROWS):
        if not np.isfinite(samples[i:i + CHUNK_ROWS]).all():
            raise ValueError("Feature vector contains NaN or infinite values; PCA cannot be applied.")

    print("Standardizing features...")
    # Z-score each feature so that power-spectrum bins and global morphology
    # parameters (very different units/ranges) contribute equally to PCA.
    standardize_inplace(samples)

    print("Applying PCA to extract highest-dispersion features...")
    samples_pca, n_keep, explained_ratio = pca_chunked(samples, n_components=0.95)
    # Free the full-width matrix as soon as the projection exists.
    del samples
    gc.collect()

    print(f"PCA complete. Reduced to {n_keep} components capturing {np.sum(explained_ratio) * 100:.2f}% variance.")
    return samples_pca, samples_ids

"""
Stage 3 of the pipeline: compress the patch set into a K-Means codebook.

There are far too many patches to run hierarchical clustering on directly, so
they are first summarised by 50,000 Mini-Batch K-Means centroids ("nodes").
Each node stands for a small region of patch-feature space; the nodes are
then grouped further by HAC in ``create_agg_model``.
"""
import numpy as np
import pandas as pd
from create_initial_features import standardize_inplace
from sklearn.cluster import MiniBatchKMeans

def reduce_dataset_kmeans(data_init, max_nodes, batch_size=200000, run_number=0):
    """Fit Mini-Batch K-Means with ``max_nodes`` centroids by repeated epochs
    of ``partial_fit`` over shuffled mini-batches, with early stopping.

    Parameters
    ----------
    data_init : ndarray or pandas.DataFrame, shape (n_samples, n_features)
        Standardised patch features.
    max_nodes : int
        Number of centroids (codebook size).
    batch_size : int
        Rows per ``partial_fit`` call.
    run_number : int or str
        Run identifier, used in log messages only.

    Returns
    -------
    ndarray, shape (max_nodes, n_features)
        The fitted cluster centres.

    Training stops after 100 epochs, or earlier once the 99th percentile of
    the per-centroid movement between consecutive epochs stays below 0.01
    for two epochs in a row.
    """
    
    print(f'Model {run_number}: Initializing Mini-Batch K-Means with {max_nodes} nodes...')
    
    mbk = MiniBatchKMeans(
        n_clusters=max_nodes,
        batch_size=batch_size,
        compute_labels=False,   # labels for the training data are not needed
        random_state=42
    )

    data_values = data_init.values if isinstance(data_init, pd.DataFrame) else data_init
    total_samples = len(data_values)
    
    # Maximum number of full passes over the data.
    epochs = 100 
    
    previous_centers = None
    no_improvement_count = 0
    
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        
        # Visit the data in a new random order every epoch.
        # (Uses the global NumPy RNG, which is not seeded here.)
        indices = np.random.permutation(total_samples)
        
        for i in range(0, total_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch = data_values[batch_idx]
            
            # One mini-batch update of the centroids.
            mbk.partial_fit(batch)
            
        # EARLY STOPPING CHECK: Measure centroid movement
        current_centers = mbk.cluster_centers_
        
        if previous_centers is not None:
            # Calculate how far each centroid moved
            shifts = np.linalg.norm(current_centers - previous_centers, axis=1)
            
            # Use the 99th percentile shift instead of absolute max to ignore 
            # a few rogue, unstable nodes on the fringes of the dataset.
            p99_center_shift = np.percentile(shifts, 99)
            
            # 0.01 is a statistically safe tolerance for Z-scored data
            if p99_center_shift < 0.01:
                no_improvement_count += 1
                print(f"Model {run_number}: Centroids stabilized (99th % shift: {p99_center_shift:.5f}).")
            else:
                no_improvement_count = 0 
                print(f"Model {run_number}: Centroids still shifting (99th % shift: {p99_center_shift:.5f}).")
                
            # Require two consecutive stable epochs before stopping.
            if no_improvement_count >= 2:
                print(f"Model {run_number}: Early stopping triggered at epoch {epoch+1}.")
                break 
        
        # Copy: cluster_centers_ is updated in place by partial_fit.
        previous_centers = np.copy(current_centers)
            
    print(f'Model {run_number}: K-Means Training Complete.')
    nodes = mbk.cluster_centers_

    return nodes

def optimized_model(run_number, samples):
    """Re-standardise the PCA features and build the K-Means codebook.

    Parameters
    ----------
    run_number : int or str
        Run identifier, used in log messages only.
    samples : ndarray, shape (n_patches, n_pca)
        PCA-projected patch features from ``create_vect``. Modified in place.

    Returns
    -------
    nodes : ndarray, shape (50000, n_pca)
        Codebook centroids.
    samples : ndarray
        The standardised patch features (same array object as the input),
        needed later to assign every patch to its nearest node.
    """

    print(f'model {run_number}: CREATING THE K-MEANS REDUCTION MODEL')

    print(f'model {run_number}: Standardizing PCA features for Euclidean distance calculations...')
    # Chunked equivalent of StandardScaler(copy=False).fit_transform(samples)
    # PCA components have very different variances (the first ones dominate);
    # scaling each to unit variance gives every component equal weight in the
    # Euclidean distances used by K-Means.
    samples, _, _ = standardize_inplace(samples)

    # Codebook size: number of K-Means centroids summarising all patches.
    max_nodes = 50000 
    
    # Increased batch size to ensure smooth statistical convergence
    batch_size = 500000

    nodes = reduce_dataset_kmeans(
        samples, 
        max_nodes=max_nodes, 
        batch_size=batch_size, 
        run_number=run_number
    )

    print(f'model {run_number}: THE REDUCTION MODEL HAS BEEN CREATED')

    return nodes, samples
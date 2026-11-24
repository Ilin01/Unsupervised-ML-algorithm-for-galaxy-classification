"""
Stage 6 of the pipeline: final morphological clustering of the galaxies.

Steps:
  1. Load the per-object TF-IDF vectors written by ``run_classify``.
  2. Reduce them with TruncatedSVD, choosing the number of dimensions
     automatically at the "knee" of the cumulative explained-variance curve.
  3. L2-normalise, so Euclidean K-Means behaves like cosine-similarity clustering.
  4. Sweep the number of clusters k from 100 to 1000 (step 10) with
     Mini-Batch K-Means and pick the k with the highest silhouette score.
  5. Fit a full K-Means with that k, compute a per-object silhouette value,
     and save the labels merged with the input catalogue.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score as score, pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

def auto_truncated_svd(feature_matrix, max_search_comps=500, min_comps=30):
    """
    Fits SVD up to max_search_comps and mathematically locates the 'knee' 
    where morphological signal flattens into background noise.
    """
    n_samples, n_features = feature_matrix.shape
    # TruncatedSVD needs n_components < min(n_samples, n_features).
    hard_limit = min(n_samples - 1, n_features - 1)
    upper_bound = min(max_search_comps, hard_limit)
    
    print(f"Scanning top {upper_bound} SVD components to locate the morphological noise floor...")
    svd = TruncatedSVD(n_components=upper_bound, random_state=42)
    # Project onto all upper_bound components once; the optimal number is
    # then just a column slice of this result.
    svd_full = svd.fit_transform(feature_matrix)
    cum_var = np.cumsum(svd.explained_variance_ratio_)
    
    # Define the curve points (the first min_comps components are always kept,
    # so the knee is searched only beyond them)
    x = np.arange(min_comps, upper_bound)
    y = cum_var[min_comps:upper_bound]
    
    # Normalize the curve to a [0, 1] bounding box to calculate geometry
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y_norm = (y - y[0]) / (y[-1] - y[0]) if (y[-1] - y[0]) > 0 else np.zeros_like(y)
    
    # The 'knee' is the point that has the maximum vertical distance from a 
    # straight diagonal line drawn from the start to the end of the curve
    deviation_from_linear = y_norm - x_norm
    optimal_k = int(x[np.argmax(deviation_from_linear)])
    
    # Slice the already-computed SVD matrix at the exact optimal dimension
    svd_optimal = svd_full[:, :optimal_k]
    retained_var = cum_var[optimal_k - 1]
    
    return svd_optimal, optimal_k, retained_var

def compute_fast_per_sample_silhouette(features, labels, chunk_size=50000):
    """Memory-safe chunked silhouette calculation for massive datasets.

    This is a centroid-based *approximation* of the silhouette: instead of the
    mean distance to all members of a cluster, it uses the distance to the
    cluster centroid. For each sample,

        a = distance to the centroid of its own cluster
        b = distance to the nearest other centroid
        s = (b - a) / max(a, b)

    which costs O(n_samples * n_clusters) instead of O(n_samples^2).

    Parameters
    ----------
    features : ndarray, shape (n_samples, n_features)
    labels : ndarray, shape (n_samples,)
    chunk_size : int
        Samples processed per distance-matrix chunk.

    Returns
    -------
    ndarray of float32, shape (n_samples,)
        Approximate silhouette value of each sample (in [-1, 1]).
    """
    n_samples = features.shape[0]
    unique_labels = np.unique(labels)
    
    # Centroid of each cluster, and labels remapped to 0..n_clusters-1 so they
    # can be used as column indices into the distance matrix.
    centroids = np.vstack([features[labels == lbl].mean(axis=0) for lbl in unique_labels])
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    mapped_labels = np.array([label_to_idx[lbl] for lbl in labels])
    
    sil_scores = np.zeros(n_samples, dtype=np.float32)
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk_data = features[start:end]
        chunk_lbls = mapped_labels[start:end]
        
        dists = pairwise_distances(chunk_data, centroids, metric='euclidean')
        # a: distance to own centroid; then mask it out with inf so the row
        # minimum gives b, the distance to the nearest *other* centroid.
        a = dists[np.arange(end - start), chunk_lbls]
        dists[np.arange(end - start), chunk_lbls] = np.inf
        b = np.min(dists, axis=1)
        
        denom = np.maximum(a, b)
        # Guard against 0/0 when a sample sits exactly on both centroids.
        chunk_sil = np.where(denom > 0, (b - a) / denom, 0.0)
        sil_scores[start:end] = chunk_sil.astype(np.float32)
        
    return sil_scores

def kmeans_model(run_number, path_model_dir, threads, object_catalogue):
    """Cluster the galaxies and attach the labels to the catalogue.

    Parameters
    ----------
    run_number : int or str
        Run identifier, used in log messages and output file names.
    path_model_dir : str
        Run folder; reads ``FINAL_FEATURE_VECT.npy`` and ``FINAL_OBJECT_IDS.npy``
        from it and writes the outputs listed below into it.
    threads : int
        Not used here (kept for a uniform stage signature).
    object_catalogue : pandas.DataFrame
        Filtered input catalogue; must contain an ``object_id`` column.

    Returns
    -------
    pandas.DataFrame
        Catalogue rows of the clustered objects (inner join on ``object_id``)
        with two extra columns: ``labels`` (final cluster) and
        ``silhouette_score`` (per-object approximate silhouette).

    Files written
    -------------
    silhouette_curve_model_<run>.png      silhouette score vs. k
    id_label_featureVect_model_<run>.npy  [object_id, label, silhouette, TF-IDF vector...]
    final_morph_clust<run>.npz            merged catalogue, one array per column
    """

    print(f'model {run_number}: THE FINAL MORPHOLOGICAL CLUSTERS ARE BEING CREATED')

    FINAL_OBJECT_IDS = np.load(os.path.join(path_model_dir, 'FINAL_OBJECT_IDS.npy'))
    FINAL_FEATURE_VECT = np.load(os.path.join(path_model_dir, 'FINAL_FEATURE_VECT.npy'))

    # --- SVD Compression via Auto-Knee Detection ---
    print(f'model {run_number}: Compressing sparse TF-IDF features using Auto-Tuned TruncatedSVD...')
    FINAL_FEATURE_VECT_COMPRESSED, optimal_dims, explained_var = auto_truncated_svd(FINAL_FEATURE_VECT)
    
    print(f'Auto-SVD detected structural noise floor after {optimal_dims} dimensions.')
    print(f'SVD retained {explained_var*100:.1f}% of core morphological variance in {optimal_dims} dimensions.')

    # L2 Normalization ensures K-Means (Euclidean) clusters exactly like Cosine similarity
    print(f'model {run_number}: L2-Normalizing features to align K-Means with Cosine metrics...')
    normalizer = Normalizer(norm='l2', copy=False)
    FINAL_FEATURE_VECT_COMPRESSED = normalizer.fit_transform(FINAL_FEATURE_VECT_COMPRESSED)

    # --- Automated Silhouette Sweep for Optimal Clusters ---
    # Candidate numbers of clusters: 150, 160, ..., 500.
    lower_clust_lim = 150
    higher_clust_lim = 500
    step = 10
    cluster_counts = np.arange(lower_clust_lim, higher_clust_lim + step, step)
    silhouette_scores = []

    print(f"Running automated Silhouette Score sweep ({lower_clust_lim} to {higher_clust_lim})...")
    for k in cluster_counts:
        # Fast approximate clustering for the sweep...
        sweep_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto', batch_size=2048)
        sweep_labels = sweep_kmeans.fit_predict(FINAL_FEATURE_VECT_COMPRESSED)
        
        # ...scored on a random subsample of 10,000 objects (the exact
        # silhouette is O(n^2) and would be too slow for every k).
        score_val = score(
            FINAL_FEATURE_VECT_COMPRESSED, 
            sweep_labels, 
            metric='euclidean',
            sample_size=10000, 
            random_state=42
        )
        silhouette_scores.append(score_val)

        print(f'model {run_number}: clusters: {k} | Silhouette Score: {score_val:.4f}')

    # --- Programmatic Maximum Detection ---
    optimal_idx = np.argmax(silhouette_scores)
    optimal_clusters = cluster_counts[optimal_idx]
    
    print(f'\n>>> AUTOMATED MICRO-CLUSTER TUNING RESULT <<<')
    print(f'The absolute peak of mathematical cluster cohesion found at: {optimal_clusters} clusters.')

    # --- Save Plot of the Silhouette Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, silhouette_scores, marker='o', linestyle='-', color='b', label='Silhouette Score')
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Optimal k={optimal_clusters}')
    plt.title('Silhouette Score For Optimal Micro-Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score (Higher is Better)')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(path_model_dir, f'silhouette_curve_model_{run_number}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Silhouette curve plot to: {plot_path}")

    # --- Final Clustering ---
    print(f"Running final highly-precise KMeans model with {optimal_clusters} clusters...")
    # Full (non-mini-batch) K-Means for the final, more accurate labels.
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')
    clust2_labels = kmeans.fit_predict(FINAL_FEATURE_VECT_COMPRESSED)

    print("Calculating final silhouette metrics...")
    try:
        # Overall model score (subsampled exact silhouette)...
        sil_score_for_model = score(
            FINAL_FEATURE_VECT_COMPRESSED, 
            clust2_labels, 
            metric='euclidean', 
            sample_size=20000, 
            random_state=42
        )
        print(f'model {run_number}: Final silhouette score at optimal k: {sil_score_for_model:.3f}')
        
        # ...and a fast centroid-based silhouette for every object, used to
        # rank objects within each cluster in the montages.
        sil_scores_per_sample = compute_fast_per_sample_silhouette(FINAL_FEATURE_VECT_COMPRESSED, clust2_labels)
    except Exception as e:
        print(f"CRITICAL: Failed calculating silhouette metrics: {e}. Falling back to zeros.")
        sil_scores_per_sample = np.zeros(len(FINAL_OBJECT_IDS))

    # Rows of FINAL_FEATURE_VECT, FINAL_OBJECT_IDS and clust2_labels are aligned.
    df2 = pd.DataFrame({
        'object_id': FINAL_OBJECT_IDS,
        'labels': clust2_labels,
        'silhouette_score': sil_scores_per_sample
    })

    # Save ID, label, silhouette and the full (uncompressed) TF-IDF vector.
    feature_df = pd.DataFrame(FINAL_FEATURE_VECT)
    df3 = pd.concat([df2, feature_df], axis=1)
    
    np.save(os.path.join(path_model_dir, f'id_label_featureVect_model_{run_number}.npy'), df3.values)

    # Attach the catalogue columns (mass, redshift, tile_index, ...). Objects
    # that produced no patches have no label and are dropped by the inner join.
    obj_catalogue = pd.merge(df2, object_catalogue, on="object_id")

    np.savez_compressed(os.path.join(path_model_dir, f'final_morph_clust{run_number}.npz'),
        **{col: obj_catalogue[col].values for col in obj_catalogue.columns})

    print(f'model {run_number}: THE FINAL CLUSTERING IS FINISHED')

    return obj_catalogue
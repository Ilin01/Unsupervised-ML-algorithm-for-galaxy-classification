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
    hard_limit = min(n_samples - 1, n_features - 1)
    upper_bound = min(max_search_comps, hard_limit)
    
    print(f"Scanning top {upper_bound} SVD components to locate the morphological noise floor...")
    svd = TruncatedSVD(n_components=upper_bound, random_state=42)
    svd_full = svd.fit_transform(feature_matrix)
    cum_var = np.cumsum(svd.explained_variance_ratio_)
    
    # Define the curve points
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
    """Memory-safe chunked silhouette calculation for massive datasets."""
    n_samples = features.shape[0]
    unique_labels = np.unique(labels)
    
    centroids = np.vstack([features[labels == lbl].mean(axis=0) for lbl in unique_labels])
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    mapped_labels = np.array([label_to_idx[lbl] for lbl in labels])
    
    sil_scores = np.zeros(n_samples, dtype=np.float32)
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk_data = features[start:end]
        chunk_lbls = mapped_labels[start:end]
        
        dists = pairwise_distances(chunk_data, centroids, metric='euclidean')
        a = dists[np.arange(end - start), chunk_lbls]
        dists[np.arange(end - start), chunk_lbls] = np.inf
        b = np.min(dists, axis=1)
        
        denom = np.maximum(a, b)
        chunk_sil = np.where(denom > 0, (b - a) / denom, 0.0)
        sil_scores[start:end] = chunk_sil.astype(np.float32)
        
    return sil_scores

def kmeans_model(run_number, path_model_dir, threads, object_catalogue):

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
    lower_clust_lim = 100
    higher_clust_lim = 1000
    step = 10
    cluster_counts = np.arange(lower_clust_lim, higher_clust_lim + step, step)
    silhouette_scores = []

    print(f"Running automated Silhouette Score sweep ({lower_clust_lim} to {higher_clust_lim})...")
    for k in cluster_counts:
        sweep_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto', batch_size=2048)
        sweep_labels = sweep_kmeans.fit_predict(FINAL_FEATURE_VECT_COMPRESSED)
        
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
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')
    clust2_labels = kmeans.fit_predict(FINAL_FEATURE_VECT_COMPRESSED)

    print("Calculating final silhouette metrics...")
    try:
        sil_score_for_model = score(
            FINAL_FEATURE_VECT_COMPRESSED, 
            clust2_labels, 
            metric='euclidean', 
            sample_size=20000, 
            random_state=42
        )
        print(f'model {run_number}: Final silhouette score at optimal k: {sil_score_for_model:.3f}')
        
        sil_scores_per_sample = compute_fast_per_sample_silhouette(FINAL_FEATURE_VECT_COMPRESSED, clust2_labels)
    except Exception as e:
        print(f"CRITICAL: Failed calculating silhouette metrics: {e}. Falling back to zeros.")
        sil_scores_per_sample = np.zeros(len(FINAL_OBJECT_IDS))

    df2 = pd.DataFrame({
        'object_id': FINAL_OBJECT_IDS,
        'labels': clust2_labels,
        'silhouette_score': sil_scores_per_sample
    })

    feature_df = pd.DataFrame(FINAL_FEATURE_VECT)
    df3 = pd.concat([df2, feature_df], axis=1)
    
    np.save(os.path.join(path_model_dir, f'id_label_featureVect_model_{run_number}.npy'), df3.values)

    obj_catalogue = pd.merge(df2, object_catalogue, on="object_id")

    np.savez_compressed(os.path.join(path_model_dir, f'final_morph_clust{run_number}.npz'),
        **{col: obj_catalogue[col].values for col in obj_catalogue.columns})

    print(f'model {run_number}: THE FINAL CLUSTERING IS FINISHED')

    return obj_catalogue
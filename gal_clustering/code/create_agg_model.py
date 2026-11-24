"""
Stage 4 of the pipeline: build the "visual word" vocabulary of patch types.

The 50,000 K-Means nodes are grouped into ``nr_hac_clusters`` clusters with
hierarchical agglomerative clustering (Ward linkage, sklearn's default). Every
patch is then labelled with the HAC cluster of its nearest node. These labels
are the "visual words" used to describe each galaxy in the next stage.
"""
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin

def agg_model(run_number, threads, nr_hac_clusters, nodes, scaled_samples, samples_ids):
    """Cluster the codebook nodes with HAC and label every patch.

    Parameters
    ----------
    run_number : int or str
        Run identifier, used in log messages only.
    threads : int
        Not used here (kept for a uniform stage signature).
    nr_hac_clusters : int
        Number of HAC clusters, i.e. the vocabulary size.
    nodes : ndarray, shape (n_nodes, n_features)
        K-Means codebook centroids.
    scaled_samples : ndarray, shape (n_patches, n_features)
        Standardised patch features, in the same space as ``nodes``.
    samples_ids : ndarray
        Not used here.

    Returns
    -------
    clust1_labels : ndarray of int32, shape (n_patches,)
        Visual-word label of every patch.
    n_clusters : int
        Number of HAC clusters actually produced.
    """

    print(f'model {run_number}: AGG CLUSTERING HAS STARTED')

    # Ward HAC needs O(n^2) memory in the number of nodes.
    if len(nodes) > 50000:
        print(f"WARNING: You are passing {len(nodes)} nodes to HAC. Keep an eye on system memory.")

    # 1. Cluster the 50,000 nodes into a smaller, cleaner set of morphological "words"
    clusterer = AgglomerativeClustering(n_clusters=nr_hac_clusters)
    node_labels = clusterer.fit_predict(nodes)

    # 2. Predict labels for the original dataset
    clust1_labels = np.empty(len(scaled_samples), dtype=np.int32)
    # Process patches in batches so the patch-to-node distance matrix
    # (batch_size x n_nodes) stays a manageable size.
    batch_size = 100000 
    
    for i in tqdm(range(0, len(scaled_samples), batch_size), desc="Assigning Labels"):
        batch = scaled_samples[i : i + batch_size]
        
        # Find the index of the closest node for each patch in the batch
        closest_node_indices = pairwise_distances_argmin(batch, nodes)
        
        # Map that node index to the overarching HAC label
        clust1_labels[i : i + batch_size] = node_labels[closest_node_indices]

    print(f'model {run_number}: AGG CLUSTERING HAS FINISHED')

    return clust1_labels, clusterer.n_clusters_
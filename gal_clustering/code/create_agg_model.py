import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin

def agg_model(run_number, threads, nr_hac_clusters, nodes, scaled_samples, samples_ids):

    print(f'model {run_number}: AGG CLUSTERING HAS STARTED')

    if len(nodes) > 30000:
        print(f"WARNING: You are passing {len(nodes)} nodes to HAC. Keep an eye on system memory.")

    # 1. Cluster the 50,000 nodes into a smaller, cleaner set of morphological "words"
    clusterer = AgglomerativeClustering(n_clusters=nr_hac_clusters)
    node_labels = clusterer.fit_predict(nodes)

    # 2. Predict labels for the massive original dataset
    clust1_labels = np.empty(len(scaled_samples), dtype=np.int32)
    batch_size = 100000 
    
    for i in tqdm(range(0, len(scaled_samples), batch_size), desc="Assigning Labels"):
        batch = scaled_samples[i : i + batch_size]
        
        # Find the index of the closest node for each patch in the batch
        closest_node_indices = pairwise_distances_argmin(batch, nodes)
        
        # Map that node index to the overarching HAC label
        clust1_labels[i : i + batch_size] = node_labels[closest_node_indices]

    print(f'model {run_number}: AGG CLUSTERING HAS FINISHED')

    return clust1_labels, clusterer.n_clusters_
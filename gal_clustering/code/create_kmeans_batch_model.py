import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

def reduce_dataset_kmeans(data_init, max_nodes, batch_size=200000, run_number=0):
    
    print(f'Model {run_number}: Initializing Mini-Batch K-Means with {max_nodes} nodes...')
    
    mbk = MiniBatchKMeans(
        n_clusters=max_nodes,
        batch_size=batch_size,
        compute_labels=False, 
        random_state=42
    )

    data_values = data_init.values if isinstance(data_init, pd.DataFrame) else data_init
    total_samples = len(data_values)
    
    epochs = 100 
    
    previous_centers = None
    no_improvement_count = 0
    
    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} ---")
        
        indices = np.random.permutation(total_samples)
        
        for i in range(0, total_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch = data_values[batch_idx]
            
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
                
            if no_improvement_count >= 2:
                print(f"Model {run_number}: Early stopping triggered at epoch {epoch+1}.")
                break 
        
        previous_centers = np.copy(current_centers)
            
    print(f'Model {run_number}: K-Means Training Complete.')
    nodes = mbk.cluster_centers_

    return nodes

def optimized_model(run_number, samples):

    print(f'model {run_number}: CREATING THE K-MEANS REDUCTION MODEL')

    print(f'model {run_number}: Standardizing PCA features for Euclidean distance calculations...')
    scaler = StandardScaler(copy=False)
    samples = scaler.fit_transform(samples)

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
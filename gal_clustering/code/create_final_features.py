import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

def final_features(run_number, samples_ids, threads, clust1_labels, nr_clust):

    print(f'model {run_number}: THE FINAL FEATURE VECTOR IS BEING CREATED')

    # 1. Map your original Object IDs to contiguous integers (0 to N-1)
    # unique_ids is the final sorted list of objects.
    # mapped_ids replaces the original IDs with their index position in unique_ids.
    unique_ids, mapped_ids = np.unique(samples_ids, return_inverse=True)
    num_unique_objects = len(unique_ids)

    # 2. Create a "flat index" for a virtual 2D grid
    # Each row is an object, each column is a cluster label.
    flat_indices = mapped_ids * nr_clust + clust1_labels

    # 3. Fast C-level counting of occurrences
    # This replaces the entire Multiprocessing Pool and DataFrame filtering loop
    counts_flat = np.bincount(flat_indices, minlength=num_unique_objects * nr_clust)

    # 4. Reshape the flat counts back into a raw 2D count matrix (Objects x Clusters)
    counts_2d = counts_flat.reshape(num_unique_objects, nr_clust).astype(np.float32)

    # 5. --- NEW: TF-IDF & L2 Normalization ---
    print(f'model {run_number}: Applying TF-IDF weighting and L2 Normalization...')
    
    # sublinear_tf=True applies logarithmic scaling to the frequencies. 
    # This ensures a galaxy with 100 spiral-arm patches isn't weighted 10x 
    # more aggressively than a galaxy with 10 spiral-arm patches.
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    
    # Fit and transform the raw counts, then convert back to a dense NumPy array
    final_feature_vect = tfidf.fit_transform(counts_2d).toarray()

    print(f'model {run_number}: FINAL FEATURE VECTOR CREATED')

    return final_feature_vect, unique_ids
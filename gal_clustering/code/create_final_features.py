import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

def final_features(run_number, samples_ids, threads, clust1_labels, nr_clust):
    print(f'model {run_number}: THE FINAL FEATURE VECTOR IS BEING CREATED')

    unique_ids, mapped_ids = np.unique(samples_ids, return_inverse=True)
    num_unique_objects = len(unique_ids)

    flat_indices = mapped_ids * nr_clust + clust1_labels
    counts_flat = np.bincount(flat_indices, minlength=num_unique_objects * nr_clust)
    counts_2d = counts_flat.reshape(num_unique_objects, nr_clust).astype(np.float32)

    print(f'model {run_number}: Applying TF-IDF weighting and L2 Normalization...')
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    final_feature_vect = tfidf.fit_transform(counts_2d).toarray()

    print(f'model {run_number}: FINAL FEATURE VECTOR CREATED')
    return final_feature_vect, unique_ids
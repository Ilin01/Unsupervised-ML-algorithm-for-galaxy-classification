"""
Stage 5 of the pipeline: describe every galaxy as a bag of visual words.

Each galaxy is represented by a histogram counting how many of its patches
fall into each visual word (HAC cluster). The histograms are then TF-IDF
weighted, as in text retrieval: words that occur in almost every galaxy are
down-weighted, and distinctive words are emphasised.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

def final_features(run_number, samples_ids, threads, clust1_labels, nr_clust):
    """Build the TF-IDF bag-of-visual-words matrix.

    Parameters
    ----------
    run_number : int or str
        Run identifier, used in log messages only.
    samples_ids : ndarray, shape (n_patches,)
        Object ID of each patch.
    threads : int
        Not used here (kept for a uniform stage signature).
    clust1_labels : ndarray of int, shape (n_patches,)
        Visual-word label (0 .. nr_clust-1) of each patch.
    nr_clust : int
        Vocabulary size.

    Returns
    -------
    final_feature_vect : ndarray, shape (n_objects, nr_clust)
        L2-normalised TF-IDF vector per object.
    unique_ids : ndarray, shape (n_objects,)
        Object ID of each row (sorted, as returned by ``np.unique``).
    """
    print(f'model {run_number}: THE FINAL FEATURE VECTOR IS BEING CREATED')

    # Map each patch's object ID to a row index 0 .. n_objects-1.
    unique_ids, mapped_ids = np.unique(samples_ids, return_inverse=True)
    num_unique_objects = len(unique_ids)

    # Vectorised 2D histogram: encode (object row, word) as one flat index,
    # count occurrences with bincount, then reshape to (n_objects, nr_clust).
    flat_indices = mapped_ids * nr_clust + clust1_labels
    counts_flat = np.bincount(flat_indices, minlength=num_unique_objects * nr_clust)
    counts_2d = counts_flat.reshape(num_unique_objects, nr_clust).astype(np.float32)

    print(f'model {run_number}: Applying TF-IDF weighting and L2 Normalization...')
    # sublinear_tf: use 1 + log(count) so a word repeated in many patches of a
    # big galaxy does not dominate; norm='l2': each object's vector has unit
    # length, removing the dependence on galaxy size (number of patches).
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    final_feature_vect = tfidf.fit_transform(counts_2d).toarray()

    print(f'model {run_number}: FINAL FEATURE VECTOR CREATED')
    return final_feature_vect, unique_ids
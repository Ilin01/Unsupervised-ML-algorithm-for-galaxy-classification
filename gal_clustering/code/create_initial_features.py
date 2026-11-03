import os
import glob
import numpy as np
import gc # Added for explicit garbage collection
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_vect(run_number, path_model_dir, VIS_band_only):

    print(f'model {run_number}: CREATION OF INITIAL FEATURE VECTOR ...')

    # --- 1. LOAD CHUNKED FILES ---
    patch_dir = os.path.join(path_model_dir, 'patches')
    sample_files = sorted(glob.glob(os.path.join(patch_dir, 'samples_chunk_*.npy')))
    id_files = sorted(glob.glob(os.path.join(patch_dir, 'samples_object_id_chunk_*.npy')))
    
    if len(sample_files) == 0:
        raise FileNotFoundError(f"No chunk files found in {patch_dir}.")

    print(f"Found {len(sample_files)} chunks. Loading and concatenating...")
    
    samples_list = []
    ids_list = []
    
    for s_file, i_file in tqdm(zip(sample_files, id_files), total=len(sample_files), desc="Loading chunks"):
        samples_list.append(np.load(s_file))
        ids_list.append(np.load(i_file))
        
    samples = np.vstack(samples_list)
    samples_ids = np.concatenate(ids_list, axis=0)
    
    # Aggressive memory cleanup
    del samples_list
    del ids_list
    gc.collect() 
    
    print(f'model {run_number}: The number of patches extracted is: {len(samples_ids)}')
    print(f'Original feature vector dimensionality: {samples.shape[1]}')

    # --- 2. STANDARDIZATION ---
    print("Standardizing features...")
    # copy=False applies the standardization in-place, saving massive amounts of RAM
    scaler = StandardScaler(copy=False) 
    samples_scaled = scaler.fit_transform(samples)
    
    # 'samples' and 'samples_scaled' now point to the same memory block, 
    # so we don't need to delete 'samples' separately.

    # --- 3. PCA COMPRESSION ---
    print("Applying PCA to extract highest-dispersion features...")
    
    # Retain 95% of the variance rather than a hardcoded number of components.
    # This automatically adapts to single-band vs multi-band dimensionality.
    pca = PCA(n_components=0.95, random_state=42)
    samples_pca = pca.fit_transform(samples_scaled)
    
    n_components_kept = pca.n_components_
    explained_variance = np.sum(pca.explained_variance_ratio_)
    
    print(f"PCA complete.")
    print(f"Reduced from {samples_scaled.shape[1]} to {n_components_kept} components.")
    print(f"These {n_components_kept} components capture {explained_variance * 100:.2f}% of the variance.")

    # We return the PCA features regardless of VIS_band_only because the 
    # concatenated multi-scale vectors require dimensionality reduction.
    print(f'model {run_number}: Vector creation complete.')
    
    return samples_pca, samples_ids
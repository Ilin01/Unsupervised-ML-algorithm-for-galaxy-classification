import os
import glob
import numpy as np
import gc
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_vect(run_number, path_model_dir):
    print(f'model {run_number}: CREATION OF INITIAL FEATURE VECTOR ...')

    patch_dir = os.path.join(path_model_dir, 'patches')
    sample_files = sorted(glob.glob(os.path.join(patch_dir, 'samples_chunk_*.npy')))
    id_files = sorted(glob.glob(os.path.join(patch_dir, 'samples_object_id_chunk_*.npy')))
    
    if len(sample_files) == 0:
        raise FileNotFoundError(f"No chunk files found in {patch_dir}.")

    samples_list, ids_list = [], []
    for s_file, i_file in tqdm(zip(sample_files, id_files), total=len(sample_files), desc="Loading chunks"):
        samples_list.append(np.load(s_file))
        ids_list.append(np.load(i_file))
        
    samples = np.vstack(samples_list)
    samples_ids = np.concatenate(ids_list, axis=0)
    
    del samples_list, ids_list
    gc.collect() 
    
    print(f'model {run_number}: The number of patches extracted is: {len(samples_ids)}')
    print(f'Original feature vector dimensionality: {samples.shape[1]}')

    print("Standardizing features...")
    scaler = StandardScaler(copy=False) 
    samples_scaled = scaler.fit_transform(samples)

    print("Applying PCA to extract highest-dispersion features...")
    pca = PCA(n_components=0.95, random_state=42)
    samples_pca = pca.fit_transform(samples_scaled)
    
    print(f"PCA complete. Reduced to {pca.n_components_} components capturing {np.sum(pca.explained_variance_ratio_) * 100:.2f}% variance.")
    return samples_pca, samples_ids
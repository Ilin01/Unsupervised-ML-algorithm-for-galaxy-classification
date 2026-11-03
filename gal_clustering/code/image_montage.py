import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# Helper function to prevent negative flux artifacts safely
def clip_negatives(band):
    band[band < 0] = 0.0
    return band

def make_plot_lupton(images_data, objectID, silhouette_score, ax):
    scale = [0.5, 0.7, 1.0]

    image_r = np.copy(images_data[0])
    image_g = np.copy(images_data[1])
    image_b = np.copy(images_data[2])

    image_r[~np.isfinite(image_r)] = 0.
    image_g[~np.isfinite(image_g)] = 0.
    image_b[~np.isfinite(image_b)] = 0.

    image = make_lupton_rgb(image_r * scale[0], image_g * scale[1], image_b * scale[2], Q=10, stretch=1.8)

    ax.imshow(image, aspect='equal', origin='lower')
    ax.set_axis_off()
    ax.set_title(f"{objectID}\nSil: {silhouette_score:.3f}", fontsize=10)

def make_plot_mono(images_data, objectID, silhouette_score, ax):
    image_mono = np.copy(images_data[0])
    image_mono[~np.isfinite(image_mono)] = 0.

    image_stretched = make_lupton_rgb(image_mono, image_mono, image_mono, Q=10, stretch=1.8)
    image_inverted = 255 - image_stretched

    ax.imshow(image_inverted, aspect='equal', origin='lower')
    ax.set_axis_off()
    ax.set_title(f"{objectID}\nSil: {silhouette_score:.3f}", fontsize=10)

def make_plot_unsharp(images_data, objectID, silhouette_score, ax, kernel_fraction):
    """
    kernel_fraction: 0.6 for Large Kernel, 0.15 for Small Kernel 
    """
    # Adjust scaling - unsharp masking heavily amplifies noise, 
    # so we back off the RGB multipliers slightly to prevent blowing out the cores.
    scale = [0.4, 0.6, 0.9]

    # Use float32 to save RAM during the heavy convolutions
    image_r = np.copy(images_data[0]).astype(np.float32)
    image_g = np.copy(images_data[1]).astype(np.float32)
    image_b = np.copy(images_data[2]).astype(np.float32)

    image_r[~np.isfinite(image_r)] = 0.
    image_g[~np.isfinite(image_g)] = 0.
    image_b[~np.isfinite(image_b)] = 0.

    Ymax, Xmax = image_r.shape
    max_dim = max(Xmax, Ymax)
    # Calculate the physical sigma once
    computed_sigma = max(1.0, max_dim * kernel_fraction) 

    if kernel_fraction==0.02:
        computed_sigma=4

    def apply_unsharp(band, sig):
        # 1. Blur
        blurred = gaussian_filter(band, sigma=sig)
        # 2. Subtract
        sharpened = (2.0 * band) - blurred
        # 3. Aggressive clip to zero to prevent Lupton RGB math errors
        sharpened = np.clip(sharpened, 0.0, None)
        return sharpened

    # Apply the mask
    r_unsharp = apply_unsharp(image_r, computed_sigma)
    g_unsharp = apply_unsharp(image_g, computed_sigma)
    b_unsharp = apply_unsharp(image_b, computed_sigma)

    # Use a lower Q and stretch parameter because the background is already gone.
    # If the stretch is too high on an unsharp image, you just see amplified sensor noise.
    image = make_lupton_rgb(
        r_unsharp * scale[0], 
        g_unsharp * scale[1], 
        b_unsharp * scale[2], 
        Q=5, 
        stretch=1.2
    )

    ax.imshow(image, aspect='equal', origin='lower')
    ax.set_axis_off()
    ax.set_title(f"{objectID}\nSil: {silhouette_score:.3f}", fontsize=10)

    
def process_single_cluster(cluster_data, path_rgb, path_mono, path_unsharp_large, path_unsharp_small, path_npy_dir):
    cluster_id, df_cluster = cluster_data
    total_objects = len(df_cluster)
    
    # 1. Setup Data
    if total_objects == 1:
        df_plot = df_cluster
        rows, cols = 1, 1
    elif total_objects >= 30:
        df_sorted = df_cluster.sort_values('silhouette_score', ascending=False)
        df_plot = pd.concat([df_sorted.head(10), df_sorted.iloc[10:-10].sample(n=10) if len(df_sorted) > 20 else pd.DataFrame(), df_sorted.tail(10)])
        rows, cols = 5, 6
    else:
        df_plot = df_cluster.sort_values('silhouette_score', ascending=False)
        cols = int(np.ceil(np.sqrt(total_objects)))
        rows = int(np.ceil(total_objects / cols))

    # 2. Setup 4 separate Figures safely
    fig_rgb, axs_rgb = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig_mono, axs_mono = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig_ul, axs_ul = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig_us, axs_us = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # Ensure array format for flat iteration
    axs_rgb = np.atleast_1d(axs_rgb)
    axs_mono = np.atleast_1d(axs_mono)
    axs_ul = np.atleast_1d(axs_ul)
    axs_us = np.atleast_1d(axs_us)

    objectIDs = df_plot['object_id'].values.astype(str)
    sil_scores = np.round(df_plot['silhouette_score'].values, 3)
    tile_indices = df_plot['tile_index'].values

    # 3. Plotting Loop
    for i, (obj_id, sil_score, tile_idx) in enumerate(zip(objectIDs, sil_scores, tile_indices)):
        npy_path = os.path.join(path_npy_dir, f'tile_{tile_idx}', f"{obj_id}.npy")

        if not os.path.exists(npy_path):
            for ax in (axs_rgb.flat[i], axs_mono.flat[i], axs_ul.flat[i], axs_us.flat[i]):
                ax.set_axis_off()
            continue
            
        images_data = np.load(npy_path, allow_pickle=True)[1:][::-1]
        
        make_plot_lupton(images_data, obj_id, sil_score, axs_rgb.flat[i])
        make_plot_mono(images_data, obj_id, sil_score, axs_mono.flat[i])
        
        # Pass the kernel fractions corresponding to extraction scales
        make_plot_unsharp(images_data, obj_id, sil_score, axs_ul.flat[i], kernel_fraction=0.3)  
        make_plot_unsharp(images_data, obj_id, sil_score, axs_us.flat[i], kernel_fraction=0.1) 

    # 4. Hide empty subplots
    for i in range(len(objectIDs), len(axs_rgb.flat)):
        axs_rgb.flat[i].set_axis_off()
        axs_mono.flat[i].set_axis_off()
        axs_ul.flat[i].set_axis_off()
        axs_us.flat[i].set_axis_off()

    # 5. Save and aggressively clear memory
    title_text = f'Cluster: {cluster_id} | Total Objects: {total_objects}'
    
    def save_and_clear(fig, path, suffix, title):
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        fig.savefig(os.path.join(path, f"cluster_{cluster_id}_{suffix}.png"))
        fig.clf() # Critical for memory
        plt.close(fig)

    save_and_clear(fig_rgb, path_rgb, "rgb", title_text + " (Lupton RGB)")
    save_and_clear(fig_mono, path_mono, "mono", title_text + " (Monochrome)")
    save_and_clear(fig_ul, path_unsharp_large, "unsharp_large", title_text + " (Unsharp: Large Kernel 0.6x)")
    save_and_clear(fig_us, path_unsharp_small, "unsharp_small", title_text + " (Unsharp: Small Kernel 0.15x)")


def image_montage(run_number, threads, path_data_dir, path_model_dir, obj_catalogue):

    print(f'model {run_number}: THE IMAGE MONTAGE IS BEING PRODUCED')

    # Setup 4 distinct output directories
    paths = {
        'path_rgb': os.path.join(path_model_dir, f'montage_{run_number}_rgb'),
        'path_mono': os.path.join(path_model_dir, f'montage_{run_number}_mono'),
        'path_unsharp_large': os.path.join(path_model_dir, f'montage_{run_number}_unsharp_large'),
        'path_unsharp_small': os.path.join(path_model_dir, f'montage_{run_number}_unsharp_small')
    }
    
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    grouped_clusters = list(obj_catalogue.groupby('labels'))

    worker_func = partial(
        process_single_cluster, 
        **paths,
        path_npy_dir=path_data_dir 
    )
    
    with Pool(threads) as pool:
        list(tqdm(pool.imap_unordered(worker_func, grouped_clusters), total=len(grouped_clusters)))

    print(f'model {run_number}: THE IMAGE MONTAGE IS FINISHED')
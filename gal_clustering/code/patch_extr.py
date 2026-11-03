import os
import glob
import time
import numpy as np
import pandas as pd
import numpy.fft as npfft
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def get_global_morphology(imgs):
    """
    Extracts CAS, radially binned m=2, a4, axis ratio, M20, Centroid Shift (D0), Ring Index (R_I),
    and a Color Gradient Index (Inner H-band / Outer VIS-band) to isolate lensed systems.
    Returns: [C, A, a4, m2_inner, m2_outer, axis_ratio, M20, D0, R_I, color_grad]
    """
    # Base structural morphology on the VIS band (bluest/highest resolution)
    img = imgs[0]
    total_light = np.sum(img)
    if total_light <= 0:
        return np.zeros(10, dtype=np.float32)

    y, x = np.indices(img.shape)
    
    cx = np.sum(img * x) / total_light
    cy = np.sum(img * y) / total_light
    
    # 1. Asymmetry (A)
    img_180 = np.rot90(img, 2)
    A = np.sum(np.abs(img - img_180)) / total_light
    
    # 2. Concentration (C) & Radii
    r = np.hypot(x - cx, y - cy)
    r_sorted_idx = np.argsort(r.flat)
    sorted_img_r = img.flat[r_sorted_idx]
    sorted_r = r.flat[r_sorted_idx]
    cumsum_r = np.cumsum(sorted_img_r)
    
    r20_idx = np.searchsorted(cumsum_r, 0.2 * total_light, side='right')
    r50_idx = np.searchsorted(cumsum_r, 0.5 * total_light, side='right')
    r80_idx = np.searchsorted(cumsum_r, 0.8 * total_light, side='right')
    
    r20 = sorted_r[r20_idx] if r20_idx < len(sorted_r) else 1e-5
    r50 = sorted_r[r50_idx] if r50_idx < len(sorted_r) else 1e-5
    r80 = sorted_r[r80_idx] if r80_idx < len(sorted_r) else 1e-5
    
    C = 5.0 * np.log10(r80 / r20) if r20 > 0 else 0.0
    
    # 3. M20 (Second-order moment of brightest 20% of pixels)
    sorted_idx_desc = np.argsort(img.flat)[::-1]
    sorted_img_desc = img.flat[sorted_idx_desc]
    cumsum_desc = np.cumsum(sorted_img_desc)
    
    idx_20 = np.searchsorted(cumsum_desc, 0.2 * total_light, side='right')
    idx_20 = max(1, min(idx_20, len(sorted_img_desc)))
    brightest_20_idx = sorted_idx_desc[:idx_20]
    
    M_tot = np.sum(img * r**2)
    M_20_val = np.sum(img.flat[brightest_20_idx] * r.flat[brightest_20_idx]**2)
    M20 = np.log10(M_20_val / M_tot) if M_tot > 0 and M_20_val > 0 else 0.0
    
    # 4. Centroid Shift (D0)
    ymax, xmax = np.unravel_index(np.argmax(img), img.shape)
    D0 = np.hypot(xmax - cx, ymax - cy) / r50
    
    # 5. Ring Index (R_I)
    num_bins = 20
    bins = np.linspace(0, r80 * 1.5, num_bins + 1)
    radial_prof, _ = np.histogram(r, bins, weights=img)
    area_prof, _ = np.histogram(r, bins)
    area_prof[area_prof == 0] = 1 
    mean_surface_brightness = radial_prof / area_prof
    
    valid_bins = bins[:-1] > r20
    if np.any(valid_bins[:-1]):
        gradients = np.diff(mean_surface_brightness)
        valid_gradients = gradients[valid_bins[:-1]]
        if len(valid_gradients) > 0 and np.mean(mean_surface_brightness) > 0:
            R_I = np.max(valid_gradients) / np.mean(mean_surface_brightness)
        else:
            R_I = 0.0
    else:
        R_I = 0.0
    R_I = max(0.0, float(R_I))
    
    # 6. Radially Binned m=2 Fourier Modes
    theta = np.arctan2(y - cy, x - cx)
    inner_mask = r <= r50
    outer_mask = r > r50
    
    light_inner = np.sum(img[inner_mask]) + 1e-5
    light_outer = np.sum(img[outer_mask]) + 1e-5
    
    m2_real_in = np.sum(img[inner_mask] * np.cos(2 * theta[inner_mask]))
    m2_imag_in = np.sum(img[inner_mask] * np.sin(2 * theta[inner_mask]))
    m2_inner = np.hypot(m2_real_in, m2_imag_in) / light_inner
    
    m2_real_out = np.sum(img[outer_mask] * np.cos(2 * theta[outer_mask]))
    m2_imag_out = np.sum(img[outer_mask] * np.sin(2 * theta[outer_mask]))
    m2_outer = np.hypot(m2_real_out, m2_imag_out) / light_outer
    
    # 7. Image Moments (for a4 and Axis Ratio)
    m20_img = np.sum(img * (x - cx)**2) / total_light
    m02_img = np.sum(img * (y - cy)**2) / total_light
    m11_img = np.sum(img * (x - cx) * (y - cy)) / total_light
    
    theta_0 = 0.5 * np.arctan2(2 * m11_img, m20_img - m02_img) if (m20_img - m02_img) != 0 else 0.0
    aligned_theta = theta - theta_0
    
    a4 = np.sum(img * np.cos(4 * aligned_theta)) / total_light

    term1 = (m20_img + m02_img) / 2.0
    term2 = np.sqrt(((m20_img - m02_img) / 2.0)**2 + m11_img**2)
    a_sq = term1 + term2
    b_sq = term1 - term2
    
    a_val = np.sqrt(max(a_sq, 1e-10))
    b_val = np.sqrt(max(b_sq, 1e-10))
    axis_ratio = b_val / a_val

    # 8. Color Gradient Index (Inner Red vs Outer Blue) for Lensing detection
    # VIS is index 0 (bluest), H is index 3 (reddest)
    h_band = imgs[3]
    vis_band = imgs[0]
    
    outer_annulus_mask = (r > r50) & (r <= r80)
    
    inner_red_flux = np.sum(h_band[inner_mask])
    outer_blue_flux = np.sum(vis_band[outer_annulus_mask])
    
    color_grad = inner_red_flux / (outer_blue_flux + 1e-5)
    
    return np.array([C, A, a4, m2_inner, m2_outer, axis_ratio, M20, D0, R_I, color_grad], dtype=np.float32)

def get_clip(x, y, window_size, im):
    top = y + window_size
    bottom = y - window_size
    left = x - window_size
    right = x + window_size
    return im[bottom:top, left:right]

def azimuthal_average_fast(image, r, bins, hist):
    radial_prof = np.histogram(r, bins, weights=image)[0] / hist.astype(float)
    return radial_prof

def get_patch_power_spectrum(patch, r, bins, hist):
    FA = npfft.fft2(patch)
    FA_FBconj = npfft.fftshift(FA * np.conjugate(FA))
    trans = np.real(FA_FBconj)
    profile2 = azimuthal_average_fast(trans, r, bins, hist)
    return profile2

def process_patch_weighted(patch, r, bins, hist, weight_large, weight_small):
    ps_raw = get_patch_power_spectrum(patch, r, bins, hist)[:-1]
    n_bins = len(ps_raw)
    
    ps_large = ps_raw * weight_large[:n_bins]
    ps_small = ps_raw * weight_small[:n_bins]
    
    return np.concatenate([ps_large, ps_small], axis=0)

def round_up(x):
    return np.round(x + 0.5)

def check_patch(patch, num_pixels):
    if np.count_nonzero(patch) < num_pixels:
        return True 
    return False

def check_patches(patches, num_pixels):
    for patch in patches:
        if np.count_nonzero(patch) < num_pixels:
            return True 
    return False

def process_single_object_fft(args, path_data_dir, path_patch_dir, window, step, min_width, min_height, r, bins, hist, VIS_band_only):
    obj_id, tile_id = args  
    
    try:
        temp_file_path = os.path.join(path_patch_dir, f"temp_{obj_id}.npz")
        
        if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
            return temp_file_path 

        target_file = os.path.join(path_data_dir, f"tile_{tile_id}", f"{obj_id}.npy")
        if not os.path.exists(target_file):
            return None
            
        stacked_array = np.load(target_file, mmap_mode='r')
        images = stacked_array[:4].astype(np.float64) 
        
        _, Ymax, Xmax = images.shape
        if Xmax <= min_width or Ymax <= min_height:
            return None

        # Extract global morphology features BEFORE log-transform and masking
        # Now passing the full 4-band array to calculate cross-band color gradients
        raw_images = np.clip(images, 0.0, None)
        global_morph_features = get_global_morphology(raw_images)

        # Apply core saturation suppression for patch extraction
        images = np.clip(images, 0.0, None)
        images = np.log1p(images)
        
        max_dim = max(Xmax, Ymax)
        sigma_large = max_dim * 0.3  
        sigma_small = max_dim * 0.1 
        
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        k = bin_centers / (window * 2) 
        
        weight_large = 2.5 - 1.5 * np.exp(-2.0 * (np.pi**2) * (sigma_large**2) * (k**2))
        weight_small = 2.5 - 1.5 * np.exp(-2.0 * (np.pi**2) * (sigma_small**2) * (k**2))
        
        weight_large /= np.max(weight_large)
        weight_small /= np.max(weight_small)

        X_range = np.linspace(window, Xmax - window, int(round_up(Xmax / step))).astype(int)
        Y_range = np.linspace(window, Ymax - window, int(round_up(Ymax / step))).astype(int)
        
        num_pixels = (window * 2) ** 2
        samples = []

        if VIS_band_only:
            vis_image = images[0]
            for y in Y_range:
                y_start, y_end = y - window, y + window
                for x in X_range:
                    x_start, x_end = x - window, x + window
                    
                    patch = vis_image[y_start:y_end, x_start:x_end]
                    
                    if not check_patch(patch, num_pixels):
                        sample = process_patch_weighted(patch, r, bins, hist, weight_large, weight_small)
                        sample = np.concatenate([sample, global_morph_features], axis=0)
                        samples.append(sample)
        else:
            for y in Y_range:
                y_start, y_end = y - window, y + window
                for x in X_range:
                    x_start, x_end = x - window, x + window
                    
                    patches = list(images[:, y_start:y_end, x_start:x_end])
                    
                    if not check_patches(patches, num_pixels):
                        band_samples = []
                        for patch in patches:
                            band_sample = process_patch_weighted(patch, r, bins, hist, weight_large, weight_small)
                            band_samples.append(band_sample)
                        
                        combined_multi_band_sample = np.concatenate(band_samples, axis=0)
                        combined_multi_band_sample = np.concatenate([combined_multi_band_sample, global_morph_features], axis=0)
                        samples.append(combined_multi_band_sample)

        if not samples:
            return None
            
        ids_array = np.full(len(samples), obj_id)
        np.savez(temp_file_path, ids=ids_array, samples=np.array(samples, dtype=np.float32)) 
        return temp_file_path

    except Exception as e:
        print(f"Error processing object {obj_id} in tile {tile_id}: {e}")
        return None

def patch_extractionn(path_model_dir, threads, window_size, radial_width, patch_step, Euclid_catalog_csv, Euclid_catalog_npz, path_data_dir, VIS_band_only):
    
    path_patch_dir = os.path.join(path_model_dir, 'patches')
    os.makedirs(path_patch_dir, exist_ok=True)

    if Euclid_catalog_npz is not None:
        data = np.load(Euclid_catalog_npz)
        object_catalogue_cosmos = pd.DataFrame({key: data[key] for key in data.files})
        object_catalogue_cosmos = object_catalogue_cosmos.loc[object_catalogue_cosmos['flag_resolved']=='yes']
        object_catalogue_cosmos = object_catalogue_cosmos.loc[(object_catalogue_cosmos['phz_pp_median_stellarmass']>=10.5) & (object_catalogue_cosmos['phz_pp_median_redshift']<0.5) ]
        data.close()
    elif Euclid_catalog_csv is not None:
        object_catalogue_cosmos = pd.read_csv(Euclid_catalog_csv)

    tile_indices = object_catalogue_cosmos['tile_index'].values
    object_ids = object_catalogue_cosmos['object_id'].values
    
    tasks = list(zip(object_ids, tile_indices))
    
    print('nr objects: ' + str(len(object_ids)))
    print('Processing sky areas...')

    patch_dim = window_size * 2
    y_idx, x_idx = np.indices((patch_dim, patch_dim))
    center = np.array([(patch_dim - 1) / 2.0, (patch_dim - 1) / 2.0])
    r = np.hypot(x_idx - center[0], y_idx - center[1])
    
    num_bins = int(np.round(r.max() / radial_width) + 1)
    max_bin = num_bins * radial_width
    bins = np.linspace(0, max_bin, num_bins + 1)
    
    hist = np.histogram(r, bins)[0]
    hist[hist == 0] = 1

    worker_func = partial(
        process_single_object_fft,
        path_data_dir=path_data_dir,
        path_patch_dir=path_patch_dir, 
        window=window_size,
        step=patch_step,
        min_width=20,
        min_height=20,
        r=r,          
        bins=bins,    
        hist=hist,     
        VIS_band_only=VIS_band_only
    )

    temp_files = []

    with Pool(threads) as pool:
        for result_path in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Extracting Patches"):
            if result_path is not None:
                temp_files.append(result_path)

    print(f"Combining {len(temp_files)} temporary files into chunks...")
    
    chunk_size = 5000  
    chunk_idx = 0
    
    all_ids = []
    all_samples = []
    
    for i, tmp_file in enumerate(tqdm(temp_files, desc="Merging Files")):
        try:
            data = np.load(tmp_file)
            all_ids.append(data['ids'])
            all_samples.append(data['samples'])
            data.close() 
            os.remove(tmp_file) 
            
        except Exception as e:
            print(f"Failed to load or delete temp file {tmp_file}: {e}")
            continue

        if (i + 1) % chunk_size == 0 or (i + 1) == len(temp_files):
            if all_samples:
                samples_stack = np.vstack(all_samples)
                ids_stack = np.concatenate(all_ids, axis=0)
                
                np.save(os.path.join(path_patch_dir, f'samples_chunk_{chunk_idx:03d}.npy'), samples_stack)
                np.save(os.path.join(path_patch_dir, f'samples_object_id_chunk_{chunk_idx:03d}.npy'), ids_stack)
                
                print(f"\nSaved chunk {chunk_idx:03d} with {len(samples_stack)} patches.")
                
                all_ids.clear()
                all_samples.clear()
                chunk_idx += 1

    print("All extraction and merging complete. Data is safely stored in chunks.")
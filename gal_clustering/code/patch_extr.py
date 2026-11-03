import os
import numpy as np
import pandas as pd
import numpy.fft as npfft
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def get_global_morphology(imgs):
    img = imgs[0]
    total_light = np.sum(img)
    if total_light <= 0:
        return np.zeros(10, dtype=np.float32)

    y, x = np.indices(img.shape)
    cx = np.sum(img * x) / total_light
    cy = np.sum(img * y) / total_light

    img_180 = np.rot90(img, 2)
    A = np.sum(np.abs(img - img_180)) / total_light

    r = np.hypot(x - cx, y - cy)
    r_sorted_idx = np.argsort(r.flat)
    sorted_img_r = img.flat[r_sorted_idx]
    sorted_r = r.flat[r_sorted_idx]
    cumsum_r = np.cumsum(sorted_img_r)

    r20_idx = np.searchsorted(cumsum_r, 0.2 * total_light, side='right')
    r50_idx = np.searchsorted(cumsum_r, 0.5 * total_light, side='right')
    r80_idx = np.searchsorted(cumsum_r, 0.8 * total_light, side='right')

    r20 = max(sorted_r[r20_idx] if r20_idx < len(sorted_r) else 1e-5, 1e-5)
    r50 = max(sorted_r[r50_idx] if r50_idx < len(sorted_r) else 1e-5, 1e-5)
    r80 = max(sorted_r[r80_idx] if r80_idx < len(sorted_r) else 1e-5, 1e-5)

    C = 5.0 * np.log10(r80 / r20) if r20 > 0 else 0.0

    sorted_idx_desc = np.argsort(img.flat)[::-1]
    sorted_img_desc = img.flat[sorted_idx_desc]
    cumsum_desc = np.cumsum(sorted_img_desc)

    idx_20 = np.searchsorted(cumsum_desc, 0.2 * total_light, side='right')
    idx_20 = max(1, min(idx_20, len(sorted_img_desc)))
    brightest_20_idx = sorted_idx_desc[:idx_20]

    M_tot = np.sum(img * r**2)
    M_20_val = np.sum(img.flat[brightest_20_idx] * r.flat[brightest_20_idx]**2)
    M20 = np.log10(M_20_val / M_tot) if M_tot > 0 and M_20_val > 0 else 0.0

    ymax, xmax = np.unravel_index(np.argmax(img), img.shape)
    D0 = np.hypot(xmax - cx, ymax - cy) / r50

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
        R_I = np.max(valid_gradients) / np.mean(mean_surface_brightness) if len(valid_gradients) > 0 and np.mean(mean_surface_brightness) > 0 else 0.0
    else:
        R_I = 0.0
    R_I = max(0.0, float(R_I))

    theta = np.arctan2(y - cy, x - cx)
    inner_mask = r <= r50
    outer_mask = r > r50

    light_inner = np.sum(img[inner_mask]) + 1e-5
    light_outer = np.sum(img[outer_mask]) + 1e-5

    m2_inner = np.hypot(np.sum(img[inner_mask] * np.cos(2 * theta[inner_mask])), np.sum(img[inner_mask] * np.sin(2 * theta[inner_mask]))) / light_inner
    m2_outer = np.hypot(np.sum(img[outer_mask] * np.cos(2 * theta[outer_mask])), np.sum(img[outer_mask] * np.sin(2 * theta[outer_mask]))) / light_outer

    m20_img = np.sum(img * (x - cx)**2) / total_light
    m02_img = np.sum(img * (y - cy)**2) / total_light
    m11_img = np.sum(img * (x - cx) * (y - cy)) / total_light

    theta_0 = 0.5 * np.arctan2(2 * m11_img, m20_img - m02_img) if (m20_img - m02_img) != 0 else 0.0
    aligned_theta = theta - theta_0
    a4 = np.sum(img * np.cos(4 * aligned_theta)) / total_light

    term1 = (m20_img + m02_img) / 2.0
    term2 = np.sqrt(((m20_img - m02_img) / 2.0)**2 + m11_img**2)
    axis_ratio = np.sqrt(max(term1 - term2, 1e-10)) / np.sqrt(max(term1 + term2, 1e-10))

    if imgs.shape[0] > 1:
        outer_annulus_mask = (r > r50) & (r <= r80)
        color_grad = np.sum(imgs[-1][inner_mask]) / (np.sum(imgs[0][outer_annulus_mask]) + 1e-5)
    else:
        color_grad = 1.0

    return np.array([C, A, a4, m2_inner, m2_outer, axis_ratio, M20, D0, R_I, color_grad], dtype=np.float32)


def azimuthal_average_fast(image, r, bins, hist):
    return np.histogram(r, bins, weights=image)[0] / hist.astype(float)


def get_patch_power_spectrum(patch, r, bins, hist):
    FA = npfft.fft2(patch)
    return azimuthal_average_fast(np.real(npfft.fftshift(FA * np.conjugate(FA))), r, bins, hist)


def process_patch_weighted(patch, r, bins, hist, weight_large, weight_small):
    ps_raw = get_patch_power_spectrum(patch, r, bins, hist)[:-1]
    n_bins = len(ps_raw)
    return np.concatenate([ps_raw * weight_large[:n_bins], ps_raw * weight_small[:n_bins]], axis=0)


def process_single_object_fft(args, path_data_dir, path_patch_dir, window, step,
                              min_width, min_height, r, bins, hist, ml_band_indices):
    obj_id, tile_id = args

    try:
        temp_file_path = os.path.join(path_patch_dir, f"temp_{obj_id}.npz")
        if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
            return temp_file_path

        if tile_id is not None and not pd.isna(tile_id):
            target_file = os.path.join(path_data_dir, f"tile_{tile_id}", f"{obj_id}.npy")
            if not os.path.exists(target_file):
                target_file = os.path.join(path_data_dir, f"{obj_id}.npy")
        else:
            target_file = os.path.join(path_data_dir, f"{obj_id}.npy")

        if not os.path.exists(target_file):
            return None

        stacked_array = np.load(target_file, mmap_mode='r')
        images = stacked_array[ml_band_indices].astype(np.float64)

        _, Ymax, Xmax = images.shape
        if Xmax <= min_width or Ymax <= min_height:
            return None

        raw_images = np.clip(images, 0.0, None)
        global_morph_features = get_global_morphology(raw_images)
        images = np.log1p(raw_images)

        max_dim = max(Xmax, Ymax)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        k = bin_centers / (window * 2)

        weight_large = 2.5 - 1.5 * np.exp(-2.0 * (np.pi**2) * ((max_dim * 0.3)**2) * (k**2))
        weight_small = 2.5 - 1.5 * np.exp(-2.0 * (np.pi**2) * ((max_dim * 0.1)**2) * (k**2))
        weight_large /= np.max(weight_large)
        weight_small /= np.max(weight_small)

        X_range = np.linspace(window, Xmax - window, int(np.round(Xmax / step + 0.5))).astype(int)
        Y_range = np.linspace(window, Ymax - window, int(np.round(Ymax / step + 0.5))).astype(int)

        num_pixels = (window * 2) ** 2
        samples = []

        for y in Y_range:
            y_start, y_end = y - window, y + window
            for x in X_range:
                x_start, x_end = x - window, x + window
                patches = list(images[:, y_start:y_end, x_start:x_end])

                if not any(np.count_nonzero(p) < num_pixels for p in patches):
                    band_samples = [process_patch_weighted(p, r, bins, hist, weight_large, weight_small) for p in patches]
                    samples.append(np.concatenate(band_samples + [global_morph_features], axis=0))

        if not samples:
            return None

        np.savez(temp_file_path, ids=np.full(len(samples), obj_id), samples=np.array(samples, dtype=np.float32))
        return temp_file_path

    except Exception as e:
        print(f"Error processing object {obj_id} (tile {tile_id}): {e}")
        return None


def patch_extractionn(path_model_dir, threads, window_size, radial_width, patch_step,
                      object_catalogue, path_data_dir, ml_band_indices,
                      id_col='object_id', tile_col='tile_index'):
    path_patch_dir = os.path.join(path_model_dir, 'patches')
    os.makedirs(path_patch_dir, exist_ok=True)

    object_ids = object_catalogue[id_col].values
    tile_indices = object_catalogue[tile_col].values if tile_col is not None and tile_col in object_catalogue.columns else [None] * len(object_ids)
    tasks = list(zip(object_ids, tile_indices))

    print(f'nr objects: {len(object_ids)}')
    patch_dim = window_size * 2
    y_idx, x_idx = np.indices((patch_dim, patch_dim))
    r = np.hypot(x_idx - (patch_dim - 1) / 2.0, y_idx - (patch_dim - 1) / 2.0)

    num_bins = int(np.round(r.max() / radial_width) + 1)
    bins = np.linspace(0, num_bins * radial_width, num_bins + 1)
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
        ml_band_indices=ml_band_indices
    )

    temp_files = []
    with Pool(threads) as pool:
        for result_path in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Extracting Patches"):
            if result_path is not None:
                temp_files.append(result_path)

    chunk_size = 5000
    chunk_idx = 0
    all_ids, all_samples = [], []

    for i, tmp_file in enumerate(tqdm(temp_files, desc="Merging Files")):
        try:
            with np.load(tmp_file) as data:
                all_ids.append(data['ids'])
                all_samples.append(data['samples'])
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
                all_ids.clear()
                all_samples.clear()
                chunk_idx += 1
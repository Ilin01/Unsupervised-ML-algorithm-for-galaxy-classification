"""
Stage 1 of the pipeline: turn every galaxy cutout into a set of patch features.

For each object:
  * the selected bands are read from its stacked ``.npy`` cutout;
  * a vector of 10 global morphology parameters is measured on the whole image
    (concentration, asymmetry, M20, axis ratio, ...);
  * the image is tiled with small square patches of side ``2 * window`` on a
    regular grid; for every patch and every band the radially averaged 2D
    Fourier power spectrum is computed, weighted by two frequency filters, and
    all bands plus the global morphology vector are concatenated into one row.

So the feature row of a single patch is laid out as

    [band_1: PS*w_large, PS*w_small | band_2: ... | ... | 10 global features]

and every patch of the same object carries the same 10 global features.

Each object is processed in its own worker process and written to a temporary
``temp_<id>.npz`` file; the temp files are then merged into
``samples_chunk_XXX.npy`` / ``samples_object_id_chunk_XXX.npy`` pairs inside
``<path_model_dir>/patches``. A JSON completion marker records the settings
used so that a rerun with identical inputs can skip extraction entirely, and a
rerun with different inputs never reuses stale patches.
"""
import os
import glob
import json
import hashlib
import numpy as np
import pandas as pd
import numpy.fft as npfft
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def get_global_morphology(imgs):
    """Measure 10 whole-galaxy morphology parameters.

    Parameters
    ----------
    imgs : ndarray, shape (n_bands, H, W)
        Non-negative (already clipped at 0) images of the selected bands,
        ordered from shortest to longest wavelength. All structural
        measurements use the first (bluest) band; the last band is only used
        for the colour-gradient term.

    Returns
    -------
    ndarray of float32, shape (10,)
        ``[C, A, a4, m2_inner, m2_outer, axis_ratio, M20, D0, R_I, color_grad]``:

        C          concentration, 5 * log10(r80 / r20)
        A          rotational (180 deg) asymmetry, sum|I - I_rot| / sum(I)
        a4         cos(4*theta) light moment about the major axis (boxy vs. disky)
        m2_inner   m=2 azimuthal Fourier amplitude inside r50 (bar / inner elongation)
        m2_outer   m=2 azimuthal Fourier amplitude outside r50 (outer elongation / spiral arms)
        axis_ratio b/a from the second-order image moments
        M20        log10 of the 2nd-order moment of the brightest 20% of the
                   light, relative to the total 2nd-order moment
        D0         offset of the brightest pixel from the light centroid, in units of r50
        R_I        strongest outward rise of the radial surface-brightness
                   profile beyond r20 (ring indicator), relative to its mean
        color_grad light of the reddest band inside r50 divided by light of the
                   bluest band in the r50-r80 annulus

        All zeros are returned if the image has no positive flux.
    """
    img = imgs[0]
    total_light = np.sum(img)
    if total_light <= 0:
        return np.zeros(10, dtype=np.float32)

    # Pixel coordinate grids and the flux-weighted centroid (cx, cy).
    y, x = np.indices(img.shape)
    cx = np.sum(img * x) / total_light
    cy = np.sum(img * y) / total_light

    # --- Asymmetry: compare the image with itself rotated by 180 degrees.
    # (np.rot90(..., 2) rotates about the array centre, i.e. the cutout centre,
    # which is assumed to coincide with the galaxy centre.)
    img_180 = np.rot90(img, 2)
    A = np.sum(np.abs(img - img_180)) / total_light

    # --- Curve of growth: sort pixels by distance from the centroid and
    # accumulate their flux to find the radii enclosing 20/50/80% of the light.
    r = np.hypot(x - cx, y - cy)
    r_sorted_idx = np.argsort(r.flat)
    sorted_img_r = img.flat[r_sorted_idx]
    sorted_r = r.flat[r_sorted_idx]
    cumsum_r = np.cumsum(sorted_img_r)

    r20_idx = np.searchsorted(cumsum_r, 0.2 * total_light, side='right')
    r50_idx = np.searchsorted(cumsum_r, 0.5 * total_light, side='right')
    r80_idx = np.searchsorted(cumsum_r, 0.8 * total_light, side='right')

    # Floor the radii at 1e-5 so later divisions/logs never hit zero.
    r20 = max(sorted_r[r20_idx] if r20_idx < len(sorted_r) else 1e-5, 1e-5)
    r50 = max(sorted_r[r50_idx] if r50_idx < len(sorted_r) else 1e-5, 1e-5)
    r80 = max(sorted_r[r80_idx] if r80_idx < len(sorted_r) else 1e-5, 1e-5)

    # --- Concentration (Bershady/Conselice definition).
    C = 5.0 * np.log10(r80 / r20) if r20 > 0 else 0.0

    # --- M20 (Lotz et al. 2004): pixels sorted from brightest to faintest;
    # take the brightest pixels that together hold 20% of the total flux.
    sorted_idx_desc = np.argsort(img.flat)[::-1]
    sorted_img_desc = img.flat[sorted_idx_desc]
    cumsum_desc = np.cumsum(sorted_img_desc)

    idx_20 = np.searchsorted(cumsum_desc, 0.2 * total_light, side='right')
    idx_20 = max(1, min(idx_20, len(sorted_img_desc)))  # keep at least one pixel
    brightest_20_idx = sorted_idx_desc[:idx_20]

    # Second-order moments about the light centroid (total, and brightest 20%).
    M_tot = np.sum(img * r**2)
    M_20_val = np.sum(img.flat[brightest_20_idx] * r.flat[brightest_20_idx]**2)
    M20 = np.log10(M_20_val / M_tot) if M_tot > 0 and M_20_val > 0 else 0.0

    # --- D0: how far the peak pixel lies from the light centroid, in r50 units
    # (large for disturbed / off-centre-nucleus systems).
    ymax, xmax = np.unravel_index(np.argmax(img), img.shape)
    D0 = np.hypot(xmax - cx, ymax - cy) / r50

    # --- Radial surface-brightness profile in 20 annuli out to 1.5 * r80.
    num_bins = 20
    bins = np.linspace(0, r80 * 1.5, num_bins + 1)
    radial_prof, _ = np.histogram(r, bins, weights=img)  # flux per annulus
    area_prof, _ = np.histogram(r, bins)                 # pixels per annulus
    area_prof[area_prof == 0] = 1                        # avoid division by zero
    mean_surface_brightness = radial_prof / area_prof

    # --- R_I (ring index): the largest increase in surface brightness between
    # consecutive annuli beyond r20. A normal declining profile gives a value
    # <= 0 (clipped to 0); a ring produces a positive bump.
    valid_bins = bins[:-1] > r20
    if np.any(valid_bins[:-1]):
        gradients = np.diff(mean_surface_brightness)
        valid_gradients = gradients[valid_bins[:-1]]
        R_I = np.max(valid_gradients) / np.mean(mean_surface_brightness) if len(valid_gradients) > 0 and np.mean(mean_surface_brightness) > 0 else 0.0
    else:
        R_I = 0.0
    R_I = max(0.0, float(R_I))

    # --- m=2 azimuthal Fourier amplitudes, separately inside and outside r50.
    # |sum I * exp(2i*theta)| / sum I measures bi-symmetric structure (bars,
    # elongation, two-armed spirals).
    theta = np.arctan2(y - cy, x - cx)
    inner_mask = r <= r50
    outer_mask = r > r50

    light_inner = np.sum(img[inner_mask]) + 1e-5
    light_outer = np.sum(img[outer_mask]) + 1e-5

    m2_inner = np.hypot(np.sum(img[inner_mask] * np.cos(2 * theta[inner_mask])), np.sum(img[inner_mask] * np.sin(2 * theta[inner_mask]))) / light_inner
    m2_outer = np.hypot(np.sum(img[outer_mask] * np.cos(2 * theta[outer_mask])), np.sum(img[outer_mask] * np.sin(2 * theta[outer_mask]))) / light_outer

    # --- Normalised second-order central moments (covariance of the light).
    m20_img = np.sum(img * (x - cx)**2) / total_light
    m02_img = np.sum(img * (y - cy)**2) / total_light
    m11_img = np.sum(img * (x - cx) * (y - cy)) / total_light

    # Position angle of the major axis, then the cos(4*theta) moment measured
    # relative to it: positive ~ disky (pointy) isophotes, negative ~ boxy.
    theta_0 = 0.5 * np.arctan2(2 * m11_img, m20_img - m02_img) if (m20_img - m02_img) != 0 else 0.0
    aligned_theta = theta - theta_0
    a4 = np.sum(img * np.cos(4 * aligned_theta)) / total_light

    # --- Axis ratio b/a = sqrt(minor eigenvalue / major eigenvalue) of the
    # moment matrix [[m20, m11], [m11, m02]].
    term1 = (m20_img + m02_img) / 2.0
    term2 = np.sqrt(((m20_img - m02_img) / 2.0)**2 + m11_img**2)
    axis_ratio = np.sqrt(max(term1 - term2, 1e-10)) / np.sqrt(max(term1 + term2, 1e-10))

    # --- Colour gradient: reddest-band light in the core vs. bluest-band light
    # in the r50-r80 annulus (e.g. red bulge + blue disc gives a large value).
    if imgs.shape[0] > 1:
        outer_annulus_mask = (r > r50) & (r <= r80)
        color_grad = np.sum(imgs[-1][inner_mask]) / (np.sum(imgs[0][outer_annulus_mask]) + 1e-5)
    else:
        color_grad = 1.0

    return np.array([C, A, a4, m2_inner, m2_outer, axis_ratio, M20, D0, R_I, color_grad], dtype=np.float32)


def azimuthal_average_fast(image, r, bins, hist):
    """Average ``image`` in annuli: sum of pixel values per radial bin of ``r``
    divided by the (precomputed) number of pixels per bin ``hist``."""
    return np.histogram(r, bins, weights=image)[0] / hist.astype(float)


def get_patch_power_spectrum(patch, r, bins, hist):
    """Radially averaged 2D power spectrum of a single square patch.

    ``|FFT|^2`` is shifted so the zero frequency sits at the patch centre,
    which is where ``r`` (distance from the patch centre) is measured from, so
    each radial bin corresponds to a range of spatial frequency |k|. The
    result is rotation invariant, so the patch orientation does not matter.
    """
    FA = npfft.fft2(patch)
    return azimuthal_average_fast(np.real(npfft.fftshift(FA * np.conjugate(FA))), r, bins, hist)


def process_patch_weighted(patch, r, bins, hist, weight_large, weight_small):
    """Power spectrum of one patch, weighted by two frequency filters.

    The last radial bin is dropped (it only contains the corner pixels of the
    square FFT grid and is poorly sampled). The spectrum is then multiplied by
    ``weight_large`` and ``weight_small`` and the two weighted copies are
    concatenated, giving ``2 * n_bins`` features per band.
    """
    ps_raw = get_patch_power_spectrum(patch, r, bins, hist)[:-1]
    n_bins = len(ps_raw)
    return np.concatenate([ps_raw * weight_large[:n_bins], ps_raw * weight_small[:n_bins]], axis=0)


def process_single_object_fft(args, path_data_dir, path_patch_dir, window, step,
                              min_width, min_height, r, bins, hist, ml_band_indices):
    """Extract all patch feature rows of one object (runs in a worker process).

    Parameters
    ----------
    args : tuple (obj_id, tile_id)
        Object ID and its tile index (``None``/NaN for a flat data folder).
    path_data_dir : str
        Folder with the ``.npy`` cutouts.
    path_patch_dir : str
        Folder where the temporary per-object ``temp_<id>.npz`` is written.
    window : int
        Patch half-width; patches are ``2*window x 2*window`` pixels.
    step : float
        Approximate spacing between patch centres.
    min_width, min_height : int
        Cutouts this small or smaller are skipped.
    r, bins, hist :
        Precomputed radial-distance grid of a patch, radial bin edges and
        pixel counts per bin (identical for every patch, so computed once).
    ml_band_indices : list of int
        Bands (axis-0 indices of the cutout) to use.

    Returns
    -------
    str or None
        Path of the temp file, or ``None`` if the object was skipped (file
        missing, too small, no valid patches, or an error occurred).
    """
    obj_id, tile_id = args

    try:
        # Resume support: if this object was already processed by an
        # interrupted run, reuse its temp file.
        temp_file_path = os.path.join(path_patch_dir, f"temp_{obj_id}.npz")
        if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
            return temp_file_path

        # Locate the cutout: prefer tile_<tile_id>/<obj_id>.npy and fall back
        # to <obj_id>.npy in the top-level folder.
        if tile_id is not None and not pd.isna(tile_id):
            target_file = os.path.join(path_data_dir, f"tile_{tile_id}", f"{obj_id}.npy")
            if not os.path.exists(target_file):
                target_file = os.path.join(path_data_dir, f"{obj_id}.npy")
        else:
            target_file = os.path.join(path_data_dir, f"{obj_id}.npy")

        if not os.path.exists(target_file):
            return None

        # Memory-map the stack and read only the selected bands.
        stacked_array = np.load(target_file, mmap_mode='r')
        images = stacked_array[ml_band_indices].astype(np.float64)

        _, Ymax, Xmax = images.shape
        if Xmax <= min_width or Ymax <= min_height:
            return None

        # Negative (background-noise) pixels are set to 0. Global morphology is
        # measured on these linear-flux images; patches use a log stretch
        # (log1p) to compress the dynamic range between core and outskirts.
        raw_images = np.clip(images, 0.0, None)
        global_morph_features = get_global_morphology(raw_images)
        images = np.log1p(raw_images)

        # --- Frequency weights, scaled to the size of this galaxy's cutout.
        # k is the spatial frequency (cycles/pixel) of each radial bin centre.
        # Each weight is 2.5 - 1.5*exp(-2 pi^2 sigma^2 k^2), i.e. 1 minus a
        # Gaussian low-pass filter: it suppresses the lowest frequencies (to
        # 1/2.5 = 0.4 after normalisation) and approaches 1 at high k. The
        # "large" filter (sigma = 0.3 * cutout size) reaches full weight at
        # lower k than the "small" one (sigma = 0.1 * cutout size), so the two
        # copies emphasise structure on different scales relative to the galaxy.
        max_dim = max(Xmax, Ymax)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        k = bin_centers / (window * 2)

        weight_large = 2.5 - 1.5 * np.exp(-2.0 * (np.pi**2) * ((max_dim * 0.3)**2) * (k**2))
        weight_small = 2.5 - 1.5 * np.exp(-2.0 * (np.pi**2) * ((max_dim * 0.1)**2) * (k**2))
        weight_large /= np.max(weight_large)
        weight_small /= np.max(weight_small)

        # --- Patch centre grid: evenly spaced from `window` to `size - window`
        # so every patch lies fully inside the image. The number of centres is
        # ~size/step, so the actual spacing is slightly below `step`.
        X_range = np.linspace(window, Xmax - window, int(np.round(Xmax / step + 0.5))).astype(int)
        Y_range = np.linspace(window, Ymax - window, int(np.round(Ymax / step + 0.5))).astype(int)

        num_pixels = (window * 2) ** 2
        samples = []

        for y in Y_range:
            y_start, y_end = y - window, y + window
            for x in X_range:
                x_start, x_end = x - window, x + window
                # One 2D patch per selected band at this position.
                patches = list(images[:, y_start:y_end, x_start:x_end])

                # Keep the patch only if every pixel in every band is > 0, i.e.
                # it lies entirely on positive flux. This discards patches that
                # touch background (clipped to 0), masked areas or the image edge.
                if not any(np.count_nonzero(p) < num_pixels for p in patches):
                    band_samples = [process_patch_weighted(p, r, bins, hist, weight_large, weight_small) for p in patches]
                    # Row = all bands' weighted spectra + the object's global morphology.
                    samples.append(np.concatenate(band_samples + [global_morph_features], axis=0))

        if not samples:
            return None

        # One temp file per object; `ids` repeats the object ID for every patch.
        np.savez(temp_file_path, ids=np.full(len(samples), obj_id), samples=np.array(samples, dtype=np.float32))
        return temp_file_path

    except Exception as e:
        # Never let one bad cutout kill the whole Pool; just skip the object.
        print(f"Error processing object {obj_id} (tile {tile_id}): {e}")
        return None

# Name of the JSON file written into the patches folder once extraction has
# finished successfully. It stores the settings and the chunk/patch counts.
DONE_MARKER = 'PATCH_EXTRACTION_DONE.json'


def _extraction_settings(object_ids, window_size, radial_width, patch_step,
                         path_data_dir, ml_band_indices):
    """Everything that determines the patch files, so a rerun with different
    inputs never silently reuses old patches."""
    # The object list is stored as a SHA-256 hash of the sorted IDs rather
    # than the full list, to keep the marker file small.
    ids_sorted = sorted(str(i) for i in object_ids)
    ids_hash = hashlib.sha256('\n'.join(ids_sorted).encode()).hexdigest()
    return {
        'window_size': int(window_size),
        'radial_width': float(radial_width),
        'patch_step': float(patch_step),
        'ml_band_indices': [int(b) for b in ml_band_indices],
        'path_data_dir': os.path.abspath(path_data_dir),
        'n_objects': len(ids_sorted),
        'object_ids_sha256': ids_hash,
    }


def _chunk_pairs(path_patch_dir):
    """Return the sorted lists of sample-chunk files and ID-chunk files."""
    sample_files = sorted(glob.glob(os.path.join(path_patch_dir, 'samples_chunk_*.npy')))
    id_files = sorted(glob.glob(os.path.join(path_patch_dir, 'samples_object_id_chunk_*.npy')))
    return sample_files, id_files


def _chunks_look_complete(path_patch_dir, check_temp=True):
    """Check the chunk files on disk: no leftover temp files from an
    interrupted run, chunk numbering 000..N-1 with no gaps, every samples
    chunk has a matching ids chunk with the same number of rows, and every
    file is readable. Returns (ok, n_chunks, n_patches, reason)."""
    if check_temp and glob.glob(os.path.join(path_patch_dir, 'temp_*.npz')):
        return False, 0, 0, 'temp files from an unfinished run are still present'

    sample_files, id_files = _chunk_pairs(path_patch_dir)
    if not sample_files:
        return False, 0, 0, 'no chunk files found'
    if len(sample_files) != len(id_files):
        return False, 0, 0, f'{len(sample_files)} sample chunks but {len(id_files)} id chunks'

    n_patches = 0
    n_features = None
    for k, (s_file, i_file) in enumerate(zip(sample_files, id_files)):
        # The k-th file in sorted order must be chunk number k.
        expected = f'_chunk_{k:03d}.npy'
        if not (s_file.endswith(expected) and i_file.endswith(expected)):
            return False, 0, 0, f'chunk numbering has a gap or mismatch near {os.path.basename(s_file)}'
        try:
            # mmap_mode only reads the header, so this is cheap even for huge files.
            s_shape = np.load(s_file, mmap_mode='r').shape
            i_shape = np.load(i_file, mmap_mode='r').shape
        except Exception as e:
            return False, 0, 0, f'cannot read {os.path.basename(s_file)}: {e}'
        if len(s_shape) != 2 or s_shape[0] != i_shape[0]:
            return False, 0, 0, f'row count mismatch in chunk {k:03d}'
        # All chunks must have the same feature width.
        if n_features is None:
            n_features = s_shape[1]
        elif s_shape[1] != n_features:
            return False, 0, 0, f'feature count changes in chunk {k:03d}'
        n_patches += s_shape[0]

    return True, len(sample_files), n_patches, ''


def patches_already_extracted(path_patch_dir, settings, object_ids):
    """True if a finished extraction for exactly these settings is on disk."""
    marker_path = os.path.join(path_patch_dir, DONE_MARKER)
    has_marker = os.path.exists(marker_path)
    # With a marker, extraction is known to have finished; temp files can only
    # be ones that failed to merge, so they don't mean the run was interrupted.
    ok, n_chunks, n_patches, reason = _chunks_look_complete(path_patch_dir, check_temp=not has_marker)
    if not ok:
        # Stay quiet on a genuinely fresh run (no marker and no chunks).
        if has_marker or reason != 'no chunk files found':
            print(f'Existing patches not reused: {reason}.')
        return False

    if has_marker:
        with open(marker_path) as f:
            marker = json.load(f)
        # Any differing setting (window, step, bands, data dir, object list...)
        # invalidates the old patches.
        changed = [k for k in settings if marker.get('settings', {}).get(k) != settings[k]]
        if changed:
            print(f'Existing patches not reused: settings changed since extraction ({", ".join(changed)}).')
            return False
        # The files on disk must also be exactly what the finished run wrote.
        if marker.get('n_chunks') != n_chunks or marker.get('n_patches') != n_patches:
            print('Existing patches not reused: chunk files differ from what the finished extraction wrote.')
            return False
        print(f'Found completed patch extraction ({n_chunks} chunks, {n_patches} patches). Skipping extraction.')
        return True

    # Patches made before completion markers existed: the files look complete,
    # but the settings they were made with are unknown. Accept them only if
    # every object in them is in the current catalogue, then write a marker.
    print('Found patch chunks without a completion marker; checking them against the catalogue...')
    catalogue_ids = set(str(i) for i in object_ids)
    _, id_files = _chunk_pairs(path_patch_dir)
    for i_file in id_files:
        chunk_ids = np.unique(np.load(i_file))
        if any(str(i) not in catalogue_ids for i in chunk_ids):
            print(f'Existing patches not reused: {os.path.basename(i_file)} has objects not in the current catalogue.')
            return False
    print('WARNING: reusing these patches. The extraction settings they were made with could not be checked; '
          f'delete {path_patch_dir} if window/step/bands/data dir have changed since.')
    _write_marker(marker_path, settings, n_chunks, n_patches)
    print(f'Found completed patch extraction ({n_chunks} chunks, {n_patches} patches). Skipping extraction.')
    return True


def _write_marker(marker_path, settings, n_chunks, n_patches):
    """Write the completion marker atomically (write to .tmp, then rename), so
    a crash mid-write can never leave a half-written marker behind."""
    tmp = marker_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'settings': settings, 'n_chunks': n_chunks, 'n_patches': n_patches}, f, indent=2)
    os.replace(tmp, marker_path)


def patch_extractionn(path_model_dir, threads, window_size, radial_width, patch_step,
                      object_catalogue, path_data_dir, ml_band_indices,
                      id_col='object_id', tile_col='tile_index'):
    """Extract patch features for every object in the catalogue.

    Parameters
    ----------
    path_model_dir : str
        Run output folder; patches are written to ``<path_model_dir>/patches``.
    threads : int
        Number of worker processes.
    window_size : int
        Patch half-width in pixels.
    radial_width : float
        Radial bin width for the azimuthally averaged power spectrum.
    patch_step : float
        Approximate spacing between patch centres.
    object_catalogue : pandas.DataFrame
        Filtered catalogue (see ``load_and_filter_catalog``).
    path_data_dir : str
        Folder with the ``.npy`` cutouts.
    ml_band_indices : list of int
        Band indices to use.
    id_col, tile_col : str
        Catalogue columns with the object ID and tile index.

    Returns
    -------
    bool
        True if extraction was run, False if a matching completed extraction
        was found on disk and reused.
    """
    path_patch_dir = os.path.join(path_model_dir, 'patches')
    os.makedirs(path_patch_dir, exist_ok=True)

    # One (object_id, tile_index) task per object; tile is None for flat folders.
    object_ids = object_catalogue[id_col].values
    tile_indices = object_catalogue[tile_col].values if tile_col is not None and tile_col in object_catalogue.columns else [None] * len(object_ids)
    tasks = list(zip(object_ids, tile_indices))

    # Skip everything if these exact settings were already extracted.
    settings = _extraction_settings(object_ids, window_size, radial_width, patch_step,
                                    path_data_dir, ml_band_indices)
    marker_path = os.path.join(path_patch_dir, DONE_MARKER)
    if patches_already_extracted(path_patch_dir, settings, object_ids):
        return False

    # Starting a (re)extraction: drop the marker and any chunk files from an
    # earlier run so stale chunks can't be mixed in with the new ones.
    # (temp_*.npz files are kept on purpose: they let an interrupted run resume.)
    if os.path.exists(marker_path):
        os.remove(marker_path)
    old_s, old_i = _chunk_pairs(path_patch_dir)
    if old_s or old_i:
        print(f'Removing {len(old_s) + len(old_i)} chunk files from an earlier, incomplete or outdated extraction.')
        for f in old_s + old_i:
            os.remove(f)

    print(f'nr objects: {len(object_ids)}')

    # --- Precompute the radial geometry shared by every patch: distance of
    # each pixel from the patch centre, the radial bin edges, and the number
    # of pixels per bin (used to turn sums into averages).
    patch_dim = window_size * 2
    y_idx, x_idx = np.indices((patch_dim, patch_dim))
    r = np.hypot(x_idx - (patch_dim - 1) / 2.0, y_idx - (patch_dim - 1) / 2.0)

    num_bins = int(np.round(r.max() / radial_width) + 1)
    bins = np.linspace(0, num_bins * radial_width, num_bins + 1)
    hist = np.histogram(r, bins)[0]
    hist[hist == 0] = 1  # empty bins: avoid division by zero

    # Freeze all shared arguments so the Pool only has to send (id, tile).
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

    # --- Parallel extraction: each worker writes one temp_<id>.npz per object.
    temp_files = []
    with Pool(threads) as pool:
        for result_path in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Extracting Patches"):
            if result_path is not None:
                temp_files.append(result_path)

    # --- Merge the per-object temp files into chunk files of up to
    # `chunk_size` objects each (not patches), deleting temp files as we go.
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

        # Flush a chunk every `chunk_size` objects and after the last object.
        if (i + 1) % chunk_size == 0 or (i + 1) == len(temp_files):
            if all_samples:
                samples_stack = np.vstack(all_samples)
                ids_stack = np.concatenate(all_ids, axis=0)
                np.save(os.path.join(path_patch_dir, f'samples_chunk_{chunk_idx:03d}.npy'), samples_stack)
                np.save(os.path.join(path_patch_dir, f'samples_object_id_chunk_{chunk_idx:03d}.npy'), ids_stack)
                all_ids.clear()
                all_samples.clear()
                chunk_idx += 1

    # If the last temp file failed to load, the batch before it was never saved.
    if all_samples:
        np.save(os.path.join(path_patch_dir, f'samples_chunk_{chunk_idx:03d}.npy'), np.vstack(all_samples))
        np.save(os.path.join(path_patch_dir, f'samples_object_id_chunk_{chunk_idx:03d}.npy'), np.concatenate(all_ids, axis=0))

    # Record success; [1:3] picks (n_chunks, n_patches) from the check result.
    _write_marker(marker_path, settings, *_chunks_look_complete(path_patch_dir, check_temp=False)[1:3])
    return True

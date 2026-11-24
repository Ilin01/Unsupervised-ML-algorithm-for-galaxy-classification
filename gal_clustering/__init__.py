"""
gal_clustering: unsupervised morphological classification of galaxies.

The package exposes two public functions:

* ``load_and_filter_catalog`` - read a .csv/.npz object catalogue and apply
  quality-flag, stellar-mass and redshift cuts.
* ``run_classify`` - run the full pipeline (patch extraction -> PCA ->
  K-Means codebook -> HAC visual words -> TF-IDF bag-of-words per galaxy ->
  final K-Means clustering -> image montages).

The individual pipeline stages live as plain modules in the ``code/``
sub-directory, which is added to ``sys.path`` below so they can import each
other by bare module name (e.g. ``from create_initial_features import ...``).
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Make the stage modules in gal_clustering/code importable by bare name.
path = os.path.abspath(__file__)
code_path = os.path.dirname(path)
sys.path.insert(1, os.path.join(code_path, 'code'))

# Silence library warnings (sklearn/pandas deprecations etc.) to keep the long
# pipeline logs readable. Note that the second filter silences *all* warnings.
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Pipeline stages, in the order they are run by run_classify().
from patch_extr import patch_extractionn                  # 1. image -> patch power-spectrum features
from create_initial_features import create_vect           # 2. load patches, standardise, PCA
from create_kmeans_batch_model import optimized_model     # 3. Mini-Batch K-Means codebook of patches
from create_agg_model import agg_model                    # 4. HAC of codebook nodes -> visual words
from create_final_features import final_features          # 5. TF-IDF bag of visual words per object
from create_kmeans_model import kmeans_model              # 6. final K-Means clustering of objects
from image_montage import image_montage                   # 7. per-cluster image montages


def load_and_filter_catalog(catalog_path, id_col='object_id', tile_col='tile_index',
                            mass_col=None, z_col=None, quality_flag_col=None,
                            quality_flag_val=None, mass_min=None, mass_max=None,
                            z_min=None, z_max=None):
    """Load an object catalogue and apply the selection cuts.

    Parameters
    ----------
    catalog_path : str
        Path to a ``.csv`` or ``.npz`` catalogue. For ``.npz`` files every
        1-D array stored in the archive becomes a column; arrays with any
        other number of dimensions are ignored.
    id_col : str
        Column holding the object IDs (must match the cutout file names).
        It is copied to a column called ``object_id`` if named differently,
        because the later stages always refer to ``object_id``.
    tile_col : str or None
        Column holding the tile index of each object, used to find cutouts
        stored in ``tile_<index>/`` sub-folders. Copied to ``tile_index`` if
        named differently. ``None`` means cutouts are in a flat folder.
    mass_col, z_col : str or None
        Stellar-mass and redshift columns used for the cuts.
    quality_flag_col, quality_flag_val : str or None
        If both are given, only rows with ``quality_flag_col == quality_flag_val``
        are kept (e.g. a resolved/star-galaxy flag).
    mass_min, mass_max : float or None
        Inclusive stellar-mass bounds; ``None`` disables that bound.
    z_min, z_max : float or None
        Redshift bounds, applied as ``z_min <= z < z_max``; ``None`` disables.

    Returns
    -------
    pandas.DataFrame
        The filtered catalogue with a fresh 0..N-1 index.
    """
    if catalog_path is None or not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalogue file not found: {catalog_path}")

    # Choose the reader from the file extension.
    ext = os.path.splitext(catalog_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(catalog_path)
    elif ext == '.npz':
        # allow_pickle is needed for object-dtype (e.g. string) columns.
        # Only 1-D arrays are valid table columns.
        with np.load(catalog_path, allow_pickle=True) as data:
            df = pd.DataFrame({key: data[key] for key in data.files if data[key].ndim == 1})
    else:
        raise ValueError(f"Unsupported catalogue extension '{ext}'. Please provide a .csv or .npz file.")

    if id_col not in df.columns:
        raise KeyError(f"ID column '{id_col}' not found in catalogue columns: {list(df.columns)}")

    # Normalise the column names the rest of the pipeline relies on.
    if id_col != 'object_id':
        df['object_id'] = df[id_col]

    if tile_col is not None and tile_col in df.columns and tile_col != 'tile_index':
        df['tile_index'] = df[tile_col]

    # --- Robust check: only filter by quality flag if the column is provided ---
    if quality_flag_col is not None and quality_flag_val is not None:
        if quality_flag_col not in df.columns:
            raise KeyError(f"Quality flag column '{quality_flag_col}' not found in catalogue.")
        df = df.loc[df[quality_flag_col] == quality_flag_val]

    # Apply stellar mass cuts (both bounds inclusive)
    if mass_col is not None and (mass_min is not None or mass_max is not None):
        if mass_col not in df.columns:
            raise KeyError(f"Stellar mass column '{mass_col}' not found in catalogue.")
        if mass_min is not None:
            df = df.loc[df[mass_col] >= mass_min]
        if mass_max is not None:
            df = df.loc[df[mass_col] <= mass_max]

    # Apply redshift cuts (lower bound inclusive, upper bound exclusive)
    if z_col is not None and (z_min is not None or z_max is not None):
        if z_col not in df.columns:
            raise KeyError(f"Redshift column '{z_col}' not found in catalogue.")
        if z_min is not None:
            df = df.loc[df[z_col] >= z_min]
        if z_max is not None:
            df = df.loc[df[z_col] < z_max]

    return df.reset_index(drop=True)


def run_classify(path_model_dir=None, path_data_dir=None, catalog_path=None,
                 available_bands=None, selected_bands_ml=None, bands_rgb_imaging=None,
                 monochrome_band=None, threads=None, run_number=1, patch_size=8,
                 nr_hac_clusters=None, radial_width=1, patch_step=8, id_col='object_id',
                 tile_col='tile_index', mass_col=None, z_col=None, quality_flag_col=None,
                 quality_flag_val=None, mass_min=None, mass_max=None, z_min=None, z_max=None):
    """Run the full unsupervised galaxy-classification pipeline.

    Parameters
    ----------
    path_model_dir : str
        Output directory for this run (created if missing).
    path_data_dir : str
        Directory containing the per-object ``.npy`` cutouts, each of shape
        ``(len(available_bands), height, width)``.
    catalog_path : str
        Object catalogue (.csv or .npz); see ``load_and_filter_catalog``.
    available_bands : list of str
        Names of the bands stacked in each ``.npy`` file, in array order.
    selected_bands_ml : list of str
        Subset of ``available_bands`` used for feature extraction (>= 2,
        because one of the global features is a colour gradient).
    bands_rgb_imaging : list of str
        Exactly 3 bands used as R, G, B in the colour montages.
    monochrome_band : str
        Band used for the monochrome montages.
    threads : int, 'all' or None
        Number of worker processes; ``None``/``0``/``'all'`` uses every core.
    run_number : int or str
        Identifier of the run, used in log messages and output file names.
    patch_size : int
        Half-width of the square patches in pixels.
    nr_hac_clusters : int
        Number of HAC clusters ("visual words") in the patch vocabulary.
    radial_width : float
        Width of the radial bins used to azimuthally average the power spectrum.
    patch_step : int
        Approximate spacing between patch centres in pixels.
    id_col, tile_col, mass_col, z_col, quality_flag_col, quality_flag_val,
    mass_min, mass_max, z_min, z_max :
        Catalogue columns and selection cuts; see ``load_and_filter_catalog``.

    Outputs (all inside ``path_model_dir``)
    ---------------------------------------
    patches/                         per-patch feature chunks + completion marker
    FINAL_FEATURE_VECT.npy           TF-IDF bag-of-words vector per object
    FINAL_OBJECT_IDS.npy             object IDs matching the rows above
    silhouette_curve_model_<n>.png   silhouette score vs number of clusters
    id_label_featureVect_model_<n>.npy, final_morph_clust<n>.npz
                                     final cluster labels (+ catalogue columns)
    montage_<n>_*/                   per-cluster image montages
    """

    # ---- Validate the required arguments ---------------------------------
    if path_model_dir is None:
        raise ValueError('path_model_dir must be specified')
    if path_data_dir is None:
        raise ValueError('path_data_dir must be specified')
    if available_bands is None or selected_bands_ml is None:
        raise ValueError('available_bands and selected_bands_ml must be specified')
    if bands_rgb_imaging is None or len(bands_rgb_imaging) != 3:
        raise ValueError('bands_rgb_imaging must specify exactly 3 bands.')
    if monochrome_band is None:
        raise ValueError('monochrome_band must be specified.')

    # Validate selected ML bands
    for b in selected_bands_ml:
        if b not in available_bands:
            raise ValueError(f"Selected ML band '{b}' not found in available_bands ({available_bands})")
    if len(selected_bands_ml) < 2:
        raise ValueError("At least 2 bands must be selected for ML feature extraction to compute color gradients.")

    # Validate RGB and Monochrome bands
    for b in bands_rgb_imaging:
        if b not in available_bands:
            raise ValueError(f"RGB band '{b}' not found in available_bands.")
    if monochrome_band not in available_bands:
        raise ValueError(f"Monochrome band '{monochrome_band}' not found in available_bands.")

    # Map band names to array indices (position along axis 0 of each .npy cutout)
    ml_band_indices = [available_bands.index(b) for b in selected_bands_ml]
    rgb_band_indices = [available_bands.index(b) for b in bands_rgb_imaging]
    mono_band_index = available_bands.index(monochrome_band)

    # Resolve CPU count
    if threads is None or threads == 0 or threads == 'all':
        threads = os.cpu_count() or 1
    else:
        threads = int(threads)

    # The patch extractor calls the half-width "window".
    window_size = patch_size
    start_time = time.time()

    # ---- Stage 0: select the objects to classify --------------------------
    object_catalogue = load_and_filter_catalog(
        catalog_path=catalog_path, id_col=id_col, tile_col=tile_col,
        mass_col=mass_col, z_col=z_col, quality_flag_col=quality_flag_col,
        quality_flag_val=quality_flag_val, mass_min=mass_min, mass_max=mass_max,
        z_min=z_min, z_max=z_max
    )

    os.makedirs(path_model_dir, exist_ok=True)

    # ---- Stage 1: patch extraction (parallel, resumable) ------------------
    print(f'model {run_number}: THE PATCH EXTRACTION IS PROCESSING ({len(object_catalogue)} objects using {threads} cores)')
    # Skips the work and returns False if a finished extraction for the same
    # settings and catalogue is already on disk.
    extracted = patch_extractionn(
        path_model_dir=path_model_dir,
        threads=threads,
        window_size=window_size,
        radial_width=radial_width,
        patch_step=patch_step,
        object_catalogue=object_catalogue,
        path_data_dir=path_data_dir,
        ml_band_indices=ml_band_indices,
        id_col=id_col,
        tile_col=tile_col
    )
    if extracted:
        print(f'model {run_number}: PATCH EXTRACTION FINISHED')
    else:
        print(f'model {run_number}: PATCH EXTRACTION SKIPPED (reusing patches on disk)')

    # ---- Stage 2: load all patch features, standardise and PCA-reduce -----
    # samples: (n_patches, n_pca) float32; samples_ids: object ID of each patch
    samples, samples_ids = create_vect(run_number, path_model_dir)

    # ---- Stage 3: compress patches into a 50k-centroid K-Means codebook ---
    # nodes: (50000, n_pca) centroids; scaled_samples: re-standardised patches
    nodes, scaled_samples = optimized_model(run_number, samples)

    # ---- Stage 4: HAC on the nodes -> visual-word label for every patch ---
    clust1_labels, nr_clust = agg_model(run_number, threads, nr_hac_clusters, nodes, scaled_samples, samples_ids)

    # ---- Stage 5: TF-IDF bag-of-visual-words vector for every object ------
    final_feature_vect, final_object_ids = final_features(run_number, samples_ids, threads, clust1_labels, nr_clust)

    print(f'model {run_number}: THE FINAL FEATURE VECTOR HAS BEEN CREATED')
    # Saved to disk because kmeans_model() reads them back from these files.
    np.save(os.path.join(path_model_dir, 'FINAL_FEATURE_VECT.npy'), final_feature_vect)
    np.save(os.path.join(path_model_dir, 'FINAL_OBJECT_IDS.npy'), final_object_ids)

    # ---- Stage 6: final clustering of objects -----------------------------
    # Returns the catalogue merged with 'labels' and 'silhouette_score' columns.
    obj_catalogue = kmeans_model(run_number, path_model_dir, threads, object_catalogue)

    # ---- Stage 7: visual inspection montages ------------------------------
    image_montage(
        run_number=run_number,
        threads=threads,
        path_data_dir=path_data_dir,
        path_model_dir=path_model_dir,
        obj_catalogue=obj_catalogue,
        rgb_band_indices=rgb_band_indices,
        mono_band_index=mono_band_index
    )

    elapsed_hrs = round((time.time() - start_time) / 3600, 2)
    print(f'model {run_number}: time taken including the image montage: {elapsed_hrs} hours')

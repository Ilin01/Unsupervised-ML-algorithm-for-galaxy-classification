import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

path = os.path.abspath(__file__)
code_path = os.path.dirname(path)
sys.path.insert(1, os.path.join(code_path, 'code'))

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

from patch_extr import patch_extractionn
from create_initial_features import create_vect
from create_kmeans_batch_model import optimized_model
from create_agg_model import agg_model
from create_final_features import final_features
from create_kmeans_model import kmeans_model
from image_montage import image_montage


def load_and_filter_catalog(catalog_path, id_col='object_id', tile_col='tile_index',
                            mass_col=None, z_col=None, quality_flag_col=None,
                            quality_flag_val=None, mass_min=None, mass_max=None,
                            z_min=None, z_max=None):
    if catalog_path is None or not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalogue file not found: {catalog_path}")

    ext = os.path.splitext(catalog_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(catalog_path)
    elif ext == '.npz':
        with np.load(catalog_path, allow_pickle=True) as data:
            df = pd.DataFrame({key: data[key] for key in data.files if data[key].ndim == 1})
    else:
        raise ValueError(f"Unsupported catalogue extension '{ext}'. Please provide a .csv or .npz file.")

    if id_col not in df.columns:
        raise KeyError(f"ID column '{id_col}' not found in catalogue columns: {list(df.columns)}")

    if id_col != 'object_id':
        df['object_id'] = df[id_col]

    if tile_col is not None and tile_col in df.columns and tile_col != 'tile_index':
        df['tile_index'] = df[tile_col]

    # --- Robust check: only filter by quality flag if the column is provided ---
    if quality_flag_col is not None and quality_flag_val is not None:
        if quality_flag_col not in df.columns:
            raise KeyError(f"Quality flag column '{quality_flag_col}' not found in catalogue.")
        df = df.loc[df[quality_flag_col] == quality_flag_val]

    # Apply stellar mass cuts
    if mass_col is not None and (mass_min is not None or mass_max is not None):
        if mass_col not in df.columns:
            raise KeyError(f"Stellar mass column '{mass_col}' not found in catalogue.")
        if mass_min is not None:
            df = df.loc[df[mass_col] >= mass_min]
        if mass_max is not None:
            df = df.loc[df[mass_col] <= mass_max]

    # Apply redshift cuts
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

    # Map band names to array indices
    ml_band_indices = [available_bands.index(b) for b in selected_bands_ml]
    rgb_band_indices = [available_bands.index(b) for b in bands_rgb_imaging]
    mono_band_index = available_bands.index(monochrome_band)

    # Resolve CPU count
    if threads is None or threads == 0 or threads == 'all':
        threads = os.cpu_count() or 1
    else:
        threads = int(threads)

    window_size = patch_size
    start_time = time.time()

    object_catalogue = load_and_filter_catalog(
        catalog_path=catalog_path, id_col=id_col, tile_col=tile_col,
        mass_col=mass_col, z_col=z_col, quality_flag_col=quality_flag_col,
        quality_flag_val=quality_flag_val, mass_min=mass_min, mass_max=mass_max,
        z_min=z_min, z_max=z_max
    )

    os.makedirs(path_model_dir, exist_ok=True)

    print(f'model {run_number}: THE PATCH EXTRACTION IS PROCESSING ({len(object_catalogue)} objects using {threads} cores)')
    patch_extractionn(
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
    print(f'model {run_number}: PATCH EXTRACTION FINISHED')

    samples, samples_ids = create_vect(run_number, path_model_dir)
    nodes, scaled_samples = optimized_model(run_number, samples)
    clust1_labels, nr_clust = agg_model(run_number, threads, nr_hac_clusters, nodes, scaled_samples, samples_ids)
    final_feature_vect, final_object_ids = final_features(run_number, samples_ids, threads, clust1_labels, nr_clust)

    print(f'model {run_number}: THE FINAL FEATURE VECTOR HAS BEEN CREATED')
    np.save(os.path.join(path_model_dir, 'FINAL_FEATURE_VECT.npy'), final_feature_vect)
    np.save(os.path.join(path_model_dir, 'FINAL_OBJECT_IDS.npy'), final_object_ids)

    obj_catalogue = kmeans_model(run_number, path_model_dir, threads, object_catalogue)

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
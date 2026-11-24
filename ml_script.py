"""
Example driver script for the unsupervised galaxy-morphology clustering pipeline.

This script only holds configuration: paths, hyperparameters, band choices and
catalogue selection cuts. All of the actual work is done by
``gal_clustering.run_classify``, which runs these stages in order:

    1. Load the object catalogue and apply quality / stellar-mass / redshift cuts.
    2. Cut every galaxy image into small square patches and describe each patch
       by its radially averaged 2D Fourier power spectrum (per band), plus a set
       of global morphology measurements of the whole galaxy.
    3. Standardise the patch features and reduce them with PCA.
    4. Compress the (very large) set of patches into a codebook of 50,000
       Mini-Batch K-Means centroids ("nodes").
    5. Group those nodes into ``nr_hac_clusters`` "visual words" with
       hierarchical agglomerative clustering (HAC), and label every patch by
       the word of its nearest node.
    6. Describe every galaxy as a TF-IDF-weighted histogram of its patch words
       (a "bag of visual words").
    7. Cluster the galaxies with K-Means, choosing the number of clusters from
       a silhouette-score sweep.
    8. Produce image montages (RGB, monochrome, unsharp-masked) for every
       final cluster so the results can be inspected by eye.

Edit the values below and run:  python ml_script.py
"""
import os
from gal_clustering import run_classify

# Model / run identifier. Used to name the output directory (model_<run_number>)
# and to tag log messages and output files, so several runs can coexist.
run_number = '1'

# Paths to main, model, and cutout directories
path_main_dir = '/.../EuclidML/'
# All outputs of this run (patches, feature vectors, plots, montages) go here.
path_model_dir = os.path.join(path_main_dir, f'model_{run_number}/')
# Directory holding the image cutouts. Each object is one .npy file containing a
# stacked array of shape (n_bands, height, width). Files are looked up either as
# <path_data_dir>/tile_<tile_index>/<object_id>.npy or <path_data_dir>/<object_id>.npy.
path_data_dir = '/.../npy_stacked_z15M9_5_5/'

# Pipeline hyperparameters
patch_size = 4          # Half-width of patch in pixels (patches are 2*patch_size x 2*patch_size)
patch_step = 4          # Step size between patches in pixels (patch_step < 2*patch_size => overlapping patches)
radial_width = 1        # Radial bin width for 2D FFT power spectrum
nr_hac_clusters = 1500  # Number of HAC codebook clusters (size of the "visual word" vocabulary)
CPUs = None             # Number of CPU cores to use (Set to None or 'all' to use all available cores)

# --- Band & Imaging Configuration ---
# All bands present in the .npy file stacked array (ordered from shortest to longest wavelength)
available_bands = ['u', 'g', 'r', 'i', 'z','y']

# Specific subset of bands you want the ML model to ingest and extract patches from (must be >= 2)
selected_bands_ml = ['u', 'g', 'r', 'i', 'z']

# 3 bands chosen for the RGB Lupton montage (must be exactly 3 from available_bands ordered from the longest to the shortest wavelength)
# i.e. the first band is shown as red, the second as green and the third as blue.
bands_rgb_imaging = ['H', 'J', 'Y']

# Specific band chosen for the monochrome image montage
monochrome_band = 'g'
# ------------------------------------

# Path to input catalogue (.csv or .npz extension is detected automatically)
catalog_path = '/.../Q1_sample_photoz_15_stellarmass_9_5.npz'

# Catalogue column definitions
id_col = 'object_id'                        # The ID column (must match the .npy file names)
tile_col = 'tile_index'                     # Set to None if cutouts are in a flat folder
mass_col = 'phz_pp_median_stellarmass'      # The stellar mass column name
z_col = 'phz_pp_median_redshift'            # The redshift column name
quality_flag_col = 'flag_resolved'          # The quality flag column name ( Set to None if your catalogue does not contain a quality/star-galaxy flag)
quality_flag_val = 'yes'                    # The value of the quality flag column name (rows with any other value are dropped)

# Physical parameter selection cuts (set any bound to None to disable)
# Mass cuts are inclusive on both ends; the redshift cut is z_min <= z < z_max.
mass_min = 9.5
mass_max = None
z_min = 0.0
z_max = 0.5

# Run classification pipeline
# (The __main__ guard is required because the pipeline uses multiprocessing:
# worker processes re-import this module and must not start the pipeline again.)
if __name__ == '__main__':
    run_classify(
        path_model_dir=path_model_dir,
        path_data_dir=path_data_dir,
        catalog_path=catalog_path,
        available_bands=available_bands,
        selected_bands_ml=selected_bands_ml,
        bands_rgb_imaging=bands_rgb_imaging,
        monochrome_band=monochrome_band,
        threads=CPUs,
        run_number=run_number,
        patch_size=patch_size,
        nr_hac_clusters=nr_hac_clusters,
        radial_width=radial_width,
        patch_step=patch_step,
        id_col=id_col,
        tile_col=tile_col,
        mass_col=mass_col,
        z_col=z_col,
        quality_flag_col=quality_flag_col,
        quality_flag_val=quality_flag_val,
        mass_min=mass_min,
        mass_max=mass_max,
        z_min=z_min,
        z_max=z_max
    )
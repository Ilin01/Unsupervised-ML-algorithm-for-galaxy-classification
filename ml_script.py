import os
from gal_clustering import run_classify

# Model / run identifier
run_number = '1'

# Paths to main, model, and cutout directories
path_main_dir = '/.../EuclidML/'
path_model_dir = os.path.join(path_main_dir, f'model_{run_number}/')
path_data_dir = '/.../npy_stacked_z15M9_5_5/'

# Pipeline hyperparameters
patch_size = 4          # Half-width of patch in pixels
patch_step = 4          # Step size between patches in pixels
radial_width = 1        # Radial bin width for 2D FFT power spectrum
nr_hac_clusters = 1500  # Number of HAC codebook clusters
CPUs = None             # Number of CPU cores to use (Set to None or 'all' to use all available cores)

# --- Band & Imaging Configuration ---
# All bands present in the .npy file stacked array (ordered from shortest to longest wavelength)
available_bands = ['u', 'g', 'r', 'i', 'z','y']

# Specific subset of bands you want the ML model to ingest and extract patches from (must be >= 2)
selected_bands_ml = ['u', 'g', 'r', 'i', 'z']

# 3 bands chosen for the RGB Lupton montage (must be exactly 3 from available_bands ordered from the longest to the shortest wavelength)
bands_rgb_imaging = ['H', 'J', 'Y']

# Specific band chosen for the monochrome image montage
monochrome_band = 'g'
# ------------------------------------

# Path to input catalogue (.csv or .npz extension is detected automatically)
catalog_path = '/.../Q1_sample_photoz_15_stellarmass_9_5.npz'

# Catalogue column definitions
id_col = 'object_id'                        # The ID column
tile_col = 'tile_index'                     # Set to None if cutouts are in a flat folder
mass_col = 'phz_pp_median_stellarmass'      # The stellar mass column name
z_col = 'phz_pp_median_redshift'            # The redshift column name
quality_flag_col = 'flag_resolved'          # The quality flag column name ( Set to None if your catalogue does not contain a quality/star-galaxy flag)
quality_flag_val = 'yes'                    # The value of the quality flag column name

# Physical parameter selection cuts (set any bound to None to disable)
mass_min = 9.5
mass_max = None
z_min = 0.0
z_max = 0.5

# Run classification pipeline
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
import glob
from gal_clustering import run_classify
import os
import numpy as np
import pandas as pd



# model number (int)
run_number='43'


# path to main directory
path_main_dir='/beegfs/car/ilazar/EuclidML/'

# path to directory for specific model
path_model_dir=path_main_dir+'model_'+str(run_number)+'/'


# path to data directory
#path_data_dir='/beegfs/car/ilazar/EuclidML/data/Q1/cutouts/npy_stacked_z015M7_clean/'
path_data_dir='/beegfs/car/ilazar/EuclidML/data/Q1/cutouts/npy_stacked_z15M9_5_5/'
#fields=['COSMOS'] # Hyper Supreme-Cam fields


patch_size = 4 # nr. bins for Fourier Transform radial profile which is half of the patch size (actual patch size is twice)
patch_step = 4 # space between patches in pixels
nr_hac_clusters = 1500
CPUs = 30 # number of CPU cores to use
bands = ['VIS','Y','J','H'] # imaging filters
bands_rgb_imaging=['H','J','Y'] # 3 bands in order from longer to smaller wavelengths (for rgb imaging purposes)

# path to catalogs with physical parameters
#catalogs_path='...'

# HSC catalogs 
#Euclid_catalog_csv='/beegfs/car/ilazar/EuclidML/catalogs/sample_final_z015M7.csv'

Euclid_catalog_csv=None

Euclid_catalog_npz='/beegfs/car/ilazar/EuclidML/catalogs/Q1_sample_photoz_15_stellarmass_9_5.npz'

VIS_band_only=False

# mass and redshift constraints to be used by the algorithm
#Z_max=0.15   # can be None or float
#Z_min=0      # can only be float or int
#mass_min=7   # can only be float or int
#mass_max=9.5  # can be None or float



# use the stars flag from the HSC catalog
#filter_stars=True


# Run classification

run_classify(path_model_dir=path_model_dir, path_data_dir=path_data_dir, 
             bands=bands,
             threads=CPUs,
             run_number=run_number,
             patch_size=patch_size,
             nr_hac_clusters=nr_hac_clusters,
             radial_width=1,
             patch_step=patch_step,bands_rgb_imaging=bands_rgb_imaging,
             Euclid_catalog_csv=Euclid_catalog_csv,Euclid_catalog_npz=Euclid_catalog_npz,VIS_band_only=VIS_band_only)

import glob
from gal_clustering import run_classify
import os
import numpy as np
import pandas as pd



# model number (int)
run_number='...'


# path to main directory
path_main_dir='...'

# path to directory for specific model
path_model_dir=path_main_dir+'model_'+str(run_number)+'/'


# path to data directory
path_data_dir='/beegfs/car/ilazar/data/HSC_DR3/DEEP/'
fields=['COSMOS','DEEP23','ELAISN1','SXDSXMMLSS'] # Hyper Supreme-Cam fields


patch_size = 4 # nr. bins for Fourier Transform radial profile which is half of the patch size (actual patch size is twice)
patch_step = 4 # space between patches in pixels
nr_hac_clusters = 1500
CPUs = 1 # number of CPU cores to use
bands = ['G','R','I','Z'] # imaging filters
bands_rgb_imaging=['I','R','G'] # 3 bands in order from longer to smaller wavelengths (for rgb imaging purposes)

# path to catalogs with physical parameters
catalogs_path='...'

# HSC catalogs 
HSC_catalogs=['hsc_cosmos_catalogue.npz',
'hsc_deep23_catalogue.npz',
'hsc_elaisn1_catalogue.npz',
'hsc_sxdsxmmlss_catalogue.npz']


# mass and redshift constraints to be used by the algorithm
Z_max=0.3   # can be None or float
Z_min=0      # can only be float or int
mass_min=8   # can only be float or int
mass_max=9.5  # can be None or float



# use the stars flag from the HSC catalog
filter_stars=True


# Run classification

run_classify(path_model_dir=path_model_dir, path_data_dir=path_data_dir, 
             bands=bands,
             threads=CPUs,
             run_number=run_number,
             patch_size=patch_size,
             nr_hac_clusters=nr_hac_clusters,
             radial_width=1,
             Z_max=Z_max,Z_min=Z_min,mass_min=mass_min,mass_max=mass_max,
             patch_step=patch_step,bands_rgb_imaging=bands_rgb_imaging,
             filter_stars=filter_stars,
             catalogs_path=catalogs_path,fields=fields,HSC_catalogs=HSC_catalogs)

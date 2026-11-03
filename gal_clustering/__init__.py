import os
import sys
import warnings
path = os.path.abspath(__file__)
code_path = os.path.dirname(path)
sys.path.insert(1, code_path+'/code')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import numpy as np
from patch_extr import patch_extractionn
from create_initial_features import create_vect
from create_kmeans_batch_model import optimized_model
from create_agg_model import agg_model
from create_final_features import final_features
from create_kmeans_model import kmeans_model
from image_montage import image_montage
import time
import pandas as pd


def run_classify(path_model_dir=None, path_data_dir=None,
    bands=None,
    threads=1,
    run_number=1,
    patch_size=8,
    nr_hac_clusters=None,
    radial_width=1,
    patch_step=8,
    bands_rgb_imaging=None,
    Euclid_catalog_csv=None,
    Euclid_catalog_npz=None,
    VIS_band_only=False):


    
    window_size=patch_size



    """
    Clusters objects found in a list of astronomical images by their visual similarity.
    """
    if path_model_dir is None:
        raise ValueError('path_model_dir must be specified and of type string')
    if path_data_dir is None:
        raise ValueError('path_data_dir must be specified and of type string')
    if bands is None:
        raise SyntaxError('Specify band names')


    start_time = time.time()

    if Euclid_catalog_npz==None:

        object_catalogue=pd.read_csv(Euclid_catalog_csv)

    if Euclid_catalog_csv==None:
        data = np.load(Euclid_catalog_npz)
        object_catalogue=pd.DataFrame({key: data[key] for key in data.files})
        object_catalogue=object_catalogue.loc[object_catalogue['flag_resolved']=='yes']
        object_catalogue = object_catalogue.loc[(object_catalogue['phz_pp_median_stellarmass']>=10.5) & (object_catalogue['phz_pp_median_redshift']<0.5) ]
        data.close()
        #print(object_catalogue['object_id'][:10])
        #print(object_catalogue['tile_index'][:10])

    
    # Run patch extraction (calculate radial power spectrums for griz galaxy images)
    
    if not os.path.exists(path_model_dir):
        os.mkdir(path_model_dir)
    print('model '+str(run_number)+': THE PATCH EXTRACTION IS PROCESSSING')
    patch_extractionn(path_model_dir,threads,window_size, radial_width, patch_step,Euclid_catalog_csv,Euclid_catalog_npz,path_data_dir,VIS_band_only)
    print('model '+str(run_number)+': PATCH EXTRACTION FINISHED')


    
    
    # Create initial feature vector (append all radial power spectrums to a data matrix)
    samples, samples_ids=create_vect(run_number,path_model_dir,VIS_band_only)

    
    # Create miniBatch Kmenas model (reduce the size of the feature space by means of minibatch kmeans)
    nodes,scaled_samples=optimized_model2(run_number,samples)

    #print(len(nodes),len(scaled_samples))

    #np.save('/beegfs/car/ilazar/EuclidML/scripts/nodes.npy',nodes)
    #np.save('/beegfs/car/ilazar/EuclidML/scripts/scaled_samples.npy',scaled_samples)

    #nodes=np.load('/beegfs/car/ilazar/EuclidML/scripts/nodes.npy')
    #scaled_samples=np.load('/beegfs/car/ilazar/EuclidML/scripts/scaled_samples.npy')





    # Perform Agglomerative Clustering on the GNG nodes created and assign a patch type label to members of each cluster 
    clust1_labels,nr_clust=agg_model(run_number,threads,nr_hac_clusters,nodes,scaled_samples,samples_ids)
    


    


    # Create the final feature data matrix where each row corresponds to a galaxy representing a histogram
    # where each bin is the patch number of a certain patch type within the galaxy in question
    final_feature_vect,final_object_ids=final_features(run_number,samples_ids,threads,clust1_labels, nr_clust)



    print('model '+str(run_number)+': THE FINAL FEATURE VECTOR HAS BEEN CREATED')



    np.save(path_model_dir+'FINAL_FEATURE_VECT.npy',final_feature_vect)
    np.save(path_model_dir+'FINAL_OBJECT_IDS.npy',final_object_ids)

    #print(object_catalogue.columns)



    #object_catalogue=object_catalogue.rename(columns={"object_id": "object_id2"})

    #object_catalogue['object_id']=object_catalogue['ind'].values


    # Create final morphological clusters (k-means clustering)
    obj_catalogue=kmeans_model(run_number,path_model_dir,threads,object_catalogue)

    

    #qwef

    #npz = np.load('/beegfs/car/ilazar/EuclidML/model_4/final_morph_clust4.npz',allow_pickle=True)
    #obj_catalogue= pd.DataFrame.from_dict({item: npz[item] for item in npz.files})

    #print(obj_catalogue['object_id'][:10])
    #print(obj_catalogue['tile_index'][:10])

    #qwef

    #print(len(obj_catalogue))


    # Create an image montage of the morphological clusters
    image_montage(run_number,threads,path_data_dir,path_model_dir,obj_catalogue)

    time_after_montage = time.time()


    print('model '+str(run_number)+': time taken including the image montage: ' + str(round((time_after_montage-start_time)/3600,2)) + ' hours' )













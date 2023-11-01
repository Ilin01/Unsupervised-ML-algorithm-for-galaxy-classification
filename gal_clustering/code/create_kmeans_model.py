import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score as score



def kmeans_model(run_number,path_model_dir,threads,object_catalogue):

    print('model '+str(run_number)+': THE FINAL MORPHOLOGICAL CLUSTERS ARE BEING CREATED')

    

    FINAL_OBJECT_IDS=np.load(path_model_dir+'FINAL_OBJECT_IDS.npy',allow_pickle=True)
    FINAL_FEATURE_VECT=np.load(path_model_dir+'FINAL_FEATURE_VECT.npy',allow_pickle=True)
    FINAL_FIELD_VECT=np.load(path_model_dir+'FINAL_FIELD_VECT.npy',allow_pickle=True)



    d = {'object_id': FINAL_OBJECT_IDS}
    objectIDs = pd.DataFrame(data=d)


    feature_vectt = pd.DataFrame(FINAL_FEATURE_VECT)

    overall_sil_scores_array=[]

    lower_clust_lim=160
    higher_clust_lim=300

    nr_final_clusters_array=np.array(range(lower_clust_lim,higher_clust_lim,10))


    # find model with highest silh score by changing the number of clusters and then redo the model with the optimal prameters
    for nr_final_clusters in nr_final_clusters_array:

        kmeans = KMeans(n_clusters=nr_final_clusters, random_state=0,verbose=0,n_jobs=threads).fit(FINAL_FEATURE_VECT)
        print('nr iter run:'+str(kmeans.n_iter_))

        clust2_labels=kmeans.labels_


        sil_score_for_model1=score(FINAL_FEATURE_VECT,clust2_labels,metric='euclidean',sample_size=10000)
        sil_score_for_model2=np.round(sil_score_for_model1,3)


        overall_sil_scores_array.append(sil_score_for_model1)

        print('model '+str(run_number)+': the silhouette score for the final clustering model with '+str(nr_final_clusters)+' clusters is: ', sil_score_for_model2)


    

    models=pd.DataFrame({'nr_clusters': nr_final_clusters_array,'silhouette_score': overall_sil_scores_array})

    max_valsIndex=models['silhouette_score'].idxmax()

    max_score_nrClust=models['nr_clusters'][max_valsIndex]

    max_silhouette_score=models['silhouette_score'][max_valsIndex]

    print('model '+str(run_number)+': the number of clusters with the highest silhouette score (which is '+str(np.round(max_silhouette_score,3))+') is ' +str(max_score_nrClust))


    kmeans = KMeans(n_clusters=max_score_nrClust, random_state=0,verbose=0,n_jobs=threads).fit(FINAL_FEATURE_VECT)

    clust2_labels=kmeans.labels_

    sil_score_for_model=score(FINAL_FEATURE_VECT,clust2_labels,metric='euclidean')

    print('model '+str(run_number)+': the actual silhouette score for the final clustering model with '+str(max_score_nrClust)+' clusters is: ', np.round(sil_score_for_model,3))


    sil_score=silhouette_samples(FINAL_FEATURE_VECT,clust2_labels,metric='euclidean')

    d = {'silhouette_score': sil_score}
    sil_score = pd.DataFrame(data=d)


    d = {'labels': clust2_labels}
    clust2_labels=pd.DataFrame(data=d)

    d = {'field': FINAL_FIELD_VECT}
    fields=pd.DataFrame(data=d)

    df10=objectIDs.join(fields)

    df2=df10.join(clust2_labels)


    df3=df2.join(feature_vectt)


    # create the feature vector for the specific model
    np.save(path_model_dir+'id_field_label_featureVect_model_'+str(run_number)+'.npy',df3.values)


    df=df2.join(sil_score)

    obj_cat=object_catalogue
    
    obj_catalogue = pd.merge(df, obj_cat, on=["object_id", "field"])


    # create a catalog containing the initial objects with their morphology label along with their physical properties
    np.savez_compressed(path_model_dir+'final_morph_clust'+str(run_number)+'.npz',object_id=obj_catalogue['object_id'].values,
    field=obj_catalogue['field'].values,
    labels=obj_catalogue.labels.values, silhouette_score=obj_catalogue.silhouette_score.values,
    ra=obj_catalogue['ra'].values,
    dec=obj_catalogue['dec'].values,skymap_id=obj_catalogue['skymap_id'].values,tract=obj_catalogue['tract'].values,patch=obj_catalogue['patch'].values,
    footprint_Xmin=obj_catalogue['footprint_Xmin'].values,footprint_Ymin=obj_catalogue['footprint_Ymin'].values,footprint_Xmax=obj_catalogue['footprint_Xmax'].values,
    footprint_Ymax=obj_catalogue['footprint_Ymax'].values,
    g_extendedness_value=obj_catalogue['g_extendedness_value'].values,r_extendedness_value=obj_catalogue['r_extendedness_value'].values,
    i_extendedness_value=obj_catalogue['i_extendedness_value'].values,z_extendedness_value=obj_catalogue['z_extendedness_value'].values,
    photoz_best=obj_catalogue['photoz_best'].values,photoz_std_best=obj_catalogue['photoz_std_best'].values,
    stellar_mass=obj_catalogue['stellar_mass'].values,stellar_mass_err68_min=obj_catalogue['stellar_mass_err68_min'].values,
    stellar_mass_err68_max=obj_catalogue['stellar_mass_err68_max'].values,sfr=obj_catalogue['sfr'].values,sfr_err68_min=obj_catalogue['sfr_err68_min'].values,
    sfr_err68_max=obj_catalogue['sfr_err68_max'].values)


    print('model '+str(run_number)+': THE FINAL CLUSTERING IS FINISHED')

    return obj_catalogue
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Manager

def final_features(run_number,samples_ids,threads,clust1_labels, fields,samples_field,nr_clust):

    print('model '+str(run_number)+': THE FINAL FEATURE VECTOR IS BEING CREATED')

    df11=pd.DataFrame({'labels': clust1_labels.astype(int), 'objectID': samples_ids.astype(int), 'field':samples_field.astype(str)})

    ALL2=Manager().list()

    n_groups=nr_clust

    bins=np.arange(-0.5,n_groups+0.5,1)

    for field in fields:
        objects=df11['objectID'][df11['field']==field].drop_duplicates().values

        global make_hist_vectors

        def make_hist_vectors(objectt):


            if len(np.array([df11['labels'][(df11['objectID']==objectt) & (df11['field']==field)]]).shape)==1:
                hist_labels=np.array([df11['labels'][(df11['objectID']==objectt) & (df11['field']==field)]])

            else:

                hist_labels=df11['labels'][(df11['objectID']==objectt) & (df11['field']==field)].values

            hist,oth=np.histogram(hist_labels, bins=bins)
            hist=hist/(np.sum(hist))
            ALL2.append([field,hist,objectt])


        with Pool(threads) as b:
            r=list(tqdm(b.imap(make_hist_vectors, [objectt for objectt in objects]),total=len(objects)))

    final_feature_vect=np.array(list(zip(*ALL2))[1])
    final_object_ids=np.array(list(zip(*ALL2))[2])
    final_field_vect=np.array(list(zip(*ALL2))[0])



    return final_feature_vect,final_object_ids,final_field_vect
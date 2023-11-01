import glob
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Manager


def create_vect(run_number,path_model_dir,threads,fields):

    print('model '+str(run_number)+': CREATION OF INITIAL FEATURE VECTOR ...')

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    FILES_samp=[]
    FILES_ID=[]
    FIELD_arr=[]

    for field in fields:
        path_patch_dir=path_model_dir+'patches_'+field+'/'

        files_samp=glob.glob(path_patch_dir+'*/samples.npy')
        files_id=glob.glob(path_patch_dir+'*/samples_object_id.npy')

        files_samp.sort(key=natural_keys)
        files_id.sort(key=natural_keys)

        field_array=[field for i in range(len(files_id))]
        FIELD_arr.extend(field_array)

        FILES_samp.extend(files_samp)
        FILES_ID.extend(files_id)


    ALL=Manager().list()

    global create_vect
    

    def create_vect(idx):

        ALL.append([np.load(FILES_samp[idx],allow_pickle=True).tolist(),np.load(FILES_ID[idx],allow_pickle=True).tolist(),[FIELD_arr[idx] for i in range(len(np.load(FILES_ID[idx],allow_pickle=True)))]])

    with Pool(threads) as b:
        r=list(tqdm(b.imap(create_vect, [idx for idx in range(len(FILES_ID))]),total=len(FILES_ID)))
    
    del create_vect

    samples=np.array([j for sub in list(zip(*ALL))[0] for j in sub])
    samples_ids=np.array([j for sub in list(zip(*ALL))[1] for j in sub])
    samples_field=np.array([j for sub in list(zip(*ALL))[2] for j in sub])
    
    print('model '+str(run_number)+': The number of patches extracted is: ',len(samples_ids))


 
    # Pick patches with highest dispersion from the mean to be first in the matrix (first vertically and next horiz)

    
    samples1=np.empty(np.shape(samples))
    
    for column_index in range(len(samples[0])):

        mean=np.mean(samples[:,column_index])

        for row_index in range(len(samples)):

            val=abs(samples[row_index][column_index]-mean)

            samples1[row_index][column_index]=val

    samples2=np.empty(len(samples))

    for row_index in range(len(samples1)):

        stdd=np.std(samples1[row_index])

        samples2[row_index]=stdd

    df=pd.DataFrame(samples)


    df['order_std']=samples2

    df['samples_ids']=samples_ids

    df['samples_field']=samples_field

    df=df.sort_values(by='order_std', ascending=True)

    df=df.reset_index()

    df=df.drop(['index'], axis=1)

    samples_ids=np.array(df['samples_ids'].values)

    samples_field=np.array(df['samples_field'].values)


    df=df.drop(columns=['samples_ids', 'order_std','samples_field'])

    samples=np.array(df.values)

    del df
    

    return samples,samples_ids,samples_field
    


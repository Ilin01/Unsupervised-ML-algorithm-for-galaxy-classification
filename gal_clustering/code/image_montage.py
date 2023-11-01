from astropy.visualization import make_lupton_rgb
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Pool

def image_montage(run_number,threads,path_data_dir,path_model_dir,bands_rgb_imaging,obj_catalogue):

    print('model '+str(run_number)+': THE IMAGE MONTAGE IS BEING PRODUCED')

    
    def make_plot_lupton(images_data,objectID,silhouette_score,i,axs,tup,obj_nr,field):

        scale = [0.5, 0.7, 1.0]

        image_r = images_data[0]
        image_g = images_data[1]
        image_b = images_data[2]

        subplot_name=str(objectID)+'_'+field[0]+'_'+str(silhouette_score)


        image_r[~np.isfinite(image_r)]=0.
        image_g[~np.isfinite(image_g)]=0.
        image_b[~np.isfinite(image_b)]=0.
        
        image = make_lupton_rgb(image_r*scale[0], image_g*scale[1], image_b*scale[2], Q=10, stretch=0.5)

        if obj_nr==1:
            axs.imshow(image, aspect='equal')
            axs.set_axis_off()
            axs.set_title(subplot_name, fontsize=20)

        if obj_nr>1 and obj_nr<30:

            if obj_nr==2:

                axs[tup[i]].imshow(image, aspect='equal')
                axs[tup[i]].set_axis_off()
                axs[tup[i]].set_title(subplot_name, fontsize=20)

            if obj_nr>2:

                axs[tup[i][0], tup[i][1]].imshow(image, aspect='equal')
                axs[tup[i][0], tup[i][1]].set_axis_off()
                axs[tup[i][0], tup[i][1]].set_title(subplot_name,fontsize=20)



        if obj_nr==30:

            axs[tup[i][0], tup[i][1]].imshow(image, aspect='equal')
            axs[tup[i][0], tup[i][1]].set_axis_off()
            axs[tup[i][0], tup[i][1]].set_title(subplot_name, fontsize=25)
            


    ind=obj_catalogue.index


    d = {'ind': ind}
    ind = pd.DataFrame(data=d)

    obj_catalogue=obj_catalogue.join(ind)


    obj_catalogue.set_index('labels',inplace=True)

    clusters=sorted(obj_catalogue.index.drop_duplicates().astype(int))

    os.chdir(path_data_dir)

    path_galaxy_montage=path_model_dir+'output_montage_'+str(run_number)
    if not os.path.exists(path_galaxy_montage):

        os.mkdir(path_galaxy_montage)

    bands=bands_rgb_imaging

    global process2

    def process2(cluster):


        path=path_galaxy_montage+'/'

        
        if len(np.array([obj_catalogue['object_id'][cluster]]).shape)==1:

            objectID=obj_catalogue['object_id'][cluster].astype(int)

            silhouette_score=obj_catalogue['silhouette_score'][cluster]
            fieldd=obj_catalogue['field'][cluster]


            images_data=np.load('/beegfs/car/ilazar/models/model_187/footpr_realigned/'+str(objectID)+'.npy',allow_pickle=True)

            images_data=images_data[::-1][1:]
            rows=1
            cols=1
            fig, axs = plt.subplots(rows, cols)
            obj_nr=1
            rows=1
            cols=1
            i=0
            tup=[]

            for row in range(rows):
                for col in range(cols):
                    tup.append([row,col])


            make_plot_lupton(images_data,objectID,silhouette_score,i,axs,tup,obj_nr,fieldd)

            fig.tight_layout()
            fig.savefig(path+str(cluster))


        if len(np.array([obj_catalogue['object_id'][cluster]]).shape)!=1 and len(obj_catalogue['object_id'][cluster].values)>=30:


            DF_MOD=obj_catalogue.loc[cluster]
            DF_MOD.set_index('ind',inplace=True)

            df_large=DF_MOD.nlargest(10, 'silhouette_score')
            df_large=df_large.sort_values(by=['silhouette_score'],ascending=False)
            df_large_index=df_large.index

            df_small=DF_MOD.drop(df_large_index)

            df_small=df_small.nsmallest(10, 'silhouette_score')
            df_small=df_small.sort_values(by=['silhouette_score'],ascending=False)
            df_small_index=df_small.index


            index_small_large=np.concatenate((df_small_index,df_large_index))
            new_df=DF_MOD.drop(index_small_large)

            new_df=new_df.sample(n = 10)

            frames = [df_large, new_df, df_small]

            result = pd.concat(frames)

            objectID=result['object_id'].values.astype(int)
            silhouette_score=np.round(result['silhouette_score'].values,3)
            fieldd=result['field'].values

            rows=5
            cols=6

            map_size=[36,30]
            fig, axs = plt.subplots(rows, cols,figsize=map_size)

            tup=[]

            for row in range(rows):
                for col in range(cols):
                    tup.append([row,col])

            obj_nr=len(objectID)




            for i in range(len(objectID)):

                objectID_loop=objectID[i]

                silhouette_score_loop=silhouette_score[i]
                fieldd_loop=fieldd[i]

                images_data=np.load('/beegfs/car/ilazar/models/model_187/footpr_realigned/'+str(objectID_loop)+'.npy',allow_pickle=True)

                
                
                images_data=images_data[::-1][1:]
            
                make_plot_lupton(images_data,objectID_loop,silhouette_score_loop,i,axs,tup,obj_nr,fieldd_loop)

            fig.suptitle('Nr. objects: '+str(len(DF_MOD)),fontsize=25,y=1.2)
            fig.tight_layout()
            fig.savefig(path+str(cluster))
            del DF_MOD



        if len(np.array([obj_catalogue['object_id'][cluster]]).shape)!=1 and len(obj_catalogue['object_id'][cluster].values)<30:


            DF_MOD=obj_catalogue.loc[cluster]
            
            DF_MOD.set_index('ind',inplace=True)


            DF_MOD=DF_MOD.sort_values(by=['silhouette_score'],ascending=False)

            objectID=DF_MOD['object_id'].values.astype(int)
            silhouette_score=np.round(DF_MOD['silhouette_score'].values,3)
            fieldd=DF_MOD['field'].values

            sqr=np.sqrt(len(objectID))

            if sqr%1>0.5:
                cols=int(np.sqrt(len(objectID))+1)
                rows=cols

            if sqr%1<0.5:
                cols=int(np.sqrt(len(objectID))+1)
                rows=cols-1


            subimage_size=5

            map_size=[cols*subimage_size,rows*subimage_size]

            fig, axs = plt.subplots(rows, cols,figsize=map_size)


            tup=[]

            for row in range(rows):
                for col in range(cols):
                    if round(rows*cols,0)==2:
                        tup.append(col)

                    else:

                        tup.append([row,col])

            obj_nr=len(objectID)



            for i in range(len(objectID)):

                objectID_loop=objectID[i]
                silhouette_score_loop=silhouette_score[i]
                fieldd_loop=fieldd[i]

                images_data=np.load('/beegfs/car/ilazar/models/model_187/footpr_realigned/'+str(objectID_loop)+'.npy',allow_pickle=True)

                images_data=images_data[::-1][1:]
                make_plot_lupton(images_data,objectID_loop,silhouette_score_loop,i,axs,tup,obj_nr,fieldd_loop)

            if len(objectID)< rows*cols:

                for j in range(i+1,len(tup)):
                    axs[tup[j][0], tup[j][1]].set_axis_off()

            fig.suptitle('Nr. objects: '+str(len(DF_MOD)),fontsize=25,y=1.2)
            fig.tight_layout()
            fig.savefig(path+str(cluster))
            del DF_MOD
            


    with Pool(threads) as b:
        r=list(tqdm(b.imap(process2, [k for k in clusters]),total=len(clusters)))

    print('model '+str(run_number)+': THE IMAGE MONTAGE IS FINISHED')


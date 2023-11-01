import sys
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import numpy.fft as npfft
from multiprocessing import Pool
from multiprocessing import Process, Manager


def get_clip(x, y, window_size,im):

    image = im

    top = y + window_size
    bottom = y - window_size
    left = x - window_size
    right = x + window_size


    return image[bottom:top,left:right]

def get_adjusted_patch_all_images(x_pos, y_pos, window_size,images):

    offset_xpos = x_pos
    offset_ypos = y_pos

    patches = []
    for i in range(len(images)):
        r_clip = get_clip(offset_xpos, offset_ypos, window_size,images[i])
        patches.append(r_clip)

    return patches


def azimuthal_average_new(image,image_shape,radial_width):
    bin_size=radial_width
    y, x = np.indices(image.shape)

    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])

    num_bins = int(np.round(r.max() / bin_size)+1)
    max_bin = num_bins * bin_size
    bins = np.linspace(0, max_bin, num_bins+1)

    hist = np.histogram(r, bins)[0]
    hist[hist == 0] = 1

    radial_prof = np.histogram(r, bins, weights=(image))[0] / hist.astype(np.float)

    return radial_prof

def get_patch_power_spectrum(patch,image_shape,radial_width):

    FA = npfft.fft2(patch)
    FA_FBconj = npfft.fftshift(FA * np.conjugate(FA))
    trans = np.real(FA_FBconj)
    profile2 = azimuthal_average_new(trans,image_shape,radial_width)

    return profile2

def process_patches(patches,image_shape,radial_width):
    d_clip = np.array([])
    num_vals = 0
    for i in range(len(patches)):
        patch = patches[i]
        ps = get_patch_power_spectrum(patch,image_shape,radial_width)
        num_vals = len(ps)
        ps = ps[0:-1]
        d_clip = np.concatenate((d_clip, ps), axis=0)
    return d_clip, num_vals - 1

def round_up(x):
    return np.round(x + 0.5)



def check_patches(patches, num_pixels):

    skip=False
    for patch_iter in range(len(patches)):

        if np.count_nonzero(patches[patch_iter]) < (num_pixels*1):
            skip = True
            break

    return skip
        


def patch_extractionn(path_model_dir, threads,window_size, radial_width,Z_max, Z_min, mass_min, mass_max, patch_step,filter_stars,catalogs_path,fields,HSC_catalogs,path_data_dir):

    patch_shape = np.array([window_size*2, window_size*2])


    
    for field in range(len(fields)):

        path_patch_dir=path_model_dir+'patches_'+fields[field]+'/'
        
        npz = np.load(catalogs_path+HSC_catalogs[field],allow_pickle=True)
        object_catalogue_cosmos=pd.DataFrame.from_dict({item: npz[item] for item in npz.files})

        object_catalogue_cosmos=object_catalogue_cosmos.drop_duplicates(subset=['object_id'])


        # define selection function for stellar mass and redshift for each field as specified in ml_script.py 
        if filter_stars:
            object_catalogue_cosmos=object_catalogue_cosmos[(object_catalogue_cosmos['g_extendedness_value']==1) & (object_catalogue_cosmos['r_extendedness_value']==1) & (object_catalogue_cosmos['i_extendedness_value']==1) & (object_catalogue_cosmos['z_extendedness_value']==1)]

        if mass_max==None and Z_max!=None:
            object_catalogue_cosmos=object_catalogue_cosmos[(object_catalogue_cosmos['photoz_best']<Z_max) & (object_catalogue_cosmos['photoz_best']>Z_min) & (object_catalogue_cosmos['stellar_mass']>mass_min)]

        if mass_max!=None and Z_max==None:
            object_catalogue_cosmos=object_catalogue_cosmos[(object_catalogue_cosmos['photoz_best']>Z_min) & (object_catalogue_cosmos['stellar_mass']>mass_min) & (object_catalogue_cosmos['stellar_mass']<mass_max)]

        if mass_max==None and Z_max==None:
            object_catalogue_cosmos=object_catalogue_cosmos[(object_catalogue_cosmos['photoz_best']>Z_min) & (object_catalogue_cosmos['stellar_mass']>mass_min)]

        if mass_max!=None and Z_max!=None:
            object_catalogue_cosmos=object_catalogue_cosmos[(object_catalogue_cosmos['photoz_best']<Z_max) & (object_catalogue_cosmos['photoz_best']>Z_min) & (object_catalogue_cosmos['stellar_mass']>mass_min) & (object_catalogue_cosmos['stellar_mass']<mass_max)]

        object_catalogue_cosmos=object_catalogue_cosmos.dropna()

        print('The sky areas are processing')


        tracts=object_catalogue_cosmos['tract'].drop_duplicates().values

        
        # use the patch size and space between patches
        step=patch_step
        window = window_size
        image_size = [window*2, window*2]

        num_pixels = image_size[0]*image_size[1]

        min_width=20 #pixels
        min_height=20 #pixels


        # loop through survey tracts
        for tract in tqdm(tracts):
            n=0
            patches=object_catalogue_cosmos['patch'][object_catalogue_cosmos['tract']==tract].drop_duplicates().values

            # loop through survey patches  (to not be confused with the patch from which power spectrums
            #  are calculated which are less than a galaxy in size)
            for patch in patches:
                n+=1
                ALL=Manager().list()
                object_ids=object_catalogue_cosmos['object_id'][(object_catalogue_cosmos['tract']==tract) & (object_catalogue_cosmos['patch']==patch)]

                sky_area_id=n
                output_folder_path = path_patch_dir  + str(sky_area_id) + '/'

    
    
                # create output dir
                if not os.path.isdir(output_folder_path):
                    os.makedirs(output_folder_path)

                
                # begin paralell processing of galaxies

                global extract_patch
                def extract_patch(obj):

                    try:

                        images=np.load(path_data_dir+fields[field]+'_footprints/'+str(obj)+'.npy',allow_pickle=True)

                        Xmin=0
                        Ymin=0
                        Xmax=len(images[0][0])-1
                        Ymax=len(images[0])-1


                        if Xmin<0 or Xmax<0 or Ymin<0 or Ymax<0:
                            raise ValueError('bad data')

                        # ignore objects with footprints less than 20 pixels in size
                        if Xmax-Xmin<=min_width or Ymax-Ymin<=min_height:
                            raise ValueError('bad data')

                        if Xmax-Xmin>min_width and Ymax-Ymin>min_height:
                            X_range=np.linspace(int(Xmin)+window,int(Xmax)-window,int(round_up((int(Xmax)-int(Xmin))/step))).astype(int)
                            Y_range=np.linspace(int(Ymin)+window,int(Ymax)-window,int(round_up((int(Ymax)-int(Ymin))/step))).astype(int)

                            for x in X_range:
                                for y in Y_range:
                                    patches = get_adjusted_patch_all_images(x, y, window,images)
                                    if check_patches(patches, num_pixels):
                                        continue
                                    sample, num_ps_values = process_patches(patches,patch_shape,radial_width)
                                    ALL.append([obj,sample.tolist()])

                    except Exception as pn:
                        pass
                with Pool(threads) as b:
                    r=list(b.imap(extract_patch, [i for i in object_ids]))

                try:
                    ALL=np.array(ALL)


                    samples_object_id=np.array(ALL[:,0])
                    samples=np.array(ALL[:,1])

                    np.save(output_folder_path + 'samples.npy',np.array(samples))
                    np.save(output_folder_path + 'samples_object_id.npy',np.array(samples_object_id))

                    pos=len(samples_object_id)

                    if pos==0:
                        os.system('rm -R '+ output_folder_path)

                except Exception:
                    os.system('rm -R '+ output_folder_path)
                    pass



#  function - compare the body track result from different only one camera
def body_part_locs_singlecam(bodyparts_camN_camNM,cameraID,singlecam_ana_type,animalnames_videotrack,bodypartnames_videotrack,date_tgt):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    # load camN h5 file
    bodyparts_camN_camNM_data = pd.read_hdf(bodyparts_camN_camNM)
    nframes = bodyparts_camN_camNM_data.shape[0]    

    # get body part location from one single camera
    body_part_locs_notfix = {}
    body_part_locs = {}
  
    nbodies = np.shape(bodypartnames_videotrack)[0]

    for iname in animalnames_videotrack:
        for ibody in np.arange(0,nbodies,1):
            ibodyname = bodypartnames_videotrack[ibody]

            xxx = bodyparts_camN_camNM_data[(singlecam_ana_type, iname, ibodyname, 'x')]
            #
	    # fill in the nan data point
            # xxx_filled = xxx.interpolate(method='nearest',limit_direction='both')
            # xxx_filled = xxx_filled.interpolate(method='linear',limit_direction='both')
            xxx_filled = xxx
            #            
            # remove outlier
            q1 = np.nanquantile(xxx_filled,0.25)
            q3 = np.nanquantile(xxx_filled,0.75)
            thres1 = q1 - 1.5*abs(q3-q1)
            thres2 = q3 + 1.5*abs(q3-q1)
            ind_x = (xxx_filled>thres2) | (xxx_filled<thres1)
            # xxx_filled[ind_x] = np.nan
            #        
            # fill in the nan data point after outlier removal
            # xxx_filled = xxx_filled.interpolate(method='linear',limit_direction='both')
            #        
            # smooth the data point   
            # std was chosen based on the data (mean std across session, animal, body part and x,y,z is ~2)
            # xxx_filtered = xxx_filled.rolling(window=10, win_type='gaussian', center=True).mean(std=2)
            #
            xxx = xxx_filled


            yyy = bodyparts_camN_camNM_data[(singlecam_ana_type, iname, ibodyname, 'y')]
	    #
            # fill in the nan data point
            # yyy_filled = yyy.interpolate(method='nearest',limit_direction='both')
            # yyy_filled = yyy_filled.interpolate(method='linear',limit_direction='both')
            yyy_filled = yyy
            #            
            # remove outlier
            q1 = np.nanquantile(yyy_filled,0.25)
            q3 = np.nanquantile(yyy_filled,0.75)
            thres1 = q1 - 1.5*abs(q3-q1)
            thres2 = q3 + 1.5*abs(q3-q1)
            ind_y = (yyy_filled>thres2) | (yyy_filled<thres1)
            # yyy_filled[ind_y] = np.nan

            # xxx_filled[ind_x] = np.nan
            # xxx_filled[ind_y] = np.nan
            # yyy_filled[ind_x] = np.nan
            # yyy_filled[ind_y] = np.nan

            #        
            # fill in the nan data point after outlier removal
            # yyy_filled = yyy_filled.interpolate(method='linear',limit_direction='both')
            #        
            # smooth the data point   
            # std was chosen based on the data (mean std across session, animal, body part and x,y,z is ~2)
            # yyy_filtered = yyy_filled.rolling(window=10, win_type='gaussian', center=True).mean(std=2)
	    #
            yyy = yyy_filled

            body_part_locs_notfix[(iname,ibodyname)] = np.transpose(np.vstack((np.array(xxx),np.array(yyy))))
 

    # fix the messed up dodson and scorch id (messed up animal1 and animal2)
    # dodson (animal1) has larger x axis number in camera 2, and smaller x axis in camera 1
    meanx_allbd_animal1 = []
    meanx_allbd_animal2 = []
    for ibody in np.arange(0,nbodies,1):
        ibodyname = bodypartnames_videotrack[ibody]
        if ibody == 0:
            meanx_allbd_animal1 = body_part_locs_notfix[('dodson',ibodyname)]
            meanx_allbd_animal2 = body_part_locs_notfix[('scorch',ibodyname)]
        else:
            meanx_allbd_animal1 = np.vstack((meanx_allbd_animal1,body_part_locs_notfix[('dodson',ibodyname)]))
            meanx_allbd_animal2 = np.vstack((meanx_allbd_animal2,body_part_locs_notfix[('scorch',ibodyname)]))
    ratio1 = (np.nanstd(meanx_allbd_animal1[:,0]))/(np.nanstd(meanx_allbd_animal1[:,0])+np.nanstd(meanx_allbd_animal2[:,0]))
    ratio2 = (np.nanstd(meanx_allbd_animal2[:,0]))/(np.nanstd(meanx_allbd_animal1[:,0])+np.nanstd(meanx_allbd_animal2[:,0]))
    animal12_x_separate = np.nanmean(meanx_allbd_animal1[:,0])*0.5+np.nanmean(meanx_allbd_animal2[:,0])*0.5
    
    swap_animals = 0
    for iframe in np.arange(0,nframes,1):

        lefteye_loc_a1 = np.array(body_part_locs_notfix[('dodson','leftEye')])[iframe,:]
        righteye_loc_a1 = np.array(body_part_locs_notfix[('dodson','rightEye')])[iframe,:]
        lefttuft_loc_a1 = np.array(body_part_locs_notfix[('dodson','leftTuft')])[iframe,:]
        righttuft_loc_a1 = np.array(body_part_locs_notfix[('dodson','rightTuft')])[iframe,:]
        whiblz_loc_a1 = np.array(body_part_locs_notfix[('dodson','whiteBlaze')])[iframe,:]
        mouth_loc_a1 = np.array(body_part_locs_notfix[('dodson','mouth')])[iframe,:]
        if (iframe>0):
            mass_loc_a1_old = mass_loc_a1
        mass_loc_a1 = np.nanmean(np.vstack((lefteye_loc_a1,righteye_loc_a1,lefteye_loc_a1,righteye_loc_a1,whiblz_loc_a1,mouth_loc_a1)),axis=0)
        if ((np.sum(np.isnan(mass_loc_a1))>0)&(iframe>0)):
            mass_loc_a1 = mass_loc_a1_old  

        lefteye_loc_a2 = np.array(body_part_locs_notfix[('scorch','leftEye')])[iframe,:]
        righteye_loc_a2 = np.array(body_part_locs_notfix[('scorch','rightEye')])[iframe,:]
        lefttuft_loc_a2 = np.array(body_part_locs_notfix[('scorch','leftTuft')])[iframe,:]
        righttuft_loc_a2 = np.array(body_part_locs_notfix[('scorch','rightTuft')])[iframe,:]
        whiblz_loc_a2 = np.array(body_part_locs_notfix[('scorch','whiteBlaze')])[iframe,:]
        mouth_loc_a2 = np.array(body_part_locs_notfix[('scorch','mouth')])[iframe,:]
        if (iframe>0):
            mass_loc_a2_old = mass_loc_a2
        mass_loc_a2 = np.nanmean(np.vstack((lefteye_loc_a2,righteye_loc_a2,lefteye_loc_a2,righteye_loc_a2,whiblz_loc_a2,mouth_loc_a2)),axis=0)
        if ((np.sum(np.isnan(mass_loc_a2))>0)&(iframe>0)):
            mass_loc_a2 = mass_loc_a2_old 


        for ibody in np.arange(0,nbodies,1):
            ibodyname = bodypartnames_videotrack[ibody] 

            if (body_part_locs_notfix[('dodson',ibodyname)][iframe,0]>body_part_locs_notfix[('scorch',ibodyname)][iframe,0]):
                swap_animals = 0
                if cameraID == 'camera-1':
                    swap_animals = 1
            elif (body_part_locs_notfix[('dodson',ibodyname)][iframe,0]<body_part_locs_notfix[('scorch',ibodyname)][iframe,0]):
                swap_animals = 1
                if cameraID == 'camera-1':
                    swap_animals = 0
            elif ((body_part_locs_notfix[('dodson',ibodyname)][iframe,0]>mass_loc_a2[0])|(body_part_locs_notfix[('scorch',ibodyname)][iframe,0]<mass_loc_a1[0])):
                swap_animals = 0
                if cameraID == 'camera-1':
                    swap_animals = 1
            elif ((body_part_locs_notfix[('dodson',ibodyname)][iframe,0]<mass_loc_a2[0])|(body_part_locs_notfix[('scorch',ibodyname)][iframe,0]>mass_loc_a1[0])):
                swap_animals = 1  
                if cameraID == 'camera-1':
                    swap_animals = 0
            #elif ((body_part_locs_notfix[('dodson',ibodyname)][iframe,0]>animal12_x_separate)|(body_part_locs_notfix[('scorch',ibodyname)][iframe,0]<animal12_x_separate)):
            #    swap_animals = 0
            #    if cameraID == 'camera-1':
            #        swap_animals = 1
            #elif ((body_part_locs_notfix[('dodson',ibodyname)][iframe,0]<animal12_x_separate)|(body_part_locs_notfix[('scorch',ibodyname)][iframe,0]>animal12_x_separate)):
            #    swap_animals = 1 
            #    if cameraID == 'camera-1':
            #        swap_animals = 0
            else:
                swap_animals = swap_animals
                
            #         
            if not swap_animals:
                if iframe == 0:
                    body_part_locs[('dodson',ibodyname)] = body_part_locs_notfix[('dodson',ibodyname)][iframe]
                    body_part_locs[('scorch',ibodyname)] = body_part_locs_notfix[('scorch',ibodyname)][iframe]
                else:
                    body_part_locs[('dodson',ibodyname)] = np.vstack((body_part_locs[('dodson',ibodyname)],body_part_locs_notfix[('dodson',ibodyname)][iframe]))
                    body_part_locs[('scorch',ibodyname)] = np.vstack((body_part_locs[('scorch',ibodyname)],body_part_locs_notfix[('scorch',ibodyname)][iframe]))
            elif swap_animals:
                if iframe == 0:
                    body_part_locs[('dodson',ibodyname)] = body_part_locs_notfix[('scorch',ibodyname)][iframe]
                    body_part_locs[('scorch',ibodyname)] = body_part_locs_notfix[('dodson',ibodyname)][iframe]
                else:
                    body_part_locs[('dodson',ibodyname)] = np.vstack((body_part_locs[('dodson',ibodyname)],body_part_locs_notfix[('scorch',ibodyname)][iframe]))
                    body_part_locs[('scorch',ibodyname)] = np.vstack((body_part_locs[('scorch',ibodyname)],body_part_locs_notfix[('dodson',ibodyname)][iframe]))
        
        
    return body_part_locs

       
   	    



        

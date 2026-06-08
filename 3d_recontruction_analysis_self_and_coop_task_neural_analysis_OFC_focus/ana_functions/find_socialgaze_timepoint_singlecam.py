# function - find social gaze time point based on one camera
def find_socialgaze_timepoint_singlecam(bodyparts_locs_camN, lever_loc_both, tube_loc_both, considerlevertube, considertubeonly, angle_thres):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    animal_names_unique = pd.unique(pd.DataFrame(bodyparts_locs_camN.keys()).iloc[:,0])
    nanimals = animal_names_unique.shape[0]
    body_parts_unique = pd.unique(pd.DataFrame(bodyparts_locs_camN.keys()).iloc[:,1])
    nbodies = body_parts_unique.shape[0]
    min_length = list(bodyparts_locs_camN.values())[0].shape[0]

    warnings.filterwarnings('ignore')
    head_vect_all_merge = {}
    other_eye_vect_all_merge = {}
    lever_eye_vect_all_merge = {}
    tube_eye_vect_all_merge = {}
    #
    # angle between head vector and each of the gaze vector
    other_eye_angle_all_merge = {}
    lever_eye_angle_all_merge = {}
    tube_eye_angle_all_merge = {}
    # 
    look_at_other_or_not_merge = {}
    look_at_tube_or_not_merge = {}
    look_at_lever_or_not_merge = {}

    
    for ianimal in np.arange(0,nanimals,1):

        iname = animal_names_unique[ianimal]
        lever_loc = lever_loc_both[iname]
        tube_loc = tube_loc_both[iname]
 
        head_vect_frames = []
        other_eye_vect_frames = []
        lever_eye_vect_frames = []
        tube_eye_vect_frames = []
        #
        # angle between head vector and each of the gaze vector
        other_eye_angle_frames = []
        lever_eye_angle_frames = []
        tube_eye_angle_frames = []
        # 
        look_at_other_frames = []
        look_at_tube_frames = []
        look_at_lever_frames = []

        for iframe in np.arange(0,min_length,1):
            lefteye_loc = np.array(bodyparts_locs_camN[(iname,'leftEye')])[iframe,:]
            righteye_loc = np.array(bodyparts_locs_camN[(iname,'rightEye')])[iframe,:]
            lefttuft_loc = np.array(bodyparts_locs_camN[(iname,'leftTuft')])[iframe,:]
            righttuft_loc = np.array(bodyparts_locs_camN[(iname,'rightTuft')])[iframe,:]
            whiblz_loc = np.array(bodyparts_locs_camN[(iname,'whiteBlaze')])[iframe,:]
            mouth_loc = np.array(bodyparts_locs_camN[(iname,'mouth')])[iframe,:]
            #
            # fill in the nan gaps
            if (iframe > 0):
                meaneye_loc_old = meaneye_loc
                meantuft_loc_old = meantuft_loc
                mass_loc_old = mass_loc
                whiblz_loc_old = whiblz_loc
            # 
            meaneye_loc = np.nanmean(np.vstack((lefteye_loc,righteye_loc)),axis=0)
            if ((np.sum(np.isnan(meaneye_loc))>0)&(iframe>0)):
                meaneye_loc = meaneye_loc_old
            # 
            meantuft_loc = np.nanmean(np.vstack((lefttuft_loc,righttuft_loc)),axis=0)
            if ((np.sum(np.isnan(meantuft_loc))>0)&(iframe>0)):
                meantuft_loc = meantuft_loc_old
            #  
            mass_loc = np.nanmean(np.vstack((lefteye_loc,righteye_loc,lefteye_loc,righteye_loc,whiblz_loc,mouth_loc)),axis=0)
            if ((np.sum(np.isnan(mass_loc))>0)&(iframe>0)):
                mass_loc = mass_loc_old   
            # 
            if ((np.sum(np.isnan(whiblz_loc))>0)&(iframe>0)):
                whiblz_loc = whiblz_loc_old         


            # examine whether this animal is looking at the other's eyes or face
            if (iname == animal_names_unique[0]): 
                iname_other = animal_names_unique[1]
            elif (iname == animal_names_unique[1]): 
                iname_other = animal_names_unique[0]
            #
            lefteye_loc_other = np.array(bodyparts_locs_camN[(iname_other,'leftEye')])[iframe,:]
            righteye_loc_other = np.array(bodyparts_locs_camN[(iname_other,'rightEye')])[iframe,:]
            lefttuft_loc_other = np.array(bodyparts_locs_camN[(iname_other,'leftTuft')])[iframe,:]
            righttuft_loc_other = np.array(bodyparts_locs_camN[(iname_other,'rightTuft')])[iframe,:]
            whiblz_loc_other = np.array(bodyparts_locs_camN[(iname_other,'whiteBlaze')])[iframe,:]
            mouth_loc_other = np.array(bodyparts_locs_camN[(iname_other,'mouth')])[iframe,:]
            #
            # fill in the nan gaps
            if (iframe > 0):
                meaneye_loc_other_old = meaneye_loc_other
                meantuft_loc_other_old = meantuft_loc_other
                mass_loc_other_old = mass_loc_other
                whiblz_loc_other_old = whiblz_loc_other
            # 
            meaneye_loc_other = np.nanmean(np.vstack((lefteye_loc_other,righteye_loc_other)),axis=0)
            if ((np.sum(np.isnan(meaneye_loc_other))>0)&(iframe>0)):
                meaneye_loc_other = meaneye_loc_other_old
            # 
            meantuft_loc_other = np.nanmean(np.vstack((lefttuft_loc_other,righttuft_loc_other)),axis=0)
            if ((np.sum(np.isnan(meantuft_loc_other))>0)&(iframe>0)):
                meantuft_loc_other = meantuft_loc_other_old
            #  
            mass_loc_other = np.nanmean(np.vstack((lefteye_loc_other,righteye_loc_other,lefteye_loc_other,righteye_loc_other,whiblz_loc_other,mouth_loc_other)),axis=0)
            if ((np.sum(np.isnan(mass_loc_other))>0)&(iframe>0)):
                mass_loc_other = mass_loc_other_old 
            # 
            if ((np.sum(np.isnan(whiblz_loc_other))>0)&(iframe>0)):
                whiblz_loc_other = whiblz_loc_other_old   

       
            # define important vectors
            # head_vect = meantuft_loc - meaneye_loc
            head_vect = meantuft_loc - whiblz_loc
            try:
                head_vect = head_vect / scipy.linalg.norm(head_vect)
            except:
                head_vect = head_vect/np.nanmax(np.absolute(head_vect))
            #
            other_eye_vect = mass_loc_other - meaneye_loc
            try:
                other_eye_vect = other_eye_vect / scipy.linalg.norm(other_eye_vect)
            except:
                other_eye_vect = other_eye_vect/np.nanmax(np.absolute(other_eye_vect))
            #
            lever_eye_vect = lever_loc - meaneye_loc
            try:
                lever_eye_vect = lever_eye_vect / scipy.linalg.norm(lever_eye_vect)
            except:
                lever_eye_vect = lever_eye_vect/np.nanmax(np.absolute(lever_eye_vect))
            #
            tube_eye_vect = tube_loc - meaneye_loc
            try:
                tube_eye_vect = tube_eye_vect / scipy.linalg.norm(tube_eye_vect)
            except:
                tube_eye_vect = tube_eye_vect/np.nanmax(np.absolute(tube_eye_vect))
            #
            head_vect_frames.append(head_vect)
            other_eye_vect_frames.append(other_eye_vect)
            lever_eye_vect_frames.append(lever_eye_vect)
            tube_eye_vect_frames.append(tube_eye_vect)
        
        
            # define important angles (attention! it is defined based on the head_vect, opposite to the head gaze direction)
            other_eye_angle=np.arccos(np.clip(np.dot(head_vect,other_eye_vect),-1,1))
            lever_eye_angle=np.arccos(np.clip(np.dot(head_vect,lever_eye_vect),-1,1))
            tube_eye_angle =np.arccos(np.clip(np.dot(head_vect,tube_eye_vect),-1,1))
            #
            other_eye_angle_frames.append(other_eye_angle)
            lever_eye_angle_frames.append(lever_eye_angle)
            tube_eye_angle_frames.append(tube_eye_angle)


            # define whether looking at the other animal/lever/tube
            look_at_other_iframe = 0
            if ((other_eye_angle<np.pi)&(other_eye_angle>angle_thres)):
                if (considerlevertube):
                    if ((other_eye_angle>lever_eye_angle)&(other_eye_angle>tube_eye_angle)):
                        look_at_other_iframe = 1
                elif (considertubeonly):
                    if (other_eye_angle>tube_eye_angle):
                        look_at_other_iframe = 1
                else:
                    look_at_other_iframe = 1
            look_at_other_frames.append(look_at_other_iframe)
            #
            look_at_lever_iframe = 0
            if ((lever_eye_angle<np.pi)&(lever_eye_angle>angle_thres)):
                if (considerlevertube):       
                    if ((lever_eye_angle>other_eye_angle)&(lever_eye_angle>tube_eye_angle)):
                        look_at_lever_iframe = 1
            look_at_lever_frames.append(look_at_lever_iframe)
            #
            look_at_tube_iframe = 0
            if ((tube_eye_angle<np.pi)&(tube_eye_angle>angle_thres)):
                if (considerlevertube):
                    if ((tube_eye_angle>other_eye_angle)&(tube_eye_angle>lever_eye_angle)):
                        look_at_tube_iframe = 1
                elif (considertubeonly):
                    if (tube_eye_angle>other_eye_angle):
                        look_at_tube_iframe = 1
            look_at_tube_frames.append(look_at_tube_iframe)
        

       # save to the summarized data
        head_vect_all_merge[(iname)] = head_vect_frames
        other_eye_vect_all_merge[(iname)] = other_eye_vect_frames
        lever_eye_vect_all_merge[(iname)] = lever_eye_vect_frames
        tube_eye_vect_all_merge[(iname)] = tube_eye_vect_frames
        #
        # angle between head vector and each of the gaze vector
        other_eye_angle_all_merge[(iname)] = other_eye_angle_frames
        lever_eye_angle_all_merge[(iname)] = lever_eye_angle_frames
        tube_eye_angle_all_merge[(iname)] = tube_eye_angle_frames 
        # 
        look_at_other_or_not_merge[(iname)] = look_at_other_frames
        look_at_tube_or_not_merge[(iname)] = look_at_tube_frames
        look_at_lever_or_not_merge[(iname)] = look_at_lever_frames


    output_allvectors = {"head_vect_all_merge":head_vect_all_merge,"other_eye_vect_all_merge":other_eye_vect_all_merge,"lever_eye_vect_all_merge":lever_eye_vect_all_merge,"tube_eye_vect_all_merge":tube_eye_vect_all_merge}       

    output_allangles = {"other_eye_angle_all_merge":other_eye_angle_all_merge,"lever_eye_angle_all_merge":lever_eye_angle_all_merge,"tube_eye_angle_all_merge":tube_eye_angle_all_merge} 

    output_look_ornot = {"look_at_other_or_not_merge":look_at_other_or_not_merge,"look_at_tube_or_not_merge":look_at_tube_or_not_merge,"look_at_lever_or_not_merge":look_at_lever_or_not_merge}
       
 
    return output_look_ornot, output_allvectors, output_allangles
        

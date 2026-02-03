# function - find social gaze time point

def normalize_vector(vector):
    import numpy as np
    """
    Normalize a vector.

    Args:
        vector (ndarray): Input vector.

    Returns:
        ndarray: Normalized vector.
    """
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector  # Avoid division by zero
    return vector / magnitude


def find_socialgaze_timepoint_Anipose_2(body_part_locs_Anipose, min_length, angle_thres,with_tubelever):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    import sys
    sys.path.append('/home/ws523/marmoset_tracking_DLCv2/following_up_analysis/3d_recontruction_analysis_joystick_task/ana_functions')
    from find_socialgaze_timepoint_Anipose_2 import normalize_vector


    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs_Anipose.keys()).iloc[:,0])
    nanimals = animal_names_unique.shape[0]    
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs_Anipose.keys()).iloc[:,1])
    nbodies = body_parts_unique.shape[0]


    warnings.filterwarnings('ignore')
    #
    # important location that will be used for next step
    meaneye_locs_all_Anipose = {}
    facemass_locs_all_Anipose = {}
    tube_locs_all_Anipose = {}
    lever_locs_all_Anipose = {}

    
    for ianimal in np.arange(0,nanimals,1):

        iname = animal_names_unique[ianimal]
        # 
        if (iname == animal_names_unique[0]): 
            iname_other = animal_names_unique[1]
        elif (iname == animal_names_unique[1]): 
            iname_other = animal_names_unique[0]
    
        # important location that will be used for next step
        meaneye_locs_frames = np.empty([min_length,3])
        facemass_locs_frames = np.empty([min_length,3])
        tube_locs_frames = np.empty([min_length,3])
        lever_locs_frames = np.empty([min_length,3])
        
        
        for iframe in np.arange(0,min_length,1):

            # define the position of the animal's face part
            #
            # fill in the nan gaps
            if (iframe > 0):
                meaneye_loc_old = meaneye_loc
                meantuft_loc_old = meantuft_loc
                mass_loc_old = mass_loc
                whiblz_loc_old = whiblz_loc
            # 
            lefteye_loc = np.array(body_part_locs_Anipose[(iname,'leftEye')])[iframe,:]
            righteye_loc = np.array(body_part_locs_Anipose[(iname,'rightEye')])[iframe,:]
            lefttuft_loc = np.array(body_part_locs_Anipose[(iname,'leftTuft')])[iframe,:]
            righttuft_loc = np.array(body_part_locs_Anipose[(iname,'rightTuft')])[iframe,:]
            whiblz_loc = np.array(body_part_locs_Anipose[(iname,'whiteBlaze')])[iframe,:]
            mouth_loc = np.array(body_part_locs_Anipose[(iname,'mouth')])[iframe,:]
            if with_tubelever:
                tube_loc = np.array(body_part_locs_Anipose[(iname,'tube')])[iframe,:]
                lever_loc = np.array(body_part_locs_Anipose[(iname,'lever')])[iframe,:]
            #
            meaneye_loc = np.nanmean(np.vstack((lefteye_loc,righteye_loc)),axis=0)
            #
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

            #
            meaneye_locs_frames[iframe,:] = meaneye_loc
            facemass_locs_frames[iframe,:] = mass_loc
            if with_tubelever:
                tube_locs_frames[iframe,:] = tube_loc
                lever_locs_frames[iframe,:] = lever_loc
       
        # 
        # important location that will be used for next step
        meaneye_locs_all_Anipose[(iname)] = meaneye_locs_frames
        facemass_locs_all_Anipose[(iname)] = facemass_locs_frames
        tube_locs_all_Anipose[(iname)] = tube_locs_frames
        lever_locs_all_Anipose[(iname)] = lever_locs_frames

    output_key_locations = {"meaneye_loc_all_Anipose":meaneye_locs_all_Anipose,"facemass_loc_all_Anipose":facemass_locs_all_Anipose,"tube_loc_all_Anipose":tube_locs_all_Anipose,"lever_loc_all_Anipose":lever_locs_all_Anipose}
    
    return output_key_locations



















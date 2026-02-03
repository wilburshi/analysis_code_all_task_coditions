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


def find_socialgaze_timepoint_Anipose(body_part_locs_Anipose, min_length, angle_thres,with_tubelever):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    import sys
    sys.path.append('/home/ws523/marmoset_tracking_DLCv2/following_up_analysis/3d_recontruction_analysis_self_and_coop_task/ana_functions')
    from find_socialgaze_timepoint_Anipose import normalize_vector


    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs_Anipose.keys()).iloc[:,0])
    nanimals = animal_names_unique.shape[0]    
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs_Anipose.keys()).iloc[:,1])
    nbodies = body_parts_unique.shape[0]


    warnings.filterwarnings('ignore')
    # important vectors
    eye_direction_Anipose = {}
    face_eye_vect_all_Anipose = {}
    selflever_eye_vect_all_Anipose = {}
    selftube_eye_vect_all_Anipose = {}
    otherlever_eye_vect_all_Anipose = {}
    othertube_eye_vect_all_Anipose = {}
    #
    # angle between eye direction vector and each of the gaze vector
    face_eye_angle_all_Anipose = {}
    selflever_eye_angle_all_Anipose = {}
    selftube_eye_angle_all_Anipose = {}
    otherlever_eye_angle_all_Anipose = {}
    othertube_eye_angle_all_Anipose = {}
    #
    # boolean variables of behavioral events
    look_at_face_or_not_Anipose = {}
    look_at_selftube_or_not_Anipose = {}
    look_at_selflever_or_not_Anipose = {}
    look_at_othertube_or_not_Anipose = {}
    look_at_otherlever_or_not_Anipose = {}

    
    for ianimal in np.arange(0,nanimals,1):

        iname = animal_names_unique[ianimal]
        # 
        if (iname == animal_names_unique[0]): 
            iname_other = animal_names_unique[1]
        elif (iname == animal_names_unique[1]): 
            iname_other = animal_names_unique[0]

        # important vectors
        eye_dir_frames = []
        face_eye_vect_frames = []
        selflever_eye_vect_frames = []
        selftube_eye_vect_frames = []
        otherlever_eye_vect_frames = []
        othertube_eye_vect_frames = []
        #
        # angle between head vector and each of the gaze vector
        face_eye_angle_frames = []
        selflever_eye_angle_frames = []
        selftube_eye_angle_frames = []
        otherlever_eye_angle_frames = []
        othertube_eye_angle_frames = []
        # 
        look_at_face_frames = []
        look_at_selftube_frames = []
        look_at_selflever_frames = []
        look_at_othertube_frames = []
        look_at_otherlever_frames = []
        
        
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


            # examine whether this animal is looking at the other's eyes or face
            #
            # fill in the nan gaps
            if (iframe > 0):
                meaneye_loc_other_old = meaneye_loc_other
                meantuft_loc_other_old = meantuft_loc_other
                mass_loc_other_old = mass_loc_other
                whiblz_loc_other_old = whiblz_loc_other
            # 
            lefteye_loc_other = np.array(body_part_locs_Anipose[(iname_other,'leftEye')])[iframe,:]
            righteye_loc_other = np.array(body_part_locs_Anipose[(iname_other,'rightEye')])[iframe,:]
            lefttuft_loc_other = np.array(body_part_locs_Anipose[(iname_other,'leftTuft')])[iframe,:]
            righttuft_loc_other = np.array(body_part_locs_Anipose[(iname_other,'rightTuft')])[iframe,:]
            whiblz_loc_other = np.array(body_part_locs_Anipose[(iname_other,'whiteBlaze')])[iframe,:]
            mouth_loc_other = np.array(body_part_locs_Anipose[(iname_other,'mouth')])[iframe,:]
            if with_tubelever:
                tube_loc_other = np.array(body_part_locs_Anipose[(iname_other,'tube')])[iframe,:]
                lever_loc_other = np.array(body_part_locs_Anipose[(iname_other,'lever')])[iframe,:]
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


            # define important vectors for defining head gaze direction
            # fill in the nan gaps
            if (iframe > 0):
                eyesight_dir_old = eyesight_dir
            #
            Vect1 = lefteye_loc - righteye_loc
            Vect2 = whiblz_loc - mouth_loc
            Vect3 = lefttuft_loc - lefteye_loc
            Vect4 = righttuft_loc - righteye_loc
            Vect5 = meantuft_loc - meaneye_loc
   
            Vect1 = normalize_vector(Vect1)
            Vect2 = normalize_vector(Vect2) 
            Vect3 = normalize_vector(Vect3)
            Vect4 = normalize_vector(Vect4)
            Vect5 = normalize_vector(Vect5)
            
            eyesight_dir = np.cross(Vect1, Vect2)
            eyesight_dir = normalize_vector(eyesight_dir)

            if (np.dot(eyesight_dir, Vect5)>0):
                eyesight_dir = -eyesight_dir
            #
            if ((np.sum(np.isnan(eyesight_dir))>0)&(iframe>0)):
                eyesight_dir = eyesight_dir_old
            
            eye_dir_frames.append(eyesight_dir)      
 
        
            # where eye is looking
            # vector between body part, tube, lever
            vect_face_eye = mass_loc_other - meaneye_loc
            vect_face_eye = normalize_vector(vect_face_eye)
            if with_tubelever:
                vect_selftube_eye = tube_loc - meaneye_loc
                vect_selflever_eye  = lever_loc - meaneye_loc
                vect_othertube_eye = tube_loc_other - meaneye_loc
                vect_otherlever_eye  = lever_loc_other - meaneye_loc
                vect_selftube_eye = normalize_vector(vect_selftube_eye)
                vect_selflever_eye  = normalize_vector(vect_selflever_eye)
                vect_othertube_eye = normalize_vector(vect_othertube_eye)
                vect_otherlever_eye  = normalize_vector(vect_otherlever_eye)

            face_eye_vect_frames.append(vect_face_eye)
            if with_tubelever:
                selflever_eye_vect_frames.append(vect_selflever_eye)
                selftube_eye_vect_frames.append(vect_selftube_eye)
                otherlever_eye_vect_frames.append(vect_otherlever_eye)
                othertube_eye_vect_frames.append(vect_othertube_eye)

            # calculate the angles
            angle_face_eye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect_face_eye/np.linalg.norm(vect_face_eye)), -1.0, 1.0))       
            if with_tubelever:
                angle_selftube_eye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect_selftube_eye/np.linalg.norm(vect_selftube_eye)), -1.0, 1.0))
                angle_selflever_eye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect_selflever_eye/np.linalg.norm(vect_selflever_eye)), -1.0, 1.0))
                angle_othertube_eye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect_othertube_eye/np.linalg.norm(vect_othertube_eye)), -1.0, 1.0))
                angle_otherlever_eye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect_otherlever_eye/np.linalg.norm(vect_otherlever_eye)), -1.0, 1.0))
            #
            face_eye_angle_frames.append(angle_face_eye)
            if with_tubelever:
                selflever_eye_angle_frames.append(angle_selflever_eye)
                selftube_eye_angle_frames.append(angle_selftube_eye)
                otherlever_eye_angle_frames.append(angle_otherlever_eye)
                othertube_eye_angle_frames.append(angle_othertube_eye)        

            # boolean variables for the behavioral events
            eye_lookface_thres = ((angle_face_eye>0)&(angle_face_eye<angle_thres))
            if with_tubelever:
                eye_lookselftube_thres = ((angle_selftube_eye>0)&(angle_selftube_eye<angle_thres))
                eye_lookselflever_thres = ((angle_selflever_eye>0)&(angle_selflever_eye<angle_thres)) 
                eye_lookothertube_thres = ((angle_othertube_eye>0)&(angle_othertube_eye<angle_thres))
                eye_lookotherlever_thres = ((angle_otherlever_eye>0)&(angle_otherlever_eye<angle_thres)) 
            #      
            look_at_face_frames.append(np.int(eye_lookface_thres))
            if with_tubelever:
                look_at_selftube_frames.append(np.int(eye_lookselftube_thres))
                look_at_selflever_frames.append(np.int(eye_lookselflever_thres))
                look_at_othertube_frames.append(np.int(eye_lookothertube_thres))
                look_at_otherlever_frames.append(np.int(eye_lookotherlever_thres))
            
        
        # save to the summarized data
        # important vectors
        eye_direction_Anipose[(iname)] = eye_dir_frames
        face_eye_vect_all_Anipose[(iname)] = face_eye_vect_frames
        selflever_eye_vect_all_Anipose[(iname)] = selflever_eye_vect_frames
        selftube_eye_vect_all_Anipose[(iname)] = selftube_eye_vect_frames
        otherlever_eye_vect_all_Anipose[(iname)] = otherlever_eye_vect_frames
        othertube_eye_vect_all_Anipose[(iname)] = othertube_eye_vect_frames
        #
        # angle between head vector and each of the gaze vector
        face_eye_angle_all_Anipose[(iname)] = face_eye_angle_frames
        selflever_eye_angle_all_Anipose[(iname)] = selflever_eye_angle_frames
        selftube_eye_angle_all_Anipose[(iname)] = selftube_eye_angle_frames 
        otherlever_eye_angle_all_Anipose[(iname)] = otherlever_eye_angle_frames
        othertube_eye_angle_all_Anipose[(iname)] = othertube_eye_angle_frames 
        # 
        # boolean variables for the behavioral events
        look_at_face_or_not_Anipose[(iname)] = look_at_face_frames
        look_at_selftube_or_not_Anipose[(iname)] = look_at_selftube_frames
        look_at_selflever_or_not_Anipose[(iname)] = look_at_selflever_frames
        look_at_othertube_or_not_Anipose[(iname)] = look_at_othertube_frames
        look_at_otherlever_or_not_Anipose[(iname)] = look_at_otherlever_frames


    output_allvectors = {"eye_direction_Anipose":eye_direction_Anipose,"face_eye_vect_all_Anipose":face_eye_vect_all_Anipose,"selflever_eye_vect_all_Anipose":selflever_eye_vect_all_Anipose,"selftube_eye_vect_all_Anipose":selftube_eye_vect_all_Anipose,"otherlever_eye_vect_all_Anipose":otherlever_eye_vect_all_Anipose,"othertube_eye_vect_all_Anipose":othertube_eye_vect_all_Anipose}       

    output_allangles = {"face_eye_angle_all_Anipose":face_eye_angle_all_Anipose,"selflever_eye_angle_all_Anipose":selflever_eye_angle_all_Anipose,"selftube_eye_angle_all_Anipose":selftube_eye_angle_all_Anipose,"otherlever_eye_angle_all_Anipose":otherlever_eye_angle_all_Anipose,"othertube_eye_angle_all_Anipose":othertube_eye_angle_all_Anipose} 

    output_look_ornot = {"look_at_face_or_not_Anipose":look_at_face_or_not_Anipose,"look_at_selftube_or_not_Anipose":look_at_selftube_or_not_Anipose,"look_at_selflever_or_not_Anipose":look_at_selflever_or_not_Anipose,"look_at_othertube_or_not_Anipose":look_at_othertube_or_not_Anipose,"look_at_otherlever_or_not_Anipose":look_at_otherlever_or_not_Anipose}
       
   
    return output_look_ornot, output_allvectors, output_allangles


















# function - find social gaze time point
def find_socialgaze_timepoint(body_part_locs_merge, min_length, angle_thres):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs_merge.keys()).iloc[:,0])
    nanimals = animal_names_unique.shape[0]    
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs_merge.keys()).iloc[:,1])
    nbodies = body_parts_unique.shape[0]


    warnings.filterwarnings('ignore')
    eye_direction_merge = {}
    eye_contact_or_not_merge = {}
    look_at_face_or_not_merge = {}

    
    for ianimal in np.arange(0,nanimals,1):

        iname = animal_names_unique[ianimal]
        # 
        if (iname == animal_names_unique[0]): 
            iname_other = animal_names_unique[1]
        elif (iname == animal_names_unique[1]): 
            iname_other = animal_names_unique[0]

        eye_dir_frames = []
        eye_contact_frames = []
        look_at_face_frames = []
        
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
            lefteye_loc = np.array(body_part_locs_merge[(iname,'leftEye')])[iframe,:]
            righteye_loc = np.array(body_part_locs_merge[(iname,'rightEye')])[iframe,:]
            lefttuft_loc = np.array(body_part_locs_merge[(iname,'leftTuft')])[iframe,:]
            righttuft_loc = np.array(body_part_locs_merge[(iname,'rightTuft')])[iframe,:]
            whiblz_loc = np.array(body_part_locs_merge[(iname,'whiteBlaze')])[iframe,:]
            mouth_loc = np.array(body_part_locs_merge[(iname,'mouth')])[iframe,:]
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
            lefteye_loc_other = np.array(body_part_locs_merge[(iname_other,'leftEye')])[iframe,:]
            righteye_loc_other = np.array(body_part_locs_merge[(iname_other,'rightEye')])[iframe,:]
            lefttuft_loc_other = np.array(body_part_locs_merge[(iname_other,'leftTuft')])[iframe,:]
            righttuft_loc_other = np.array(body_part_locs_merge[(iname_other,'rightTuft')])[iframe,:]
            whiblz_loc_other = np.array(body_part_locs_merge[(iname_other,'whiteBlaze')])[iframe,:]
            mouth_loc_other = np.array(body_part_locs_merge[(iname_other,'mouth')])[iframe,:]
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
            try:       
                Vect1 = Vect1 / scipy.linalg.norm(Vect1)
                Vect2 = Vect2 / scipy.linalg.norm(Vect2) 
                Vect3 = Vect3 / scipy.linalg.norm(Vect3)
                Vect4 = Vect4 / scipy.linalg.norm(Vect4)
                Vect5 = Vect5 / scipy.linalg.norm(Vect5)
            except:
                Vect1 = Vect1 / np.nanmax(np.absolute(Vect1))
                Vect2 = Vect2 / np.nanmax(np.absolute(Vect2))
                Vect3 = Vect3 / np.nanmax(np.absolute(Vect3))
                Vect4 = Vect4 / np.nanmax(np.absolute(Vect4))
                Vect5 = Vect5 / np.nanmax(np.absolute(Vect5))
            eyesight_dir = np.cross(Vect1, Vect2)
            try:
                eyesight_dir = eyesight_dir / scipy.linalg.norm(eyesight_dir)
            except:
                eyesight_dir = eyesight_dir / np.nanmax(np.absolute(eyesight_dir))
            #
            if (np.dot(eyesight_dir, Vect5)>0):
                eyesight_dir = -eyesight_dir
            #
            if ((np.sum(np.isnan(eyesight_dir))>0)&(iframe>0)):
                eyesight_dir = eyesight_dir_old
            eye_dir_frames.append(eyesight_dir)      
 
        
            # where left eye is looking
            # vector between body part
            vect_face_eye = mass_loc_other - meaneye_loc
            vect_eye_eye = meaneye_loc_other - meaneye_loc
            try:
                vect_face_eye = vect_face_eye / scipy.linalg.norm(vect_face_eye)
            except:
                vect_face_eye = vect_face_eye / np.nanmax(np.absolute(vect_face_eye))
            try:
                vect_eye_eye = vect_eye_eye / scipy.linalg.norm(vect_eye_eye)
            except:
                vect_eye_eye = vect_eye_eye / np.nanmax(np.absolute(vect_eye_eye))


            angle_face_eye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect_face_eye/np.linalg.norm(vect_face_eye)), -1.0, 1.0))       
            angle_eye_eye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect_eye_eye/np.linalg.norm(vect_eye_eye)), -1.0, 1.0))
            
        
            eye_contact_thres = ((angle_eye_eye>0)&(angle_eye_eye<angle_thres))
            eye_lookface_thres = ((angle_face_eye>0)&(angle_face_eye<angle_thres))
                  
            eye_contact_frames.append(np.int(eye_contact_thres))
            look_at_face_frames.append(np.int(eye_lookface_thres))
        
        # save to the summarized data
        eye_direction_merge[(iname)] = eye_dir_frames
        eye_contact_or_not_merge[(iname)] = eye_contact_frames
        look_at_face_or_not_merge[(iname)] = look_at_face_frames
        
    return eye_direction_merge, eye_contact_or_not_merge, look_at_face_or_not_merge
        

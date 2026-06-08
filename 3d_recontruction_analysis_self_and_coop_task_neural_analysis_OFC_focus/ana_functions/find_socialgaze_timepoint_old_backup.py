# function - find social gaze time point
def find_socialgaze_timepoint(body_part_locs_camera23, body_part_locs_camera12, body_part_locs_merge, angle_thres):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs_camera23.keys()).iloc[:,0])
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs_camera23.keys()).iloc[:,1])
    min_length = np.min([list(body_part_locs_camera23.values())[0].shape[0],list(body_part_locs_camera12.values())[0].shape[0]])

    warnings.filterwarnings('ignore')
    eye_direction_merge = {}
    eye_contact_or_not_merge = {}
    look_at_face_or_not_merge = {}
    
    for iname in animal_names_unique:
        eye_dir_frames = []
        eye_contact_frames = []
        look_at_face_frames = []
        
        for iframe in np.arange(0,min_length,1):
            lefteye_loc = np.array(body_part_locs_merge[(iname,'leftEye')])[iframe,:]
            righteye_loc = np.array(body_part_locs_merge[(iname,'rightEye')])[iframe,:]
            lefttuft_loc = np.array(body_part_locs_merge[(iname,'leftTuft')])[iframe,:]
            righttuft_loc = np.array(body_part_locs_merge[(iname,'rightTuft')])[iframe,:]
            whiblz_loc = np.array(body_part_locs_merge[(iname,'whiteBlaze')])[iframe,:]
            mouth_loc = np.array(body_part_locs_merge[(iname,'mouth')])[iframe,:]
        
            Vect1 = lefteye_loc - righteye_loc
            Vect2 = whiblz_loc - mouth_loc
            Vect3 = lefttuft_loc - lefteye_loc
            Vect4 = righttuft_loc - righteye_loc
        
            try:       
                Vect1 = Vect1 / scipy.linalg.norm(Vect1)
                Vect2 = Vect2 / scipy.linalg.norm(Vect2) 
            except:
                Vect1 = Vect1
                Vect2 = Vect2
            eyesight_dir = np.cross(Vect1, Vect2)
        
            if ((np.dot(eyesight_dir, Vect3)>0) | (np.dot(eyesight_dir, Vect4)>0)):
                eyesight_dir = -eyesight_dir
        
            eye_dir_frames.append(eyesight_dir)      
        
            # examine whether this animal is looking at the other's eyes or face
            if (iname == animal_names_unique[0]): 
                iname_other = animal_names_unique[1]
            elif (iname == animal_names_unique[1]): 
                iname_other = animal_names_unique[0]
            
            lefteye_loc_other = np.array(body_part_locs_merge[(iname_other,'leftEye')])[iframe,:]
            righteye_loc_other = np.array(body_part_locs_merge[(iname_other,'rightEye')])[iframe,:]
            lefttuft_loc_other = np.array(body_part_locs_merge[(iname_other,'leftTuft')])[iframe,:]
            righttuft_loc_other = np.array(body_part_locs_merge[(iname_other,'rightTuft')])[iframe,:]
            whiblz_loc_other = np.array(body_part_locs_merge[(iname_other,'whiteBlaze')])[iframe,:]
            mouth_loc_other = np.array(body_part_locs_merge[(iname_other,'mouth')])[iframe,:]
        
            # where left eye is looking
            # vector between body part
            vect1_lefteye = lefteye_loc_other - lefteye_loc
            vect2_lefteye = righteye_loc_other - lefteye_loc
            vect3_lefteye = lefttuft_loc_other - lefteye_loc
            vect4_lefteye = righttuft_loc_other - lefteye_loc
            vect5_lefteye = whiblz_loc_other - lefteye_loc
            vect6_lefteye = mouth_loc_other - lefteye_loc
            # angle between body part vector and eyesight direction
            # angle1_lefteye =  np.sign(np.dot(eyesight_dir,vect1_lefteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect1_lefteye/np.linalg.norm(vect1_lefteye)), -1.0, 1.0))       
            # angle2_lefteye =  np.sign(np.dot(eyesight_dir,vect2_lefteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect2_lefteye/np.linalg.norm(vect2_lefteye)), -1.0, 1.0))
            # angle3_lefteye =  np.sign(np.dot(eyesight_dir,vect3_lefteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect3_lefteye/np.linalg.norm(vect3_lefteye)), -1.0, 1.0))
            # angle4_lefteye =  np.sign(np.dot(eyesight_dir,vect4_lefteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect4_lefteye/np.linalg.norm(vect4_lefteye)), -1.0, 1.0))
            # angle5_lefteye =  np.sign(np.dot(eyesight_dir,vect5_lefteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect5_lefteye/np.linalg.norm(vect5_lefteye)), -1.0, 1.0))
            # angle6_lefteye =  np.sign(np.dot(eyesight_dir,vect6_lefteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect6_lefteye/np.linalg.norm(vect6_lefteye)), -1.0, 1.0))
            angle1_lefteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect1_lefteye/np.linalg.norm(vect1_lefteye)), -1.0, 1.0))       
            angle2_lefteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect2_lefteye/np.linalg.norm(vect2_lefteye)), -1.0, 1.0))
            angle3_lefteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect3_lefteye/np.linalg.norm(vect3_lefteye)), -1.0, 1.0))
            angle4_lefteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect4_lefteye/np.linalg.norm(vect4_lefteye)), -1.0, 1.0))
            angle5_lefteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect5_lefteye/np.linalg.norm(vect5_lefteye)), -1.0, 1.0))
            angle6_lefteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect6_lefteye/np.linalg.norm(vect6_lefteye)), -1.0, 1.0))
        
            # where right eye is looking
            # vector between body part
            vect1_righteye = lefteye_loc_other - righteye_loc
            vect2_righteye = righteye_loc_other - righteye_loc
            vect3_righteye = lefttuft_loc_other - righteye_loc
            vect4_righteye = righttuft_loc_other - righteye_loc
            vect5_righteye = whiblz_loc_other - righteye_loc
            vect6_righteye = mouth_loc_other - righteye_loc
            # angle between body part vector and eyesight direction
            # angle1_righteye =  np.sign(np.dot(eyesight_dir,vect1_righteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect1_righteye/np.linalg.norm(vect1_righteye)), -1.0,1.0))       
            # angle2_righteye =  np.sign(np.dot(eyesight_dir,vect2_righteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect2_righteye/np.linalg.norm(vect2_righteye)), -1.0, 1.0))
            # angle3_righteye =  np.sign(np.dot(eyesight_dir,vect3_righteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect3_righteye/np.linalg.norm(vect3_righteye)), -1.0, 1.0))
            # angle4_righteye =  np.sign(np.dot(eyesight_dir,vect4_righteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect4_righteye/np.linalg.norm(vect4_righteye)), -1.0, 1.0))
            # angle5_righteye =  np.sign(np.dot(eyesight_dir,vect5_righteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect5_righteye/np.linalg.norm(vect5_righteye)), -1.0, 1.0))
            # angle6_righteye =  np.sign(np.dot(eyesight_dir,vect6_righteye))*np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect6_righteye/np.linalg.norm(vect6_righteye)), -1.0, 1.0))
            angle1_righteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect1_righteye/np.linalg.norm(vect1_righteye)), -1.0, 1.0))       
            angle2_righteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect2_righteye/np.linalg.norm(vect2_righteye)), -1.0, 1.0))
            angle3_righteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect3_righteye/np.linalg.norm(vect3_righteye)), -1.0, 1.0))
            angle4_righteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect4_righteye/np.linalg.norm(vect4_righteye)), -1.0, 1.0))
            angle5_righteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect5_righteye/np.linalg.norm(vect5_righteye)), -1.0, 1.0))
            angle6_righteye =  np.arccos(np.clip(np.dot(eyesight_dir/np.linalg.norm(eyesight_dir), vect6_righteye/np.linalg.norm(vect6_righteye)), -1.0, 1.0))
        
            lefteye_contact_thres = ((angle1_lefteye>0)&(angle1_lefteye<angle_thres))|((angle2_lefteye>0)&(angle2_lefteye<angle_thres))
            lefteye_lookface_thres = ((angle3_lefteye>0)&(angle3_lefteye<angle_thres))|((angle4_lefteye>0)&(angle4_lefteye<angle_thres))|((angle5_lefteye>0)&(angle5_lefteye<angle_thres))|((angle6_lefteye>0)&(angle6_lefteye<angle_thres))
            righteye_contact_thres = ((angle1_righteye>0)&(angle1_righteye<angle_thres))|((angle2_righteye>0)&(angle2_righteye<angle_thres))
            righteye_lookface_thres = ((angle3_righteye>0)&(angle3_righteye<angle_thres))|((angle4_righteye>0)&(angle4_righteye<angle_thres))|((angle5_righteye>0)&(angle5_righteye<angle_thres))|((angle6_righteye>0)&(angle6_righteye<angle_thres))
        
            eye_contact_frames.append(np.int(lefteye_contact_thres|righteye_contact_thres))
            look_at_face_frames.append(np.int(lefteye_contact_thres|righteye_contact_thres|lefteye_lookface_thres|righteye_lookface_thres))
        
        # save to the summarized data
        eye_direction_merge[(iname)] = eye_dir_frames
        eye_contact_or_not_merge[(iname)] = eye_contact_frames
        look_at_face_or_not_merge[(iname)] = look_at_face_frames
        
    return eye_direction_merge, eye_contact_or_not_merge, look_at_face_or_not_merge
        

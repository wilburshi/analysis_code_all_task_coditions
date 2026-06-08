# function - find social gaze time point based on one camera
def find_socialgaze_timepoint_singlecam_wholebody(bodyparts_locs_camN, lever_loc_both, tube_loc_both, considerlevertube, considertubeonly,sqr_thres_tubelever,sqr_thres_face,sqr_thres_body):
    
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
    otherlever_eye_vect_all_merge = {}
    othertube_eye_vect_all_merge = {}
    #
    # angle between head vector and each of the gaze vector
    other_eye_angle_all_merge = {}
    lever_eye_angle_all_merge = {}
    tube_eye_angle_all_merge = {}
    otherlever_eye_angle_all_merge = {}
    othertube_eye_angle_all_merge = {}
    # 
    look_at_other_or_not_merge = {}
    look_at_tube_or_not_merge = {}
    look_at_lever_or_not_merge = {}
    look_at_otherface_or_not_merge = {}
    look_at_othertube_or_not_merge = {}
    look_at_otherlever_or_not_merge = {}

    
    for ianimal in np.arange(0,nanimals,1):

        iname = animal_names_unique[ianimal]
        lever_loc = lever_loc_both[iname]
        tube_loc = tube_loc_both[iname]
        # 
        if (iname == animal_names_unique[0]): 
            iname_other = animal_names_unique[1]
        elif (iname == animal_names_unique[1]): 
            iname_other = animal_names_unique[0]
        otherlever_loc = lever_loc_both[iname_other]
        othertube_loc = tube_loc_both[iname_other]
 
        head_vect_frames = []
        other_eye_vect_frames = []
        lever_eye_vect_frames = []
        tube_eye_vect_frames = []
        otherlever_eye_vect_frames = []
        othertube_eye_vect_frames = []
        #
        # angle between head vector and each of the gaze vector
        other_eye_angle_frames = []
        lever_eye_angle_frames = []
        tube_eye_angle_frames = []
        otherlever_eye_angle_frames = []
        othertube_eye_angle_frames = []
        # 
        look_at_other_frames = []
        look_at_otherface_frames = []
        look_at_tube_frames = []
        look_at_lever_frames = []
        look_at_othertube_frames = []
        look_at_otherlever_frames = []

        for iframe in np.arange(0,min_length,1):

            # define the position of the animal's face part
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
            head_gaze_vect = whiblz_loc - meantuft_loc
            try:
                head_vect = head_vect / scipy.linalg.norm(head_vect)
                head_gaze_vect = head_gaze_vect / scipy.linalg.norm(head_gaze_vect)
            except:
                head_vect = head_vect/np.nanmax(np.absolute(head_vect))
                head_gaze_vect = head_gaze_vect/np.nanmax(np.absolute(head_gaze_vect))
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
            otherlever_eye_vect = otherlever_loc - meaneye_loc
            try:
                otherlever_eye_vect = otherlever_eye_vect / scipy.linalg.norm(otherlever_eye_vect)
            except:
                otherlever_eye_vect = otherlever_eye_vect/np.nanmax(np.absolute(otherlever_eye_vect))
            #
            othertube_eye_vect = othertube_loc - meaneye_loc
            try:
                othertube_eye_vect = othertube_eye_vect / scipy.linalg.norm(othertube_eye_vect)
            except:
                othertube_eye_vect = othertube_eye_vect/np.nanmax(np.absolute(othertube_eye_vect))
            #
            head_vect_frames.append(head_vect)
            other_eye_vect_frames.append(other_eye_vect)
            lever_eye_vect_frames.append(lever_eye_vect)
            tube_eye_vect_frames.append(tube_eye_vect)
            otherlever_eye_vect_frames.append(otherlever_eye_vect)
            othertube_eye_vect_frames.append(othertube_eye_vect)
        
        
            # define important angles (a little different from the none-wholebody condition)
            other_eye_angle=np.arccos(np.clip(np.dot(head_gaze_vect,other_eye_vect),-1,1))
            lever_eye_angle=np.arccos(np.clip(np.dot(head_gaze_vect,lever_eye_vect),-1,1))
            tube_eye_angle =np.arccos(np.clip(np.dot(head_gaze_vect,tube_eye_vect),-1,1))
            otherlever_eye_angle=np.arccos(np.clip(np.dot(head_gaze_vect,otherlever_eye_vect),-1,1))
            othertube_eye_angle =np.arccos(np.clip(np.dot(head_gaze_vect,othertube_eye_vect),-1,1))
            #
            other_eye_angle_frames.append(other_eye_angle)
            lever_eye_angle_frames.append(lever_eye_angle)
            tube_eye_angle_frames.append(tube_eye_angle)
            otherlever_eye_angle_frames.append(otherlever_eye_angle)
            othertube_eye_angle_frames.append(othertube_eye_angle)


            # define whether looking at the other animal/lever/tube
            #
            # define whether looking at the other animal's face
            look_at_otherface_iframe = 0
            # four corner of the face boundary
            dist7 = np.linalg.norm(righteye_loc_other - righttuft_loc_other)
            dist8 = np.linalg.norm(lefteye_loc_other - lefttuft_loc_other)
            dist9 = np.linalg.norm(righttuft_loc_other - whiblz_loc_other)
            dist10 = np.linalg.norm(lefttuft_loc_other - whiblz_loc_other)
            # fill in the nan gaps
            if (iframe > 0):
                face_offset_old = face_offset          
            face_offset = np.nanmax([dist7,dist8,dist9,dist10])*sqr_thres_face
            if ((np.sum(np.isnan(face_offset))>0)&(iframe>0)):
                face_offset = face_offset_old
            corner1_of = np.array([mass_loc_other[0]-face_offset, mass_loc_other[1]-face_offset])
            corner2_of = np.array([mass_loc_other[0]-face_offset, mass_loc_other[1]+face_offset])
            corner3_of = np.array([mass_loc_other[0]+face_offset, mass_loc_other[1]-face_offset])
            corner4_of = np.array([mass_loc_other[0]+face_offset, mass_loc_other[1]+face_offset])
            crossprod1 = np.cross(head_gaze_vect,corner1_of-meaneye_loc)*np.cross(head_gaze_vect,corner2_of-meaneye_loc)
            crossprod2 = np.cross(head_gaze_vect,corner1_of-meaneye_loc)*np.cross(head_gaze_vect,corner3_of-meaneye_loc)
            crossprod3 = np.cross(head_gaze_vect,corner1_of-meaneye_loc)*np.cross(head_gaze_vect,corner4_of-meaneye_loc)
            crossprod4 = np.cross(head_gaze_vect,corner2_of-meaneye_loc)*np.cross(head_gaze_vect,corner3_of-meaneye_loc)
            crossprod5 = np.cross(head_gaze_vect,corner2_of-meaneye_loc)*np.cross(head_gaze_vect,corner4_of-meaneye_loc)
            crossprod6 = np.cross(head_gaze_vect,corner3_of-meaneye_loc)*np.cross(head_gaze_vect,corner4_of-meaneye_loc)
            if (((crossprod1<0)|(crossprod2<0)|(crossprod3<0)|(crossprod4<0)|(crossprod5<0)|(crossprod6<0))&(other_eye_angle<np.pi/2)):
                if (considerlevertube):
                    if ((other_eye_angle<lever_eye_angle)&(other_eye_angle<tube_eye_angle)):
                        look_at_otherface_iframe = 1
                elif (considertubeonly):
                    if (other_eye_angle<tube_eye_angle):
                        look_at_otherface_iframe = 1
                else:
                        look_at_otherface_iframe = 1
            look_at_otherface_frames.append(look_at_otherface_iframe)
            #
            # define whether looking at the other animal (whole estimated body)
            look_at_other_iframe = 0
            # four corner of the body boundary
            corner1_oa = np.array([mass_loc_other[0]-face_offset, mass_loc_other[1]-face_offset])
            corner2_oa = np.array([mass_loc_other[0]-face_offset, mass_loc_other[1]+sqr_thres_body*face_offset])
            corner3_oa = np.array([mass_loc_other[0]+face_offset, mass_loc_other[1]-face_offset])
            corner4_oa = np.array([mass_loc_other[0]+face_offset, mass_loc_other[1]+sqr_thres_body*face_offset])
            crossprod1 = np.cross(head_gaze_vect,corner1_oa-meaneye_loc)*np.cross(head_gaze_vect,corner2_oa-meaneye_loc)
            crossprod2 = np.cross(head_gaze_vect,corner1_oa-meaneye_loc)*np.cross(head_gaze_vect,corner3_oa-meaneye_loc)
            crossprod3 = np.cross(head_gaze_vect,corner1_oa-meaneye_loc)*np.cross(head_gaze_vect,corner4_oa-meaneye_loc)
            crossprod4 = np.cross(head_gaze_vect,corner2_oa-meaneye_loc)*np.cross(head_gaze_vect,corner3_oa-meaneye_loc)
            crossprod5 = np.cross(head_gaze_vect,corner2_oa-meaneye_loc)*np.cross(head_gaze_vect,corner4_oa-meaneye_loc)
            crossprod6 = np.cross(head_gaze_vect,corner3_oa-meaneye_loc)*np.cross(head_gaze_vect,corner4_oa-meaneye_loc)
            if (((crossprod1<0)|(crossprod2<0)|(crossprod3<0)|(crossprod4<0)|(crossprod5<0)|(crossprod6<0))&(other_eye_angle<np.pi/2)):    
                look_at_other_iframe = 1
            look_at_other_frames.append(look_at_other_iframe)
            #
            # define whehter looking at the self lever
            look_at_lever_iframe = 0
            # four corner of the lever boundary
            corner1_lv = np.array([lever_loc[0]-sqr_thres_tubelever, lever_loc[1]-sqr_thres_tubelever])
            corner2_lv = np.array([lever_loc[0]-sqr_thres_tubelever, lever_loc[1]+sqr_thres_tubelever])
            corner3_lv = np.array([lever_loc[0]+sqr_thres_tubelever, lever_loc[1]-sqr_thres_tubelever])
            corner4_lv = np.array([lever_loc[0]+sqr_thres_tubelever, lever_loc[1]+sqr_thres_tubelever])
            crossprod1 = np.cross(head_gaze_vect,corner1_lv-meaneye_loc)*np.cross(head_gaze_vect,corner2_lv-meaneye_loc)
            crossprod2 = np.cross(head_gaze_vect,corner1_lv-meaneye_loc)*np.cross(head_gaze_vect,corner3_lv-meaneye_loc)
            crossprod3 = np.cross(head_gaze_vect,corner1_lv-meaneye_loc)*np.cross(head_gaze_vect,corner4_lv-meaneye_loc)
            crossprod4 = np.cross(head_gaze_vect,corner2_lv-meaneye_loc)*np.cross(head_gaze_vect,corner3_lv-meaneye_loc)
            crossprod5 = np.cross(head_gaze_vect,corner2_lv-meaneye_loc)*np.cross(head_gaze_vect,corner4_lv-meaneye_loc)
            crossprod6 = np.cross(head_gaze_vect,corner3_lv-meaneye_loc)*np.cross(head_gaze_vect,corner4_lv-meaneye_loc)
            if (((crossprod1<0)|(crossprod2<0)|(crossprod3<0)|(crossprod4<0)|(crossprod5<0)|(crossprod6<0))&(lever_eye_angle<np.pi/2)):
                if (considerlevertube):       
                    if ((lever_eye_angle<other_eye_angle)&(lever_eye_angle<tube_eye_angle)):
                        look_at_lever_iframe = 1
            look_at_lever_frames.append(look_at_lever_iframe)
            #
            # define whether looking at the self tube
            look_at_tube_iframe = 0
            # four corner of the tube boundary
            corner1_tb = np.array([tube_loc[0]-sqr_thres_tubelever, tube_loc[1]-sqr_thres_tubelever])
            corner2_tb = np.array([tube_loc[0]-sqr_thres_tubelever, tube_loc[1]+sqr_thres_tubelever])
            corner3_tb = np.array([tube_loc[0]+sqr_thres_tubelever, tube_loc[1]-sqr_thres_tubelever])
            corner4_tb = np.array([tube_loc[0]+sqr_thres_tubelever, tube_loc[1]+sqr_thres_tubelever])
            crossprod1 = np.cross(head_gaze_vect,corner1_tb-meaneye_loc)*np.cross(head_gaze_vect,corner2_tb-meaneye_loc)
            crossprod2 = np.cross(head_gaze_vect,corner1_tb-meaneye_loc)*np.cross(head_gaze_vect,corner3_tb-meaneye_loc)
            crossprod3 = np.cross(head_gaze_vect,corner1_tb-meaneye_loc)*np.cross(head_gaze_vect,corner4_tb-meaneye_loc)
            crossprod4 = np.cross(head_gaze_vect,corner2_tb-meaneye_loc)*np.cross(head_gaze_vect,corner3_tb-meaneye_loc)
            crossprod5 = np.cross(head_gaze_vect,corner2_tb-meaneye_loc)*np.cross(head_gaze_vect,corner4_tb-meaneye_loc)
            crossprod6 = np.cross(head_gaze_vect,corner3_tb-meaneye_loc)*np.cross(head_gaze_vect,corner4_tb-meaneye_loc)
            if (((crossprod1<0)|(crossprod2<0)|(crossprod3<0)|(crossprod4<0)|(crossprod5<0)|(crossprod6<0))&(tube_eye_angle<np.pi/2)):
                if (considerlevertube):
                    if ((tube_eye_angle<other_eye_angle)&(tube_eye_angle<lever_eye_angle)):
                        look_at_tube_iframe = 1
                elif (considertubeonly):
                    if (tube_eye_angle<other_eye_angle):
                        look_at_tube_iframe = 1
            look_at_tube_frames.append(look_at_tube_iframe)
            #
            # define whehter looking at the other lever
            look_at_otherlever_iframe = 0
            # four corner of the other lever boundary
            corner1_lv = np.array([otherlever_loc[0]-sqr_thres_tubelever, otherlever_loc[1]-sqr_thres_tubelever])
            corner2_lv = np.array([otherlever_loc[0]-sqr_thres_tubelever, otherlever_loc[1]+sqr_thres_tubelever])
            corner3_lv = np.array([otherlever_loc[0]+sqr_thres_tubelever, otherlever_loc[1]-sqr_thres_tubelever])
            corner4_lv = np.array([otherlever_loc[0]+sqr_thres_tubelever, otherlever_loc[1]+sqr_thres_tubelever])
            crossprod1 = np.cross(head_gaze_vect,corner1_lv-meaneye_loc)*np.cross(head_gaze_vect,corner2_lv-meaneye_loc)
            crossprod2 = np.cross(head_gaze_vect,corner1_lv-meaneye_loc)*np.cross(head_gaze_vect,corner3_lv-meaneye_loc)
            crossprod3 = np.cross(head_gaze_vect,corner1_lv-meaneye_loc)*np.cross(head_gaze_vect,corner4_lv-meaneye_loc)
            crossprod4 = np.cross(head_gaze_vect,corner2_lv-meaneye_loc)*np.cross(head_gaze_vect,corner3_lv-meaneye_loc)
            crossprod5 = np.cross(head_gaze_vect,corner2_lv-meaneye_loc)*np.cross(head_gaze_vect,corner4_lv-meaneye_loc)
            crossprod6 = np.cross(head_gaze_vect,corner3_lv-meaneye_loc)*np.cross(head_gaze_vect,corner4_lv-meaneye_loc)
            if (((crossprod1<0)|(crossprod2<0)|(crossprod3<0)|(crossprod4<0)|(crossprod5<0)|(crossprod6<0))&(otherlever_eye_angle<np.pi/2)):
                if (considerlevertube):       
                    if ((otherlever_eye_angle<other_eye_angle)&(otherlever_eye_angle<tube_eye_angle)&(otherlever_eye_angle<lever_eye_angle)):
                        look_at_otherlever_iframe = 1
            look_at_otherlever_frames.append(look_at_otherlever_iframe)
            #
            # define whether looking at the other tube
            look_at_othertube_iframe = 0
            # four corner of the other tube boundary
            corner1_tb = np.array([othertube_loc[0]-sqr_thres_tubelever, othertube_loc[1]-sqr_thres_tubelever])
            corner2_tb = np.array([othertube_loc[0]-sqr_thres_tubelever, othertube_loc[1]+sqr_thres_tubelever])
            corner3_tb = np.array([othertube_loc[0]+sqr_thres_tubelever, othertube_loc[1]-sqr_thres_tubelever])
            corner4_tb = np.array([othertube_loc[0]+sqr_thres_tubelever, othertube_loc[1]+sqr_thres_tubelever])
            crossprod1 = np.cross(head_gaze_vect,corner1_tb-meaneye_loc)*np.cross(head_gaze_vect,corner2_tb-meaneye_loc)
            crossprod2 = np.cross(head_gaze_vect,corner1_tb-meaneye_loc)*np.cross(head_gaze_vect,corner3_tb-meaneye_loc)
            crossprod3 = np.cross(head_gaze_vect,corner1_tb-meaneye_loc)*np.cross(head_gaze_vect,corner4_tb-meaneye_loc)
            crossprod4 = np.cross(head_gaze_vect,corner2_tb-meaneye_loc)*np.cross(head_gaze_vect,corner3_tb-meaneye_loc)
            crossprod5 = np.cross(head_gaze_vect,corner2_tb-meaneye_loc)*np.cross(head_gaze_vect,corner4_tb-meaneye_loc)
            crossprod6 = np.cross(head_gaze_vect,corner3_tb-meaneye_loc)*np.cross(head_gaze_vect,corner4_tb-meaneye_loc)
            if (((crossprod1<0)|(crossprod2<0)|(crossprod3<0)|(crossprod4<0)|(crossprod5<0)|(crossprod6<0))&(othertube_eye_angle<np.pi/2)):
                if (considerlevertube):
                    if ((othertube_eye_angle<other_eye_angle)&(othertube_eye_angle<lever_eye_angle)&(othertube_eye_angle<tube_eye_angle)):
                        look_at_othertube_iframe = 1
                elif (considertubeonly):
                    if ((othertube_eye_angle<other_eye_angle)&(othertube_eye_angle<tube_eye_angle)):
                        look_at_othertube_iframe = 1
            look_at_othertube_frames.append(look_at_othertube_iframe)
        

       # save to the summarized data
        head_vect_all_merge[(iname)] = head_vect_frames
        other_eye_vect_all_merge[(iname)] = other_eye_vect_frames
        lever_eye_vect_all_merge[(iname)] = lever_eye_vect_frames
        tube_eye_vect_all_merge[(iname)] = tube_eye_vect_frames
        otherlever_eye_vect_all_merge[(iname)] = otherlever_eye_vect_frames
        othertube_eye_vect_all_merge[(iname)] = othertube_eye_vect_frames
        #
        # angle between head vector and each of the gaze vector
        other_eye_angle_all_merge[(iname)] = other_eye_angle_frames
        lever_eye_angle_all_merge[(iname)] = lever_eye_angle_frames
        tube_eye_angle_all_merge[(iname)] = tube_eye_angle_frames 
        otherlever_eye_angle_all_merge[(iname)] = otherlever_eye_angle_frames
        othertube_eye_angle_all_merge[(iname)] = othertube_eye_angle_frames 
        # 
        look_at_other_or_not_merge[(iname)] = look_at_other_frames
        look_at_otherface_or_not_merge[(iname)] = look_at_otherface_frames
        look_at_tube_or_not_merge[(iname)] = look_at_tube_frames
        look_at_lever_or_not_merge[(iname)] = look_at_lever_frames
        look_at_othertube_or_not_merge[(iname)] = look_at_othertube_frames
        look_at_otherlever_or_not_merge[(iname)] = look_at_otherlever_frames


    output_allvectors = {"head_vect_all_merge":head_vect_all_merge,"other_eye_vect_all_merge":other_eye_vect_all_merge,"lever_eye_vect_all_merge":lever_eye_vect_all_merge,"tube_eye_vect_all_merge":tube_eye_vect_all_merge,"otherlever_eye_vect_all_merge":otherlever_eye_vect_all_merge,"othertube_eye_vect_all_merge":othertube_eye_vect_all_merge}       

    output_allangles = {"other_eye_angle_all_merge":other_eye_angle_all_merge,"lever_eye_angle_all_merge":lever_eye_angle_all_merge,"tube_eye_angle_all_merge":tube_eye_angle_all_merge,"otherlever_eye_angle_all_merge":otherlever_eye_angle_all_merge,"othertube_eye_angle_all_merge":othertube_eye_angle_all_merge} 

    output_look_ornot = {"look_at_other_or_not_merge":look_at_other_or_not_merge,"look_at_otherface_or_not_merge":look_at_otherface_or_not_merge,"look_at_tube_or_not_merge":look_at_tube_or_not_merge,"look_at_lever_or_not_merge":look_at_lever_or_not_merge,"look_at_othertube_or_not_merge":look_at_othertube_or_not_merge,"look_at_otherlever_or_not_merge":look_at_otherlever_or_not_merge}
       
 
    return output_look_ornot, output_allvectors, output_allangles
        

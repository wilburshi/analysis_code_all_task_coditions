# function - define time point of behavioral events

def bhv_events_timepoint(bhv_data, animal1, animal2, eye_direction_merge, eye_contact_or_not_merge, look_at_face_or_not_merge):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    time_point_pull1 = bhv_data["time_points"][bhv_data["behavior_events"]==1]
    time_point_pull2 = bhv_data["time_points"][bhv_data["behavior_events"]==2]
    ind_lookatotherface1 = np.where(np.array(look_at_face_or_not_merge['dodson'])==1)
    time_point_lookatotherface1 = look_at_face_or_not_merge["time_in_second"][ind_lookatotherface1]
    ind_lookatotherface2 = np.where(np.array(look_at_face_or_not_merge['scorch'])==1)
    time_point_lookatotherface2 = look_at_face_or_not_merge["time_in_second"][ind_lookatotherface2]

    ind_eyecontact1 = np.where(np.array(eye_contact_or_not_merge['dodson'])==1)
    time_point_eyecontact1 = eye_contact_or_not_merge["time_in_second"][ind_eyecontact1]
    ind_eyecontact2 = np.where(np.array(eye_contact_or_not_merge['scorch'])==1)
    time_point_eyecontact2 = eye_contact_or_not_merge["time_in_second"][ind_eyecontact2]

    # calculate the oneway gaze or mutual gaze
    animal1_gaze = np.round(np.concatenate((time_point_eyecontact1,time_point_lookatotherface1)),1)
    animal1_gaze = np.unique(np.sort(animal1_gaze))
    animal2_gaze = np.round(np.concatenate((time_point_eyecontact2,time_point_lookatotherface2)),1)
    animal2_gaze = np.unique(np.sort(animal2_gaze))

    ngaze1 = len(animal1_gaze)
    ngaze2 = len(animal2_gaze)
    oneway_gaze1 = []
    oneway_gaze2 = []
    mutual_gaze1 = []
    mutual_gaze2 = []
    # 
    for igaze1 in np.arange(0, ngaze1, 1):
        for igaze2 in np.arange(0,ngaze2,1):
            if abs(animal1_gaze[igaze1]-animal2_gaze[igaze2])<1:
                mutual_gaze1.append(animal1_gaze[igaze1])
                mutual_gaze2.append(animal2_gaze[igaze2])
    mutual_gaze1 = np.unique(mutual_gaze1)   
    mutual_gaze2 = np.unique(mutual_gaze2)
    oneway_gaze1 = animal1_gaze[~np.isin(animal1_gaze,mutual_gaze1)]
    oneway_gaze2 = animal2_gaze[~np.isin(animal2_gaze,mutual_gaze2)]
    
    time_point_pull1 = np.round(time_point_pull1,2)
    time_point_pull2 = np.round(time_point_pull2,2)
    
    return time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2

# function - define time point of behavioral events

def bhv_events_timepoint_Anipose(bhv_data, look_at_Anipose):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    time_point_pull1 = bhv_data["time_points"][bhv_data["behavior_events"]==1]
    time_point_pull2 = bhv_data["time_points"][bhv_data["behavior_events"]==2]

    time_point_pull1 = np.round(time_point_pull1,1)
    time_point_pull2 = np.round(time_point_pull2,1)
    
    time_point_juice1 = bhv_data["time_points"][bhv_data["behavior_events"]==3]
    time_point_juice2 = bhv_data["time_points"][bhv_data["behavior_events"]==4]

    time_point_juice1 = np.round(time_point_juice1,1)
    time_point_juice2 = np.round(time_point_juice2,1)

    look_at_face_or_not_Anipose = look_at_Anipose['face']
    look_at_selftube_or_not_Anipose = look_at_Anipose['selftube']
    look_at_selflever_or_not_Anipose = look_at_Anipose['selflever']
    look_at_othertube_or_not_Anipose = look_at_Anipose['othertube']
    look_at_otherlever_or_not_Anipose = look_at_Anipose['otherlever']


    # calculate the oneway gaze or mutual gaze
    # ind_lookatother1 = np.where((np.array(look_at_face_or_not_Anipose['dodson'])==1))
    ind_lookatother1 = np.where((np.array(look_at_face_or_not_Anipose['dodson'])==1)|(np.array(look_at_otherlever_or_not_Anipose['dodson'])==1)|(np.array(look_at_othertube_or_not_Anipose['dodson'])==1))
    time_point_lookatother1 = look_at_face_or_not_Anipose["time_in_second"][ind_lookatother1]
    # ind_lookatother2 = np.where((np.array(look_at_face_or_not_Anipose['scorch'])==1))
    ind_lookatother2 = np.where((np.array(look_at_face_or_not_Anipose['scorch'])==1)|(np.array(look_at_otherlever_or_not_Anipose['scorch'])==1)|(np.array(look_at_othertube_or_not_Anipose['scorch'])==1))
    time_point_lookatother2 = look_at_face_or_not_Anipose["time_in_second"][ind_lookatother2]


    # 
    animal1_gaze = np.round(time_point_lookatother1,1)
    animal1_gaze = np.unique(np.sort(animal1_gaze))
    animal2_gaze = np.round(time_point_lookatother2,1)
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
    

    # round the time of looking at the levers or tubes
    ind_lookatlever1 = np.where(np.array(look_at_selflever_or_not_Anipose['dodson'])==1)
    time_point_lookatlever1 = look_at_selflever_or_not_Anipose["time_in_second"][ind_lookatlever1] 
    ind_lookatlever2 = np.where(np.array(look_at_selflever_or_not_Anipose['scorch'])==1)
    time_point_lookatlever2 = look_at_selflever_or_not_Anipose["time_in_second"][ind_lookatlever2]

    ind_lookattube1 = np.where(np.array(look_at_selftube_or_not_Anipose['dodson'])==1)
    time_point_lookattube1 = look_at_selftube_or_not_Anipose["time_in_second"][ind_lookattube1] 
    ind_lookattube2 = np.where(np.array(look_at_selftube_or_not_Anipose['scorch'])==1)
    time_point_lookattube2 = look_at_selftube_or_not_Anipose["time_in_second"][ind_lookattube2]

    #
    time_point_lookatlever1 = np.round(time_point_lookatlever1,1)
    time_point_lookatlever1 = np.unique(np.sort(time_point_lookatlever1))
    time_point_lookatlever2 = np.round(time_point_lookatlever2,1)
    time_point_lookatlever2 = np.unique(np.sort(time_point_lookatlever2))
    
    time_point_lookattube1 = np.round(time_point_lookattube1,1)
    time_point_lookattube1 = np.unique(np.sort(time_point_lookattube1))
    time_point_lookattube2 = np.round(time_point_lookattube2,1)
    time_point_lookattube2 = np.unique(np.sort(time_point_lookattube2))


    output_time_points_socialgaze = {"time_point_pull1":time_point_pull1,"time_point_pull2":time_point_pull2,"time_point_juice1":time_point_juice1,"time_point_juice2":time_point_juice2,"oneway_gaze1":oneway_gaze1,"oneway_gaze2":oneway_gaze2, "mutual_gaze1":mutual_gaze1,"mutual_gaze2":mutual_gaze2}

    output_time_points_levertube = {"time_point_pull1":time_point_pull1,"time_point_pull2":time_point_pull2,"time_point_juice1":time_point_juice1,"time_point_juice2":time_point_juice2,"time_point_lookatlever1":time_point_lookatlever1,"time_point_lookatlever2":time_point_lookatlever2, "time_point_lookattube1":time_point_lookattube1,"time_point_lookattube2":time_point_lookattube2}
    
    return output_time_points_socialgaze, output_time_points_levertube

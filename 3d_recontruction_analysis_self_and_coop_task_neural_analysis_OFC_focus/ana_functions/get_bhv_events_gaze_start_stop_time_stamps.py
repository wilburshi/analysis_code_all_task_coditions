# # function - get the gaze start and stop time stamps
def get_bhv_events_gaze_start_stop_time_stamps(animal1, animal2, time_point_pull1, time_point_pull2, time_point_pulls_succfail, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, gaze_thresold, totalsess_time_forFR):
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import scipy.stats as st
    import os

    time_point_pull1 = np.array(time_point_pull1)
    time_point_pull2 = np.array(time_point_pull2)
    # merge oneway gaze and mutual gaze
    oneway_gaze1 = np.sort(np.hstack((oneway_gaze1,mutual_gaze1)))
    oneway_gaze2 = np.sort(np.hstack((oneway_gaze2,mutual_gaze2)))

    # get the gaze start and stop
    #animal1_gaze = np.concatenate([oneway_gaze1, mutual_gaze1])
    animal1_gaze = oneway_gaze1
    animal1_gaze = np.sort(np.unique(animal1_gaze))
    animal1_gaze_stop = animal1_gaze[np.concatenate(((animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal1_gaze_start = np.concatenate(([animal1_gaze[0]],animal1_gaze[np.where(animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)[0]+1]))
    animal1_gaze_flash = np.intersect1d(animal1_gaze_start, animal1_gaze_stop)
    animal1_gaze_start = animal1_gaze_start[~np.isin(animal1_gaze_start,animal1_gaze_flash)]
    animal1_gaze_stop = animal1_gaze_stop[~np.isin(animal1_gaze_stop,animal1_gaze_flash)]
    #
    #animal2_gaze = np.concatenate([oneway_gaze2, mutual_gaze2])
    animal2_gaze = oneway_gaze2
    animal2_gaze = np.sort(np.unique(animal2_gaze))
    animal2_gaze_stop = animal2_gaze[np.concatenate(((animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal2_gaze_start = np.concatenate(([animal2_gaze[0]],animal2_gaze[np.where(animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)[0]+1]))
    animal2_gaze_flash = np.intersect1d(animal2_gaze_start, animal2_gaze_stop)
    animal2_gaze_start = animal2_gaze_start[~np.isin(animal2_gaze_start,animal2_gaze_flash)]
    animal2_gaze_stop = animal2_gaze_stop[~np.isin(animal2_gaze_stop,animal2_gaze_flash)] 

    # get the successful and failed pull time point
    time_point_pull1_succ = np.array(time_point_pulls_succfail['pull1_succ'])
    time_point_pull2_succ = np.array(time_point_pulls_succfail['pull2_succ'])
    time_point_pull1_fail = np.array(time_point_pulls_succfail['pull1_fail'])
    time_point_pull2_fail = np.array(time_point_pulls_succfail['pull2_fail'])


    # keep the total time consistent
    time_point_pull1 = np.unique(time_point_pull1[time_point_pull1<totalsess_time_forFR])
    time_point_pull2 = np.unique(time_point_pull2[time_point_pull2<totalsess_time_forFR])
    oneway_gaze1 = np.unique(oneway_gaze1[oneway_gaze1<totalsess_time_forFR])
    oneway_gaze2 = np.unique(oneway_gaze2[oneway_gaze2<totalsess_time_forFR])
    animal1_gaze_start = np.unique(animal1_gaze_start[animal1_gaze_start<totalsess_time_forFR])
    animal2_gaze_start = np.unique(animal2_gaze_start[animal2_gaze_start<totalsess_time_forFR])
    animal1_gaze_stop = np.unique(animal1_gaze_stop[animal1_gaze_stop<totalsess_time_forFR])
    animal2_gaze_stop = np.unique(animal2_gaze_stop[animal2_gaze_stop<totalsess_time_forFR])
    #
    time_point_pull1_succ = np.unique(time_point_pull1_succ[time_point_pull1_succ<totalsess_time_forFR])
    time_point_pull2_succ = np.unique(time_point_pull2_succ[time_point_pull2_succ<totalsess_time_forFR])
    time_point_pull1_fail = np.unique(time_point_pull1_fail[time_point_pull1_fail<totalsess_time_forFR])
    time_point_pull2_fail = np.unique(time_point_pull2_fail[time_point_pull2_fail<totalsess_time_forFR])

    
    return animal1_gaze_start, animal1_gaze_stop, animal2_gaze_start, animal2_gaze_stop
          

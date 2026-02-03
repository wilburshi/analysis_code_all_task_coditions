# # function - run firing rate from pull onset to pull action 
def plot_bhv_events_aligned_FR_PullStartToPull_variedSection(animal1, animal2, time_point_pull1, time_point_pull2, time_point_pulls_succfail, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, gaze_thresold, totalsess_time_forFR, pull1_rt, pull2_rt, fps, FR_timepoint_allch, FR_zscore_allch, clusters_info_data):
    
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import scipy.stats as st
    import os

    fps_FR = int(np.ceil(np.shape(FR_timepoint_allch)[0]/np.max(FR_timepoint_allch)))

    time_point_pull1 = np.array(time_point_pull1)
    time_point_pull2 = np.array(time_point_pull2)
    # merge oneway gaze and mutual gaze
    oneway_gaze1 = np.sort(np.hstack((oneway_gaze1,mutual_gaze1)))
    oneway_gaze2 = np.sort(np.hstack((oneway_gaze2,mutual_gaze2)))

    # get the gaze start and stop
    #animal1_gaze = np.concatenate([oneway_gaze1, mutual_gaze1])
    animal1_gaze = oneway_gaze1
    animal1_gaze = np.sort(np.unique(animal1_gaze))
    try:
        animal1_gaze_stop = animal1_gaze[np.concatenate(((animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
        animal1_gaze_start = np.concatenate(([animal1_gaze[0]],animal1_gaze[np.where(animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)[0]+1]))
        animal1_gaze_flash = np.intersect1d(animal1_gaze_start, animal1_gaze_stop)
        animal1_gaze_start = animal1_gaze_start[~np.isin(animal1_gaze_start,animal1_gaze_flash)]
        animal1_gaze_stop = animal1_gaze_stop[~np.isin(animal1_gaze_stop,animal1_gaze_flash)]
    except:
        animal1_gaze_flash = np.nan
        animal1_gaze_start = np.nan
        animal1_gaze_stop = np.nan
    #
    #animal2_gaze = np.concatenate([oneway_gaze2, mutual_gaze2])
    animal2_gaze = oneway_gaze2
    animal2_gaze = np.sort(np.unique(animal2_gaze))
    try:
        animal2_gaze_stop = animal2_gaze[np.concatenate(((animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
        animal2_gaze_start = np.concatenate(([animal2_gaze[0]],animal2_gaze[np.where(animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)[0]+1]))
        animal2_gaze_flash = np.intersect1d(animal2_gaze_start, animal2_gaze_stop)
        animal2_gaze_start = animal2_gaze_start[~np.isin(animal2_gaze_start,animal2_gaze_flash)]
        animal2_gaze_stop = animal2_gaze_stop[~np.isin(animal2_gaze_stop,animal2_gaze_flash)] 
    except:
        animal2_gaze_flash = np.nan
        animal2_gaze_start = np.nan
        animal2_gaze_stop = np.nan 

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
    try:
        animal1_gaze_start = np.unique(animal1_gaze_start[animal1_gaze_start<totalsess_time_forFR])
        animal2_gaze_start = np.unique(animal2_gaze_start[animal2_gaze_start<totalsess_time_forFR])
        animal1_gaze_stop = np.unique(animal1_gaze_stop[animal1_gaze_stop<totalsess_time_forFR])
        animal2_gaze_stop = np.unique(animal2_gaze_stop[animal2_gaze_stop<totalsess_time_forFR])
    except:
        animal2_gaze_start = np.nan
        animal2_gaze_stop = np.nan        

    #
    time_point_pull1_succ = np.unique(time_point_pull1_succ[time_point_pull1_succ<totalsess_time_forFR])
    time_point_pull2_succ = np.unique(time_point_pull2_succ[time_point_pull2_succ<totalsess_time_forFR])
    time_point_pull1_fail = np.unique(time_point_pull1_fail[time_point_pull1_fail<totalsess_time_forFR])
    time_point_pull2_fail = np.unique(time_point_pull2_fail[time_point_pull2_fail<totalsess_time_forFR])


    # unit clusters
    clusterIDs = list(FR_zscore_allch.keys())
    ncells = np.shape(clusterIDs)[0]


    # change the bhv event time point to reflect the batch on both side (add 4s batch)
    time_point_pull1_align = time_point_pull1 + 4
    time_point_pull2_align = time_point_pull2 + 4
    


    # plot the figures
    # align to the bhv events
    bhv_events_anatypes = ['pull1','pull2',
                          ]
    bhv_events_names = [animal1+' pull', animal2+' pull',
                       ]
    timepoint_bhvevents = {'pull1':time_point_pull1_align,
                           'pull2':time_point_pull2_align,
                          }
    alignwins = {'pull1':pull1_rt,
                 'pull2':pull2_rt,
                }
    
    clrs_plot = ['r','y',
                 ]
    nanatypes = np.shape(bhv_events_anatypes)[0]

    #
    bhvevents_aligned_FR_allevents_all = dict.fromkeys(bhv_events_names,[])

    # 
    # loop for all bhv events
    for ianatype in np.arange(0,nanatypes,1):

        bhvevent_anatype = bhv_events_anatypes[ianatype]
        bhvevent_name = bhv_events_names[ianatype]
        bhv_event_timepoint = timepoint_bhvevents[bhvevent_anatype]
        alignwin = alignwins[bhvevent_anatype]

        try:
            nevents = np.shape(bhv_event_timepoint)[0]
        except:
            nevents = 0
        #
        bhvevents_aligned_FR_allevents_all[bhvevent_name] = dict.fromkeys(clusterIDs,[])
        
        # loop for all units/cells/neurons
        for icell in np.arange(0,ncells,1):

            clusterID = clusterIDs[icell]
            
            try:
                spike_channel = list(clusters_info_data[clusters_info_data['cluster_id']==int(clusterID)]['ch'])[0]
            except:
                spike_channel = list(clusters_info_data[clusters_info_data['id']==int(clusterID)]['ch'])[0]
            
            # load the FR
            FR_target = FR_zscore_allch[clusterID]

            # add nan batch to both sides
            FR_target_leftbatch = np.hstack((np.full((1,4*fps_FR),np.nan)[0],FR_target))
            FR_target_bothbatch = np.hstack((FR_target_leftbatch,np.full((1,4*fps_FR),np.nan)[0]))

            #
            alltraces_icell = []

            # loop for all individual event
            for ievent in np.arange(0,nevents,1):

                time_point_ievent = bhv_event_timepoint[ievent]
                alignwin_ievent = alignwin[ievent]
                #
                xxx_trigevent = np.arange(0,alignwin_ievent,1/fps) 
                
                try:
                    itrace_icell = FR_target_bothbatch[int((time_point_ievent)*fps_FR):int((time_point_ievent+alignwin_ievent)*fps_FR)]
                except:
                    itrace_icell = np.full((1,np.shape(xxx_trigevent)[0]),np.nan)[0]
                    
                alltraces_icell.append(itrace_icell)
                    

            

            # put the data in the summarizing dataset
            bhvevents_aligned_FR_allevents_all[bhvevent_name][str(clusterID)] = {'ch': spike_channel,'FR_allevents': alltraces_icell}   
                
                
   

    
    return bhvevents_aligned_FR_allevents_all
          

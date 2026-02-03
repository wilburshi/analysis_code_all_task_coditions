# # function - run firing rate around behavior events
def plot_bhv_events_aligned_FR(date_tgt,savefig,save_path, animal1, animal2,time_point_pull1,time_point_pull2,time_point_pulls_succfail,oneway_gaze1,oneway_gaze2,mutual_gaze1,mutual_gaze2,gaze_thresold,totalsess_time_forFR,aligntwins,fps,FR_timepoint_allch,FR_zscore_allch,clusters_info_data):
    
    
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


    # unit clusters
    clusterIDs = list(FR_zscore_allch.keys())
    ncells = np.shape(clusterIDs)[0]


    # change the bhv event time point to reflect the batch on both side
    time_point_pull1_align = time_point_pull1 + aligntwins
    time_point_pull2_align = time_point_pull2 + aligntwins
    oneway_gaze1_align = oneway_gaze1 + aligntwins
    oneway_gaze2_align = oneway_gaze2 + aligntwins
    animal1_gaze_start_align = animal1_gaze_start + aligntwins
    animal2_gaze_start_align = animal2_gaze_start + aligntwins
    animal1_gaze_stop_align = animal1_gaze_stop + aligntwins
    animal2_gaze_stop_align = animal2_gaze_stop + aligntwins
    #
    time_point_pull1_succ_align = time_point_pull1_succ + aligntwins
    time_point_pull2_succ_align = time_point_pull2_succ + aligntwins
    time_point_pull1_fail_align = time_point_pull1_fail + aligntwins
    time_point_pull2_fail_align = time_point_pull2_fail + aligntwins


    # plot the figures
    # align to the bhv events
    bhv_events_anatypes = ['pull1','pull2',
                           'pull1_succ','pull1_fail',
                           'pull2_succ','pull2_fail',
                           'gaze1','gaze2',
                           'gaze1_start','gaze1_stop',
                           'gaze2_start','gaze2_stop',
                          ]
    bhv_events_names = [animal1+' pull', animal2+' pull',
                        animal1+' succpull', animal1+' failpull',
                        animal2+' succpull', animal2+' failpull',
                        animal1+' gaze', animal2+' gaze',
                        animal1+' gazestart', animal1+' gazestop',
                        animal2+' gazestart', animal2+' gazestop',
                       ]
    timepoint_bhvevents = {'pull1':time_point_pull1_align,
                           'pull2':time_point_pull2_align,
                           'pull1_succ':time_point_pull1_succ_align,
                           'pull2_succ':time_point_pull2_succ_align,
                           'pull1_fail':time_point_pull1_fail_align,
                           'pull2_fail':time_point_pull2_fail_align,
                           'gaze1':oneway_gaze1_align,
                           'gaze2':oneway_gaze2_align,
                           'gaze1_start':animal1_gaze_start_align,
                           'gaze1_stop':animal1_gaze_stop_align,
                           'gaze2_start':animal2_gaze_start_align,
                           'gaze2_stop':animal2_gaze_stop_align,
                          }
    clrs_plot = ['r','y',
                 'g','b',
                 'c','m',
                 '#7BC8F6','#9467BD',
                 '#458B74','#FFC710',
                 '#FF1493','#A9A9A9',
                 '#8B4513','#FFC0CB',]
    nanatypes = np.shape(bhv_events_anatypes)[0]

    #
    bhvevents_aligned_FR_average_all = dict.fromkeys(bhv_events_names,[])
    bhvevents_aligned_FR_allevents_all = dict.fromkeys(bhv_events_names,[])

    # 
    fig2, axs2 = plt.subplots(nanatypes,ncells)
    fig2.set_figheight(2*nanatypes)
    fig2.set_figwidth(3*ncells)

    # loop for all bhv events
    for ianatype in np.arange(0,nanatypes,1):

        bhvevent_anatype = bhv_events_anatypes[ianatype]
        bhvevent_name = bhv_events_names[ianatype]
        bhv_event_timepoint = timepoint_bhvevents[bhvevent_anatype]

        nevents = np.shape(bhv_event_timepoint)[0]

        #
        bhvevents_aligned_FR_average_all[bhvevent_name] = dict.fromkeys(clusterIDs,[])
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
            FR_target_leftbatch = np.hstack((np.full((1,aligntwins*fps_FR),np.nan)[0],FR_target))
            FR_target_bothbatch = np.hstack((FR_target_leftbatch,np.full((1,aligntwins*fps_FR),np.nan)[0]))

            #
            xxx_forplot = np.arange(-aligntwins*fps_FR,aligntwins*fps_FR,1)
            alltraces_icell = np.ones((np.shape(xxx_forplot)[0],nevents))*np.nan

            # loop for all individual event
            for ievent in np.arange(0,nevents,1):

                time_point_ievent = bhv_event_timepoint[ievent]
                try:
                    alltraces_icell[:,ievent] = FR_target_bothbatch[int((time_point_ievent-aligntwins)*fps_FR):int((time_point_ievent+aligntwins)*fps_FR)]
                except:
                    continue

            #
            mean_trig_trace = np.nanmean(alltraces_icell,axis=1)
            std_trig_trace = np.nanstd(alltraces_icell,axis=1)
            sem_trig_trace = np.nanstd(alltraces_icell,axis=1)/np.sqrt(np.shape(alltraces_icell)[1])
            itv95_trig_trace = 1.96*sem_trig_trace

            #
            axs2[ianatype,icell].errorbar(xxx_forplot,mean_trig_trace,yerr=itv95_trig_trace,color='#212121',ecolor=clrs_plot[ianatype])
            axs2[ianatype,icell].plot([0,0],[np.nanmin(mean_trig_trace-itv95_trig_trace),np.nanmax(mean_trig_trace+itv95_trig_trace)],'--k')

            #
            if icell == 0:
                axs2[ianatype,icell].set_ylabel(bhvevent_name)
            # else:
            #     axs2[ianatype,icell].set_yticklabels([])

            if ianatype == nanatypes-1:
                axs2[ianatype,icell].set_xlabel('time (s)')
                axs2[ianatype,icell].set_xticks(np.arange(-aligntwins*fps_FR,aligntwins*fps_FR,60))
                axs2[ianatype,icell].set_xticklabels(list(map(str,np.arange(-aligntwins,aligntwins,2))))
            else:
                axs2[ianatype,icell].set_xticklabels([])

            if ianatype == 0:
                axs2[ianatype,icell].set_title('cluster#'+str(clusterID))

            # put the data in the summarizing dataset
            bhvevents_aligned_FR_average_all[bhvevent_name][str(clusterID)] = {'ch':spike_channel,'FR_average':mean_trig_trace}
            bhvevents_aligned_FR_allevents_all[bhvevent_name][str(clusterID)] = {'ch':spike_channel,'FR_allevents':alltraces_icell}   
                
                
    #
    if savefig:
        fig2.savefig(save_path+"/"+date_tgt+'_bhv_events_aligned_FR.pdf')

    
    return bhvevents_aligned_FR_average_all, bhvevents_aligned_FR_allevents_all
          
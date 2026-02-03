#  function - plot spike triggered averge of continuous bhv variables

def plot_spike_triggered_continuous_bhv(date_tgt,savefig,save_path, animal1, animal2, session_start_time, min_length, time_point_pull1, time_point_pull2, time_point_pulls_succfail,
                          oneway_gaze1,oneway_gaze2,mutual_gaze1,mutual_gaze2,gaze_thresold,animalnames_videotrack, output_look_ornot, output_allvectors, output_allangles, output_key_locations, spike_clusters_data, spike_time_data,spike_channels_data,do_shuffle):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import scipy.stats as st
    import string
    import warnings
    import pickle    
       
    
    # prepare some time stamp data
    # merge oneway gaze and mutual gaze # note!! the time stamps are already aligned to the start of session (instead of the start of video recording)
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
    
    
    nanimals = np.shape(animalnames_videotrack)[0]

    gausKernelsize = 3

    fps = 30     

    trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s

    con_vars_plot = ['gaze_other_angle','gaze_tube_angle','gaze_lever_angle','animal_animal_dist','animal_tube_dist','animal_lever_dist',
                     'othergaze_self_angle','mass_move_speed','gaze_angle_speed','leverpull_prob','socialgaze_prob']
    nconvarplots = np.shape(con_vars_plot)[0]

    clrs_plot = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493','#A9A9A9','#8B4513']
    yaxis_labels = ['degree','degree','degree','dist(a.u.)','dist(a.u.)','dist(a.u.)','degree','pixel/s','degree/s','','']

    spike_trig_average_all = dict.fromkeys([animal1,animal2],[])

    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]

        # initialize the summarize data 
        if ianimal == 0:
            spike_trig_average_all[animal1] = dict.fromkeys(con_vars_plot,[])
        elif ianimal == 1:
            spike_trig_average_all[animal2] = dict.fromkeys(con_vars_plot,[])

        # get the variables
        # keep the tracking time to the start of video recording
        xxx_time = np.arange(0,min_length,1)/fps

        gaze_other_angle = output_allangles['face_eye_angle_all_Anipose'][animal_name]
        gaze_other_angle = scipy.ndimage.gaussian_filter1d(gaze_other_angle,gausKernelsize)  
        gaze_tube_angle = output_allangles['selftube_eye_angle_all_Anipose'][animal_name]
        gaze_tube_angle = scipy.ndimage.gaussian_filter1d(gaze_tube_angle,gausKernelsize)  
        gaze_lever_angle = output_allangles['selflever_eye_angle_all_Anipose'][animal_name]
        gaze_lever_angle = scipy.ndimage.gaussian_filter1d(gaze_lever_angle,gausKernelsize)  

        othergaze_self_angle = output_allangles['face_eye_angle_all_Anipose'][animal_name_other]
        othergaze_self_angle = scipy.ndimage.gaussian_filter1d(othergaze_self_angle,5)  


        a = output_key_locations['facemass_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['facemass_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_animal_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_animal_dist = scipy.ndimage.gaussian_filter1d(animal_animal_dist,gausKernelsize)  

        a = output_key_locations['tube_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_tube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_tube_dist = scipy.ndimage.gaussian_filter1d(animal_tube_dist,gausKernelsize)  

        a = output_key_locations['lever_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_lever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_lever_dist = scipy.ndimage.gaussian_filter1d(animal_lever_dist,gausKernelsize)  

        a = output_key_locations['facemass_loc_all_Anipose'][animal_name].transpose()
        a = np.hstack((a,[[np.nan],[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        mass_move_speed = scipy.ndimage.gaussian_filter1d(mass_move_speed,gausKernelsize)  

        a = np.array(output_allvectors['eye_direction_Anipose'][animal_name]).transpose()
        a = np.hstack((a,[[np.nan],[np.nan],[np.nan]]))
        at1 = a[:,1:]
        at0 = a[:,:-1] 
        nframes = np.shape(at1)[1]
        gaze_angle_speed = np.empty((1,nframes,))
        gaze_angle_speed[:] = np.nan
        gaze_angle_speed = gaze_angle_speed[0]
        #
        for iframe in np.arange(0,nframes,1):
            gaze_angle_speed[iframe] = np.arccos(np.clip(np.dot(at1[:,iframe]/np.linalg.norm(at1[:,iframe]), at0[:,iframe]/np.linalg.norm(at0[:,iframe])), -1.0, 1.0))    
        gaze_angle_speed = scipy.ndimage.gaussian_filter1d(gaze_angle_speed,gausKernelsize)  

        #
        # get the pull time series
        # align to the start of the video recording
        if ianimal == 0:
            timepoint_pull = time_point_pull1+session_start_time
        elif ianimal == 1:
            timepoint_pull = time_point_pull2+session_start_time
        #
        try:
            timeseries_pull = np.zeros(np.shape(gaze_angle_speed))
            timeseries_pull[list(map(int,list(np.round((timepoint_pull))*fps)))]=1
        except: # some videos are shorter than the task 
            timeseries_pull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_pull*fps))))+1,))
            timeseries_pull[list(map(int,list(np.round(timepoint_pull*fps))))]=1
        leverpull_prob = scipy.ndimage.gaussian_filter1d(timeseries_pull,gausKernelsize)

        #
        # get the self social gaze time series
        # align to the start of the video recording
        if ianimal == 0:
            timepoint_gaze = oneway_gaze1+session_start_time
        elif ianimal == 1:
            timepoint_gaze = oneway_gaze2+session_start_time
        #
        try:
            timeseries_gaze = np.zeros(np.shape(gaze_angle_speed))
            timeseries_gaze[list(map(int,list(np.round((timepoint_gaze))*fps)))]=1
        except: # some videos are shorter than the task 
            timeseries_gaze = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_gaze*fps))))+1,))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        socialgaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,gausKernelsize)


        # put all the data together in the same order as the con_vars_plot
        data_summary = [gaze_other_angle,gaze_tube_angle,gaze_lever_angle,animal_animal_dist,animal_tube_dist,animal_lever_dist,
                        othergaze_self_angle,mass_move_speed,gaze_angle_speed,leverpull_prob,socialgaze_prob]


        # get the spike related data
        spike_clusters_unique = np.unique(spike_clusters_data)

        # align the spike time to the start of the video recording
        spike_time_data_new = spike_time_data + session_start_time*fps

        nclusters = np.shape(spike_clusters_unique)[0]

        # plot the pull triggered continuous variables, plot for each bhv variable and each spike cluster
        fig2, axs2 = plt.subplots(nconvarplots,nclusters)
        fig2.set_figheight(2*nconvarplots)
        fig2.set_figwidth(3*nclusters)


        # if do_shuffle, plot the shuffled plot first
        # shuffle the cluster id information (keep the channel information the same)
        if do_shuffle:

            # shuffle the cluster id
            shuf_ind = np.arange(0,np.shape(spike_clusters_data)[0],1)
            np.random.shuffle(shuf_ind)
            spike_clusters_data_shf = spike_clusters_data[shuf_ind]
            spike_channels_data_shf = spike_channels_data[shuf_ind]

            # calculate based on each spike cluster
            #
            for icluster in np.arange(0,nclusters,1):
                iclusterID = spike_clusters_unique[icluster]

                ind_clusterID = np.isin(spike_clusters_data_shf,iclusterID)

                spike_time_icluster = spike_time_data_new[ind_clusterID]

                spike_time_icluster = np.unique(spike_time_icluster)

                spike_channel = np.unique(spike_channels_data_shf[ind_clusterID])[0]

                # calculate the spike triggered average
                nspikes = np.shape(spike_time_icluster)[0]


                # plot for each bhv variables
                for iplot in np.arange(0,nconvarplots,1):

                    # behavioral variables
                    yyy_bhv = data_summary[iplot]
                    # 
                    # normalize the yyy_bhv
                    yyy_bhv = (yyy_bhv-np.nanmin(yyy_bhv))/(np.nanmax(yyy_bhv)-np.nanmin(yyy_bhv))


                    xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)

                    alltraces_ispike = np.ones((np.shape(xxx_forplot)[0],nspikes))*np.nan

                    # plot for each spike cluster
                    for ispike in np.arange(0,nspikes,1):
                        ispike_time = spike_time_icluster[ispike]

                        ispike_trig_twins = [int(ispike_time+trig_twins[0]*fps),int(ispike_time+trig_twins[1]*fps)]

                        ispike_trig_trace = yyy_bhv[ispike_trig_twins[0]:ispike_trig_twins[1]]

                        try:
                            alltraces_ispike[:,ispike] = ispike_trig_trace
                            # axs2[iplot,icluster].plot(xxx_forplot,ispike_trig_trace)
                        except:
                            continue

                    mean_trig_trace = np.nanmean(alltraces_ispike,axis=1)
                    std_trig_trace = np.nanstd(alltraces_ispike,axis=1)
                    sem_trig_trace = np.nanstd(alltraces_ispike,axis=1)/np.sqrt(np.shape(alltraces_ispike)[1])
                    itv95_trig_trace = 1.96*sem_trig_trace

                    # axs2[iplot,icluster].plot(xxx_forplot,mean_trig_trace)
                    axs2[iplot,icluster].errorbar(xxx_forplot,mean_trig_trace,yerr=itv95_trig_trace,color='#E0E0E0',ecolor='#EEEEEE')
                    axs2[iplot,icluster].plot([0,0],[np.nanmin(mean_trig_trace-itv95_trig_trace),np.nanmax(mean_trig_trace+itv95_trig_trace)],'--k')



        # calculate based on each spike cluster
        #
        # plot for each bhv variables
        for iplot in np.arange(0,nconvarplots,1):

            # intialize summarized data
            if ianimal == 0:
                spike_trig_average_all[animal1][con_vars_plot[iplot]] = dict.fromkeys(list(map(str,spike_clusters_unique)),[])
            elif ianimal == 1:
                spike_trig_average_all[animal2][con_vars_plot[iplot]] = dict.fromkeys(list(map(str,spike_clusters_unique)),[])


            for icluster in np.arange(0,nclusters,1):
                iclusterID = spike_clusters_unique[icluster]

                ind_clusterID = np.isin(spike_clusters_data,iclusterID)

                spike_time_icluster = spike_time_data_new[ind_clusterID]

                spike_time_icluster = np.unique(spike_time_icluster)

                spike_channel = np.unique(spike_channels_data[ind_clusterID])[0]


                # calculate the spike triggered average
                nspikes = np.shape(spike_time_icluster)[0]


                # behavioral variables
                yyy_bhv = data_summary[iplot]
                # 
                # normalize the yyy_bhv
                yyy_bhv = (yyy_bhv-np.nanmin(yyy_bhv))/(np.nanmax(yyy_bhv)-np.nanmin(yyy_bhv))

                xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)

                alltraces_ispike = np.ones((np.shape(xxx_forplot)[0],nspikes))*np.nan

                # plot for each spike cluster
                for ispike in np.arange(0,nspikes,1):
                    ispike_time = spike_time_icluster[ispike]

                    ispike_trig_twins = [int(ispike_time+trig_twins[0]*fps),int(ispike_time+trig_twins[1]*fps)]

                    ispike_trig_trace = yyy_bhv[ispike_trig_twins[0]:ispike_trig_twins[1]]

                    try:
                        alltraces_ispike[:,ispike] = ispike_trig_trace
                        # axs2[iplot,icluster].plot(xxx_forplot,ispike_trig_trace)
                    except:
                        continue

                mean_trig_trace = np.nanmean(alltraces_ispike,axis=1)
                std_trig_trace = np.nanstd(alltraces_ispike,axis=1)
                sem_trig_trace = np.nanstd(alltraces_ispike,axis=1)/np.sqrt(np.shape(alltraces_ispike)[1])
                itv95_trig_trace = 1.96*sem_trig_trace

                # axs2[iplot,icluster].plot(xxx_forplot,mean_trig_trace)
                axs2[iplot,icluster].errorbar(xxx_forplot,mean_trig_trace,yerr=itv95_trig_trace,color='#212121',ecolor=clrs_plot[iplot])
                axs2[iplot,icluster].plot([0,0],[np.nanmin(mean_trig_trace-itv95_trig_trace),np.nanmax(mean_trig_trace+itv95_trig_trace)],'--k')

                if icluster == 0:
                    axs2[iplot,icluster].set_ylabel(con_vars_plot[iplot]+"\n"+yaxis_labels[iplot])
                # else:
                #     axs2[iplot,icluster].set_yticklabels([])

                if iplot == nconvarplots-1:
                    axs2[iplot,icluster].set_xlabel('time (s)')
                    axs2[iplot,icluster].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
                    axs2[iplot,icluster].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
                else:
                    axs2[iplot,icluster].set_xticklabels([])

                if iplot == 0:
                    axs2[iplot,icluster].set_title('cluster#'+str(iclusterID)+' channel#'+str(spike_channel))


                # put the data in the summarizing dataset
                if ianimal == 0:
                    spike_trig_average_all[animal1][con_vars_plot[iplot]][str(iclusterID)] = {'ch':spike_channel,'st_average':mean_trig_trace}
                elif ianimal == 1:
                    spike_trig_average_all[animal2][con_vars_plot[iplot]][str(iclusterID)] = {'ch':spike_channel,'st_average':mean_trig_trace}

        #
        if savefig:
            if ianimal == 0:
                fig2.savefig(save_path+"/"+date_tgt+'_'+animal1+"_spike_triggered_bhv_average.pdf")
            elif ianimal == 1:
                fig2.savefig(save_path+"/"+date_tgt+'_'+animal2+"_spike_triggered_bhv_average.pdf")









                
    return spike_trig_average_all

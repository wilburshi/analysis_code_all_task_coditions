#  function - plot_gaze_triggered_continuous_bhv_var

def plot_gaze_triggered_continuous_bhv_var(date_tgt, savefig, animal1, animal2, session_start_time, succpulls_ornot, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, animalnames_videotrack,  output_look_ornot, output_allvectors, output_allangles, output_key_locations, doGazeStart):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    

    fps = 30 

    gausKernelsize = 15

    gaze_thresold = 0.25 # min length threshold to define if a gaze is real gaze or noise, in the unit of second 

    nanimals = np.shape(animalnames_videotrack)[0]

    con_vars_plot_all = ['gaze_other_angle','gaze_tube_angle','gaze_lever_angle',
                         'animal_animal_dist','animal_tube_dist','animal_lever_dist',
                         'othergaze_self_angle','mass_move_speed','gaze_angle_speed',
                         'otherani_otherlever_dist','leverpull_prob','otherpull_prob']

    # con_vars_plot = [
    #                  'animal_animal_dist','animal_lever_dist','otherani_otherlever_dist',
    # ]
    con_vars_plot = con_vars_plot_all

    nconvarplots = np.shape(con_vars_plot)[0]

    clrs_plot = ['r','y','g',
                 'b','c','m',
                 '#458B74','#FFC710','#FF1493',
                 '#7c7c7c','#A9A9A9','#8B4513',
                ]

    yaxis_labels = ['degree','degree','degree',
                    'dist(a.u.)','dist(a.u.)','dist(a.u.)',
                    'degree','pixel/s','degree/s',
                    'dust(a.u.)','','']


    gaze_trig_events_summary = {}

    # get the discrete behavioral events
    # aligned to the start of the video recording
    time_point_pull1_frames = (np.array(time_point_pull1)+session_start_time)*fps 
    time_point_pull2_frames = (np.array(time_point_pull2)+session_start_time)*fps 

    
    if doGazeStart:
        # define gaze as the flash gaze and gaze start
        # get the gaze start and stop
        #animal1_gaze = np.concatenate([oneway_gaze1, mutual_gaze1])
        animal1_gaze = np.sort(np.concatenate((oneway_gaze1,mutual_gaze1)))
        if len(animal1_gaze) == 0:
            animal1_gaze_flash = np.array([]) 
            animal1_gaze_start = np.array([])
            animal1_gaze_stop = np.array([])
        else:
            animal1_gaze_stop = animal1_gaze[np.concatenate(((animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
            animal1_gaze_start = np.concatenate(([animal1_gaze[0]],animal1_gaze[np.where(animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)[0]+1]))
            animal1_gaze_flash = np.intersect1d(animal1_gaze_start, animal1_gaze_stop)
            animal1_gaze_start = animal1_gaze_start[~np.isin(animal1_gaze_start,animal1_gaze_flash)]
            animal1_gaze_stop = animal1_gaze_stop[~np.isin(animal1_gaze_stop,animal1_gaze_flash)]
        #
        #animal2_gaze = np.concatenate([oneway_gaze2, mutual_gaze2])
        animal2_gaze = np.sort(np.concatenate((oneway_gaze2,mutual_gaze2)))
        if len(animal2_gaze) == 0:
            animal2_gaze_flash = np.array([]) 
            animal2_gaze_start = np.array([])
            animal2_gaze_stop = np.array([])
        else:
            animal2_gaze_stop = animal2_gaze[np.concatenate(((animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
            animal2_gaze_start = np.concatenate(([animal2_gaze[0]],animal2_gaze[np.where(animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)[0]+1]))
            animal2_gaze_flash = np.intersect1d(animal2_gaze_start, animal2_gaze_stop)
            animal2_gaze_start = animal2_gaze_start[~np.isin(animal2_gaze_start,animal2_gaze_flash)]
            animal2_gaze_stop = animal2_gaze_stop[~np.isin(animal2_gaze_stop,animal2_gaze_flash)] 
        #
        oneway_gaze1_frames = (np.sort(np.concatenate((animal1_gaze_start,animal1_gaze_flash)))+session_start_time)*fps 
        oneway_gaze2_frames = (np.sort(np.concatenate((animal2_gaze_start,animal2_gaze_flash)))+session_start_time)*fps 
    
    else:
        # define gazes are all gaze entry
        #
        oneway_gaze1_frames = (np.sort(np.concatenate((oneway_gaze1,mutual_gaze1)))+session_start_time)*fps 
        oneway_gaze2_frames = (np.sort(np.concatenate((oneway_gaze2,mutual_gaze2)))+session_start_time)*fps 

     
    # plot the pull triggered continuous variables
    trig_twin = [-6,6] # time window to plot the pull triggered events; unit: s
    fig2, axs2 = plt.subplots(nconvarplots,nanimals)
    fig2.set_figheight(2*nconvarplots)
    fig2.set_figwidth(6*nanimals)
    

    #
    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]
                      

        # get the variables

        gaze_other_angle = output_allangles['face_eye_angle_all_Anipose'][animal_name]
        gaze_other_angle = scipy.ndimage.gaussian_filter1d(gaze_other_angle,gausKernelsize)  # smooth the curve, use 30 before, change to 3 
        gaze_tube_angle = output_allangles['selftube_eye_angle_all_Anipose'][animal_name]
        gaze_tube_angle = scipy.ndimage.gaussian_filter1d(gaze_tube_angle,gausKernelsize)  
        gaze_lever_angle = output_allangles['selflever_eye_angle_all_Anipose'][animal_name]
        gaze_lever_angle = scipy.ndimage.gaussian_filter1d(gaze_lever_angle,gausKernelsize)  

        othergaze_self_angle = output_allangles['face_eye_angle_all_Anipose'][animal_name_other]
        othergaze_self_angle = scipy.ndimage.gaussian_filter1d(othergaze_self_angle,gausKernelsize)  


        a = output_key_locations['facemass_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['facemass_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_animal_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_animal_dist = scipy.ndimage.gaussian_filter1d(animal_animal_dist,gausKernelsize)  

        a = output_key_locations['tube_loc_all_Anipose'][animal_name].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_tube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_tube_dist = scipy.ndimage.gaussian_filter1d(animal_tube_dist,gausKernelsize)  

        a = output_key_locations['lever_loc_all_Anipose'][animal_name].transpose()
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

        a = output_key_locations['lever_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name_other].transpose()
        a_min_b = a - b
        otherani_otherlever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        otherani_otherlever_dist = scipy.ndimage.gaussian_filter1d(otherani_otherlever_dist,gausKernelsize)
        
        #
        # get the pull time series
        # align to the start of the video recording
        # self pull
        if ianimal == 0:
            timepoint_pull = time_point_pull1_frames/fps
        elif ianimal == 1:
            timepoint_pull = time_point_pull2_frames/fps
        #
        try:
            timeseries_pull = np.zeros(np.shape(gaze_angle_speed))
            timeseries_pull[list(map(int,list(np.round((timepoint_pull))*fps)))]=1
        except: # some videos are shorter than the task 
            timeseries_pull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_pull*fps))))+1,))
            timeseries_pull[list(map(int,list(np.round(timepoint_pull*fps))))]=1
        leverpull_prob = scipy.ndimage.gaussian_filter1d(timeseries_pull,gausKernelsize)
        
        # other pull
        if ianimal == 0:
            timepoint_pull = time_point_pull2_frames/fps
        elif ianimal == 1:
            timepoint_pull = time_point_pull1_frames/fps
        #
        try:
            timeseries_pull = np.zeros(np.shape(gaze_angle_speed))
            timeseries_pull[list(map(int,list(np.round((timepoint_pull))*fps)))]=1
        except: # some videos are shorter than the task 
            timeseries_pull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_pull*fps))))+1,))
            timeseries_pull[list(map(int,list(np.round(timepoint_pull*fps))))]=1
        otherpull_prob = scipy.ndimage.gaussian_filter1d(timeseries_pull,gausKernelsize)
        
        # put all the data together in the same order as the con_vars_plot
        data_summary = [gaze_other_angle, gaze_tube_angle, gaze_lever_angle,
                        animal_animal_dist, animal_tube_dist, animal_lever_dist,
                        othergaze_self_angle, mass_move_speed, gaze_angle_speed,
                        otherani_otherlever_dist,leverpull_prob,otherpull_prob,
                       ]

        for iplot in np.arange(0,nconvarplots,1):
            

            # plot the gaze aligned
            if ianimal == 0:
                timepoint_gaze = oneway_gaze1_frames/fps - session_start_time
            elif ianimal == 1:
                timepoint_gaze = oneway_gaze2_frames/fps - session_start_time
            #
            ngazes = np.shape(timepoint_gaze)[0]

            gaze_trigevent_data = []

           
            for igaze in np.arange(0,ngazes,1):

                timestemp_igaze = (np.array(timepoint_gaze)[igaze]+session_start_time)
                # yrange = [np.floor(np.nanmin(data_summary[iplot])),np.ceil(np.nanmax(data_summary[iplot]))]
                yrange = [np.floor(np.nanmin(data_summary[iplot])),(np.nanmax(data_summary[iplot]))*1.1]

                # plot pull triggered events
                frame_win_igaze = (timestemp_igaze+trig_twin)*fps
                xxx_trigevent = np.arange(trig_twin[0],trig_twin[1],1/fps) 

                try:
                    axs2[iplot,ianimal].plot(xxx_trigevent,data_summary[iplot][int(frame_win_igaze[0]):int(frame_win_igaze[1])],'-',color = clrs_plot[iplot])
                    #
                    gaze_trigevent_data.append(data_summary[iplot][int(frame_win_igaze[0]):int(frame_win_igaze[1])])
                except:
                    gaze_trigevent_data.append(np.full((1,np.shape(xxx_trigevent)[0]),np.nan)[0])


            # save data into the summary data set
            if ianimal == 0:
                gaze_trig_events_summary[(animal1,con_vars_plot[iplot])] = gaze_trigevent_data
            elif ianimal == 1:
                gaze_trig_events_summary[(animal2,con_vars_plot[iplot])] = gaze_trigevent_data


            # plot settings for axs2        
            try: 
                axs2[iplot,ianimal].plot([0,0],yrange,'k-')
                axs2[iplot,ianimal].plot(xxx_trigevent,np.nanmean(gaze_trigevent_data,axis=0),'k')
            except:
                axs2[iplot,ianimal].plot(xxx_trigevent,gaze_trigevent_data,'k')
            #
            axs2[iplot,ianimal].set_xlim(trig_twin)
            axs2[iplot,ianimal].set_xlabel('')
            axs2[iplot,ianimal].set_xticklabels('')
            axs2[iplot,ianimal].set_ylabel(yaxis_labels[iplot])
            if ianimal == 0:
                axs2[iplot,ianimal].set_title('all gazes '+animal1+' '+con_vars_plot[iplot])
            elif ianimal == 1:
                axs2[iplot,ianimal].set_title('all gazes '+animal2+' '+con_vars_plot[iplot])
            #
            if iplot == nconvarplots-1:
                axs2[iplot,ianimal].set_xlabel('time(s)', fontsize = 14)
                axs2[iplot,ianimal].set_xticks(np.linspace(trig_twin[0],trig_twin[1],5)) 
                axs2[iplot,ianimal].set_xticklabels(list(map(str,np.linspace(trig_twin[0],trig_twin[1],5))))
                
            
    if savefig:

        fig2.savefig(date_tgt+'_gaze_triggered_bhv_variables.pdf')
            

    return gaze_trig_events_summary

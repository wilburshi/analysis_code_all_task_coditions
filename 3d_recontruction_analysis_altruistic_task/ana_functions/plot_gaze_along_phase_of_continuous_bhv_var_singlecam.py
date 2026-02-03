#  function - plot behavioral events; save the gize distribution along the phase of continuous bhv variables

def plot_gaze_along_phase_of_continuous_bhv_var_singlecam(fig_savepath, savefig, animal1, animal2, session_start_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, animalnames_videotrack, output_look_ornot, output_allvectors, output_allangles, output_key_locations, doActvePeri, doGazeStart):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle   
    import seaborn

    fps = 30 

    gausKernelsize = 9

    gaze_thresold = 0.25 # min length threshold to define if a gaze is real gaze or noise, in the unit of second 

    if doActvePeri:
        activeTwin = 5 # the active time window around animal1 or animal2's pulls, in the unit of second


    nanimals = np.shape(animalnames_videotrack)[0]

    con_vars_plot_all = ['gaze_other_angle','gaze_tube_angle','gaze_lever_angle',
                         'animal_animal_dist','animal_tube_dist','animal_lever_dist',
                         'othergaze_self_angle','mass_move_speed','gaze_angle_speed',
                         'otherani_otherlever_dist']

    con_vars_plot = [
                     'animal_animal_dist','animal_lever_dist','otherani_otherlever_dist',
                    ]


    nconvarplots = np.shape(con_vars_plot)[0]

    clrs_plot = ['r','y','g',
                 'b','c','m',
                 '#458B74','#FFC710','#FF1493',
                 '#7c7c7c',
                ]

    yaxis_labels = ['degree','degree','degree',
                    'dist(a.u.)','dist(a.u.)','dist(a.u.)',
                    'degree','pixel/s','degree/s',
                    'dust(a.u.)']

    gazeDist_phaseof_contbhvvar_summary = {}

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



    if doActvePeri:
        # define active frames based on time point pull1 and pull2
        nallframes = np.shape(output_allangles['other_eye_angle_all_merge']['dodson'])[0]
        allframeIDs = np.ones((nallframes,))*np.nan

        for ipullframe in time_point_pull1_frames:
            ipull_actstartframe = np.round(ipullframe - activeTwin*fps)
            ipull_actendframe = np.round(ipullframe + activeTwin*fps)
            #
            ipull_actstartframe = ipull_actstartframe.astype(int)
            ipull_actendframe = ipull_actendframe.astype(int)
            #
            allframeIDs[ipull_actstartframe:ipull_actendframe] = 1

        for ipullframe in time_point_pull2_frames:
            ipull_actstartframe = np.round(ipullframe - activeTwin*fps)
            ipull_actendframe = np.round(ipullframe + activeTwin*fps)
            #
            ipull_actstartframe = ipull_actstartframe.astype(int)
            ipull_actendframe = ipull_actendframe.astype(int)
            #
            allframeIDs[ipull_actstartframe:ipull_actendframe] = 1

        activeframeIDs = allframeIDs == 1
        activeframeInds = np.where(activeframeIDs)[0]


    fig, axs = plt.subplots(nconvarplots,nanimals)
    fig.set_figheight(5*nconvarplots)
    fig.set_figwidth(5*nanimals)

    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
            timepoint_pull_frame_tgt = time_point_pull1_frames
            timepoint_gaze_frame_tgt = oneway_gaze1_frames
            #
            gazeDist_phaseof_contbhvvar_summary[animal1] = {}

        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]
            timepoint_pull_frame_tgt = time_point_pull2_frames
            timepoint_gaze_frame_tgt = oneway_gaze2_frames
            #
            gazeDist_phaseof_contbhvvar_summary[animal2] = {}


        # get the variables
        gaze_other_angle = output_allangles['other_eye_angle_all_merge'][animal_name]
        gaze_other_angle = scipy.ndimage.gaussian_filter1d(gaze_other_angle,gausKernelsize)  # smooth the curve, use 30 before, change to 3 

        gaze_tube_angle = output_allangles['tube_eye_angle_all_merge'][animal_name]
        gaze_tube_angle = scipy.ndimage.gaussian_filter1d(gaze_tube_angle,gausKernelsize)  

        gaze_lever_angle = output_allangles['lever_eye_angle_all_merge'][animal_name]
        gaze_lever_angle = scipy.ndimage.gaussian_filter1d(gaze_lever_angle,gausKernelsize)  

        othergaze_self_angle = output_allangles['other_eye_angle_all_merge'][animal_name_other]
        othergaze_self_angle = scipy.ndimage.gaussian_filter1d(othergaze_self_angle,gausKernelsize)  

        a = output_key_locations['facemass_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['facemass_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_animal_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_animal_dist = scipy.ndimage.gaussian_filter1d(animal_animal_dist,gausKernelsize)  

        a = output_key_locations['tube_loc_all_merge'][animal_name].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_tube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_tube_dist = scipy.ndimage.gaussian_filter1d(animal_tube_dist,gausKernelsize)  

        a = output_key_locations['lever_loc_all_merge'][animal_name].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_lever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_lever_dist = scipy.ndimage.gaussian_filter1d(animal_lever_dist,gausKernelsize)  

        a = output_key_locations['facemass_loc_all_merge'][animal_name].transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        mass_move_speed = scipy.ndimage.gaussian_filter1d(mass_move_speed,gausKernelsize)  

        a = np.array(output_allvectors['head_vect_all_merge'][animal_name]).transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
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

        a = output_key_locations['lever_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name_other].transpose()
        a_min_b = a - b
        otherani_otherlever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        otherani_otherlever_dist = scipy.ndimage.gaussian_filter1d(otherani_otherlever_dist,gausKernelsize)


        # put all the data together in the same order as the con_vars_plot_all and con_vars_plot
        data_summary_all = [gaze_other_angle, gaze_tube_angle, gaze_lever_angle,
                            animal_animal_dist, animal_tube_dist, animal_lever_dist,
                            othergaze_self_angle, mass_move_speed, gaze_angle_speed,
                            otherani_otherlever_dist,
                           ]
        #
        data_summary_forplot = [animal_animal_dist, animal_lever_dist, otherani_otherlever_dist,]

        #
        for iplot in np.arange(0,nconvarplots,1):

            yyy = data_summary_forplot[iplot]
            data_summary_forplot[iplot] = (yyy-np.nanmin(yyy))/(np.nanmax(yyy)-np.nanmin(yyy))

            contvar_tgt = yyy
            contvar_tgt_name = con_vars_plot[iplot]
            # 
            diff = np.insert(np.diff(contvar_tgt), 0, np.nan)
            # 
            frames = np.arange(0,np.shape(contvar_tgt)[0],1)
            # 
            df = pd.DataFrame({'frameID': frames, 'value': contvar_tgt,'diff': diff})
            df['phase'] = np.select(
            [
                df['diff'] > 0,   # Increasing condition
                df['diff'] < 0    # Decreasing condition
            ],
            [
                'increasing',     # Label for increasing
                'decreasing'      # Label for decreasing
            ],
            default='undefined'   # Label for undefined (including NaN or diff == 0)
            )

            # only analyze the "active" period
            if doActvePeri:
                df = df[np.isin(df['frameID'],activeframeInds)]

            #
            ind = np.isin(df['frameID'],timepoint_gaze_frame_tgt.astype(int))
            df_gaze = df[ind]
            #
            phase_order = ['increasing', 'decreasing', 'undefined']
            df_gaze['phase'] = pd.Categorical(df_gaze['phase'], categories=phase_order, ordered=True)
            df_gaze = df_gaze.sort_values('phase')

            #
            seaborn.countplot(ax=axs[iplot,ianimal],x='phase', data=df_gaze, palette='Set2')

            # Add labels and title
            axs[iplot,ianimal].set_xlabel('Phase of the '+ contvar_tgt_name)
            axs[iplot,ianimal].set_ylabel('Social gaze count')
            if ianimal == 0:
                axs[iplot,ianimal].set_title('gaze action animal '+animal1)
            else:
                axs[iplot,ianimal].set_title('gaze action animal '+animal2)



            # save data into the summary data set
            if ianimal == 0:
                gazeDist_phaseof_contbhvvar_summary[animal1][contvar_tgt_name] = df_gaze
            elif ianimal == 1:
                gazeDist_phaseof_contbhvvar_summary[animal2][contvar_tgt_name] = df_gaze

    if savefig:                    
        fig.savefig(fig_savepath+"_gazeDistribution_along_phase_of_continueous_varibles.pdf")



    return gazeDist_phaseof_contbhvvar_summary

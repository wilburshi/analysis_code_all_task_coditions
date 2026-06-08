# # function - run PCA around behavior events
def PCA_around_bhv_events_video(FR_timepoint_allch,FR_zscore_allch_np_merged,time_point_pull1,time_point_pull2,time_point_pulls_succfail,
                          oneway_gaze1,oneway_gaze2,mutual_gaze1,mutual_gaze2,gaze_thresold,totalsess_time_forFR,PCAtwins,fps,
                          data_saved_folder,cameraID,animal1_filename,animal2_filename,date_tgt):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.animation as animation
    import scipy.stats as st
    import os
    import cv2
    from sklearn.decomposition import PCA

    from ana_functions.confidence_ellipse import confidence_ellipse

    fps_FR = int(np.ceil(np.shape(FR_timepoint_allch)[0]/np.max(FR_timepoint_allch)))

    nframes = PCAtwins*fps_FR*2

    # Settings
    save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # video file path for saving
    video_file = save_path+"/"+date_tgt+"_neuralFR_PCAprojections_aligned_at_pull_video.mp4"
    clear_frames = True     # Should it clear the figure between each frame?   

    # Output video writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='PCA trejactory', artist='Matplotlib', comment='')
    writer = FFMpegWriter(fps=fps_FR, metadata=metadata)



    # # # # 
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
    time_point_pull1 = time_point_pull1[time_point_pull1<totalsess_time_forFR]
    time_point_pull2 = time_point_pull2[time_point_pull2<totalsess_time_forFR]
    oneway_gaze1 = oneway_gaze1[oneway_gaze1<totalsess_time_forFR]
    oneway_gaze2 = oneway_gaze2[oneway_gaze2<totalsess_time_forFR]
    animal1_gaze_start = animal1_gaze_start[animal1_gaze_start<totalsess_time_forFR]
    animal2_gaze_start = animal2_gaze_start[animal2_gaze_start<totalsess_time_forFR]
    animal1_gaze_stop = animal1_gaze_stop[animal1_gaze_stop<totalsess_time_forFR]
    animal2_gaze_stop = animal2_gaze_stop[animal2_gaze_stop<totalsess_time_forFR]
    #
    time_point_pull1_succ = time_point_pull1_succ[time_point_pull1_succ<totalsess_time_forFR]
    time_point_pull2_succ = time_point_pull2_succ[time_point_pull2_succ<totalsess_time_forFR]
    time_point_pull1_fail = time_point_pull1_fail[time_point_pull1_fail<totalsess_time_forFR]
    time_point_pull2_fail = time_point_pull2_fail[time_point_pull2_fail<totalsess_time_forFR]


    ncells = np.shape(FR_zscore_allch_np_merged)[0]

    # add nan batch to both sides
    FR_zscore_allch_leftbatch = np.hstack((np.full((ncells,PCAtwins*fps_FR),np.nan),FR_zscore_allch_np_merged))
    FR_zscore_allch_bothbatch = np.hstack((FR_zscore_allch_leftbatch,np.full((ncells,PCAtwins*fps_FR),np.nan)))
    # change the bhv event time point to reflect the batch on both side
    time_point_pull1_align = time_point_pull1 + PCAtwins
    time_point_pull2_align = time_point_pull2 + PCAtwins
    oneway_gaze1_align = oneway_gaze1 + PCAtwins
    oneway_gaze2_align = oneway_gaze2 + PCAtwins
    animal1_gaze_start_align = animal1_gaze_start + PCAtwins
    animal2_gaze_start_align = animal2_gaze_start + PCAtwins
    animal1_gaze_stop_align = animal1_gaze_stop + PCAtwins
    animal2_gaze_stop_align = animal2_gaze_stop + PCAtwins
    #
    time_point_pull1_succ_align = time_point_pull1_succ + PCAtwins
    time_point_pull2_succ_align = time_point_pull2_succ + PCAtwins
    time_point_pull1_fail_align = time_point_pull1_fail + PCAtwins
    time_point_pull2_fail_align = time_point_pull2_fail + PCAtwins


    # plot the figures
    # align to the bhv events
    bhv_events_anatypes = [# 'pull1','pull2',
                           'pull1_succ','pull1_fail',
                           'pull2_succ','pull2_fail',
                           # 'gaze1','gaze2',
                           # 'gaze1_start','gaze1_stop',
                           # 'gaze2_start','gaze2_stop',
                          ]
    bhv_events_names = [   # 'pull1','pull2',
                           'self successful pulls','self failed pulls',
                           'other successful pulls','other failed pulls',
                           # 'gaze1','gaze2',
                           # 'gaze1_start','gaze1_stop',
                           # 'gaze2_start','gaze2_stop',
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
    nanatypes = np.shape(bhv_events_anatypes)[0]

    # plot the trajectories, highlight the start, middle(event time), end
    fig = plt.figure(figsize = (3*8,nanatypes*8))
    axs = np.full((nanatypes,3),{})
    i = 0
    for iplot in np.arange(0,nanatypes,1):
        for jplot in np.arange(0,3,1):
            axs[iplot,jplot] = fig.add_subplot(nanatypes,3, i+1)
            i = i + 1
    fig.tight_layout()

    #
    PC1_lims = [-3,7]
    PC2_lims = [-4,3]
    PC3_lims = [-5,10]


    with writer.saving(fig, video_file, 100):
        for iframe in np.arange(0,nframes,1):    

            print("printing frame ",str(iframe+1),"/",str(nframes))

            if clear_frames:
                fig.clear()
                fig.subplots_adjust(wspace=0, hspace=0)
                #
                axs = np.full((nanatypes,3),{})
                i = 0
                for iplot in np.arange(0,nanatypes,1):
                    for jplot in np.arange(0,3,1):
                        axs[iplot,jplot] = fig.add_subplot(nanatypes,3, i+1)
                        i = i + 1
                fig.tight_layout()


            # test PCA and plot
            for ianatype in np.arange(0,nanatypes,1):

                bhvevent_anatype = bhv_events_anatypes[ianatype]
                bhv_event_name = bhv_events_names[ianatype]
                bhv_event_timepoint = timepoint_bhvevents[bhvevent_anatype]

                # concatenate around time_point_bhvevent
                nevents = np.shape(bhv_event_timepoint)[0]
                # initialize the data
                FR_aligned_concat_train = np.full((ncells,PCAtwins*2*fps_FR*nevents),np.nan) # -PCAtwins to PCAtwins around the events as one trial
                # FR_aligned_concat_test = dict.fromkeys({'0','1','2','3','4'},[])
                # ntests = 5
                FR_aligned_concat_test = dict.fromkeys(np.char.mod('%d', np.arange(0,nevents,1)),[])
                ntests = nevents
                # train the PCA
                for ievent in np.arange(0,nevents,1):
                    time_point_ievent = bhv_event_timepoint[ievent]
                    try:
                        FR_aligned_concat_train[:,PCAtwins*2*fps_FR*ievent:PCAtwins*2*fps_FR*(ievent+1)] = FR_zscore_allch_bothbatch[:,int((time_point_ievent-PCAtwins)*fps_FR):int((time_point_ievent+PCAtwins)*fps_FR)]
                    except:
                        continue
                #
                pca_eventtype = PCA(n_components=3)
                FR_aligned_concat_train = FR_aligned_concat_train[:,~np.isnan(np.sum(FR_aligned_concat_train,axis=0))]
                FR_zscore_allch_np_merged = FR_zscore_allch_np_merged[~np.isnan(np.sum(FR_zscore_allch_np_merged,axis=1)),:]
                # # PCA on the aligned concatanated data set 
                # pca_eventtype.fit(FR_aligned_concat_train.transpose())
                # PCA on the entire FR trace
                pca_eventtype.fit(FR_zscore_allch_np_merged.transpose())
                #
                # test the PCA
                #
                for ievent in np.arange(0,ntests,1):
                # for ievent in np.arange(1,50,3):
                    time_point_ievent = bhv_event_timepoint[ievent]
                    FR_test_ievent = FR_zscore_allch_bothbatch[:,int((time_point_ievent-PCAtwins)*fps_FR):int((time_point_ievent-PCAtwins)*fps_FR)+iframe+1]
                    #
                    try:
                        if ~np.isnan(np.sum(FR_test_ievent)):
                            FR_aligned_concat_test[str(ievent)] = pca_eventtype.transform(FR_test_ievent.transpose())
                            #
                            # plot the 3D trajectories
                            xline = FR_aligned_concat_test[str(ievent)][:,0]
                            yline = FR_aligned_concat_test[str(ievent)][:,1]
                            zline = FR_aligned_concat_test[str(ievent)][:,2]
                            #
                            # axs[ianatype,0].plot(xline[0:PCAtwins*fps_FR+1], yline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                            axs[ianatype,0].plot(xline, yline,color=[0.7,0.7,0.7],alpha=0.2,linewidth=3)
                            axs[ianatype,0].plot(xline[iframe], yline[iframe],'ko',markersize = 15)
                            axs[ianatype,0].set_xlabel('PC1',fontsize=30)
                            axs[ianatype,0].set_ylabel('PC2',fontsize=30)
                            axs[ianatype,0].set_xlim(PC1_lims)
                            axs[ianatype,0].set_ylim(PC2_lims)
                            axs[ianatype,0].set_xticks([])
                            axs[ianatype,0].set_yticks([])
                            axs[ianatype,0].spines['top'].set_visible(False)
                            axs[ianatype,0].spines['right'].set_visible(False)
                            axs[ianatype,0].set_title(bhv_event_name,fontsize=35)
                            #
                            # axs[ianatype,1].plot(xline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                            axs[ianatype,1].plot(xline, zline,color=[0.7,0.7,0.7],alpha=0.2,linewidth=3)
                            axs[ianatype,1].plot(xline[iframe], zline[iframe],'ko',markersize = 15)
                            axs[ianatype,1].set_xlabel('PC1',fontsize=30)
                            axs[ianatype,1].set_ylabel('PC3',fontsize=30)
                            axs[ianatype,1].set_xlim(PC1_lims)
                            axs[ianatype,1].set_ylim(PC3_lims)
                            axs[ianatype,1].set_xticks([])
                            axs[ianatype,1].set_yticks([])
                            axs[ianatype,1].spines['top'].set_visible(False)
                            axs[ianatype,1].spines['right'].set_visible(False)
                            axs[ianatype,1].set_title(bhv_event_name,fontsize=35)
                            #
                            axs[ianatype,2].plot(yline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                            axs[ianatype,2].plot(yline, zline,color=[0.7,0.7,0.7],alpha=0.2,linewidth=3)
                            axs[ianatype,2].plot(yline[iframe], zline[iframe],'ko',markersize = 15)
                            axs[ianatype,2].set_xlabel('PC2',fontsize=30)
                            axs[ianatype,2].set_ylabel('PC3',fontsize=30)
                            axs[ianatype,2].set_xlim(PC2_lims)
                            axs[ianatype,2].set_ylim(PC3_lims)
                            axs[ianatype,2].set_xticks([])
                            axs[ianatype,2].set_yticks([])
                            axs[ianatype,2].spines['top'].set_visible(False)
                            axs[ianatype,2].spines['right'].set_visible(False)
                            axs[ianatype,2].set_title(bhv_event_name,fontsize=35)

                            if ianatype == 0:
                                axs[ianatype,2].text(2,8,f"{iframe/fps_FR-PCAtwins:.2f}"+"s",fontsize=40)
                    except:
                        continue
                    
                fig.tight_layout()

            # fig.tight_layout()
            writer.grab_frame()         



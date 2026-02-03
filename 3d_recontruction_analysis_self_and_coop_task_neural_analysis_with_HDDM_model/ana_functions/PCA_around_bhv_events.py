# # function - run PCA around behavior events
def PCA_around_bhv_events(FR_timepoint_allch,FR_zscore_allch_np_merged,time_point_pull1,time_point_pull2,time_point_pulls_succfail,
                          oneway_gaze1,oneway_gaze2,mutual_gaze1,mutual_gaze2,gaze_thresold,totalsess_time_forFR,PCAtwins,fps,
                          savefigs,data_saved_folder,cameraID,animal1_filename,animal2_filename,date_tgt):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import scipy.stats as st
    import os
    from sklearn.decomposition import PCA

    from ana_functions.confidence_ellipse import confidence_ellipse


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
    bhv_events_anatypes = ['pull1','pull2',
                           'pull1_succ','pull1_fail',
                           'pull2_succ','pull2_fail',
                           # 'gaze1','gaze2',
                           'gaze1_start','gaze1_stop',
                           'gaze2_start','gaze2_stop',
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

    # test PCA and plot
    for ianatype in np.arange(0,nanatypes,1):

        try:
            bhvevent_anatype = bhv_events_anatypes[ianatype]
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
            # plot the trajectories, highlight the start, middle(event time), end
            fig = plt.figure(figsize=(8*4,8*1))
            ax1 = fig.add_subplot(1, 4, 1, projection='3d')
            ax2 = fig.add_subplot(1, 4, 2)
            ax3 = fig.add_subplot(1, 4, 3)
            ax4 = fig.add_subplot(1, 4, 4)
            #
            PC1_threepoints = np.full((ntests, 3), np.nan)
            PC2_threepoints = np.full((ntests, 3), np.nan)
            PC3_threepoints = np.full((ntests, 3), np.nan)
            #
            PC1_lims = [-3,7]
            PC2_lims = [-4,3]
            PC3_lims = [-5,10]
            #
            for ievent in np.arange(0,ntests,1):
            # for ievent in np.arange(1,50,3):
                time_point_ievent = bhv_event_timepoint[ievent]
                FR_test_ievent = FR_zscore_allch_bothbatch[:,int((time_point_ievent-PCAtwins)*fps_FR):int((time_point_ievent+PCAtwins)*fps_FR)]
                #
                try:
                    if ~np.isnan(np.sum(FR_test_ievent)):
                        FR_aligned_concat_test[str(ievent)] = pca_eventtype.transform(FR_test_ievent.transpose())
                        #
                        # plot the 3D trajectories
                        xline = FR_aligned_concat_test[str(ievent)][:,0]
                        yline = FR_aligned_concat_test[str(ievent)][:,1]
                        zline = FR_aligned_concat_test[str(ievent)][:,2]
                        # ax.plot3D(xline, yline, zline)
                        ax1.plot3D(xline[0:PCAtwins*fps_FR+1], yline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax1.plot3D(xline[0], yline[0], zline[0],'ko')
                        ax1.plot3D(xline[PCAtwins*fps_FR], yline[PCAtwins*fps_FR], zline[PCAtwins*fps_FR],'go')
                        ax1.plot3D(xline[-1], yline[-1], zline[-1],'ro')
                        ax1.set_xlabel('PC1',fontsize=15)
                        ax1.set_ylabel('PC2',fontsize=15)
                        ax1.set_zlabel('PC3',fontsize=15)
                        ax1.set_xlim(PC1_lims)
                        ax1.set_ylim(PC2_lims)
                        ax1.set_zlim(PC3_lims)
                        ax1.set_title(bhvevent_anatype,fontsize=20)
                        #
                        ax2.plot(xline[0:PCAtwins*fps_FR+1], yline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax2.plot(xline[0], yline[0],'ko')
                        ax2.plot(xline[PCAtwins*fps_FR], yline[PCAtwins*fps_FR],'go')
                        ax2.plot(xline[-1], yline[-1],'ro')
                        ax2.set_xlabel('PC1',fontsize=15)
                        ax2.set_ylabel('PC2',fontsize=15)
                        ax2.set_xlim(PC1_lims)
                        ax2.set_ylim(PC2_lims)
                        ax2.set_title(bhvevent_anatype,fontsize=20)
                        #
                        ax3.plot(xline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax3.plot(xline[0], zline[0],'ko')
                        ax3.plot(xline[PCAtwins*fps_FR], zline[PCAtwins*fps_FR],'go')
                        ax3.plot(yline[-1], zline[-1],'ro')
                        ax3.set_xlabel('PC1',fontsize=15)
                        ax3.set_ylabel('PC3',fontsize=15)
                        ax3.set_xlim(PC1_lims)
                        ax3.set_ylim(PC3_lims)
                        ax3.set_title(bhvevent_anatype,fontsize=20)
                        #
                        ax4.plot(yline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax4.plot(yline[0], zline[0],'ko')
                        ax4.plot(yline[PCAtwins*fps_FR], zline[PCAtwins*fps_FR],'go')
                        ax4.plot(yline[-1], zline[-1],'ro')
                        ax4.set_xlabel('PC2',fontsize=15)
                        ax4.set_ylabel('PC3',fontsize=15)
                        ax4.set_xlim(PC2_lims)
                        ax4.set_ylim(PC3_lims)
                        ax4.set_title(bhvevent_anatype,fontsize=20)
                        #
                        PC1_threepoints[ievent,:] = [xline[0],xline[PCAtwins*fps_FR],xline[-1]]
                        PC2_threepoints[ievent,:] = [yline[0],yline[PCAtwins*fps_FR],yline[-1]]
                        PC3_threepoints[ievent,:] = [zline[0],zline[PCAtwins*fps_FR],zline[-1]]

                except:
                    continue

            fig.tight_layout()

            if savefigs:
                save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(save_path+"/"+date_tgt+"_neuralFR_PCAprojections_aligned_at_"+bhvevent_anatype+".pdf")


            # plot the confidence ellipse
            # plot the trajectories, highlight the start, middle(event time), end
            FR_aligned_concat_test_list = list(FR_aligned_concat_test.values())
            non_empty_arrays = [arr for arr in FR_aligned_concat_test_list if np.shape(arr)[0] == PCAtwins*2*fps_FR]
            FR_aligned_concat_test_ar = np.array(non_empty_arrays)
            FR_aligned_concat_test_mean = np.nanmean(FR_aligned_concat_test_ar,axis=0)

            fig = plt.figure(figsize=(8*3,8*1))
            ax1 = fig.add_subplot(1, 3, 1,)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)
            #
            ind_good=~np.isnan(np.sum(PC1_threepoints,axis=1)+np.sum(PC2_threepoints,axis=1)+np.sum(PC3_threepoints,axis=1))
            PC1_threepoints = PC1_threepoints[ind_good,:]
            PC2_threepoints = PC2_threepoints[ind_good,:]
            PC3_threepoints = PC3_threepoints[ind_good,:]
            #
            ax1.plot(FR_aligned_concat_test_mean[:,0],FR_aligned_concat_test_mean[:,1],'-',color=[0.5,0.5,0.5])
            confidence_ellipse(PC1_threepoints[:,0], PC2_threepoints[:,0], ax1, n_std=1.0, facecolor='none',edgecolor='k')
            confidence_ellipse(PC1_threepoints[:,1], PC2_threepoints[:,1], ax1, n_std=1.0, facecolor='none',edgecolor='g')
            confidence_ellipse(PC1_threepoints[:,2], PC2_threepoints[:,2], ax1, n_std=1.0, facecolor='none',edgecolor='r')
            ax1.plot(PC1_threepoints[:,0], PC2_threepoints[:,0],'ko',label=str(PCAtwins)+'s before event')
            ax1.plot(PC1_threepoints[:,1], PC2_threepoints[:,1],'go',label='at the event')
            ax1.plot(PC1_threepoints[:,2], PC2_threepoints[:,2],'ro',label=str(PCAtwins)+'s after event')
            ax1.set_xlabel('PC1',fontsize=15)
            ax1.set_ylabel('PC2',fontsize=15)
            # ax1.set_xlim(np.floor(np.nanmin(PC1_threepoints)),np.ceil(np.nanmax(PC1_threepoints)))
            # ax1.set_ylim(np.floor(np.nanmin(PC2_threepoints)),np.ceil(np.nanmax(PC2_threepoints)))
            ax1.set_xlim(PC1_lims)
            ax1.set_ylim(PC2_lims)
            ax1.legend()
            ax1.set_title(bhvevent_anatype,fontsize=20)
            #
            ax2.plot(FR_aligned_concat_test_mean[:,0],FR_aligned_concat_test_mean[:,2],'-',color=[0.5,0.5,0.5])
            confidence_ellipse(PC1_threepoints[:,0], PC3_threepoints[:,0], ax2, n_std=1.0, facecolor='none',edgecolor='k')
            confidence_ellipse(PC1_threepoints[:,1], PC3_threepoints[:,1], ax2, n_std=1.0, facecolor='none',edgecolor='g')
            confidence_ellipse(PC1_threepoints[:,2], PC3_threepoints[:,2], ax2, n_std=1.0, facecolor='none',edgecolor='r')
            ax2.plot(PC1_threepoints[:,0], PC3_threepoints[:,0],'ko')
            ax2.plot(PC1_threepoints[:,1], PC3_threepoints[:,1],'go')
            ax2.plot(PC1_threepoints[:,2], PC3_threepoints[:,2],'ro')
            ax2.set_xlabel('PC1',fontsize=15)
            ax2.set_ylabel('PC3',fontsize=15)
            ax2.set_xlim(PC1_lims)
            ax2.set_ylim(PC3_lims)
            # ax2.set_xlim(np.floor(np.nanmin(PC1_threepoints)),np.ceil(np.nanmax(PC1_threepoints)))
            # ax2.set_ylim(np.floor(np.nanmin(PC3_threepoints)),np.ceil(np.nanmax(PC3_threepoints)))
            ax2.set_title(bhvevent_anatype,fontsize=20)
            #
            ax3.plot(FR_aligned_concat_test_mean[:,1],FR_aligned_concat_test_mean[:,2],'-',color=[0.5,0.5,0.5])
            confidence_ellipse(PC2_threepoints[:,0], PC3_threepoints[:,0], ax3, n_std=1.0, facecolor='none',edgecolor='k')
            confidence_ellipse(PC2_threepoints[:,1], PC3_threepoints[:,1], ax3, n_std=1.0, facecolor='none',edgecolor='g')
            confidence_ellipse(PC2_threepoints[:,2], PC3_threepoints[:,2], ax3, n_std=1.0, facecolor='none',edgecolor='r')
            ax3.plot(PC2_threepoints[:,0], PC3_threepoints[:,0],'ko')
            ax3.plot(PC2_threepoints[:,1], PC3_threepoints[:,1],'go')
            ax3.plot(PC2_threepoints[:,2], PC3_threepoints[:,2],'ro')
            ax3.set_xlabel('PC2',fontsize=15)
            ax3.set_ylabel('PC3',fontsize=15)
            ax3.set_xlim(PC2_lims)
            ax3.set_ylim(PC3_lims)
            # ax3.set_xlim(np.floor(np.nanmin(PC2_threepoints)),np.ceil(np.nanmax(PC2_threepoints)))
            # ax3.set_ylim(np.floor(np.nanmin(PC3_threepoints)),np.ceil(np.nanmax(PC3_threepoints)))
            ax3.set_title(bhvevent_anatype,fontsize=20)
            #
            fig.tight_layout()
            #
            if savefigs:
                save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(save_path+"/"+date_tgt+"_neuralFR_PCAprojections_aligned_at_"+bhvevent_anatype+"_confEllipse.pdf")

        except:
            continue

    # # # 
    # plot the figures - plot the successful and failed pulls
    # align to the bhv events
    bhv_events_anatypes = ['pull1','pull2',
                          ]
    timepoint_bhvevents = {'pull1':time_point_pull1_align,
                           'pull2':time_point_pull2_align,
                           'pull1_succ':time_point_pull1_succ_align,
                           'pull2_succ':time_point_pull2_succ_align,
                           'pull1_fail':time_point_pull1_fail_align,
                           'pull2_fail':time_point_pull2_fail_align,
                          }
    nanatypes = np.shape(bhv_events_anatypes)[0]

    # plot
    for ianatype in np.arange(0,nanatypes,1):

        fig = plt.figure(figsize=(8*4,8*1))
        ax1 = fig.add_subplot(1, 4, 1, projection='3d')
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)

        try: 
            # successful pulls
            # 
            bhvevent_anatype = bhv_events_anatypes[ianatype]+"_succ"
            bhv_event_timepoint = timepoint_bhvevents[bhvevent_anatype]
            #
            # concatenate around time_point_bhvevent
            nevents = np.shape(bhv_event_timepoint)[0]
            # initialize the data
            FR_aligned_concat_train = np.full((ncells,PCAtwins*2*fps_FR*nevents),np.nan) # -PCAtwins to PCAtwins around the events as one trial
            # FR_aligned_concat_test = dict.fromkeys({'0','1','2','3','4'},[])
            # ntests = 5
            FR_aligned_concat_test = dict.fromkeys(np.char.mod('%d', np.arange(0,nevents,1)),[])
            ntests = nevents
            #
            PC1_succ_threepoints = np.full((ntests, 3), np.nan)
            PC2_succ_threepoints = np.full((ntests, 3), np.nan)
            PC3_succ_threepoints = np.full((ntests, 3), np.nan)
            #
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
                FR_test_ievent = FR_zscore_allch_bothbatch[:,int((time_point_ievent-PCAtwins)*fps_FR):int((time_point_ievent+PCAtwins)*fps_FR)]
                #
                try:
                    if ~np.isnan(np.sum(FR_test_ievent)):
                        FR_aligned_concat_test[str(ievent)] = pca_eventtype.transform(FR_test_ievent.transpose())
                        #
                        # plot the 3D trajectories
                        xline = FR_aligned_concat_test[str(ievent)][:,0]
                        yline = FR_aligned_concat_test[str(ievent)][:,1]
                        zline = FR_aligned_concat_test[str(ievent)][:,2]
                        # ax.plot3D(xline, yline, zline)
                        # ax1.plot3D(xline[0:PCAtwins*fps_FR+1], yline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax1.plot3D(xline[0], yline[0], zline[0],'k^')
                        ax1.plot3D(xline[PCAtwins*fps_FR], yline[PCAtwins*fps_FR], zline[PCAtwins*fps_FR],'g^')
                        ax1.plot3D(xline[-1], yline[-1], zline[-1],'r^')
                        ax1.set_xlabel('PC1',fontsize=15)
                        ax1.set_ylabel('PC2',fontsize=15)
                        ax1.set_zlabel('PC3',fontsize=15)
                        ax1.set_xlim(PC1_lims)
                        ax1.set_ylim(PC2_lims)
                        ax1.set_zlim(PC3_lims)
                        ax1.set_title(bhv_events_anatypes[ianatype]+'_bothSuccAndFail',fontsize=20)
                        #
                        # ax2.plot(xline[0:PCAtwins*fps_FR+1], yline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax2.plot(xline[0], yline[0],'k^')
                        ax2.plot(xline[PCAtwins*fps_FR], yline[PCAtwins*fps_FR],'g^')
                        ax2.plot(xline[-1], yline[-1],'r^')
                        ax2.set_xlabel('PC1',fontsize=15)
                        ax2.set_ylabel('PC2',fontsize=15)
                        ax2.set_xlim(PC1_lims)
                        ax2.set_ylim(PC2_lims)
                        ax2.set_title(bhv_events_anatypes[ianatype]+'_bothSuccAndFail',fontsize=20)
                        #
                        # ax3.plot(xline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax3.plot(xline[0], zline[0],'k^')
                        ax3.plot(xline[PCAtwins*fps_FR], zline[PCAtwins*fps_FR],'g^')
                        ax3.plot(yline[-1], zline[-1],'r^')
                        ax3.set_xlabel('PC1',fontsize=15)
                        ax3.set_ylabel('PC3',fontsize=15)
                        ax3.set_xlim(PC1_lims)
                        ax3.set_ylim(PC3_lims)
                        ax3.set_title(bhv_events_anatypes[ianatype]+'_bothSuccAndFail',fontsize=20)
                        #
                        # ax4.plot(yline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax4.plot(yline[0], zline[0],'k^')
                        ax4.plot(yline[PCAtwins*fps_FR], zline[PCAtwins*fps_FR],'g^')
                        ax4.plot(yline[-1], zline[-1],'r^')
                        ax4.set_xlabel('PC2',fontsize=15)
                        ax4.set_ylabel('PC3',fontsize=15)
                        ax4.set_xlim(PC2_lims)
                        ax4.set_ylim(PC3_lims)
                        ax4.set_title(bhv_events_anatypes[ianatype]+'_bothSuccAndFail',fontsize=20)
                except:
                    continue
                #
                PC1_succ_threepoints[ievent,:] = [xline[0],xline[PCAtwins*fps_FR],xline[-1]]
                PC2_succ_threepoints[ievent,:] = [yline[0],yline[PCAtwins*fps_FR],yline[-1]]
                PC3_succ_threepoints[ievent,:] = [zline[0],zline[PCAtwins*fps_FR],zline[-1]]
            #
            FR_aligned_concat_test_list = list(FR_aligned_concat_test.values())
            non_empty_arrays = [arr for arr in FR_aligned_concat_test_list if np.shape(arr)[0] == PCAtwins*2*fps_FR]
            FR_aligned_concat_test_ar = np.array(non_empty_arrays)
            FR_aligned_concat_test_succ_mean = np.nanmean(FR_aligned_concat_test_ar,axis=0)

            # failed pulls
            # 
            bhvevent_anatype = bhv_events_anatypes[ianatype]+"_fail"
            bhv_event_timepoint = timepoint_bhvevents[bhvevent_anatype]
            #
            # concatenate around time_point_bhvevent
            nevents = np.shape(bhv_event_timepoint)[0]
            # initialize the data
            FR_aligned_concat_train = np.full((ncells,PCAtwins*2*fps_FR*nevents),np.nan) # -PCAtwins to PCAtwins around the events as one trial
            # FR_aligned_concat_test = dict.fromkeys({'0','1','2','3','4'},[])
            # ntests = 5
            FR_aligned_concat_test = dict.fromkeys(np.char.mod('%d', np.arange(0,nevents,1)),[])
            ntests = nevents
            #
            PC1_fail_threepoints = np.full((ntests, 3), np.nan)
            PC2_fail_threepoints = np.full((ntests, 3), np.nan)
            PC3_fail_threepoints = np.full((ntests, 3), np.nan)
            #
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
                FR_test_ievent = FR_zscore_allch_bothbatch[:,int((time_point_ievent-PCAtwins)*fps_FR):int((time_point_ievent+PCAtwins)*fps_FR)]
                #
                try:
                    if ~np.isnan(np.sum(FR_test_ievent)):
                        FR_aligned_concat_test[str(ievent)] = pca_eventtype.transform(FR_test_ievent.transpose())
                        #
                        # plot the 3D trajectories
                        xline = FR_aligned_concat_test[str(ievent)][:,0]
                        yline = FR_aligned_concat_test[str(ievent)][:,1]
                        zline = FR_aligned_concat_test[str(ievent)][:,2]
                        # ax.plot3D(xline, yline, zline)
                        # ax1.plot3D(xline[0:PCAtwins*fps_FR+1], yline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax1.plot3D(xline[0], yline[0], zline[0],'bv')
                        ax1.plot3D(xline[PCAtwins*fps_FR], yline[PCAtwins*fps_FR], zline[PCAtwins*fps_FR],'yv')
                        ax1.plot3D(xline[-1], yline[-1], zline[-1],'mv')
                        ax1.set_xlabel('PC1',fontsize=15)
                        ax1.set_ylabel('PC2',fontsize=15)
                        ax1.set_zlabel('PC3',fontsize=15)
                        ax1.set_xlim(PC1_lims)
                        ax1.set_ylim(PC2_lims)
                        ax1.set_zlim(PC3_lims)
                        ax1.set_title(bhv_events_anatypes[ianatype]+'_bothSuccAndFail',fontsize=20)
                        #
                        # ax2.plot(xline[0:PCAtwins*fps_FR+1], yline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax2.plot(xline[0], yline[0],'bv')
                        ax2.plot(xline[PCAtwins*fps_FR], yline[PCAtwins*fps_FR],'yv')
                        ax2.plot(xline[-1], yline[-1],'mv')
                        ax2.set_xlabel('PC1',fontsize=15)
                        ax2.set_ylabel('PC2',fontsize=15)
                        ax2.set_xlim(PC1_lims)
                        ax2.set_ylim(PC2_lims)
                        ax2.set_title(bhv_events_anatypes[ianatype]+'_bothSuccAndFail',fontsize=20)
                        #
                        # ax3.plot(xline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax3.plot(xline[0], zline[0],'bv')
                        ax3.plot(xline[PCAtwins*fps_FR], zline[PCAtwins*fps_FR],'yv')
                        ax3.plot(yline[-1], zline[-1],'mv')
                        ax3.set_xlabel('PC1',fontsize=15)
                        ax3.set_ylabel('PC3',fontsize=15)
                        ax3.set_xlim(PC1_lims)
                        ax3.set_ylim(PC3_lims)
                        ax3.set_title(bhv_events_anatypes[ianatype]+'_bothSuccAndFail',fontsize=20)
                        #
                        # ax4.plot(yline[0:PCAtwins*fps_FR+1], zline[0:PCAtwins*fps_FR+1],color=[0.8,0.8,0.8],alpha=0.2)
                        ax4.plot(yline[0], zline[0],'bv')
                        ax4.plot(yline[PCAtwins*fps_FR], zline[PCAtwins*fps_FR],'yv')
                        ax4.plot(yline[-1], zline[-1],'mv')
                        ax4.set_xlabel('PC2',fontsize=15)
                        ax4.set_ylabel('PC3',fontsize=15)
                        ax4.set_xlim(PC2_lims)
                        ax4.set_ylim(PC3_lims)
                        ax4.set_title(bhv_events_anatypes[ianatype]+'_bothSuccAndFail',fontsize=20)
                except:
                    continue
                #
                PC1_fail_threepoints[ievent,:] = [xline[0],xline[PCAtwins*fps_FR],xline[-1]]
                PC2_fail_threepoints[ievent,:] = [yline[0],yline[PCAtwins*fps_FR],yline[-1]]
                PC3_fail_threepoints[ievent,:] = [zline[0],zline[PCAtwins*fps_FR],zline[-1]]
            #  
            FR_aligned_concat_test_list = list(FR_aligned_concat_test.values())
            non_empty_arrays = [arr for arr in FR_aligned_concat_test_list if np.shape(arr)[0] == PCAtwins*2*fps_FR] 
            FR_aligned_concat_test_ar = np.array(non_empty_arrays)
            FR_aligned_concat_test_fail_mean = np.nanmean(FR_aligned_concat_test_ar,axis=0)


            fig.tight_layout()

            if savefigs:
                save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(save_path+"/"+date_tgt+"_neuralFR_PCAprojections_aligned_at_"+bhv_events_anatypes[ianatype]+"_bothsucc_and_fail.pdf")


            # plot the confidence ellipse
            # plot the trajectories, highlight the start, middle(event time), end
            fig = plt.figure(figsize=(8*3,8*1))
            ax1 = fig.add_subplot(1, 3, 1,)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)
            # 
            ax1.plot(FR_aligned_concat_test_succ_mean[:,0],FR_aligned_concat_test_succ_mean[:,1],'-',color=[0.8,0.8,0.8])
            confidence_ellipse(PC1_succ_threepoints[:,0], PC2_succ_threepoints[:,0], ax1, n_std=1.0, facecolor='none',edgecolor='k',label=str(PCAtwins)+'s before successful event')
            confidence_ellipse(PC1_succ_threepoints[:,1], PC2_succ_threepoints[:,1], ax1, n_std=1.0, facecolor='none',edgecolor='g',label='at the successful event')
            confidence_ellipse(PC1_succ_threepoints[:,2], PC2_succ_threepoints[:,2], ax1, n_std=1.0, facecolor='none',edgecolor='r',label=str(PCAtwins)+'s after successful event')
            ax1.plot(FR_aligned_concat_test_fail_mean[:,0],FR_aligned_concat_test_fail_mean[:,1],'-',color=[0.2,0.2,0.2])
            confidence_ellipse(PC1_fail_threepoints[:,0], PC2_fail_threepoints[:,0], ax1, n_std=1.0, facecolor='none',edgecolor='b',label=str(PCAtwins)+'s before failed event')
            confidence_ellipse(PC1_fail_threepoints[:,1], PC2_fail_threepoints[:,1], ax1, n_std=1.0, facecolor='none',edgecolor='y',label='at the failed event')
            confidence_ellipse(PC1_fail_threepoints[:,2], PC2_fail_threepoints[:,2], ax1, n_std=1.0, facecolor='none',edgecolor='m',label=str(PCAtwins)+'s after failed event')
            ax1.plot(PC1_succ_threepoints[:,0], PC2_succ_threepoints[:,0],'k^')
            ax1.plot(PC1_succ_threepoints[:,1], PC2_succ_threepoints[:,1],'g^')
            ax1.plot(PC1_succ_threepoints[:,2], PC2_succ_threepoints[:,2],'r^')
            ax1.plot(PC1_fail_threepoints[:,0], PC2_fail_threepoints[:,0],'bv')
            ax1.plot(PC1_fail_threepoints[:,1], PC2_fail_threepoints[:,1],'yv')
            ax1.plot(PC1_fail_threepoints[:,2], PC2_fail_threepoints[:,2],'mv')
            ax1.set_xlabel('PC1',fontsize=15)
            ax1.set_ylabel('PC2',fontsize=15)
            ax1.set_xlim(PC1_lims)
            ax1.set_ylim(PC2_lims)
            # ax1.set_xlim(np.floor(np.nanmin(PC1_succ_threepoints)),np.ceil(np.nanmax(PC1_succ_threepoints)))
            # ax1.set_ylim(np.floor(np.nanmin(PC2_succ_threepoints)),np.ceil(np.nanmax(PC2_succ_threepoints)))
            ax1.legend()
            ax1.set_title(bhv_events_anatypes[ianatype],fontsize=20)
            #
            ax2.plot(FR_aligned_concat_test_succ_mean[:,0],FR_aligned_concat_test_succ_mean[:,2],'-',color=[0.8,0.8,0.8])
            confidence_ellipse(PC1_succ_threepoints[:,0], PC3_succ_threepoints[:,0], ax2, n_std=1.0, facecolor='none',edgecolor='k')
            confidence_ellipse(PC1_succ_threepoints[:,1], PC3_succ_threepoints[:,1], ax2, n_std=1.0, facecolor='none',edgecolor='g')
            confidence_ellipse(PC1_succ_threepoints[:,2], PC3_succ_threepoints[:,2], ax2, n_std=1.0, facecolor='none',edgecolor='r')
            ax2.plot(FR_aligned_concat_test_fail_mean[:,0],FR_aligned_concat_test_fail_mean[:,2],'-',color=[0.2,0.2,0.2])
            confidence_ellipse(PC1_fail_threepoints[:,0], PC3_fail_threepoints[:,0], ax2, n_std=1.0, facecolor='none',edgecolor='b')
            confidence_ellipse(PC1_fail_threepoints[:,1], PC3_fail_threepoints[:,1], ax2, n_std=1.0, facecolor='none',edgecolor='y')
            confidence_ellipse(PC1_fail_threepoints[:,2], PC3_fail_threepoints[:,2], ax2, n_std=1.0, facecolor='none',edgecolor='m')
            ax2.plot(PC1_succ_threepoints[:,0], PC3_succ_threepoints[:,0],'k^')
            ax2.plot(PC1_succ_threepoints[:,1], PC3_succ_threepoints[:,1],'g^')
            ax2.plot(PC1_succ_threepoints[:,2], PC3_succ_threepoints[:,2],'r^')
            ax2.plot(PC1_fail_threepoints[:,0], PC3_fail_threepoints[:,0],'bv')
            ax2.plot(PC1_fail_threepoints[:,1], PC3_fail_threepoints[:,1],'yv')
            ax2.plot(PC1_fail_threepoints[:,2], PC3_fail_threepoints[:,2],'mv')
            ax2.set_xlabel('PC1',fontsize=15)
            ax2.set_ylabel('PC3',fontsize=15)
            ax2.set_xlim(PC1_lims)
            ax2.set_ylim(PC3_lims)
            # ax2.set_xlim(np.floor(np.nanmin(PC1_succ_threepoints)),np.ceil(np.nanmax(PC1_succ_threepoints)))
            # ax2.set_ylim(np.floor(np.nanmin(PC3_succ_threepoints)),np.ceil(np.nanmax(PC3_succ_threepoints)))
            ax2.set_title(bhv_events_anatypes[ianatype],fontsize=20)
            #
            ax3.plot(FR_aligned_concat_test_succ_mean[:,1],FR_aligned_concat_test_succ_mean[:,2],'-',color=[0.8,0.8,0.8])
            confidence_ellipse(PC2_succ_threepoints[:,0], PC3_succ_threepoints[:,0], ax3, n_std=1.0, facecolor='none',edgecolor='k')
            confidence_ellipse(PC2_succ_threepoints[:,1], PC3_succ_threepoints[:,1], ax3, n_std=1.0, facecolor='none',edgecolor='g')
            confidence_ellipse(PC2_succ_threepoints[:,2], PC3_succ_threepoints[:,2], ax3, n_std=1.0, facecolor='none',edgecolor='r')
            ax3.plot(FR_aligned_concat_test_fail_mean[:,1],FR_aligned_concat_test_fail_mean[:,2],'-',color=[0.2,0.2,0.2])
            confidence_ellipse(PC2_fail_threepoints[:,0], PC3_fail_threepoints[:,0], ax3, n_std=1.0, facecolor='none',edgecolor='b')
            confidence_ellipse(PC2_fail_threepoints[:,1], PC3_fail_threepoints[:,1], ax3, n_std=1.0, facecolor='none',edgecolor='y')
            confidence_ellipse(PC2_fail_threepoints[:,2], PC3_fail_threepoints[:,2], ax3, n_std=1.0, facecolor='none',edgecolor='m')
            ax3.plot(PC2_succ_threepoints[:,0], PC3_succ_threepoints[:,0],'k^')
            ax3.plot(PC2_succ_threepoints[:,1], PC3_succ_threepoints[:,1],'g^')
            ax3.plot(PC2_succ_threepoints[:,2], PC3_succ_threepoints[:,2],'r^')
            ax3.plot(PC2_fail_threepoints[:,0], PC3_fail_threepoints[:,0],'bv')
            ax3.plot(PC2_fail_threepoints[:,1], PC3_fail_threepoints[:,1],'yv')
            ax3.plot(PC2_fail_threepoints[:,2], PC3_fail_threepoints[:,2],'mv')
            ax3.set_xlabel('PC2',fontsize=15)
            ax3.set_ylabel('PC3',fontsize=15)
            ax3.set_xlim(PC2_lims)
            ax3.set_ylim(PC3_lims)
            # ax3.set_xlim(np.floor(np.nanmin(PC2_succ_threepoints)),np.ceil(np.nanmax(PC2_succ_threepoints)))
            # ax3.set_ylim(np.floor(np.nanmin(PC3_succ_threepoints)),np.ceil(np.nanmax(PC3_succ_threepoints)))
            ax3.set_title(bhv_events_anatypes[ianatype],fontsize=20)
            #
            fig.tight_layout()
            #
            if savefigs:
                save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(save_path+"/"+date_tgt+"_neuralFR_PCAprojections_aligned_at_"+bhv_events_anatypes[ianatype]+"_bothsucc_and_fail_confEllipse.pdf")

        except:
            continue


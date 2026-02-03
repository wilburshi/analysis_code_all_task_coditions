#  function - plot behavioral events; save the pull-triggered events

def plot_continuous_bhv_var_and_neuralFR(date_tgt,savefig,save_path, animal1, animal2, session_start_time, min_length, mintime_forplot, maxtime_forplot, succpulls_ornot, time_point_pull1, time_point_pull2, animalnames_videotrack, output_look_ornot, output_allvectors, output_allangles, output_key_locations, FR_timepoint_allch, FR_zscore_allch_PCs):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import scipy.stats as st
    import string
    import warnings
    import pickle    


    nanimals = np.shape(animalnames_videotrack)[0]

    gausKernelsize = 3

    fps = 30     

    con_vars_plot = ['gaze_other_angle','gaze_tube_angle','gaze_lever_angle','animal_animal_dist','animal_tube_dist','animal_lever_dist','othergaze_self_angle','mass_move_speed','gaze_angle_speed']
    nconvarplots = np.shape(con_vars_plot)[0]

    clrs_plot = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493']
    yaxis_labels = ['degree','degree','degree','dist(a.u.)','dist(a.u.)','dist(a.u.)','degree','pixel/s','degree/s']

    PCtraces = ['PC1','PC2','PC3']
    nPCtraces = np.shape(PCtraces)[0]

    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]

        # get the variables
        # align the tracking time to the session start
        xxx_time = np.arange(0,min_length,1)/fps - session_start_time

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


        # put all the data together in the same order as the con_vars_plot
        data_summary = [gaze_other_angle,gaze_tube_angle,gaze_lever_angle,animal_animal_dist,animal_tube_dist,animal_lever_dist,othergaze_self_angle,mass_move_speed,gaze_angle_speed]


        # plot the pull triggered PC1 2 3
        trig_twin = [-6,6] # time window to plot the pull triggered events; unit: s
        fig2, axs2 = plt.subplots(nPCtraces,1)
        fig2.set_figheight(2*nPCtraces)
        fig2.set_figwidth(6)

        # plot the pull triggered continuous variables, successful pulls
        fig3, axs3 = plt.subplots(nPCtraces,1)
        fig3.set_figheight(2*nPCtraces)
        fig3.set_figwidth(6)

        # plot the pull triggered continuous variables, failed pulls
        fig4, axs4 = plt.subplots(nPCtraces,1)
        fig4.set_figheight(2*nPCtraces)
        fig4.set_figwidth(6)

        #
        for iPCtrace in np.arange(0,nPCtraces,1):

            fig, axs = plt.subplots(nconvarplots,1)
            fig.set_figheight(2*nconvarplots)
            fig.set_figwidth(20)


            for iplot in np.arange(0,nconvarplots,1):

                #
                # plot the pull time stamp
                if ianimal == 0:
                    timepoint_pull = time_point_pull1
                    timepoint_pull_other = time_point_pull2
                elif ianimal == 1:
                    timepoint_pull = time_point_pull2
                    timepoint_pull_other = time_point_pull1

                npulls = np.shape(timepoint_pull)[0]
                #
                for ipull in np.arange(0,npulls,1):
                    timestemp_ipull = (np.array(timepoint_pull)[ipull])
                    yrange = [-0.1,1.1]
                    if ipull == npulls-1:
                        axs[iplot].plot([timestemp_ipull,timestemp_ipull],yrange,'-',color = '0.85',label='self pull')
                    else:
                        axs[iplot].plot([timestemp_ipull,timestemp_ipull],yrange,'-',color = '0.85')

                npulls = np.shape(timepoint_pull_other)[0]
                #
                for ipull in np.arange(0,npulls,1):
                    timestemp_ipull = (np.array(timepoint_pull_other)[ipull])
                    yrange = [-0.1,1.1]
                    if ipull == npulls-1:
                        axs[iplot].plot([timestemp_ipull,timestemp_ipull],yrange,'--',color = '0.85',label='other pull')
                    else:
                        axs[iplot].plot([timestemp_ipull,timestemp_ipull],yrange,'--',color = '0.85')

                #
                # plot the behavioral tracking result
                timepoint_bhv = xxx_time
                yyy_bhv = data_summary[iplot]
                # select a smaller time window
                ind_bhv = (timepoint_bhv>mintime_forplot)&(timepoint_bhv<maxtime_forplot)
                timepoint_bhv = timepoint_bhv[ind_bhv]
                yyy_bhv = yyy_bhv[ind_bhv]
                # normalize y
                yyy_bhv = (yyy_bhv-np.nanmin(yyy_bhv))/(np.nanmax(yyy_bhv)-np.nanmin(yyy_bhv))

                axs[iplot].plot(timepoint_bhv,yyy_bhv,'-',color = clrs_plot[iplot])
                axs[iplot].set_xlim(mintime_forplot,maxtime_forplot)
                axs[iplot].set_xlabel('')
                axs[iplot].set_xticks(np.arange(mintime_forplot,maxtime_forplot,round((maxtime_forplot-mintime_forplot)/5))) 
                axs[iplot].set_xticklabels('')
                axs[iplot].set_ylabel(yaxis_labels[iplot])
                if ianimal == 0:
                    axs[iplot].set_title(animal1+' '+con_vars_plot[iplot])
                elif ianimal == 1:
                    axs[iplot].set_title(animal2+' '+con_vars_plot[iplot])

                if iplot == nconvarplots-1:
                    axs[iplot].set_xlabel('time(s)', fontsize = 14)
                    axs[iplot].set_xticks(np.arange(mintime_forplot,maxtime_forplot,round((maxtime_forplot-mintime_forplot)/5))) 
                    axs[iplot].set_xticklabels(list(map(str,np.arange(mintime_forplot,maxtime_forplot,round((maxtime_forplot-mintime_forplot)/5)))))


                #   
                # plot the neural result, e.g PC1,2,3    
                timepoint_fr = FR_timepoint_allch
                yyy_fr = FR_zscore_allch_PCs[:,iPCtrace]
                # select a smaller time window
                ind_fr = (timepoint_fr>mintime_forplot)&(timepoint_fr<maxtime_forplot)
                timepoint_fr = timepoint_fr[ind_fr]
                yyy_fr = yyy_fr[ind_fr]
                # normalize y 
                yyy_fr = (yyy_fr-np.nanmin(yyy_fr))/(np.nanmax(yyy_fr)-np.nanmin(yyy_fr))
                #
                axs[iplot].plot(timepoint_fr,yyy_fr,color = '0.5', label=PCtraces[iPCtrace], alpha=0.7)


                # 
                axs[iplot].legend(loc = 'upper right')

                # add the basic correlation value 
                if np.shape(yyy_fr)[0] < np.shape(yyy_bhv)[0]:
                    yyy_bhv = yyy_bhv[1:]
                elif np.shape(yyy_fr)[0] > np.shape(yyy_bhv)[0]:
                    yyy_fr = yyy_fr[1:]
                try:
                    corr,pcorr = st.pearsonr(yyy_fr,yyy_bhv)
                    #
                    axs[iplot].text(mintime_forplot*1.02,1,'corr ='+"{:.2f}".format(corr))
                    axs[iplot].text(mintime_forplot*1.02,0.90,'corr p ='+"{:.2f}".format(pcorr))
                except:
                    corr = np.nan
                    pcorr = np.nan



            # plot the pull triggered PC1,2,3
            #
            if ianimal == 0:
                timepoint_pull = time_point_pull1
            elif ianimal == 1:
                timepoint_pull = time_point_pull2
            npulls = np.shape(timepoint_pull)[0]

            pull_trigevent_data = []
            pull_trigevent_data_succtrial = []
            pull_trigevent_data_errtrial = []

            for ipull in np.arange(0,npulls,1):
                timestemp_ipull = (np.array(timepoint_pull)[ipull])
                yrange = [np.floor(np.nanmin(FR_zscore_allch_PCs[:,iPCtrace])),np.ceil(np.nanmax(FR_zscore_allch_PCs[:,iPCtrace]))]

                # plot pull triggered events
                frame_win_ipull = (timestemp_ipull+trig_twin)*fps
                xxx_trigevent = np.arange(trig_twin[0],trig_twin[1],1/fps) 

                try:
                    axs2[iPCtrace].plot(xxx_trigevent,FR_zscore_allch_PCs[:,iPCtrace][int(frame_win_ipull[0]):int(frame_win_ipull[1])],'-',color = clrs_plot[iPCtrace])
                    #
                    pull_trigevent_data.append(FR_zscore_allch_PCs[:,iPCtrace][int(frame_win_ipull[0]):int(frame_win_ipull[1])])
                except:
                    pull_trigevent_data.append(np.full((1,np.shape(xxx_trigevent)[0]),np.nan)[0])

                # separate trace for the successful pulls
                if succpulls_ornot[ianimal][ipull]==1:
                    try:
                        axs3[iPCtrace].plot(xxx_trigevent,FR_zscore_allch_PCs[:,iPCtrace][int(frame_win_ipull[0]):int(frame_win_ipull[1])],'-',color = clrs_plot[iPCtrace])
                        #
                        pull_trigevent_data_succtrial.append(FR_zscore_allch_PCs[:,iPCtrace][int(frame_win_ipull[0]):int(frame_win_ipull[1])])
                    except:
                        pull_trigevent_data_succtrial.append(np.full((1,np.shape(xxx_trigevent)[0]),np.nan)[0])

                # separate trace for the failed pulls
                if succpulls_ornot[ianimal][ipull]==0:
                    try:
                        axs4[iPCtrace].plot(xxx_trigevent,FR_zscore_allch_PCs[:,iPCtrace][int(frame_win_ipull[0]):int(frame_win_ipull[1])],'-',color = clrs_plot[iPCtrace])
                        #
                        pull_trigevent_data_errtrial.append(FR_zscore_allch_PCs[:,iPCtrace][int(frame_win_ipull[0]):int(frame_win_ipull[1])])
                    except:
                        pull_trigevent_data_errtrial.append(np.full((1,np.shape(xxx_trigevent)[0]),np.nan)[0])



            # save the trace; for each PC separately
            if savefig:
                if ianimal == 0:
                    fig.savefig(save_path+"/"+date_tgt+"_neuralFR_"+PCtraces[iPCtrace]+"_and_"+animal1+"_continuousTrackingVari.pdf")
                elif ianimal == 1:
                    fig.savefig(save_path+"/"+date_tgt+"_neuralFR_"+PCtraces[iPCtrace]+"_and_"+animal2+"_continuousTrackingVari.pdf")


            # plot settings for the pull triggered PC123
            # plot settings for axs2        
            axs2[iPCtrace].plot([0,0],yrange,'k-')
            try: 
                axs2[iPCtrace].plot(xxx_trigevent,np.nanmean(pull_trigevent_data,axis=0),'k')
            except:
                axs2[iPCtrace].plot(xxx_trigevent,pull_trigevent_data,'k')
            #
                axs2[iPCtrace].set_xlim(trig_twin)
            axs2[iPCtrace].set_xlabel('')
            axs2[iPCtrace].set_xticklabels('')
            axs2[iPCtrace].set_ylabel('')
            if ianimal == 0:
                axs2[iPCtrace].set_title('all pulls '+animal1+' '+PCtraces[iPCtrace])
            elif ianimal == 1:
                axs2[iPCtrace].set_title('all pulls '+animal2+' '+PCtraces[iPCtrace])
            #
            if iplot == nconvarplots-1:
                axs2[iPCtrace].set_xlabel('time(s)', fontsize = 14)
                axs2[iPCtrace].set_xticks(np.linspace(trig_twin[0],trig_twin[1],5)) 
                axs2[iPCtrace].set_xticklabels(list(map(str,np.linspace(trig_twin[0],trig_twin[1],5))))

            # plot settings for axs3        
            axs3[iPCtrace].plot([0,0],yrange,'k-')
            try:
                axs3[iPCtrace].plot(xxx_trigevent,np.nanmean(pull_trigevent_data_succtrial,axis=0),'k')
            except:
                try:
                    axs3[iPCtrace].plot(xxx_trigevent,pull_trigevent_data_succtrial,'k')
                except:
                    pass
            #
            axs3[iPCtrace].set_xlim(trig_twin)
            axs3[iPCtrace].set_xlabel('')
            axs3[iPCtrace].set_xticklabels('')
            axs3[iPCtrace].set_ylabel('')
            if ianimal == 0:
                axs3[iPCtrace].set_title('successful pulls '+animal1+' '+PCtraces[iPCtrace])
            elif ianimal == 1:
                axs3[iPCtrace].set_title('successful pulls '+animal2+' '+PCtraces[iPCtrace])
            #
            if iplot == nconvarplots-1:
                axs3[iPCtrace].set_xlabel('time(s)', fontsize = 14)
                axs3[iPCtrace].set_xticks(np.linspace(trig_twin[0],trig_twin[1],5)) 
                axs3[iPCtrace].set_xticklabels(list(map(str,np.linspace(trig_twin[0],trig_twin[1],5))))

            # plot settings for axs4        
            axs4[iPCtrace].plot([0,0],yrange,'k-')
            try:
                axs4[iPCtrace].plot(xxx_trigevent,np.nanmean(pull_trigevent_data_errtrial,axis=0),'k')
            except:
                try:
                    axs4[iPCtrace].plot(xxx_trigevent,pull_trigevent_data_errtrial,'k')
                except:
                    pass
            #
            axs4[iPCtrace].set_xlim(trig_twin)
            axs4[iPCtrace].set_xlabel('')
            axs4[iPCtrace].set_xticklabels('')
            axs4[iPCtrace].set_ylabel('')
            if ianimal == 0:
                axs4[iPCtrace].set_title('failed pulls '+animal1+' '+PCtraces[iPCtrace])
            elif ianimal == 1:
                axs4[iPCtrace].set_title('failed pulls '+animal2+' '+PCtraces[iPCtrace])
            #
            if iPCtrace == nPCtraces-1:
                axs4[iPCtrace].set_xlabel('time(s)', fontsize = 14)
                axs4[iPCtrace].set_xticks(np.linspace(trig_twin[0],trig_twin[1],5)) 
                axs4[iPCtrace].set_xticklabels(list(map(str,np.linspace(trig_twin[0],trig_twin[1],5))))

            if savefig:
                if ianimal == 0:
                    fig2.savefig(save_path+"/"+date_tgt+'_'+animal1+"_pull_triggered_PC123traces.pdf")
                    fig3.savefig(save_path+"/"+date_tgt+'_'+animal1+"_successfulpull_triggered_PC123traces.pdf")
                    fig4.savefig(save_path+"/"+date_tgt+'_'+animal1+"_failedpull_triggered_PC123traces.pdf")
                elif ianimal == 1:
                    fig2.savefig(save_path+"/"+date_tgt+'_'+animal2+"_pull_triggered_PC123traces.pdf")
                    fig3.savefig(save_path+"/"+date_tgt+'_'+animal2+"_successfulpull_triggered_PC123traces.pdf")
                    fig4.savefig(save_path+"/"+date_tgt+'_'+animal2+"_failedpull_triggered_PC123traces.pdf")





#  function - make an example frame (saved as PDF) for the body part tracking based on single camera, also show the important axes
def tracking_frame_singlecam_wholebody_demo(bodyparts_locs_camN,output_look_ornot,output_allvectors,output_allangles,lever_loc_both,
tube_loc_both,time_point_pull1,time_point_pull2,animalnames_videotrack,bodypartnames_videotrack,date_tgt,animal1_filename,animal2_filename,
session_start_time,fps,nframes,cameraID,video_file_original,sqr_thres_tubelever,sqr_thres_face,sqr_thres_body,iframe_plot,window_sec):
    """
    Same as tracking_video_singlecam_wholebody_demo, but instead of looping over
    all frames and writing a video, this renders a single example figure and
    saves it as a PDF:

      - TOP panel: the tracked video frame (with body parts, skeleton, face box,
        and head-gaze vector overlaid) at the targeted time point.
      - BOTTOM panels: the four behavioral event time series (animal1 social
        gaze, animal1 pull, animal2 social gaze, animal2 pull) shown in a
        window from -window_sec to +window_sec around the targeted time,
        with a vertical line marking the target time itself.

    New parameters
    ---------------
    iframe_plot : int or None
        The (absolute, 0-indexed) frame number of the original video to use as
        the targeted time point. If None, defaults to iframe_min (i.e. the
        frame at session_start_time).
    window_sec : float
        Half-width, in seconds, of the behavioral event time series window
        around the target time (default 15, i.e. -15s to +15s).
    """

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import scipy
    import string
    import warnings
    import pickle
    import cv2

    skeletons = [ ['rightTuft','rightEye'],
                  ['rightTuft','whiteBlaze'],
                  ['leftTuft','leftEye'],
                  ['leftTuft','whiteBlaze'],
                  ['rightEye','whiteBlaze'],
                  ['leftEye','whiteBlaze'],
                  ['rightEye','mouth'],
                  ['leftEye','mouth'],
                  ['leftEye','rightEye']
                ]
    nskeletons = np.shape(skeletons)[0]

    colors = ['#bf3eff','#f5911e','#5c5c5c']
    # colors = ['#a6a6a6','#000000','k']

    linewidthss = 5
    markersizess = 15

    # Settings
    # PDF file path for saving the example frame
    pdf_file = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/3d_recontruction_analysis_self_and_coop_task_data_saved/example_videos_singlecam_wholebody_demo/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt+"_"+animal1_filename+animal2_filename+"_singlecam_wholebody_tracking_exampleframe.pdf"

    # load the original video
    vidcap = cv2.VideoCapture(video_file_original)

    animal_names_unique = animalnames_videotrack
    body_parts_unique = bodypartnames_videotrack

    nanimals = np.shape(animal_names_unique)[0]
    nbodyparts = np.shape(body_parts_unique)[0]

    # align the plot with the session start
    iframe_min = int(np.round(session_start_time*fps))
    iframe_max = int(nframes+iframe_min)

    # which frame to actually render as the targeted time point
    if iframe_plot is None:
        iframe = iframe_min
    else:
        iframe = int(iframe_plot)

    # window (in frames) around the target time for the behavioral event time series
    window_frames = int(np.round(window_sec*fps))
    iwindow_min = iframe - window_frames
    iwindow_max = iframe + window_frames

    # set up the figure setting
    fig = plt.figure(figsize = (15,15))
    gs=GridSpec(9,4) # 9 rows, 4 columns

    ax1=fig.add_subplot(gs[0:5,:]) # animal tracking frame
    ax2=fig.add_subplot(gs[5,:]) # animal1 gaze
    ax3=fig.add_subplot(gs[6,:]) # animal1 pull
    ax4=fig.add_subplot(gs[7,:]) # animal2 gaze
    ax5=fig.add_subplot(gs[8,:]) # animal2 pull

    ax1.set_xlim([0,1920])
    ax1.set_ylim([0,1080])
    ax1.set_xlabel('x (pixel)',fontsize=24)
    ax1.set_ylabel('y (pixel)',fontsize=24)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.invert_yaxis()
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    ax1.axis('off')

    # x-axis ticks every 5s within the +/- window_sec window, labeled relative to target time (s)
    tick_step_frames = int(np.round(5*fps))
    xticks_window = np.arange(iwindow_min,iwindow_max+1,tick_step_frames)
    xticklabels_window = [str(int(np.round((t-iframe)/fps))) for t in xticks_window]

    ax2.set_xlim([iwindow_min,iwindow_max])
    ax2.set_xticks(xticks_window)
    ax2.set_xticklabels('')
    ax2.set_ylim([0,1])
    ax2.set_yticklabels('')
    ax2.set_xlabel('')
    ax2.set_ylabel('animal1\nsocial gaze',fontsize=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax2.axvline(iframe,color='r',linewidth=2,linestyle='-')

    ax3.set_xlim([iwindow_min,iwindow_max])
    ax3.set_xticks(xticks_window)
    ax3.set_xticklabels('')
    ax3.set_ylim([0,1])
    ax3.set_yticklabels('')
    ax3.set_xlabel('')
    ax3.set_ylabel('animal1\npull',fontsize=15)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.get_xaxis().set_ticks([])
    ax3.get_yaxis().set_ticks([])
    ax3.axvline(iframe,color='r',linewidth=2,linestyle='-')

    ax4.set_xlim([iwindow_min,iwindow_max])
    ax4.set_xticks(xticks_window)
    ax4.set_xticklabels('')
    ax4.set_ylim([0,1])
    ax4.set_yticklabels('')
    ax4.set_xlabel('')
    ax4.set_ylabel('animal2\nsocial gaze',fontsize=15)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.get_xaxis().set_ticks([])
    ax4.get_yaxis().set_ticks([])
    ax4.axvline(iframe,color='r',linewidth=2,linestyle='-')

    ax5.set_xlim([iwindow_min,iwindow_max])
    ax5.set_xticks(xticks_window)
    ax5.set_xticklabels(xticklabels_window,fontsize=16)
    ax5.set_ylim([0,1])
    ax5.set_yticklabels('')
    ax5.set_xlabel('time relative to target (s)',fontsize = 24)
    ax5.set_ylabel('animal2\npull',fontsize=15)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    ax5.get_yaxis().set_ticks([])
    ax5.axvline(iframe,color='r',linewidth=2,linestyle='-')

    print("printing example frame ",str(iframe+1),"/",str(iframe_max))

    # plot the original video frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, iframe)
    ret, image_original = vidcap.read()
    ax1.imshow(cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB))

    for ianimal in np.arange(0,nanimals,1):
        ianimal_name = animal_names_unique[ianimal]
        # draw body part
        bodypart_loc_iframe = np.zeros((nbodyparts,2))
        #
        for ibdpart in np.arange(0,nbodyparts,1):

            ibdpart_name = body_parts_unique[ibdpart]
            bodypart_loc_iframe[ibdpart,:] = np.array(bodyparts_locs_camN[(ianimal_name,ibdpart_name)])[iframe,:]
        # plot the body parts
        ax1.plot(bodypart_loc_iframe[:,0], bodypart_loc_iframe[:,1], '.', markersize=markersizess, color=colors[ianimal])

        # draw skeleton
        withlabel = 0
        for iskel in np.arange(0,nskeletons,1):
            try:
                iskeleton_name = skeletons[iskel]
                skelbody12_loc_iframe = np.zeros((2,2))
                #
                skel_body1_name = iskeleton_name[0]
                skel_body2_name = iskeleton_name[1]
                #
                skelbody12_loc_iframe[0,:] = np.array(bodyparts_locs_camN[(ianimal_name,skel_body1_name)])[iframe,:]
                skelbody12_loc_iframe[1,:] = np.array(bodyparts_locs_camN[(ianimal_name,skel_body2_name)])[iframe,:]
                # plot one skeleton
                # add the label for legend
                if not withlabel:
                    if (ianimal==0):
                        ax1.plot(skelbody12_loc_iframe[:,0],skelbody12_loc_iframe[:,1],'-',linewidth=linewidthss, color=colors[ianimal],label ='animal 1')
                    else:
                        ax1.plot(skelbody12_loc_iframe[:,0],skelbody12_loc_iframe[:,1],'-',linewidth=linewidthss, color=colors[ianimal],label ='animal 2')
                    withlabel = 1
                else:
                    ax1.plot(skelbody12_loc_iframe[:,0],skelbody12_loc_iframe[:,1],'-',linewidth=linewidthss, color=colors[ianimal])
            except:
                continue

        # draw face rectangle
        face_mass = np.nanmean(np.vstack((np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:],np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:],
                                          np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:],np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:],
                                          np.array(bodyparts_locs_camN[(ianimal_name,'mouth')])[iframe,:],np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:])),axis=0)

        dist7 = np.linalg.norm(np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:]-np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:])
        dist8 = np.linalg.norm(np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:]-np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:])
        dist9 = np.linalg.norm(np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:]-np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:])
        dist10 = np.linalg.norm(np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:]-np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:])

        face_offset = np.nanmax([dist7,dist8,dist9,dist10])*sqr_thres_face
        ax1.plot([face_mass[0]-face_offset,face_mass[0]+face_offset],[face_mass[1]-face_offset,face_mass[1]-face_offset],'--',color=colors[ianimal])
        ax1.plot([face_mass[0]-face_offset,face_mass[0]+face_offset],[face_mass[1]+face_offset,face_mass[1]+face_offset],'--',color=colors[ianimal])
        ax1.plot([face_mass[0]-face_offset,face_mass[0]-face_offset],[face_mass[1]-face_offset,face_mass[1]+face_offset],'--',color=colors[ianimal])
        ax1.plot([face_mass[0]+face_offset,face_mass[0]+face_offset],[face_mass[1]-face_offset,face_mass[1]+face_offset],'--',color=colors[ianimal])

        # draw head vector (gaze direction)
        rightEye_loc_iframe = np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:]
        leftEye_loc_iframe = np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:]
        meaneye_loc_iframe = np.nanmean(np.vstack([rightEye_loc_iframe,leftEye_loc_iframe]),axis=0)
        # head gaze direction is assumed to be opposite to the head axis
        head_loc_iframe = meaneye_loc_iframe - 400*np.array(output_allvectors['head_vect_all_merge'][ianimal_name])[iframe,:]
        if (ianimal==1):
            ax1.plot([meaneye_loc_iframe[0],head_loc_iframe[0]],[meaneye_loc_iframe[1],head_loc_iframe[1]],'--',linewidth=linewidthss, color = '#00a6d0',label='head gaze')
        else:
            ax1.plot([meaneye_loc_iframe[0],head_loc_iframe[0]],[meaneye_loc_iframe[1],head_loc_iframe[1]],'--',linewidth=linewidthss, color = '#00a6d0')

        # draw animal behavioral events that fall within the +/- window_sec window around the target time
        look_at_other_framenum_all = np.where(np.array(output_look_ornot["look_at_other_or_not_merge"][ianimal_name])==1)[0]
        look_at_other_framenum_plot = look_at_other_framenum_all[(look_at_other_framenum_all>=iwindow_min)&(look_at_other_framenum_all<=iwindow_max)]
        look_at_lever_framenum_all = np.where(np.array(output_look_ornot["look_at_lever_or_not_merge"][ianimal_name])==1)[0]
        look_at_lever_framenum_plot = look_at_lever_framenum_all[(look_at_lever_framenum_all>=iwindow_min)&(look_at_lever_framenum_all<=iwindow_max)]
        look_at_tube_framenum_all = np.where(np.array(output_look_ornot["look_at_tube_or_not_merge"][ianimal_name])==1)[0]
        look_at_tube_framenum_plot = look_at_tube_framenum_all[(look_at_tube_framenum_all>=iwindow_min)&(look_at_tube_framenum_all<=iwindow_max)]

        pull1_framenum = (time_point_pull1 + session_start_time)*fps
        pull1_framenum_plot = pull1_framenum[(pull1_framenum>=iwindow_min)&(pull1_framenum<=iwindow_max)]
        pull2_framenum = (time_point_pull2 + session_start_time)*fps
        pull2_framenum_plot = pull2_framenum[(pull2_framenum>=iwindow_min)&(pull2_framenum<=iwindow_max)]

        bhv_events_plot = np.hstack([look_at_other_framenum_plot,look_at_lever_framenum_plot,look_at_tube_framenum_plot,pull1_framenum_plot,pull2_framenum_plot])
        nplotframes = np.shape(bhv_events_plot)[0]

        for iplotframe in np.arange(0,nplotframes,1):
            bhv_events_iframe = bhv_events_plot[iplotframe]
            if (ianimal == 0):
                if (np.isin(bhv_events_iframe,look_at_other_framenum_plot)):
                    ax2.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = colors[np.absolute(ianimal)])
                if (np.isin(bhv_events_iframe,pull1_framenum_plot)):
                    ax3.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = 'k')
            elif (ianimal == 1):
                if (np.isin(bhv_events_iframe,look_at_other_framenum_plot)):
                    ax4.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = colors[np.absolute(ianimal)])
                if (np.isin(bhv_events_iframe,pull2_framenum_plot)):
                    ax5.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = 'k')

    plt.savefig(pdf_file, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return pdf_file
#  function - make demo videos for the body part tracking based on single camera, also show the important axes
def tracking_video_singlecam_wholebody_demo(bodyparts_locs_camN,output_look_ornot,output_allvectors,output_allangles,lever_loc_both,
tube_loc_both,time_point_pull1,time_point_pull2,animalnames_videotrack,bodypartnames_videotrack,date_tgt,animal1_filename,animal2_filename,
session_start_time,fps,nframes,cameraID,video_file_original,sqr_thres_tubelever,sqr_thres_face,sqr_thres_body):

    import pandas as pd
    import numpy as np
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
    
    colors = ['b','r','k']
    
    import matplotlib.animation as animation

    # Settings
    # video file path for saving
    video_file = "/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/3d_recontruction_analysis_self_and_coop_task_data_saved/example_videos_singlecam_wholebody_demo/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt+"_"+animal1_filename+animal2_filename+"_singlecam_wholebody_tracking_demo.mp4"
    clear_frames = True     # Should it clear the figure between each frame?
    fps = 30 

    # load the original video
    vidcap = cv2.VideoCapture(video_file_original)       


    # Output video writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Animal tracking demo', artist='Matplotlib', comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # animal_names_unique = pd.unique(pd.DataFrame(bodyparts_locs_camN.keys()).iloc[:,0])
    # body_parts_unique = pd.unique(pd.DataFrame(bodyparts_locs_camN.keys()).iloc[:,1])
    animal_names_unique = animalnames_videotrack
    body_parts_unique = bodypartnames_videotrack    

    nanimals = np.shape(animal_names_unique)[0]  
    nbodyparts = np.shape(body_parts_unique)[0]
    
    # align the plot with the session start
    iframe_min = int(np.round(session_start_time*fps))
    iframe_max = nframes+iframe_min   


    # set up the figure setting  
    fig = plt.figure(figsize = (15,12))
    gs=GridSpec(7,4) # 7 rows, 4 columns

    ax1=fig.add_subplot(gs[0:5,:]) # animal tracking frame
    ax2=fig.add_subplot(gs[5,:]) # animal1 behavioral events
    ax3=fig.add_subplot(gs[6,:]) # animal2 behavioral events

    ax1.set_xlim([0,1920])
    ax1.set_ylim([0,1080])
    ax1.set_xlabel('x (pixel)',fontsize = 24)
    ax1.set_ylabel('y (pixel)',fontsize = 24)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.invert_yaxis()
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    
    ax2.set_xlim([iframe_min,iframe_max]) 
    ax2.set_xticks(np.arange(iframe_min,iframe_max,300)) 
    ax2.set_xticklabels('')
    ax2.set_ylim([0,1])
    ax2.set_yticklabels('')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    # ax2.set_title('animal 1 behavioral events')
    
    ax3.set_xlim([iframe_min,iframe_max])  
    ax3.set_xticks(np.arange(iframe_min,iframe_max,300)) 
    ax3.set_xticklabels(list(map(str,np.arange(0/fps,nframes/fps,300/fps))))
    ax3.tick_params(axis='x', labelsize=20)
    ax3.set_ylim([0,1])
    ax3.set_yticklabels('')
    ax3.set_xlabel('time (s)',fontsize = 24)
    ax3.set_ylabel('')
    # ax3.set_title('animal 2 behavioral events')



    with writer.saving(fig, video_file, 100):
        # for iframe in np.arange(0,nframes,1):    
        for iframe in np.arange(iframe_min,iframe_max,1):
           
            print("printing frame ",str(iframe+1),"/",str(iframe_max))
        
            if clear_frames:
                fig.clear()
                gs=GridSpec(7,4) # 7 rows, 4 columns

                ax1=fig.add_subplot(gs[0:5,:]) # animal tracking frame
                ax2=fig.add_subplot(gs[5,:]) # animal1 behavioral events
                ax3=fig.add_subplot(gs[6,:]) # animal2 behavioral events

                ax1.set_xlim([0,1920])
                ax1.set_ylim([0,1080])
                ax1.set_xlabel('x (pixel)',fontsize=24)
                ax1.set_ylabel('y (pixel)',fontsize=24)
                ax1.tick_params(axis='x', labelsize=20)
                ax1.tick_params(axis='y', labelsize=20)
                ax1.invert_yaxis()
                ax1.xaxis.set_ticks_position('top')
                ax1.xaxis.set_label_position('top')

                
                ax2.set_xlim([iframe_min,iframe_max]) 
                ax2.set_xticks(np.arange(iframe_min,iframe_max,300)) 
                ax2.set_xticklabels('')
                ax2.set_ylim([0,1])
                ax2.set_yticklabels('')
                ax2.set_xlabel('')
                ax2.set_ylabel('')
                #ax2.set_title('animal 1 behavioral events')
    
                ax3.set_xlim([iframe_min,iframe_max])  
                ax3.set_xticks(np.arange(iframe_min,iframe_max,300)) 
                ax3.set_xticklabels(list(map(str,np.arange(0/fps,nframes/fps,300/fps))))
                ax3.tick_params(axis='x', labelsize=20)
                ax3.set_ylim([0,1])
                ax3.set_yticklabels('')
                ax3.set_xlabel('time (s)',fontsize = 24)
                ax3.set_ylabel('')
                #ax3.set_title('animal 2 behavioral events')

                # plot the original videos
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
                if (ianimal==0): 
                    ax1.plot(bodypart_loc_iframe[:,0], bodypart_loc_iframe[:,1], '.', color=colors[ianimal],label ='animal1')
                else:
                    ax1.plot(bodypart_loc_iframe[:,0], bodypart_loc_iframe[:,1], '.', color=colors[ianimal],label ='animal2')
                
                # draw skeleton                
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
                        ax1.plot(skelbody12_loc_iframe[:,0],skelbody12_loc_iframe[:,1],'-',color=colors[ianimal])
                    except:
                        continue


                # draw face and body rectagon
                face_mass = np.nanmean(np.vstack((np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:],np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:],
                                                  np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:],np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:],
                                                  np.array(bodyparts_locs_camN[(ianimal_name,'mouth')])[iframe,:],np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:])),axis=0)     
                
                dist1 = np.linalg.norm(face_mass-np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:])
                dist2 = np.linalg.norm(face_mass-np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:])
                dist3 = np.linalg.norm(face_mass-np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:])
                dist4 = np.linalg.norm(face_mass-np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:])
                dist5 = np.linalg.norm(face_mass-np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:])
                dist6 = np.linalg.norm(face_mass-np.array(bodyparts_locs_camN[(ianimal_name,'mouth')])[iframe,:])   
                dist7 = np.linalg.norm(np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:]-np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:])
                dist8 = np.linalg.norm(np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:]-np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:])
                dist9 = np.linalg.norm(np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:]-np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:])
                dist10 = np.linalg.norm(np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:]-np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:])

                # face_offset = np.nanmax([dist1,dist2,dist3,dist4,dist5,dist6])*sqr_thres_face # draw square around face
                face_offset = np.nanmax([dist7,dist8,dist9,dist10])*sqr_thres_face
                ax1.plot([face_mass[0]-face_offset,face_mass[0]+face_offset],[face_mass[1]-face_offset,face_mass[1]-face_offset],'--',color=colors[ianimal])
                ax1.plot([face_mass[0]-face_offset,face_mass[0]+face_offset],[face_mass[1]+face_offset,face_mass[1]+face_offset],'--',color=colors[ianimal])
                ax1.plot([face_mass[0]-face_offset,face_mass[0]-face_offset],[face_mass[1]-face_offset,face_mass[1]+face_offset],'--',color=colors[ianimal])
                ax1.plot([face_mass[0]+face_offset,face_mass[0]+face_offset],[face_mass[1]-face_offset,face_mass[1]+face_offset],'--',color=colors[ianimal])       

                # draw the estimated body
                ax1.plot([face_mass[0]-face_offset,face_mass[0]+face_offset],[face_mass[1]-face_offset,face_mass[1]-face_offset],'--',color=colors[ianimal])
                ax1.plot([face_mass[0]-face_offset,face_mass[0]+face_offset],[face_mass[1]+sqr_thres_body*face_offset,face_mass[1]+sqr_thres_body*face_offset],'--',color=colors[ianimal])
                ax1.plot([face_mass[0]-face_offset,face_mass[0]-face_offset],[face_mass[1]-face_offset,face_mass[1]+sqr_thres_body*face_offset],'--',color=colors[ianimal])
                ax1.plot([face_mass[0]+face_offset,face_mass[0]+face_offset],[face_mass[1]-face_offset,face_mass[1]+sqr_thres_body*face_offset],'--',color=colors[ianimal])                       

                
                # draw lever and tube location
                if (ianimal==1): 
                    ax1.plot(lever_loc_both[ianimal_name][0],lever_loc_both[ianimal_name][1],'o',color='g',label='lever')
                    ax1.plot(tube_loc_both[ianimal_name][0],tube_loc_both[ianimal_name][1],'o',color='y',label='tube')
                else:
                    ax1.plot(lever_loc_both[ianimal_name][0],lever_loc_both[ianimal_name][1],'o',color='g')
                    ax1.plot(tube_loc_both[ianimal_name][0],tube_loc_both[ianimal_name][1],'o',color='y')

                
                # draw lever square
                sqr_offset = sqr_thres_tubelever # usually set as 75: draw a 150x150 pixel around lever 
                ax1.plot([lever_loc_both[ianimal_name][0]-sqr_offset,lever_loc_both[ianimal_name][0]+sqr_offset],[lever_loc_both[ianimal_name][1]-sqr_offset,lever_loc_both[ianimal_name][1]-sqr_offset],'--',color='g')
                ax1.plot([lever_loc_both[ianimal_name][0]-sqr_offset,lever_loc_both[ianimal_name][0]+sqr_offset],[lever_loc_both[ianimal_name][1]+sqr_offset,lever_loc_both[ianimal_name][1]+sqr_offset],'--',color='g')
                ax1.plot([lever_loc_both[ianimal_name][0]-sqr_offset,lever_loc_both[ianimal_name][0]-sqr_offset],[lever_loc_both[ianimal_name][1]-sqr_offset,lever_loc_both[ianimal_name][1]+sqr_offset],'--',color='g')
                ax1.plot([lever_loc_both[ianimal_name][0]+sqr_offset,lever_loc_both[ianimal_name][0]+sqr_offset],[lever_loc_both[ianimal_name][1]-sqr_offset,lever_loc_both[ianimal_name][1]+sqr_offset],'--',color='g')


                # draw tube square
                sqr_offset = sqr_thres_tubelever # usually set as 75: draw a 150x150 pixel around lever 
                ax1.plot([tube_loc_both[ianimal_name][0]-sqr_offset,tube_loc_both[ianimal_name][0]+sqr_offset],[tube_loc_both[ianimal_name][1]-sqr_offset,tube_loc_both[ianimal_name][1]-sqr_offset],'--',color='y')
                ax1.plot([tube_loc_both[ianimal_name][0]-sqr_offset,tube_loc_both[ianimal_name][0]+sqr_offset],[tube_loc_both[ianimal_name][1]+sqr_offset,tube_loc_both[ianimal_name][1]+sqr_offset],'--',color='y')
                ax1.plot([tube_loc_both[ianimal_name][0]-sqr_offset,tube_loc_both[ianimal_name][0]-sqr_offset],[tube_loc_both[ianimal_name][1]-sqr_offset,tube_loc_both[ianimal_name][1]+sqr_offset],'--',color='y')
                ax1.plot([tube_loc_both[ianimal_name][0]+sqr_offset,tube_loc_both[ianimal_name][0]+sqr_offset],[tube_loc_both[ianimal_name][1]-sqr_offset,tube_loc_both[ianimal_name][1]+sqr_offset],'--',color='y')
              
           
                # draw head vector
                rightEye_loc_iframe = np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:]
                leftEye_loc_iframe = np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:]
                meaneye_loc_iframe = np.nanmean(np.vstack([rightEye_loc_iframe,leftEye_loc_iframe]),axis=0)
                # head_loc_iframe = meaneye_loc_iframe + 400*np.array(output_allvectors['head_vect_all_merge'][ianimal_name])[iframe,:]
                # # head gaze direction is assumed to be opposite to the head axis
                head_loc_iframe = meaneye_loc_iframe - 400*np.array(output_allvectors['head_vect_all_merge'][ianimal_name])[iframe,:]
                if (ianimal==1):
                    # ax1.plot([meaneye_loc_iframe[0],head_loc_iframe[0]],[meaneye_loc_iframe[1],head_loc_iframe[1]],'-',color = '0.75',label='head axis')
                    ax1.plot([meaneye_loc_iframe[0],head_loc_iframe[0]],[meaneye_loc_iframe[1],head_loc_iframe[1]],'-',color = '0.75',label='head gaze')
                else:
                    ax1.plot([meaneye_loc_iframe[0],head_loc_iframe[0]],[meaneye_loc_iframe[1],head_loc_iframe[1]],'-',color = '0.75')     

                # draw other - eye vector   
                if 0:         
                    other_loc_iframe = meaneye_loc_iframe + 200*np.array(output_allvectors['other_eye_vect_all_merge'][ianimal_name])[iframe,:]
                    if (ianimal==1):
                        ax1.plot([meaneye_loc_iframe[0],other_loc_iframe[0]],[meaneye_loc_iframe[1],other_loc_iframe[1]],'-',color = colors[np.absolute(ianimal-1)],label="eye-other")
                    else:
                        ax1.plot([meaneye_loc_iframe[0],other_loc_iframe[0]],[meaneye_loc_iframe[1],other_loc_iframe[1]],'-',color = colors[np.absolute(ianimal-1)])

                # draw tube - eye vector
                if 0: 
                    tube_loc_iframe = meaneye_loc_iframe + 200*np.array(output_allvectors['tube_eye_vect_all_merge'][ianimal_name])[iframe,:]
                    if (ianimal==1):
                        ax1.plot([meaneye_loc_iframe[0],tube_loc_iframe[0]],[meaneye_loc_iframe[1],tube_loc_iframe[1]],'-',color = 'y',label="eye-tube")
                    else:
                        ax1.plot([meaneye_loc_iframe[0],tube_loc_iframe[0]],[meaneye_loc_iframe[1],tube_loc_iframe[1]],'-',color = 'y')

                # draw lever - eye vector
                if 0:
                    lever_loc_iframe = meaneye_loc_iframe + 200*np.array(output_allvectors['lever_eye_vect_all_merge'][ianimal_name])[iframe,:]
                    if (ianimal==1):
                        ax1.plot([meaneye_loc_iframe[0],lever_loc_iframe[0]],[meaneye_loc_iframe[1],lever_loc_iframe[1]],'-',color = 'g',label="eye-lever")
                    else:
                        ax1.plot([meaneye_loc_iframe[0],lever_loc_iframe[0]],[meaneye_loc_iframe[1],lever_loc_iframe[1]],'-',color = 'g')


                ax1.legend(loc='upper right',fontsize=15)
                         


                # draw animal behavioral events
                look_at_other_framenum_all = np.where(np.array(output_look_ornot["look_at_other_or_not_merge"][ianimal_name])==1)[0]
                look_at_other_framenum_plot = look_at_other_framenum_all[(look_at_other_framenum_all<=iframe)&(look_at_other_framenum_all>iframe_min)]
                look_at_lever_framenum_all = np.where(np.array(output_look_ornot["look_at_lever_or_not_merge"][ianimal_name])==1)[0]
                look_at_lever_framenum_plot = look_at_lever_framenum_all[(look_at_lever_framenum_all<=iframe)&(look_at_lever_framenum_all>iframe_min)]
                look_at_tube_framenum_all = np.where(np.array(output_look_ornot["look_at_tube_or_not_merge"][ianimal_name])==1)[0]
                look_at_tube_framenum_plot = look_at_tube_framenum_all[(look_at_tube_framenum_all<=iframe)&(look_at_tube_framenum_all>iframe_min)]

                pull1_framenum = (time_point_pull1 + session_start_time)*fps
                pull1_framenum_plot = pull1_framenum[(pull1_framenum<=iframe)&(pull1_framenum>iframe_min)]
                pull2_framenum = (time_point_pull2 + session_start_time)*fps
                pull2_framenum_plot = pull2_framenum[(pull2_framenum<=iframe)&(pull2_framenum>iframe_min)]

                bhv_events_plot = np.hstack([look_at_other_framenum_plot,look_at_lever_framenum_plot,look_at_tube_framenum_plot,pull1_framenum_plot,pull2_framenum_plot])
                nplotframes = np.shape(bhv_events_plot)[0]
                

                for iplotframe in np.arange(0,nplotframes,1):
                    bhv_events_iframe = bhv_events_plot[iplotframe]
                    if (ianimal == 0):
                        if (np.isin(bhv_events_iframe,look_at_other_framenum_plot)): 
                            ax2.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = colors[np.absolute(ianimal-1)])
                        elif (np.isin(bhv_events_iframe,look_at_lever_framenum_plot)): 
                            ax2.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = 'g')
                        elif (np.isin(bhv_events_iframe,look_at_tube_framenum_plot)): 
                            ax2.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = 'y')
                        elif (np.isin(bhv_events_iframe,pull1_framenum_plot)): 
                            ax2.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = 'k')
                        # else:
                        #     ax2.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = '0.5')
                    elif (ianimal == 1):
                        if (np.isin(bhv_events_iframe,look_at_other_framenum_plot)): 
                            ax3.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = colors[np.absolute(ianimal-1)])
                        elif (np.isin(bhv_events_iframe,look_at_lever_framenum_plot)): 
                            ax3.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = 'g')
                        elif (np.isin(bhv_events_iframe,look_at_tube_framenum_plot)): 
                            ax3.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = 'y')
                        elif (np.isin(bhv_events_iframe,pull2_framenum_plot)): 
                            ax3.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = 'k')
                        # else:
                        #     ax3.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = '0.5')                          
                    

            writer.grab_frame()            

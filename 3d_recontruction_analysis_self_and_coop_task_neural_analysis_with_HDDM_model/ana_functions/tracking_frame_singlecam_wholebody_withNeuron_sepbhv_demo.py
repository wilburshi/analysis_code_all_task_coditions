#  function - make demo videos for the body part tracking based on single camera, also show the important axes
#  separate trace for each behavioral events
def tracking_frame_singlecam_wholebody_withNeuron_sepbhv_demo(bodyparts_locs_camN,output_look_ornot,output_allvectors,output_allangles,lever_loc_both,
tube_loc_both,time_point_pull1,time_point_pull2,animalnames_videotrack,bodypartnames_videotrack,date_tgt,animal1_filename,animal2_filename,
session_start_time,fps,start_frame,example_frame,cameraID,video_file_original,sqr_thres_tubelever,sqr_thres_face,sqr_thres_body,spike_time_data,lfp_filt_sess_aligned,
spike_channels_data,channel_to_depth):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import scipy
    from scipy.ndimage import gaussian_filter1d
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
    trace_colors = ['#cfe2f3','#f4cccc','k']

    # Settings
    fps = 30  

    clear_frames = True     # Should it clear the figure between each frame?

    # load the original video
    vidcap = cv2.VideoCapture(video_file_original)

    # animal_names_unique = pd.unique(pd.DataFrame(bodyparts_locs_camN.keys()).iloc[:,0])
    # body_parts_unique = pd.unique(pd.DataFrame(bodyparts_locs_camN.keys()).iloc[:,1])
    animal_names_unique = animalnames_videotrack
    body_parts_unique = bodypartnames_videotrack    

    nanimals = np.shape(animal_names_unique)[0]  
    nbodyparts = np.shape(body_parts_unique)[0]

    # align the plot with the session start
    iframe_min = int(np.round(session_start_time*fps)+start_frame)
    iframe_max = int(np.round(session_start_time*fps)+example_frame)  


    # set up the figure setting  
    fig = plt.figure(figsize = (15,25))
    gs=GridSpec(14,4) # 14 rows, 4 columns

    fig.subplots_adjust(wspace=0, hspace=0)

    # fig.constrained_layout = True

    ax1=fig.add_subplot(gs[0:6,:]) # animal tracking frame
    ax2=fig.add_subplot(gs[5,:]) # animal1 gaze
    ax3=fig.add_subplot(gs[6,:]) # animal1 pull
    ax4=fig.add_subplot(gs[7,:]) # animal2 gaze
    ax5=fig.add_subplot(gs[8,:]) # animal2 pull
    ax6=fig.add_subplot(gs[9:13,:]) # recording trace

    ax1.set_xlim([0,1920])
    ax1.set_ylim([0,1080])
    ax1.set_xlabel('x (pixel)',fontsize = 24)
    ax1.set_ylabel('y (pixel)',fontsize = 24)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.invert_yaxis()
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    ax1.axis('off')

    ax2.set_xlim([iframe_min,iframe_max]) 
    ax2.set_xticks(np.arange(iframe_min,iframe_max,300)) 
    ax2.set_xticklabels('')
    ax2.set_ylim([0,1])
    ax2.set_yticklabels('')
    ax2.set_xlabel('')
    ax2.set_ylabel('animal1\ngaze',fontsize=15)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    ax3.set_xlim([iframe_min,iframe_max])  
    ax3.set_xticks(np.arange(iframe_min,iframe_max,300)) 
    ax3.set_xticklabels('')
    ax3.set_ylim([0,1])
    ax3.set_yticklabels('')
    ax3.set_xlabel('')
    ax3.set_ylabel('animal1\npull',fontsize=15)
    ax3.tick_params(axis='y', labelsize=20)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.get_xaxis().set_ticks([])
    ax3.get_yaxis().set_ticks([])

    ax4.set_xlim([iframe_min,iframe_max]) 
    ax4.set_xticks(np.arange(iframe_min,iframe_max,300)) 
    ax4.set_xticklabels('')
    ax4.set_ylim([0,1])
    ax4.set_yticklabels('')
    ax4.set_xlabel('')
    ax4.set_ylabel('animal2\ngaze',fontsize=15)
    ax4.tick_params(axis='y', labelsize=20)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.get_xaxis().set_ticks([])
    ax4.get_yaxis().set_ticks([])

    ax5.set_xlim([iframe_min,iframe_max])  
    ax5.set_xticks(np.arange(iframe_min,iframe_max,300)) 
    ax5.set_xticklabels('')
    ax5.set_ylim([0,1])
    ax5.set_yticklabels('')
    ax5.set_xlabel('')
    ax5.set_ylabel('animal2\npull',fontsize=15)
    ax5.tick_params(axis='y', labelsize=20)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    ax5.get_xaxis().set_ticks([])
    ax5.get_yaxis().set_ticks([])

    ax6.set_xlim([start_frame,example_frame])  
    ax6.set_xticks(np.arange(start_frame,example_frame,150)) 
    ax6.set_xticklabels(list(map(str,np.arange(start_frame/fps,example_frame/fps,150/fps))))
    ax6.tick_params(axis='x', labelsize=20)
    ax6.set_ylim([-65*2,0]) # all 64 channels
    ax6.set_yticks(np.arange(-64*2,0,2)) 
    ax6.set_yticklabels(list(map(str,np.arange(-64,0,1))))
    ax6.set_xlabel('time (s)',fontsize = 24)
    ax6.set_ylabel('')
    # ax6.set_title('neural activity trace')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    # ax6.spines['bottom'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    # ax6.get_xaxis().set_ticks([])
    ax6.get_yaxis().set_ticks([])


    # for iframe in np.arange(0,nframes,1):    
    # for iframe in np.arange(iframe_min,iframe_max,1):
    for iframe in np.arange(iframe_max-1,iframe_max,1):

        print("printing frame ",str(iframe+1),"/",str(iframe_max))

        if clear_frames:
            fig.clear()
            fig.subplots_adjust(wspace=0, hspace=0)

            gs=GridSpec(14,4) # 14 rows, 4 columns

            ax1=fig.add_subplot(gs[0:6,:]) # animal tracking frame
            ax2=fig.add_subplot(gs[5,:]) # animal1 gaze
            ax3=fig.add_subplot(gs[6,:]) # animal1 pull
            ax4=fig.add_subplot(gs[7,:]) # animal2 gaze
            ax5=fig.add_subplot(gs[8,:]) # animal2 pull
            ax6=fig.add_subplot(gs[9:13,:]) # recording trace

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

            ax2.set_xlim([iframe_min,iframe_max]) 
            ax2.set_xticks(np.arange(iframe_min,iframe_max,300)) 
            ax2.set_xticklabels('')
            ax2.set_ylim([0,1])
            ax2.set_yticklabels('')
            ax2.set_xlabel('')
            ax2.set_ylabel('animal1\ngaze',fontsize=15)
            ax2.tick_params(axis='y', labelsize=20)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])

            ax3.set_xlim([iframe_min,iframe_max])  
            ax3.set_xticks(np.arange(iframe_min,iframe_max,300)) 
            ax3.set_xticklabels('')
            ax3.set_ylim([0,1])
            ax3.set_yticklabels('')
            ax3.set_xlabel('')
            ax3.set_ylabel('animal1\npull',fontsize=15)
            ax3.tick_params(axis='y', labelsize=20)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.get_xaxis().set_ticks([])
            ax3.get_yaxis().set_ticks([])

            ax4.set_xlim([iframe_min,iframe_max]) 
            ax4.set_xticks(np.arange(iframe_min,iframe_max,300)) 
            ax4.set_xticklabels('')
            ax4.set_ylim([0,1])
            ax4.set_yticklabels('')
            ax4.set_xlabel('')
            ax4.set_ylabel('animal2\ngaze',fontsize=15)
            ax4.tick_params(axis='y', labelsize=20)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.spines['bottom'].set_visible(False)
            ax4.spines['left'].set_visible(False)
            ax4.get_xaxis().set_ticks([])
            ax4.get_yaxis().set_ticks([])

            ax5.set_xlim([iframe_min,iframe_max])  
            ax5.set_xticks(np.arange(iframe_min,iframe_max,300)) 
            ax5.set_xticklabels('')
            ax5.set_ylim([0,1])
            ax5.set_yticklabels('')
            ax5.set_xlabel('')
            ax5.set_ylabel('animal2\npull',fontsize=15)
            ax5.tick_params(axis='y', labelsize=20)
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['bottom'].set_visible(False)
            ax5.spines['left'].set_visible(False)
            ax5.get_xaxis().set_ticks([])
            ax5.get_yaxis().set_ticks([])

            # ax6.set_xlim([iframe_min,iframe_max])  
            # ax6.set_xticks(np.arange(iframe_min,iframe_max,300)) 
            # ax6.set_xlim([0,iframe_max-iframe_min])  
            # ax6.set_xticks(np.arange(0,iframe_max-iframe_min,300)) 
            ax6.set_xlim([start_frame,example_frame])  
            ax6.set_xticks(np.arange(start_frame,example_frame,150)) 
            ax6.set_xticklabels(list(map(str,np.arange(start_frame/fps,example_frame/fps,150/fps))))
            ax6.tick_params(axis='x', labelsize=20)
            ax6.set_ylim([-65*2,0]) # all 64 channels
            ax6.set_yticks(np.arange(-64*2,0,2)) 
            ax6.set_yticklabels(list(map(str,np.arange(-64,0,1))))
            ax6.set_xlabel('time (s)',fontsize = 24)
            ax6.set_ylabel('channels',fontsize=15)
            # ax6.set_title('neural activity trace')
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
            # ax6.spines['bottom'].set_visible(False)
            ax6.spines['left'].set_visible(False)
            # ax6.get_xaxis().set_ticks([])
            ax6.get_yaxis().set_ticks([])

            # plot the original videos
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, iframe)
            ret, image_original = vidcap.read()
            ax1.imshow(cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB))


        for ianimal in np.arange(0,nanimals,1):    
            ianimal_name = animal_names_unique[ianimal] 

            # draw body part tracking history
            bodypart_loc_ibdy_allframes = np.zeros((nbodyparts,iframe_max-iframe_min,2))
            #
            for ibdpart in np.arange(0,nbodyparts,1):           

                ibdpart_name = body_parts_unique[ibdpart]
                bodypart_loc_ibdy_allframes[ibdpart,:,:] = np.array(bodyparts_locs_camN[(ianimal_name,ibdpart_name)])[iframe_min:iframe_max,:]
                # ax1.plot(bodypart_loc_ibdy_allframes[ibdpart,:,0],bodypart_loc_ibdy_allframes[ibdpart,:,1],'-',color=colors[ianimal])
            #
            bodypart_loc_mass_allframes = np.nanmean(bodypart_loc_ibdy_allframes,axis=0)
            ax1.plot(gaussian_filter1d(bodypart_loc_mass_allframes[:,0],3),
                     gaussian_filter1d(bodypart_loc_mass_allframes[:,1],3),
                     '-',color=colors[ianimal],linewidth=10,alpha=0.3)


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
                        # ax2.plot([bhv_events_iframe,bhv_events_iframe],[0.2,0.8],'-',color = colors[np.absolute(ianimal-1)])
                        ax2.plot([bhv_events_iframe,bhv_events_iframe],[0.2,0.8],'-',color = colors[np.absolute(ianimal)])
                    if (np.isin(bhv_events_iframe,pull1_framenum_plot)): 
                        ax3.plot([bhv_events_iframe,bhv_events_iframe],[0.2,0.8],'-',color = 'k')
                    # else:
                    #     ax2.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = '0.5')
                elif (ianimal == 1):
                    if (np.isin(bhv_events_iframe,look_at_other_framenum_plot)): 
                        # ax4.plot([bhv_events_iframe,bhv_events_iframe],[0.2,0.8],'-',color = colors[np.absolute(ianimal-1)])
                        ax4.plot([bhv_events_iframe,bhv_events_iframe],[0.2,0.8],'-',color = colors[np.absolute(ianimal)])
                    if (np.isin(bhv_events_iframe,pull2_framenum_plot)): 
                        ax5.plot([bhv_events_iframe,bhv_events_iframe],[0.2,0.8],'-',color = 'k')
                    # else:
                    #     ax3.plot([bhv_events_iframe,bhv_events_iframe],[0,1],'-',color = '0.5')             


            # plot the neural activity
            # neural activities are already aligned to the start of the sessions, so don't need to take the iframe_min time (session_start_time) out  
            if (ianimal == 0):
                for ichannel_iplot in np.arange(0,64,1):

                    #
                    # get the depth of the ichannel
                    ichannel_depth = channel_to_depth[1,channel_to_depth[0]==ichannel_iplot][0]

                    #
                    # plot LFP
                    # lfp_filt_ichannel = lfp_filt_sess_aligned[ichannel_iplot,iframe_min:iframe+1]
                    # lfp_filt_ichannel = lfp_filt_sess_aligned[ichannel_iplot,0:iframe-iframe_min+1]
                    lfp_filt_ichannel = lfp_filt_sess_aligned[ichannel_iplot,start_frame:example_frame]
                    lfp_filt_ichannel_forplot = lfp_filt_ichannel*4*0.2 + ichannel_depth*2-0.2*2
                    ax6.plot(np.arange(start_frame,example_frame,1),
                             lfp_filt_ichannel_forplot,'-',color = (0.7,0.7,0.7))

                    
                    #
                    # plot the spikes
                    #
                    spike_time_data_ichannel = spike_time_data[spike_channels_data == ichannel_iplot]
                    spike_time_data_ichannel = np.unique(spike_time_data_ichannel)
                    #
                    # spike_time_data_ichannel = spike_time_data_ichannel[spike_time_data_ichannel>iframe_min]
                    # spike_time_data_ichannel = spike_time_data_ichannel[spike_time_data_ichannel<iframe]
                    # spike_time_data_ichannel = spike_time_data_ichannel[spike_time_data_ichannel>0]
                    # spike_time_data_ichannel = spike_time_data_ichannel[spike_time_data_ichannel<iframe-iframe_min]
                    spike_time_data_ichannel = spike_time_data_ichannel[spike_time_data_ichannel>start_frame]
                    spike_time_data_ichannel = spike_time_data_ichannel[spike_time_data_ichannel<example_frame]
                    #  
                    for ispike in spike_time_data_ichannel:
                        ax6.plot([ispike,ispike],[(ichannel_depth-0.7)*2,(ichannel_depth+0.7)*2],'k-')




        # fig.subplots_adjust(wspace=0, hspace=0)
        #plt.show()


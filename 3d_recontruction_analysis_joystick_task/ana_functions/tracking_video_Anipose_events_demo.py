#  function - make demo videos for the body part tracking after 3d reconrstruct

def find_optimal_transform(source_points, target_points):
    import numpy as np
    """
    Finds the optimal rotation and translation between corresponding 3D points using SVD.

    Args:
        source_points (ndarray): Array of shape (n, 3) representing the source points.
        target_points (ndarray): Array of shape (n, 3) representing the target points.

    Returns:
        R (ndarray): 3x3 rotation matrix.
        t (ndarray): 1x3 translation vector.
    """

    # Compute centroids
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Center the points
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    # Compute the cross-covariance matrix
    H = centered_source.T @ centered_target

    # Perform SVD
    U, _, Vt = np.linalg.svd(H)

    # Compute the optimal rotation matrix
    R = Vt.T @ U.T

    # Compute the translation vector
    t = centroid_target - centroid_source.dot(R.T)

    return R, t

def normalize_vector(vector):
    import numpy as np
    """
    Normalize a vector.

    Args:
        vector (ndarray): Input vector.

    Returns:
        ndarray: Normalized vector.
    """
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector  # Avoid division by zero
    return vector / magnitude


def tracking_video_Anipose_events_demo(bodyparts_locs_3d,output_look_ornot,output_allvectors,output_allangles,time_point_pull1,time_point_pull2,
animalnames_videotrack,bodypartnames_videotrack,date_tgt,animal1_filename,animal2_filename,animal1_real,animal2_real,session_start_time,fps,nframes,video_file,withboxCorner):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import scipy
    import string
    import warnings
    import pickle
    import cv2
    import os   
    import matplotlib.animation as animation

    import sys
    sys.path.append('/home/ws523/marmoset_tracking_DLCv2/following_up_analysis/3d_recontruction_analysis_joystick_task/ana_functions')
    from tracking_video_Anipose_events_demo import find_optimal_transform 
    from tracking_video_Anipose_events_demo import normalize_vector

    import warnings
    warnings.filterwarnings("ignore")    

    if withboxCorner:
        skeletons = [ ['rightTuft','rightEye'],
                      ['rightTuft','whiteBlaze'],
                      ['leftTuft','leftEye'],
                      ['leftTuft','whiteBlaze'],
                      ['rightEye','whiteBlaze'],
                      ['leftEye','whiteBlaze'],
                      ['rightEye','mouth'],
                      ['leftEye','mouth'],
                      ['leftEye','rightEye'],
                      #['boxCorner1','boxCorner2'],
                      #['boxCorner2','boxCorner3'],
                      #['boxCorner3','boxCorner4']
                    ]
    else:
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
    

    # Settings
    clear_frames = True     # Should it clear the figure between each frame?
    fps = 30

    # Output video writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Animal tracking demo', artist='Matplotlib', comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # animal_names_unique = pd.unique(pd.DataFrame(bodyparts_locs_3d.keys()).iloc[:,0])
    # body_parts_unique = pd.unique(pd.DataFrame(bodyparts_locs_3d.keys()).iloc[:,1])
    animal_names_unique = animalnames_videotrack
    body_parts_unique = bodypartnames_videotrack    

    nanimals = np.shape(animal_names_unique)[0]  
    nbodyparts = np.shape(body_parts_unique)[0]
    
    for iname in np.arange(0,nanimals,1):
        for ibody in np.arange(0,nbodyparts,1):
            if (iname == 0) & (ibody == 0):
                xxx = np.array(bodyparts_locs_3d[(animal_names_unique[iname],body_parts_unique[ibody])])
            else:
                xxx2 = np.array(bodyparts_locs_3d[(animal_names_unique[iname],body_parts_unique[ibody])])
                xxx = np.concatenate([xxx,xxx2])
    
    if withboxCorner:
        # rotate the x y z axis
        old_axis_x = np.array([1, 0, 0])
        old_axis_y = np.array([0, 1, 0])
        old_axis_z = np.array([0, 0, 1])
        old_axis = np.vstack([old_axis_x,old_axis_y,old_axis_z])

        new_axis_x = np.array(bodyparts_locs_3d[('dodson','boxCorner3')])[0,:]-np.array(bodyparts_locs_3d[('scorch','boxCorner3')])[0,:]
        new_axis_y = np.array(bodyparts_locs_3d[('dodson','boxCorner4')])[0,:]-np.array(bodyparts_locs_3d[('dodson','boxCorner3')])[0,:] 
        new_axis_z = np.array(bodyparts_locs_3d[('dodson','boxCorner1')])[0,:]-np.array(bodyparts_locs_3d[('dodson','boxCorner2')])[0,:] 
        new_axis_x = normalize_vector(new_axis_x)
        new_axis_y = normalize_vector(new_axis_y)
        new_axis_z = normalize_vector(new_axis_z)
        new_axis = np.vstack([new_axis_x,new_axis_y,new_axis_z])

        # calculate the rotation matrix and transition vector
        R,t = find_optimal_transform(new_axis,old_axis)

        # apply the rotation matrix and transition vector
        xxx = np.dot(xxx, R.T)+t

    #
    xyz_min = np.nanmin(xxx,axis=0)
    xyz_max = np.nanmax(xxx,axis=0)


    # align the plot with the session start
    iframe_min = int(np.round(session_start_time*fps))
    iframe_max = nframes+iframe_min   


    # set up the figure setting  
    fig = plt.figure(figsize = (36,16))
    gs=GridSpec(4,8) # 5 rows, 3 columns

    ax1=fig.add_subplot(gs[0:4,0:4],projection='3d') # animal tracking frame
    ax2=fig.add_subplot(gs[0,4:7]) # animal1 behavioral events
    ax3=fig.add_subplot(gs[2,4:7]) # animal2 behavioral events

    #ax1.set_xlim([-10,10])
    #ax1.set_ylim([-10,10])
    #ax1.set_zlim([70,110])
    ax1.set_xlim([xyz_min[0],xyz_max[0]])
    ax1.set_ylim([xyz_min[1],xyz_max[1]])
    ax1.set_zlim([xyz_min[2],xyz_max[2]])
    ax1.tick_params(axis='x', labelsize=20) 
    ax1.tick_params(axis='y', labelsize=20) 
    ax1.tick_params(axis='z', labelsize=20) 
    ax1.set_xlabel('x',fontsize = 24)
    ax1.set_ylabel('y',fontsize = 24)
    ax1.set_zlabel('z',fontsize = 24)
    
    ax2.set_xlim([iframe_min,iframe_max]) 
    ax2.set_xticks(np.arange(iframe_min,iframe_max,300))
    ax2.tick_params(axis='x', labelsize=20) 
    ax2.set_xticklabels('')
    ax2.set_ylim([0,1])
    ax2.set_yticklabels('')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_title('animal 1 behavioral events',fontsize = 26)

    ax3.set_xlim([iframe_min,iframe_max])  
    ax3.set_xticks(np.arange(iframe_min,iframe_max,300))
    ax3.tick_params(axis='x', labelsize=20)
    ax3.set_xticklabels(list(map(str,np.arange(0/fps,nframes/fps,300/fps))))
    ax3.set_ylim([0,1])
    ax3.set_yticklabels('')
    ax3.set_xlabel('time (s)',fontsize = 24)
    ax3.set_ylabel('')
    ax3.set_title('animal 2 behavioral events',fontsize = 26)



    with writer.saving(fig, video_file, 100):
        # for iframe in np.arange(0,nframes,1):    
        for iframe in np.arange(iframe_min,iframe_max,1):
           
            print("printing frame ",str(iframe+1),"/",str(iframe_max))
        
            if clear_frames:
                fig.clear()
                #gs=GridSpec(5,3) # 5 rows, 3 columns

                ax1=fig.add_subplot(gs[0:4,0:4],projection='3d') # animal tracking frame
                ax2=fig.add_subplot(gs[0,4:7]) # animal1 behavioral events
                ax3=fig.add_subplot(gs[2,4:7]) # animal2 behavioral events
                # ax1 = fig.add_subplot(projection='3d')

                #ax1.set_xlim([-10,10])
                #ax1.set_ylim([-10,10])
                #ax1.set_zlim([70,110])
                ax1.set_xlim([xyz_min[0],xyz_max[0]])
                ax1.set_ylim([xyz_min[1],xyz_max[1]])
                ax1.set_zlim([xyz_min[2],xyz_max[2]])
                ax1.tick_params(axis='x', labelsize=20) 
                ax1.tick_params(axis='y', labelsize=20) 
                ax1.tick_params(axis='z', labelsize=20) 
                ax1.set_xlabel('x',fontsize = 24)
                ax1.set_ylabel('y',fontsize = 24)
                ax1.set_zlabel('z',fontsize = 24)
                
                ax2.set_xlim([iframe_min,iframe_max]) 
                ax2.set_xticks(np.arange(iframe_min,iframe_max,300)) 
                ax2.tick_params(axis='x', labelsize=20)
                ax2.set_xticklabels('')
                ax2.set_ylim([0,1])
                ax2.set_yticklabels('')
                ax2.set_xlabel('')
                ax2.set_ylabel('')
                ax2.set_title('animal 1 behavioral events',fontsize = 26)
    
                ax3.set_xlim([iframe_min,iframe_max])  
                ax3.set_xticks(np.arange(iframe_min,iframe_max,300)) 
                ax3.tick_params(axis='x', labelsize=20)
                ax3.set_xticklabels(list(map(str,np.arange(0/fps,nframes/fps,300/fps))))
                ax3.set_ylim([0,1])
                ax3.set_yticklabels('')
                ax3.set_xlabel('time (s)',fontsize = 24)
                ax3.set_ylabel('')
                ax3.set_title('animal 2 behavioral events',fontsize = 26)

            
            for ianimal in np.arange(0,nanimals,1):    
                ianimal_name = animal_names_unique[ianimal] 
                # draw body part
                bodypart_loc_iframe = np.zeros((nbodyparts,3))
                #
                for ibdpart in np.arange(0,nbodyparts,1):           

                    ibdpart_name = body_parts_unique[ibdpart]
                    bodypart_loc_iframe[ibdpart,:] = np.array(bodyparts_locs_3d[(ianimal_name,ibdpart_name)])[iframe,:]

                if withboxCorner:
                    # rotate the x y z axis
                    bodypart_loc_iframe = np.dot(bodypart_loc_iframe, R.T)+t


                # plot the body parts
                if (ianimal==0): 
                    ax1.plot3D(bodypart_loc_iframe[:,0], bodypart_loc_iframe[:,1],bodypart_loc_iframe[:,2], '.', color=colors[ianimal],label ='animal1')
                else:
                    ax1.plot3D(bodypart_loc_iframe[:,0], bodypart_loc_iframe[:,1],bodypart_loc_iframe[:,2], '.', color=colors[ianimal],label ='animal2')
                
                # draw skeleton                
                for iskel in np.arange(0,nskeletons,1):
                    try:
                        iskeleton_name = skeletons[iskel]
                        skelbody12_loc_iframe = np.zeros((2,3))
                        #
                        skel_body1_name = iskeleton_name[0]
                        skel_body2_name = iskeleton_name[1]
                        #
                        skelbody12_loc_iframe[0,:] = np.array(bodyparts_locs_3d[(ianimal_name,skel_body1_name)])[iframe,:]
                        skelbody12_loc_iframe[1,:] = np.array(bodyparts_locs_3d[(ianimal_name,skel_body2_name)])[iframe,:]

                        if withboxCorner:
                            # rotate the x y z axis
                            skelbody12_loc_iframe = np.dot(skelbody12_loc_iframe, R.T)+t

                        # plot one skeleton
                        ax1.plot3D(skelbody12_loc_iframe[:,0],skelbody12_loc_iframe[:,1],skelbody12_loc_iframe[:,2],'-',color=colors[ianimal])
                    except:
                        continue
               
                # draw lever and tube
                lever_loc_iframe = np.array(bodyparts_locs_3d[(ianimal_name,'lever')])[iframe,:]
                if withboxCorner:
                    # rotate the x y z axis
                    lever_loc_iframe = R.dot(lever_loc_iframe)+t
                tube_loc_iframe = np.array(bodyparts_locs_3d[(ianimal_name,'tube')])[iframe,:]
                if withboxCorner:
                    # rotate the x y z axis
                    tube_loc_iframe = R.dot(tube_loc_iframe)+t
                if (ianimal==1): 
                    ax1.plot3D(lever_loc_iframe[0],lever_loc_iframe[1],lever_loc_iframe[2], 'o', color='g',label ='lever')
                    ax1.plot3D(tube_loc_iframe[0], tube_loc_iframe[1], tube_loc_iframe[2], 'o', color='y',label ='tube')
                else:
                    ax1.plot3D(lever_loc_iframe[0],lever_loc_iframe[1],lever_loc_iframe[2], 'o', color='g')
                    ax1.plot3D(tube_loc_iframe[0], tube_loc_iframe[1], tube_loc_iframe[2], 'o', color='y')
                
           
                # draw the eye direction 
                rightEye_loc_iframe = np.array(bodyparts_locs_3d[(ianimal_name,'rightEye')])[iframe,:]
                leftEye_loc_iframe = np.array(bodyparts_locs_3d[(ianimal_name,'leftEye')])[iframe,:]              
                gaze_dir_iframe = np.array(output_allvectors['eye_direction_Anipose'][ianimal_name])[iframe,:]
                if withboxCorner:
                    # rotate the x y z axis
                    gaze_dir_iframe = R.dot(gaze_dir_iframe)+t
                    rightEye_loc_iframe = R.dot(rightEye_loc_iframe)+t
                    leftEye_loc_iframe = R.dot(leftEye_loc_iframe)+t
                meaneye_loc_iframe = np.nanmean(np.vstack([rightEye_loc_iframe,leftEye_loc_iframe]),axis=0)
                gaze_dir_iframe = meaneye_loc_iframe + 4*gaze_dir_iframe


                if (ianimal==1):
                    ax1.plot3D([meaneye_loc_iframe[0],gaze_dir_iframe[0]],[meaneye_loc_iframe[1],gaze_dir_iframe[1]],[meaneye_loc_iframe[2],gaze_dir_iframe[2]],'-',color = '0.25',label='head gaze dir')
                else:
                    ax1.plot3D([meaneye_loc_iframe[0],gaze_dir_iframe[0]],[meaneye_loc_iframe[1],gaze_dir_iframe[1]],[meaneye_loc_iframe[2],gaze_dir_iframe[2]],'-',color = '0.25')     


                # ax1.legend(loc='upper right',fontsize = 28)
                         


                # draw animal behavioral events
                # look_at_other_framenum_all = np.where(np.array(output_look_ornot["look_at_face_or_not_Anipose"][ianimal_name])==1)[0]
                look_at_other_framenum_all = np.where((np.array(output_look_ornot["look_at_face_or_not_Anipose"][ianimal_name])==1)|(np.array(output_look_ornot["look_at_otherlever_or_not_Anipose"][ianimal_name])==1)|(np.array(output_look_ornot["look_at_othertube_or_not_Anipose"][ianimal_name])==1))[0]
                look_at_other_framenum_plot = look_at_other_framenum_all[(look_at_other_framenum_all<=iframe)&(look_at_other_framenum_all>iframe_min)]
                look_at_lever_framenum_all = np.where(np.array(output_look_ornot["look_at_selflever_or_not_Anipose"][ianimal_name])==1)[0]
                look_at_lever_framenum_plot = look_at_lever_framenum_all[(look_at_lever_framenum_all<=iframe)&(look_at_lever_framenum_all>iframe_min)]
                look_at_tube_framenum_all = np.where(np.array(output_look_ornot["look_at_selftube_or_not_Anipose"][ianimal_name])==1)[0]
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

#  function - make demo videos for the body part tracking
def tracking_video_demo(body_part_locs,date_tgt,camerapair,nframes):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

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
    video_file = "example_videos_demo/"+date_tgt+"_"+camerapair+"_tracking_demo.mp4"
    clear_frames = True     # Should it clear the figure between each frame?
    fps = 30

    # Output video writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Animal tracking demo', artist='Matplotlib', comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs.keys()).iloc[:,0])
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs.keys()).iloc[:,1])
    nanimals = np.shape(animal_names_unique)[0]  
    nbodyparts = np.shape(body_parts_unique)[0]

    # axislim = [np.nanmin(list(body_part_locs.values())), np.nanmax(list(body_part_locs.values()))]
    # axislim = [-25,25]
    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs.keys()).iloc[:,0])
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs.keys()).iloc[:,1])
    nanimals = np.shape(animal_names_unique)[0]  
    nbodyparts = np.shape(body_parts_unique)[0]
    for iname in np.arange(0,nanimals,1):
        for ibody in np.arange(0,nbodyparts,1):
            if (iname == 0) & (ibody == 0):
                xxx = np.array(body_part_locs[(animal_names_unique[iname],body_parts_unique[ibody])])
            else:
                xxx2 = np.array(body_part_locs[(animal_names_unique[iname],body_parts_unique[ibody])])
                xxx = np.concatenate([xxx,xxx2])
    xyz_min = np.nanmin(xxx,axis=0)
    xyz_max = np.nanmax(xxx,axis=0)
    
    
    fig = plt.figure(figsize = (5,5))
    ax = plt.axes(projection='3d')
    # ax.set_xlim(axislim)
    # ax.set_ylim(axislim)
    # ax.set_zlim(axislim)
    ax.set_xlim([xyz_min[0],xyz_max[0]])
    ax.set_ylim([xyz_min[1],xyz_max[1]])
    ax.set_zlim([xyz_min[2],xyz_max[2]])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    with writer.saving(fig, video_file, 100):
        for iframe in np.arange(0,nframes,1):    
            if clear_frames:
                fig.clear()
                ax = plt.axes(projection='3d')
                # ax.set_xlim(axislim)
                # ax.set_ylim(axislim)
                # ax.set_zlim(axislim)
                ax.set_xlim([xyz_min[0],xyz_max[0]])
                ax.set_ylim([xyz_min[1],xyz_max[1]])
                ax.set_zlim([xyz_min[2],xyz_max[2]])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
            
            for ianimal in np.arange(0,nanimals,1):    
                ianimal_name = animal_names_unique[ianimal] 
                # draw body part
                bodypart_loc_iframe = np.zeros((nbodyparts,3))
                #
                for ibdpart in np.arange(0,nbodyparts,1):           

                    ibdpart_name = body_parts_unique[ibdpart]
                    bodypart_loc_iframe[ibdpart,:] = np.array(body_part_locs[(ianimal_name,ibdpart_name)])[iframe,:]
                # plot the body parts
                ax.plot3D(bodypart_loc_iframe[:,0], bodypart_loc_iframe[:,1], bodypart_loc_iframe[:,2], '.', color=colors[ianimal])
            
                # draw skeleton                
                for iskel in np.arange(0,nskeletons,1):
                    try:
                        iskeleton_name = skeletons[iskel]
                        skelbody12_loc_iframe = np.zeros((2,3))
                        #
                        skel_body1_name = iskeleton_name[0]
                        skel_body2_name = iskeleton_name[1]
                        #
                        skelbody12_loc_iframe[0,:] = np.array(body_part_locs[(ianimal_name,skel_body1_name)])[iframe,:]
                        skelbody12_loc_iframe[1,:] = np.array(body_part_locs[(ianimal_name,skel_body2_name)])[iframe,:]
                        # plot one skeleton
                        ax.plot3D(skelbody12_loc_iframe[:,0],skelbody12_loc_iframe[:,1],skelbody12_loc_iframe[:,2],'-',color=colors[ianimal])
                    except:
                        continue
            
            writer.grab_frame()            

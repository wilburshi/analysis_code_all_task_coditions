#  function - compare the body track result from different camera pairs
def compare_bodytracking_allcamerapairs(body_part_locs_camera12, body_part_locs_camera23, animal1_fixedorder, animal2_fixedorder, date_tgt, saveornot):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

# plot and compare the body track results from different camera pairs
    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs_camera23.keys()).iloc[:,0])
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs_camera23.keys()).iloc[:,1])
    nbodies = np.shape(body_parts_unique)[0]

    for iname in animal_names_unique:
        fig, axs = plt.subplots(nbodies+1,3,figsize=(15, 30))


        for ibody in np.arange(0,nbodies,1):
            ibodyname = body_parts_unique[ibody]
            xxx = np.array(body_part_locs_camera12[(iname,ibodyname)])
            yyy = np.array(body_part_locs_camera23[(iname,ibodyname)])

            if ibody == 0:
                mess_center_x_12 = xxx[:,0]
                mess_center_y_12 = xxx[:,1]
                mess_center_z_12 = xxx[:,2]
                mess_center_x_23 = yyy[:,0]
                mess_center_y_23 = yyy[:,1]
                mess_center_z_23 = yyy[:,2]
            else:
                mess_center_x_12 = np.vstack((mess_center_x_12,xxx[:,0]))
                mess_center_y_12 = np.vstack((mess_center_y_12,xxx[:,1]))
                mess_center_z_12 = np.vstack((mess_center_z_12,xxx[:,2]))
                mess_center_x_23 = np.vstack((mess_center_x_23,yyy[:,0]))
                mess_center_y_23 = np.vstack((mess_center_y_23,yyy[:,1]))
                mess_center_z_23 = np.vstack((mess_center_z_23,yyy[:,2]))

            axs[ibody,0].plot(xxx[:,0])
            axs[ibody,0].plot(yyy[:,0])
            axs[ibody,0].set_title(iname+" "+ibodyname+" "+"x")
            axs[ibody,0].set_ylabel('position')
            axs[ibody,1].plot(xxx[:,1])
            axs[ibody,1].plot(yyy[:,1])
            axs[ibody,1].set_title(iname+" "+ibodyname+" "+"y")
            axs[ibody,1].set_ylabel('position')
            axs[ibody,2].plot(xxx[:,2])
            axs[ibody,2].plot(yyy[:,2])
            axs[ibody,2].set_title(iname+" "+ibodyname+" "+"z")
            axs[ibody,2].set_ylabel('position')
            axs[ibody,2].legend(['camera12','camera23'])

        axs[ibody+1,0].plot(np.nanmean(mess_center_x_12,axis=0))
        axs[ibody+1,0].plot(np.nanmean(mess_center_x_23,axis=0))
        axs[ibody+1,0].set_title(iname+" mean mess center x")
        axs[ibody+1,0].set_ylabel('position')    
        axs[ibody+1,0].set_xlabel('frame') 
        axs[ibody+1,1].plot(np.nanmean(mess_center_y_12,axis=0))
        axs[ibody+1,1].plot(np.nanmean(mess_center_y_23,axis=0))
        axs[ibody+1,1].set_title(iname+" mean mess center y")
        axs[ibody+1,1].set_ylabel('position')
        axs[ibody+1,1].set_xlabel('frame')
        axs[ibody+1,2].plot(np.nanmean(mess_center_z_12,axis=0))
        axs[ibody+1,2].plot(np.nanmean(mess_center_z_23,axis=0))
        axs[ibody+1,2].set_title(iname+" mean mess center z")
        axs[ibody+1,2].set_ylabel('position')
        axs[ibody+1,2].set_xlabel('frame')
        axs[ibody+1,2].legend(['camera12','camera23'])
        
        if saveornot:
            plt.savefig("bodypart_camerapair_comparison_"+animal1_fixedorder+animal2_fixedorder+"/"+date_tgt+"_"+iname+".pdf")
        

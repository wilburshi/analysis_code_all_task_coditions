#  function - compare the body track result from different camera pairs
def bodytracking_singlecameras(body_part_locs_camera12,body_part_locs_camera23,bodyparts_cam1_cam12,bodyparts_cam2_cam12,bodyparts_cam2_cam23,bodyparts_cam3_cam23,animal1_fixedorder,animal2_fixedorder,date_tgt,saveornot):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

# load cam1 h5 file
    try:
        bodyparts_cam1_cam12_data = pd.read_hdf(bodyparts_cam1_cam12)
    except:
        try:
            bodyparts_cam2_cam12_data = pd.read_hdf(bodyparts_cam2_cam12)
        except:
            try:
                bodyparts_cam2_cam23_data = pd.read_hdf(bodyparts_cam2_cam23)
            except:
                try:
                    bodyparts_cam3_cam23_data = pd.read_hdf(bodyparts_cam3_cam23)
                except:
                    return 0


# plot and compare the body track results from different camera pairs
    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs_camera12.keys()).iloc[:,0])
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs_camera12.keys()).iloc[:,1])
    nbodies = np.shape(body_parts_unique)[0]

    
    for iname in animal_names_unique:
        fig, axs = plt.subplots(nbodies,4,figsize=(15, 30))

        for ibody in np.arange(0,nbodies,1):
            ibodyname = body_parts_unique[ibody]
            try:
                xxx = np.array(bodyparts_cam1_cam12_data[('DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000', iname, ibodyname, 'x')])
                yyy = np.array(bodyparts_cam1_cam12_data[('DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000', iname, ibodyname, 'y')])
                axs[ibody,0].plot(xxx[:],'.',markersize=3)
                axs[ibody,0].plot(yyy[:],'.',markersize=3)
                axs[ibody,0].legend(['x','y'])
                axs[ibody,0].set_title(iname+" "+ibodyname+" cam1 in cam12")
            except:
                try:
                    xxx = np.array(bodyparts_cam2_cam12_data[('DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000', iname, ibodyname, 'x')])
                    yyy = np.array(bodyparts_cam2_cam12_data[('DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000', iname, ibodyname, 'y')])
                    axs[ibody,1].plot(xxx[:],'.',markersize=3)
                    axs[ibody,1].plot(yyy[:],'.',markersize=3)
                    axs[ibody,1].legend(['x','y'])
                    axs[ibody,1].set_title(iname+" "+ibodyname+" cam2 in cam12")
                except:
                    try:
                        xxx = np.array(bodyparts_cam2_cam23_data[('DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000', iname, ibodyname, 'x')])
                        yyy = np.array(bodyparts_cam2_cam23_data[('DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000', iname, ibodyname, 'y')])
                        axs[ibody,2].plot(xxx[:],'.',markersize=3)
                        axs[ibody,2].plot(yyy[:],'.',markersize=3)
                        axs[ibody,2].legend(['x','y'])
                        axs[ibody,2].set_title(iname+" "+ibodyname+" cam2 in cam23")
                    except:
                        try:
                            xxx = np.array(bodyparts_cam3_cam23_data[('DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000', iname, ibodyname, 'x')])
                            yyy = np.array(bodyparts_cam3_cam23_data[('DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000', iname, ibodyname, 'y')])
                            axs[ibody,3].plot(xxx[:],'.',markersize=3)
                            axs[ibody,3].plot(yyy[:],'.',markersize=3)
                            axs[ibody,3].legend(['x','y'])
                            axs[ibody,3].set_title(iname+" "+ibodyname+" cam3 in cam23")
                        except:
                            continue

        if saveornot:
            plt.savefig("bodypart_singlecamera_comparison_"+animal1_fixedorder+animal2_fixedorder+"/"+date_tgt+"_"+iname+".pdf")



        

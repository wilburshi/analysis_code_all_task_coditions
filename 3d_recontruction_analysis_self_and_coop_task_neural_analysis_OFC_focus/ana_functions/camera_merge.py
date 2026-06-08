#  function - merge the two pairs of cameras
def camera_merge(body_part_locs_camera23, body_part_locs_camera12, RR_sum, tt_sum, err_sum):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

# merge the two pairs of cameras
    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs_camera23.keys()).iloc[:,0])
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs_camera23.keys()).iloc[:,1])
    
    body_part_locs_merge = {}
    for iname in animal_names_unique:
        for ibody in body_parts_unique:
            xxx = body_part_locs_camera23[(iname,ibody)]
            yyy = body_part_locs_camera12[(iname,ibody)]
            min_length = np.min([xxx.shape[0],yyy.shape[0]])
            
            ## transpose the two pair of cameras
        
            # RR = RR_sum[min(err_sum, key=err_sum.get)]
            # tt = tt_sum[min(err_sum, key=err_sum.get)]
            RR = RR_sum[(iname,ibody)]
            tt = tt_sum[(iname,ibody)]
            body_part_x = np.transpose(xxx.loc[np.arange(0,min_length,1),:])
            body_part_project = np.transpose(np.dot(RR,body_part_x) + np.dot(tt, np.ones((1,np.shape(body_part_x)[1]))))
            body_part_origin = yyy.loc[np.arange(0,min_length,1),:].values
            body_part_origin[np.sum(np.isnan(body_part_origin),axis=1)>0,:] = body_part_project[np.sum(np.isnan(body_part_origin),axis=1)>0,:]
            #
            # body_part_locs_merge[(iname,ibody)] = body_part_origin
    
            ## average across the two pair of cameras
            body_part_locs_merge[(iname,ibody)] = pd.concat([xxx, yyy]).groupby(level=0).mean()          
    
    return body_part_locs_merge

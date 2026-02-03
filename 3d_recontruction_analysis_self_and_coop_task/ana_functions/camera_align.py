#  function - align the two cameras
def camera_align(body_part_locs_camera23, body_part_locs_camera12):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    # try the rotation on all possible pairs
    animal_names_unique = pd.unique(pd.DataFrame(body_part_locs_camera23.keys()).iloc[:,0])
    body_parts_unique = pd.unique(pd.DataFrame(body_part_locs_camera23.keys()).iloc[:,1])
    
    RR_sum = {}
    tt_sum = {}
    err_sum = {}
    for iname in animal_names_unique:
        for ibody in body_parts_unique:
            xxx = body_part_locs_camera23[(iname,ibody)]
            yyy = body_part_locs_camera12[(iname,ibody)]
            min_length = np.min([xxx.shape[0],yyy.shape[0]])
        
            xxx = xxx.loc[np.arange(0,min_length,1),:]
            yyy = yyy.loc[np.arange(0,min_length,1),:]     
        
            ind_good = (~np.isnan(xxx.iloc[:,0]) & ~np.isnan(xxx.iloc[:,1]) & ~np.isnan(xxx.iloc[:,2])) & (~np.isnan(yyy.iloc[:,0]) & ~np.isnan(yyy.iloc[:,1]) & ~np.isnan(yyy.iloc[:,2])) 
            xxx_values = pd.DataFrame.transpose(xxx.loc[ind_good,:]).values
            yyy_values = pd.DataFrame.transpose(yyy.loc[ind_good,:]).values
        
            xxx_centroid = np.dot(np.mean(xxx_values,axis = 1).reshape(3,1), np.ones((1,np.shape(xxx_values)[1])))
            yyy_centroid = np.dot(np.mean(yyy_values,axis = 1).reshape(3,1), np.ones((1,np.shape(xxx_values)[1])))
            HH = np.dot((xxx_values - xxx_centroid), np.transpose(yyy_values - yyy_centroid))
            u, s, vh = np.linalg.svd(HH, full_matrices=True)
            RR = np.dot(np.transpose(vh),np.transpose(u))
            tt= yyy_centroid - np.dot(RR,xxx_centroid)
            tt = tt[:,1].reshape(3,1)
        
            RR_sum[(iname,ibody)] = RR
            tt_sum[(iname,ibody)] = tt
            err_sum[(iname,ibody)] = np.sum(np.square(yyy_values - (np.dot(RR,xxx_values)+np.dot(tt, np.ones((1,np.shape(xxx_values)[1]))))))
    
    return RR_sum, tt_sum, err_sum

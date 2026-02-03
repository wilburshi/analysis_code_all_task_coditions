# function - get body part location for each pair of cameras
def body_part_locs_eachpair(cameraAB_h5_data):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    # clean the data first
    ncols = cameraAB_h5_data.shape[1]
    nframes = cameraAB_h5_data.shape[0]
    animal_names = []
    body_parts = []
    xyz_axis = []

    for i in np.arange(0,ncols,1):
        animal_names.append(cameraAB_h5_data.columns[i][1])
        body_parts.append(cameraAB_h5_data.columns[i][2])
        xyz_axis.append(cameraAB_h5_data.columns[i][3])
  
        # fill in the nan data point
        data_point = cameraAB_h5_data.iloc[:,i]
        # data_point_filled = data_point.interpolate(method='nearest',limit_direction='both')
        # data_point_filled = data_point_filled.interpolate(method='linear',limit_direction='both')
        data_point_filled = data_point
        
        # remove outlier
        q1 = np.nanquantile(data_point_filled,0.25)
        q3 = np.nanquantile(data_point_filled,0.75)
        thres1 = q1 - 1.5*abs(q3-q1)
        thres2 = q3 + 1.5*abs(q3-q1)
        ind = (data_point_filled>thres2) | (data_point_filled<thres1)
        data_point_filled[ind] = np.nan
        
        # fill in the nan data point after outlier removal
        # data_point_filled = data_point_filled.interpolate(method='linear',limit_direction='both')
        
        # smooth the data point   
        # std was chosen based on the data (mean std across session, animal, body part and x,y,z is ~2)
        # data_point_filtered = data_point_filled.rolling(window=10, win_type='gaussian', center=True).mean(std=2)
        #
        # cameraAB_h5_data.iloc[:,i] = data_point_filtered
        # cameraAB_h5_data.iloc[:,i] = data_point_filled
        cameraAB_h5_data.iloc[:,i] = data_point
    
    animal_names_unique = pd.unique(animal_names)
    body_parts_unique = pd.unique(body_parts)
    xyz_axis_unique = pd.unique(xyz_axis)

    #
    # find the location of each body part
    body_part_locs = {}
    for iname in animal_names_unique:
        for ibody in body_parts_unique:
            ind = np.isin(animal_names,iname) & np.isin(body_parts,ibody)
            body_part_locs[(iname,ibody)] = cameraAB_h5_data.iloc[:,ind]            
            
            # # remove position based on z
            # ind_badz = (body_part_locs[(iname,ibody)].iloc[:,2]>30)|(body_part_locs[(iname,ibody)].iloc[:,2]<-30) 
            # body_part_locs[(iname,ibody)].loc[ind_badz,:] = np.nan
            
    return body_part_locs

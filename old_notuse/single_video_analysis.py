import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

h5file_one_camera = "/ysm-gpfs/pi/jadi/VideoTracker_SocialInter/test_video_3d/20220915_Dodson_Scorch_camera12/20220915_Dodson_Scorch_1000frames_camera-1DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000_el_filtered_old.h5"
h5file_one_camera_save = "/ysm-gpfs/pi/jadi/VideoTracker_SocialInter/test_video_3d/20220915_Dodson_Scorch_camera12/20220915_Dodson_Scorch_1000frames_camera-1DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000_el_filtered.h5"
h5file_two_camera = "/ysm-gpfs/pi/jadi/VideoTracker_SocialInter/test_video_3d/20220912_Dodson_Scorch_camera23/20220912_Dodson_Scorch_1000frames_weikang.h5"

data_one_camera = pd.read_hdf(h5file_one_camera)  
data_two_camera = pd.read_hdf(h5file_two_camera)

# one camera data
ncols = data_one_camera.columns.shape[0]
animal_names = []
body_parts = []
xy_likelihoods = []

for i in np.arange(0,ncols,1):
  animal_names.append(data_one_camera.columns[i][1])
  body_parts.append(data_one_camera.columns[i][2])
  xy_likelihoods.append(data_one_camera.columns[i][3])
  # if (data_one_camera.columns[i][3] != 'likelihood'):
     # fill in the nan data point
  data_point = data_one_camera.iloc[:,i]
  data_point_filled = data_point.interpolate(method='nearest',limit_direction='both'
  data_point_filled = data_point_filled.interpolate(method='linear',limit_direction='both')
  # smooth the data point   
  # data_point_filtered = data_point_filled.rolling(window=5, win_type='gaussian', center=True).mean(std=0.5)
  #
  data_one_camera.iloc[:,i] = data_point_filled

data_one_camera.to_hdf(h5file_one_camera_save, key = "data_one_camera")




  
      
      

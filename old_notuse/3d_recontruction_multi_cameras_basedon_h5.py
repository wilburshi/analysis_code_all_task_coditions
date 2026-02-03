import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 3d config.yaml file path
camera12_config_path = "/home/ws523/marmoset_tracking_DLCv2/marmoset_tracking_middle_camera_new_pos2_camera12-weikang-2022-09-15-3d/"
camera13_config_path = "/home/ws523/marmoset_tracking_DLCv2/marmoset_tracking_middle_camera_new_pos2_camera13-weikang-2022-09-15-3d/" 
camera23_config_path = "/home/ws523/marmoset_tracking_DLCv2/marmoset_tracking_middle_camera_new_pos2_camera23-weikang-2022-09-15-3d/" 

# 3d analyzed video path
camera12_analyzed_path = "/ysm-gpfs/pi/jadi/VideoTracker_SocialInter/test_video_3d/20220915_Dodson_Scorch_camera12/"
camera13_analyzed_path = "/ysm-gpfs/pi/jadi/VideoTracker_SocialInter/test_video_3d/20220915_Dodson_Scorch_camera13/"
camera23_analyzed_path = "/ysm-gpfs/pi/jadi/VideoTracker_SocialInter/test_video_3d/20220915_Dodson_Scorch_camera23/"
      
# h5 files for the analyzed videos
camera12_h5_file = camera12_analyzed_path + "20220915_Dodson_Scorch_1000frames_weikang.h5"
camera13_h5_file = camera13_analyzed_path + "20220915_Dodson_Scorch_1000frames_weikang.h5"
camera23_h5_file = camera23_analyzed_path + "20220915_Dodson_Scorch_1000frames_weikang.h5"

# meta pickle data for the analyzed videos
camera12_metapickle_file = camera12_analyzed_path + "20220915_Dodson_Scorch_1000frames_weikang_meta.pickle"
camera13_metapickle_file = camera13_analyzed_path + "20220915_Dodson_Scorch_1000frames_weikang_meta.pickle"
camera23_metapickle_file = camera23_analyzed_path + "20220915_Dodson_Scorch_1000frames_weikang_meta.pickle"

# load data
camera12_metapickle_data = pd.read_pickle(camera12_metapickle_file)
camera13_metapickle_data = pd.read_pickle(camera13_metapickle_file)
camera23_metapickle_data = pd.read_pickle(camera23_metapickle_file)

camera12_h5_data = pd.read_hdf(camera12_h5_file)
camera13_h5_data = pd.read_hdf(camera13_h5_file)
camera23_h5_data = pd.read_hdf(camera23_h5_file)






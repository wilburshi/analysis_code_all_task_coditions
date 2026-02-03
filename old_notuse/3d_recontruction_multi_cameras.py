import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

camera12_params_file = "/home/ws523/marmoset_tracking_DLCv2/marmoset_tracking_middle_camera_new_pos2_camera12-weikang-2022-09-15-3d/"
camera23_params_file = "/home/ws523/marmoset_tracking_DLCv2/marmoset_tracking_middle_camera_new_pos2_camera23-weikang-2022-09-15-3d/" 
      
cam1_param_with12 = camera12_params_file + "camera_matrix/camera-1_intrinsic_params.pickle"     
cam2_param_with12 = camera12_params_file + "camera_matrix/camera-2_intrinsic_params.pickle"     
stereo_param_with12 = camera12_params_file + "camera_matrix/stereo_params.pickle"     
triangulate_with12 = camera12_params_file + "triangulate.pickle"

cam2_param_with23 = camera23_params_file + "camera_matrix/camera-2_intrinsic_params.pickle"     
cam3_param_with23 = camera23_params_file + "camera_matrix/camera-3_intrinsic_params.pickle"     
stereo_param_with23 = camera23_params_file + "camera_matrix/stereo_params.pickle"    
triangulate_with23 = camera23_params_file + "triangulate.pickle"

data_cam1_param_with12 = pd.read_pickle(cam1_param_with12)
data_cam2_param_with12 = pd.read_pickle(cam2_param_with12)
data_stereo_param_with12 = pd.read_pickle(stereo_param_with12)
data_triangulate_with12 = pd.read_pickle(triangulate_with12)

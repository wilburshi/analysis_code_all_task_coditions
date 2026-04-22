#!/usr/bin/env python
# coding: utf-8

# ### Basic neural activity analysis with single camera tracking
# #### use GLM model to analyze spike count trains, based on discrete behavioral variables

# In[1]:


import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import scipy
import scipy.stats as st
import scipy.io
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import string
import warnings
import pickle
import json

import statsmodels.api as sm

import os
import glob
import random
from time import time


# ### function - get body part location for each pair of cameras

# In[2]:


from ana_functions.body_part_locs_eachpair import body_part_locs_eachpair
from ana_functions.body_part_locs_singlecam import body_part_locs_singlecam


# ### function - align the two cameras

# In[3]:


from ana_functions.camera_align import camera_align       


# ### function - merge the two pairs of cameras

# In[4]:


from ana_functions.camera_merge import camera_merge


# ### function - find social gaze time point

# In[5]:


from ana_functions.find_socialgaze_timepoint import find_socialgaze_timepoint
from ana_functions.find_socialgaze_timepoint_singlecam import find_socialgaze_timepoint_singlecam
from ana_functions.find_socialgaze_timepoint_singlecam_wholebody import find_socialgaze_timepoint_singlecam_wholebody


# ### function - define time point of behavioral events

# In[6]:


from ana_functions.bhv_events_timepoint import bhv_events_timepoint
from ana_functions.bhv_events_timepoint_singlecam import bhv_events_timepoint_singlecam


# ### function - plot behavioral events

# In[7]:


from ana_functions.plot_bhv_events import plot_bhv_events
from ana_functions.plot_bhv_events_levertube import plot_bhv_events_levertube
from ana_functions.draw_self_loop import draw_self_loop
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection


# ### function - make force quantiles

# In[8]:


from ana_functions.make_force_quantiles import make_force_quantiles


# ### function - interval between all behavioral events

# In[9]:


from ana_functions.bhv_events_interval import bhv_events_interval


# ### function - GLM fitting for spike trains based on the discrete variables from single camera

# In[10]:


from ana_functions.singlecam_bhv_var_neuralGLM_fitting import get_singlecam_bhv_var_for_neuralGLM_fitting
from ana_functions.singlecam_bhv_var_neuralGLM_fitting import neuralGLM_fitting


# ## Analyze each session

# ### prepare the basic behavioral data (especially the time stamps for each bhv events)

# In[25]:


# instead of using gaze angle threshold, use the target rectagon to deside gaze info
# ...need to update
sqr_thres_tubelever = 75 # draw the square around tube and lever
sqr_thres_face = 1.15 # a ratio for defining face boundary
sqr_thres_body = 4 # how many times to enlongate the face box boundry to the body


# get the fps of the analyzed video
fps = 30

# get the fs for neural recording
fs_spikes = 20000
fs_lfp = 1000

# frame number of the demo video
# nframes = 0.5*30 # second*30fps
nframes = 45*30 # second*30fps

# re-analyze the video or not
reanalyze_video = 0
redo_anystep = 1

# do OFC sessions or DLPFC sessions
do_OFC = 0
do_DLPFC  = 1
if do_OFC:
    savefile_sufix = '_OFCs'
elif do_DLPFC:
    savefile_sufix = '_DLPFCs'
else:
    savefile_sufix = ''
    
# all the videos (no misaligned ones)
# aligned with the audio
# get the session start time from "videosound_bhv_sync.py/.ipynb"
# currently the session_start_time will be manually typed in. It can be updated after a better method is used

# dannon kanga
if 1:
    if do_DLPFC:
        neural_record_conditions = [
                                    '20240910_Kanga_EffortBasedMC',
                                    '20240911_Kanga_EffortBasedMC',
                                    '20240912_Kanga_EffortBasedSR',
                                    '20240913_Kanga_EffortBasedSR',
                                    '20240916_Kanga_EffortBasedMC',
                                    '20240917_Kanga_EffortBasedSR',
                                    '20240918_Kanga_EffortBasedMC',
                                    '20241008_Kanga_EffortBasedMC',
                                    '20241009_Kanga_DannonEffortBasedMC',
                                    '20241010_Kanga_EffortBasedMC',
                                    '20241011_Kanga_DannonEffortBasedMC',
                                    '20241014_Kanga_EffortBasedMC',
                                    '20241016_Kanga_DannonEffortBasedMC',
                                    '20241017_Kanga_EffortBasedMC',
                                    '20241018_Kanga_DannonEffortBasedMC',
                                    '20241022_Kanga_DannonEffortBasedMC',
                                    '20241025_Kanga_DannonEffortBasedMC',
                                    '20241101_Kanga_EffortBasedSR',
                                    '20241104_Kanga_EffortBasedSR',
                                   ]
        task_conditions = [
                            'self_EffortBasedMC',
                            'self_EffortBasedMC',
                            'self_EffortBasedSR',
                            'self_EffortBasedSR',
                            'self_EffortBasedMC',
                            'self_EffortBasedSR',
                            'self_EffortBasedMC',
                            'self_EffortBasedMC',
                            'other_EffortBasedMC',
                            'self_EffortBasedMC',
                            'other_EffortBasedMC',
                            'self_EffortBasedMC',
                            'other_EffortBasedMC',
                            'self_EffortBasedMC',
                            'other_EffortBasedMC',
                            'other_EffortBasedMC',
                            'other_EffortBasedMC',
                            'self_EffortBasedSR',
                            'self_EffortBasedSR',
                          ]
        dates_list = [
                        '20240910',
                        '20240911',
                        '20240912',
                        '20240913',
                        '20240916',
                        '20240917',
                        '20240918',
                        '20241008',
                        '20241009',
                        '20241010',
                        '20241011',
                        '20241014',
                        '20241016',
                        '20241017',
                        '20241018',
                        '20241022',
                        '20241025',
                        '20241101',
                        '20241104',
                     ]
        videodates_list = [
                            '20240910',
                            '20240911',
                            '20240912',
                            '20240913',
                            '20240916',
                            '20240917',
                            '20240918',
                            '20241008',
                            '20241009',
                            '20241010',
                            '20241011',
                            '20241014',
                            '20241016',
                            '20241017',
                            '20241018',
                            '20241022',
                            '20241025',
                            '20241101',
                            '20241104',
                          ] # to deal with the sessions that MC and SR were in the same session
        session_start_times = [ 
                                0.00,
                                0.00,
                                90.1,
                                69.5,
                                62.5,
                                0.00,
                                43.5,
                                59.6,
                                0.00,
                                66.0,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
            
                              ] # in second
        kilosortvers = [ 
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                         4,
                       ]
        animal1_fixedorders = ['dannon']*np.shape(dates_list)[0]
        animal2_fixedorders = ['kanga']*np.shape(dates_list)[0]

        animal1_filenames = ["Dannon"]*np.shape(dates_list)[0]
        animal2_filenames = ["Kanga"]*np.shape(dates_list)[0]
        
    elif do_OFC:
        neural_record_conditions = [
                                
                                   ]
        task_conditions = [
                            
                          ]
        dates_list = [
                      
                     ]
        videodates_list = dates_list
        session_start_times = [ 
                                
                              ] # in second
        kilosortvers = [ 
                       
                       ]
    
        animal1_fixedorders = ['dannon']*np.shape(dates_list)[0]
        animal2_fixedorders = ['kanga']*np.shape(dates_list)[0]

        animal1_filenames = ["Dannon"]*np.shape(dates_list)[0]
        animal2_filenames = ["Kanga"]*np.shape(dates_list)[0]

    
#
    
# a test case
if 0:
    neural_record_conditions = ['20240910_Kanga_EffortBasedMC',]
    dates_list = ['20240910',]
    videodates_list = dates_list
    task_conditions = ['self_EffortBasedMC',]
    session_start_times = [0.00] # in second
    kilosortvers = [4]
    animal1_fixedorders = ['dannon']
    animal2_fixedorders = ['kanga']
    animal1_filenames = ["Dannon"]
    animal2_filenames = ["Kanga"]

    
ndates = np.shape(dates_list)[0]

session_start_frames = session_start_times * fps # fps is 30Hz

# video tracking results info
animalnames_videotrack = ['dodson','scorch'] # does not really mean dodson and scorch, instead, indicate animal1 and animal2
bodypartnames_videotrack = ['rightTuft','whiteBlaze','leftTuft','rightEye','leftEye','mouth']


# which camera to analyzed
cameraID = 'camera-2'
cameraID_short = 'cam2'


# location of levers and tubes for camera 2
# get this information using DLC animal tracking GUI, the results are stored: 
# /home/ws523/marmoset_tracking_DLCv2/marmoset_tracking_with_lever_tube-weikang-2023-04-13/labeled-data/
considerlevertube = 1
considertubeonly = 0
# # camera 1
# lever_locs_camI = {'dodson':np.array([645,600]),'scorch':np.array([425,435])}
# tube_locs_camI  = {'dodson':np.array([1350,630]),'scorch':np.array([555,345])}
# # camera 2
lever_locs_camI = {'dodson':np.array([1335,715]),'scorch':np.array([550,715])}
tube_locs_camI  = {'dodson':np.array([1550,515]),'scorch':np.array([350,515])}
# # lever_locs_camI = {'dodson':np.array([1335,715]),'scorch':np.array([550,715])}
# # tube_locs_camI  = {'dodson':np.array([1650,490]),'scorch':np.array([250,490])}
# # camera 3
# lever_locs_camI = {'dodson':np.array([1580,440]),'scorch':np.array([1296,540])}
# tube_locs_camI  = {'dodson':np.array([1470,375]),'scorch':np.array([805,475])}


if np.shape(session_start_times)[0] != np.shape(dates_list)[0]:
    exit()

    
# define bhv events summarizing variables  

# GLM related variables
Kernel_coefs_all_dates = dict.fromkeys(dates_list, [])
Kernel_spikehist_all_dates = dict.fromkeys(dates_list, [])
Kernel_selfforce_all_dates = dict.fromkeys(dates_list, [])
Kernel_partnerforce_all_dates = dict.fromkeys(dates_list, [])
#
Kernel_coefs_all_shuffled_dates = dict.fromkeys(dates_list, [])
Kernel_spikehist_all_shuffled_dates = dict.fromkeys(dates_list, [])
Kernel_selfforce_all_shuffled_dates = dict.fromkeys(dates_list, [])
Kernel_partnerforce_all_shuffled_dates = dict.fromkeys(dates_list, [])

Kernel_coefs_stretagy_all_dates = dict.fromkeys(dates_list, [])
Kernel_spikehist_stretagy_all_dates = dict.fromkeys(dates_list, [])
Kernel_selfforce_stretagy_all_dates = dict.fromkeys(dates_list, [])
Kernel_partnerforce_stretagy_all_dates = dict.fromkeys(dates_list, [])
#
Kernel_coefs_stretagy_all_shuffled_dates = dict.fromkeys(dates_list, [])
Kernel_spikehist_stretagy_all_shuffled_dates = dict.fromkeys(dates_list, [])
Kernel_selfforce_stretagy_all_shuffled_dates = dict.fromkeys(dates_list, [])
Kernel_partnerforce_stretagy_all_shuffled_dates = dict.fromkeys(dates_list, [])

# where to save the summarizing data
data_saved_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/3d_recontruction_analysis_forceManipulation_task_data_saved/'

# neural data folder
neural_data_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/Marmoset_neural_recording/'

    


# In[27]:


# basic behavior analysis (define time stamps for each bhv events, etc)

try:
    if redo_anystep:
        dummy
    
    # load saved data
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'
    
    #
    with open(data_saved_subfolder+'/Kernel_coefs_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_coefs_all_dates = pickle.load(f)         
    with open(data_saved_subfolder+'/Kernel_spikehist_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_spikehist_all_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/Kernel_coefs_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_coefs_all_shuffled_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/Kernel_spikehist_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_spikehist_all_shuffled_dates = pickle.load(f) 
    
    with open(data_saved_subfolder+'/Kernel_coefs_stretagy_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_coefs_stretagy_all_dates = pickle.load(f)         
    with open(data_saved_subfolder+'/Kernel_spikehist_stretagy_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_spikehist_stretagy_all_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/Kernel_coefs_stretagy_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_coefs_stretagy_all_shuffled_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/Kernel_spikehist_stretagy_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_spikehist_stretagy_all_shuffled_dates = pickle.load(f) 
    
    print('all data from all dates are loaded')

except:

    print('analyze all dates')

    for idate in np.arange(0,ndates,1):
    
        date_tgt = dates_list[idate]
        videodate_tgt = videodates_list[idate]
        
        neural_record_condition = neural_record_conditions[idate]
        
        session_start_time = session_start_times[idate]
        
        kilosortver = kilosortvers[idate]
      
        animal1_filename = animal1_filenames[idate]
        animal2_filename = animal2_filenames[idate]
        
        animal1_fixedorder = [animal1_fixedorders[idate]]
        animal2_fixedorder = [animal2_fixedorders[idate]]
        
        #####
        # load behavioral results
        try:
            bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_forceManipulation_task/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
            trial_record_json = glob.glob(bhv_data_path +date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_TrialRecord_" + "*.json")
            bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_bhv_data_" + "*.json")
            session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_session_info_" + "*.json")
            lever_reading_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_lever_reading_" + "*.json") 
            ni_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_ni_data_" + "*.json")
            #
            trial_record = pd.read_json(trial_record_json[0])
            bhv_data = pd.read_json(bhv_data_json[0])
            session_info = pd.read_json(session_info_json[0])
            lever_reading = pd.read_json(lever_reading_json[0])
            # 
            with open(ni_data_json[0]) as f:
                for line in f:
                    ni_data=json.loads(line) 
        except:
            bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_forceManipulation_task/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
            trial_record_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_TrialRecord_" + "*.json")
            bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_bhv_data_" + "*.json")
            session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_session_info_" + "*.json")
            lever_reading_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_lever_reading_" + "*.json")             
            ni_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_ni_data_" + "*.json")
            #
            trial_record = pd.read_json(trial_record_json[0])
            bhv_data = pd.read_json(bhv_data_json[0])
            session_info = pd.read_json(session_info_json[0])
            lever_reading = pd.read_json(lever_reading_json[0])
            # 
            with open(ni_data_json[0]) as f:
                for line in f:
                    ni_data=json.loads(line) 

        # get animal info from the session information
        animal1 = session_info['lever1_animal'][0].lower()
        animal2 = session_info['lever2_animal'][0].lower()

        # clean up the trial_record
        warnings.filterwarnings('ignore')
        trial_record_clean = pd.DataFrame(columns=trial_record.columns)
        for itrial in np.arange(0,np.max(trial_record['trial_number']),1):
            # trial_record_clean.loc[itrial] = trial_record[trial_record['trial_number']==itrial+1].iloc[[0]]
            trial_record_clean = trial_record_clean.append(trial_record[trial_record['trial_number']==itrial+1].iloc[[0]])
        trial_record_clean = trial_record_clean.reset_index(drop = True)

        # change bhv_data time to the absolute time
        time_points_new = pd.DataFrame(np.zeros(np.shape(bhv_data)[0]),columns=["time_points_new"])
        for itrial in np.arange(0,np.max(trial_record_clean['trial_number']),1):
            ind = bhv_data["trial_number"]==itrial+1
            new_time_itrial = bhv_data[ind]["time_points"] + trial_record_clean["trial_starttime"].iloc[itrial]
            time_points_new["time_points_new"][ind] = new_time_itrial
        bhv_data["time_points"] = time_points_new["time_points_new"]
        bhv_data = bhv_data[bhv_data["time_points"] != 0]

        # change lever reading time to the absolute time
        time_points_new = pd.DataFrame(np.zeros(np.shape(lever_reading)[0]),columns=["time_points_new"])
        for itrial in np.arange(0,np.max(trial_record_clean['trial_number']),1):
            ind = lever_reading["trial_number"]==itrial+1
            new_time_itrial = lever_reading[ind]["readout_timepoint"] + trial_record_clean["trial_starttime"].iloc[itrial]
            time_points_new["time_points_new"][ind] = new_time_itrial
        lever_reading["readout_timepoint"] = time_points_new["time_points_new"]
        lever_reading = lever_reading[lever_reading["readout_timepoint"] != 0]
        #
        lever1_pull = lever_reading[(lever_reading['lever_id']==1)&(lever_reading['pull_or_release']==1)]
        lever1_release = lever_reading[(lever_reading['lever_id']==1)&(lever_reading['pull_or_release']==0)]
        lever2_pull = lever_reading[(lever_reading['lever_id']==2)&(lever_reading['pull_or_release']==1)]
        lever2_release = lever_reading[(lever_reading['lever_id']==2)&(lever_reading['pull_or_release']==0)]
        #
        if np.shape(lever1_release)[0]<np.shape(lever1_pull)[0]:
            lever1_pull = lever1_pull.iloc[0:-1]
        if np.shape(lever2_release)[0]<np.shape(lever2_pull)[0]:
            lever2_pull = lever2_pull.iloc[0:-1]
        #
        lever1_pull_release = lever1_pull
        lever1_pull_release['delta_timepoint'] = np.array(lever1_release['readout_timepoint'].reset_index(drop=True)-lever1_pull['readout_timepoint'].reset_index(drop=True))
        lever1_pull_release['delta_gauge'] = np.array(lever1_release['strain_gauge'].reset_index(drop=True)-lever1_pull['strain_gauge'].reset_index(drop=True))
        lever2_pull_release = lever2_pull
        lever2_pull_release['delta_timepoint'] = np.array(lever2_release['readout_timepoint'].reset_index(drop=True)-lever2_pull['readout_timepoint'].reset_index(drop=True))
        lever2_pull_release['delta_gauge'] = np.array(lever2_release['strain_gauge'].reset_index(drop=True)-lever2_pull['strain_gauge'].reset_index(drop=True))
        
        
        #####
        # load behavioral event results from the tracking analysis
        # folder and file path
        camera12_analyzed_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/test_video_forceManipulation_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera12/"
        camera23_analyzed_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/test_video_forceManipulation_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera23/"

        try:
            singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000"
            try: 
                bodyparts_camI_camIJ = camera12_analyzed_path+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"
                # get the bodypart data from files
                bodyparts_locs_camI = body_part_locs_singlecam(bodyparts_camI_camIJ,singlecam_ana_type,animalnames_videotrack,bodypartnames_videotrack,date_tgt)
                video_file_original = camera12_analyzed_path+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+".mp4"
            except:
                bodyparts_camI_camIJ = camera23_analyzed_path+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"
                # get the bodypart data from files
                bodyparts_locs_camI = body_part_locs_singlecam(bodyparts_camI_camIJ,singlecam_ana_type,animalnames_videotrack,bodypartnames_videotrack,date_tgt)
                video_file_original = camera23_analyzed_path+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+".mp4"        
        except:
            singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_middle_camera_withHeadchamberFeb28shuffle1_167500"
            try: 
                bodyparts_camI_camIJ = camera12_analyzed_path+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"
                # get the bodypart data from files
                bodyparts_locs_camI = body_part_locs_singlecam(bodyparts_camI_camIJ,singlecam_ana_type,animalnames_videotrack,bodypartnames_videotrack,date_tgt)
                video_file_original = camera12_analyzed_path+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+".mp4"
            except:
                bodyparts_camI_camIJ = camera23_analyzed_path+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"
                # get the bodypart data from files
                bodyparts_locs_camI = body_part_locs_singlecam(bodyparts_camI_camIJ,singlecam_ana_type,animalnames_videotrack,bodypartnames_videotrack,date_tgt)
                video_file_original = camera23_analyzed_path+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+".mp4"        
                            
        try:
            # dummy
            print('load social gaze with '+cameraID+' only of '+date_tgt)
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_look_ornot.pkl', 'rb') as f:
                output_look_ornot = pickle.load(f)
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_allvectors.pkl', 'rb') as f:
                output_allvectors = pickle.load(f)
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_allangles.pkl', 'rb') as f:
                output_allangles = pickle.load(f)  
        except:   
            print('analyze social gaze with '+cameraID+' only of '+date_tgt)
            # get social gaze information 
            output_look_ornot, output_allvectors, output_allangles = find_socialgaze_timepoint_singlecam_wholebody(bodyparts_locs_camI,lever_locs_camI,tube_locs_camI,
                                                                                                                   considerlevertube,considertubeonly,sqr_thres_tubelever,                                                                                                               sqr_thres_face,sqr_thres_body)
            # save data
            current_dir = data_saved_folder+'/bhv_events_singlecam_wholebody/'+animal1_fixedorder[0]+animal2_fixedorder[0]
            add_date_dir = os.path.join(current_dir,cameraID+'/'+date_tgt)
            if not os.path.exists(add_date_dir):
                os.makedirs(add_date_dir)
            #
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_look_ornot.pkl', 'wb') as f:
                pickle.dump(output_look_ornot, f)
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_allvectors.pkl', 'wb') as f:
                pickle.dump(output_allvectors, f)
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_allangles.pkl', 'wb') as f:
                pickle.dump(output_allangles, f)

        look_at_other_or_not_merge = output_look_ornot['look_at_other_or_not_merge']
        look_at_tube_or_not_merge = output_look_ornot['look_at_tube_or_not_merge']
        look_at_lever_or_not_merge = output_look_ornot['look_at_lever_or_not_merge']
        # change the unit to second
        session_start_time = session_start_times[idate]
        look_at_other_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_other_or_not_merge['dodson'])[0],1)/fps - session_start_time
        look_at_lever_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_lever_or_not_merge['dodson'])[0],1)/fps - session_start_time
        look_at_tube_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_tube_or_not_merge['dodson'])[0],1)/fps - session_start_time 

        # find time point of behavioral events
        output_time_points_socialgaze ,output_time_points_levertube = bhv_events_timepoint_singlecam(bhv_data,look_at_other_or_not_merge,look_at_lever_or_not_merge,look_at_tube_or_not_merge)
        time_point_pull1 = output_time_points_socialgaze['time_point_pull1']
        time_point_pull2 = output_time_points_socialgaze['time_point_pull2']
        time_point_juice1 = output_time_points_socialgaze['time_point_juice1']
        time_point_juice2 = output_time_points_socialgaze['time_point_juice2']
        oneway_gaze1 = output_time_points_socialgaze['oneway_gaze1']
        oneway_gaze2 = output_time_points_socialgaze['oneway_gaze2']
        mutual_gaze1 = output_time_points_socialgaze['mutual_gaze1']
        mutual_gaze2 = output_time_points_socialgaze['mutual_gaze2']

        # define successful pulls and failed pulls
        trialnum_succ = np.array(trial_record_clean['trial_number'][trial_record_clean['rewarded']>0])
        bhv_data_succ = bhv_data[np.isin(bhv_data['trial_number'],trialnum_succ)]
        #
        time_point_pull1_succ = bhv_data_succ["time_points"][bhv_data_succ["behavior_events"]==1]
        time_point_pull2_succ = bhv_data_succ["time_points"][bhv_data_succ["behavior_events"]==2]
        time_point_pull1_succ = np.round(time_point_pull1_succ,1)
        time_point_pull2_succ = np.round(time_point_pull2_succ,1)
        #
        trialnum_fail = np.array(trial_record_clean['trial_number'][trial_record_clean['rewarded']==0])
        bhv_data_fail = bhv_data[np.isin(bhv_data['trial_number'],trialnum_fail)]
        #
        time_point_pull1_fail = bhv_data_fail["time_points"][bhv_data_fail["behavior_events"]==1]
        time_point_pull2_fail = bhv_data_fail["time_points"][bhv_data_fail["behavior_events"]==2]
        time_point_pull1_fail = np.round(time_point_pull1_fail,1)
        time_point_pull2_fail = np.round(time_point_pull2_fail,1)
        # 
        time_point_pulls_succfail = { "pull1_succ":time_point_pull1_succ,
                                      "pull2_succ":time_point_pull2_succ,
                                      "pull1_fail":time_point_pull1_fail,
                                      "pull2_fail":time_point_pull2_fail,
                                    }
        
        # new total session time (instead of a fix time) - total time of the video recording
        totalsess_time = np.floor(np.shape(output_look_ornot['look_at_lever_or_not_merge']['dodson'])[0]/30) # 30 is the fps, in the unit of second
              
        
        #####     
        # load neural recording data
        # session starting time compared with the neural recording
        session_start_time_niboard_offset = ni_data['session_t0_offset'] # in the unit of second
        neural_start_time_niboard_offset = ni_data['trigger_ts'][0]['elapsed_time'] # in the unit of second
        neural_start_time_session_start_offset = neural_start_time_niboard_offset-session_start_time_niboard_offset
    
        # load channel maps
        channel_map_file = '/home/ws523/kilisort_spikesorting/Channel-Maps/Neuronexus_whitematter_2x32.mat'
        # channel_map_file = '/home/ws523/kilisort_spikesorting/Channel-Maps/Neuronexus_whitematter_2x32_kilosort4_new.mat'
        channel_map_data = scipy.io.loadmat(channel_map_file)
                  
        # # load spike sorting results
        print('load spike data for '+neural_record_condition)
        if kilosortver == 2:
            spike_time_file = neural_data_folder+neural_record_condition+'/Kilosort/spike_times.npy'
            spike_time_data = np.load(spike_time_file)
        elif kilosortver == 4:
            spike_time_file = neural_data_folder+neural_record_condition+'/kilosort4_6500HzNotch/spike_times.npy'
            spike_time_data = np.load(spike_time_file)
        # 
        # align the FR recording time stamps
        spike_time_data = spike_time_data + fs_spikes*neural_start_time_session_start_offset
        # down-sample the spike recording resolution to 30Hz
        spike_time_data = spike_time_data/fs_spikes*fps
        spike_time_data = np.round(spike_time_data)
        #
        if kilosortver == 2:
            spike_clusters_file = neural_data_folder+neural_record_condition+'/Kilosort/spike_clusters.npy'
            spike_clusters_data = np.load(spike_clusters_file)
            spike_channels_data = np.copy(spike_clusters_data)
        elif kilosortver == 4:
            spike_clusters_file = neural_data_folder+neural_record_condition+'/kilosort4_6500HzNotch/spike_clusters.npy'
            spike_clusters_data = np.load(spike_clusters_file)
            spike_channels_data = np.copy(spike_clusters_data)
        #
        if kilosortver == 2:
            channel_maps_file = neural_data_folder+neural_record_condition+'/Kilosort/channel_map.npy'
            channel_maps_data = np.load(channel_maps_file)
        elif kilosortver == 4:
            channel_maps_file = neural_data_folder+neural_record_condition+'/kilosort4_6500HzNotch/channel_map.npy'
            channel_maps_data = np.load(channel_maps_file)
        #
        if kilosortver == 2:
            channel_pos_file = neural_data_folder+neural_record_condition+'/Kilosort/channel_positions.npy'
            channel_pos_data = np.load(channel_pos_file)
        elif kilosortver == 4:
            channel_pos_file = neural_data_folder+neural_record_condition+'/kilosort4_6500HzNotch/channel_positions.npy'
            channel_pos_data = np.load(channel_pos_file)
        #
        if kilosortver == 2:
            clusters_info_file = neural_data_folder+neural_record_condition+'/Kilosort/cluster_info.tsv'
            clusters_info_data = pd.read_csv(clusters_info_file,sep="\t")
        elif kilosortver == 4:
            clusters_info_file = neural_data_folder+neural_record_condition+'/kilosort4_6500HzNotch/cluster_info.tsv'
            clusters_info_data = pd.read_csv(clusters_info_file,sep="\t")
        #
        # only get the spikes that are manually checked
        try:
            good_clusters = clusters_info_data[(clusters_info_data.group=='good')|(clusters_info_data.group=='mua')]['cluster_id'].values
        except:
            good_clusters = clusters_info_data[(clusters_info_data.group=='good')|(clusters_info_data.group=='mua')]['id'].values
        #
        clusters_info_data = clusters_info_data[~pd.isnull(clusters_info_data.group)]
        #
        spike_time_data = spike_time_data[np.isin(spike_clusters_data,good_clusters)]
        spike_channels_data = spike_channels_data[np.isin(spike_clusters_data,good_clusters)]
        spike_clusters_data = spike_clusters_data[np.isin(spike_clusters_data,good_clusters)]

        #
        nclusters = np.shape(clusters_info_data)[0]
        #
        for icluster in np.arange(0,nclusters,1):
            try:
                cluster_id = clusters_info_data['id'].iloc[icluster]
            except:
                cluster_id = clusters_info_data['cluster_id'].iloc[icluster]
            spike_channels_data[np.isin(spike_clusters_data,cluster_id)] = clusters_info_data['ch'].iloc[icluster]   
        # 
        # get the channel to depth information, change 2 shanks to 1 shank 
        try:
            channel_depth=np.hstack([channel_pos_data[channel_pos_data[:,0]==0,1]*2,channel_pos_data[channel_pos_data[:,0]==1,1]*2+1])
            # channel_depth=np.hstack([channel_pos_data[channel_pos_data[:,0]==0,1],channel_pos_data[channel_pos_data[:,0]==31.2,1]])            
            # channel_to_depth = np.vstack([channel_maps_data.T[0],channel_depth])
            channel_to_depth = np.vstack([channel_maps_data.T,channel_depth])
        except:
            channel_depth=np.hstack([channel_pos_data[channel_pos_data[:,0]==0,1],channel_pos_data[channel_pos_data[:,0]==31.2,1]])            
            # channel_to_depth = np.vstack([channel_maps_data.T[0],channel_depth])
            channel_to_depth = np.vstack([channel_maps_data.T,channel_depth])
            channel_to_depth[1] = channel_to_depth[1]/30-64 # make the y axis consistent

            
            
        #####    
        #####    
        # after all the analysis and data loading, separate them based on different subblock    
        # get task type and cooperation threshold
        # tasktype: 1-normal SR, 2-force changed SR, 3-normal coop, 4-force changed coop
        trialID_list = np.array(trial_record_clean['trial_number'],dtype = 'int')
        tasktype_list = np.array(trial_record_clean['task_type'],dtype = 'int')
        coop_thres_list = np.array(trial_record_clean['pulltime_thres'],dtype = 'int')
        lever1force_list = np.array(trial_record_clean['lever1_force'],dtype = 'int')
        lever2force_list = np.array(trial_record_clean['lever2_force'],dtype = 'int')
        
        # use the combination of lever 1/2 forces to separate trials
        force12_uniques,indices = np.unique(np.vstack((lever1force_list,lever2force_list)),axis=1,return_index=True)
        force12_uniques = force12_uniques[:,np.argsort(indices)]
        ntrialtypes = np.shape(force12_uniques)[1]
        
        # 
        # put them in the summarizing result, organized for each day
        force1_all_idate = np.zeros((0,)) 
        force2_all_idate = np.zeros((0,)) 

        subblockID_all_idate = np.zeros((0,))

        succ_rate_all_idate = np.zeros((0,))
        trialnum_all_idate = np.zeros((0,))
        blockstarttime_all_idate = np.zeros((0,))
        blockendtime_all_idate = np.zeros((0,))
         
        #    
        for itrialtype in np.arange(0,ntrialtypes,1):
            force1_unique = force12_uniques[0,itrialtype]
            force2_unique = force12_uniques[1,itrialtype]

            ind = np.isin(lever1force_list,force1_unique) & np.isin(lever2force_list,force2_unique)
            
            trialID_itrialtype = trialID_list[ind]
            
            tasktype_itrialtype = np.unique(tasktype_list[ind])
            coop_thres_itrialtype = np.unique(coop_thres_list[ind])
            
            # save some simple measures
            force1_all_idate = np.append(force1_all_idate,force1_unique)
            force2_all_idate = np.append(force2_all_idate,force2_unique)
            #
            trialnum_all_idate = np.append(trialnum_all_idate,np.sum(ind))
            subblockID_all_idate = np.append(subblockID_all_idate,itrialtype)
            
            # analyze behavior results
            bhv_data_itrialtype = bhv_data[np.isin(bhv_data['trial_number'],trialID_itrialtype)]
            
            #
            # successful rates
            succ_rate_itrialtype = np.sum((bhv_data_itrialtype['behavior_events']==3)|(bhv_data_itrialtype['behavior_events']==4))/np.sum((bhv_data_itrialtype['behavior_events']==1)|(bhv_data_itrialtype['behavior_events']==2))
            succ_rate_all_idate = np.append(succ_rate_all_idate,succ_rate_itrialtype)
            #
            # block time
            block_starttime = bhv_data_itrialtype[bhv_data_itrialtype['behavior_events']==0]['time_points'].iloc[0]
            blockstarttime_all_idate = np.append(blockstarttime_all_idate,block_starttime)
            block_endtime = bhv_data_itrialtype[bhv_data_itrialtype['behavior_events']==9]['time_points'].iloc[-1]
            blockendtime_all_idate = np.append(blockendtime_all_idate,block_endtime)
        
        
        # translate force level to quantiles
        force1_all_idate = make_force_quantiles(force1_all_idate)
        force2_all_idate = make_force_quantiles(force2_all_idate)
        
        
        # get the dataset for GLM and run GLM
        starttime = bhv_data[bhv_data['behavior_events']==0]['time_points'].iloc[0]
        endtime = bhv_data[bhv_data['behavior_events']==9]['time_points'].iloc[-1]
        # 
        gaze_threshold = 0.5 # min length threshold to define if a gaze is real gaze or noise, in the unit of second 

        #
        stg_twins = 3 # 3s, the behavioral event interval used to define strategy, consistent with DBN 3s time lags

        # get the organized data for GLM
        print('get '+neural_record_condition+' data for single camera GLM fitting')
        #
        data_summary, data_summary_names, spiketrain_summary = get_singlecam_bhv_var_for_neuralGLM_fitting(animal1, animal2, animalnames_videotrack, 
                                                                session_start_time, starttime, endtime, totalsess_time, 
                                                                blockstarttime_all_idate, blockendtime_all_idate, force1_all_idate, force2_all_idate,
                                                                stg_twins, time_point_pull1, time_point_pull2, time_point_juice1, time_point_juice2,
                                                                time_point_pulls_succfail, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, gaze_threshold, 
                                                                spike_clusters_data, spike_time_data, spike_channels_data)

        
        
        
        
        # GLM to behavioral events (actions)
        if 1:
            print('do single camera GLM fitting (behavioral events) for '+neural_record_condition)

            nbootstraps = 20 # cannot be 1, will introduce error
            traintestperc = 0.6

            # select the behavioral variables that want to be in the GLM
            dostrategies = 0
            # bhvvaris_toGLM = ['self leverpull_prob', 'self socialgaze_prob', 'self juice_prob', 
            #                   'other leverpull_prob', 'other socialgaze_prob', 'other juice_prob', ]
            bhvvaris_toGLM = ['self leverpull_prob', 'self socialgaze_prob', ]

            # the time window for behavioral variables, 0 means the spike time
            trig_twin = [-4,4] # in the unit of second

            # if consider the spike history
            dospikehist = 1
            spikehist_twin = 3 # the length the spike history to consider, in the unit of second; # has to smallerthan trig_twin

            doforcelevel = 1
            
            # if do a spike time shuffle to generate null distribution 
            donullshuffle = 1

            doplots = 0 # plot the kernel for each variables and each cell clusters
            savefig = 0 # save the plotted kernel function
            #
            save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents_GLMfitting_singlecam/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            (Kernel_coefs_allboots_allcells, Kernel_spikehist_allboots_allcells,
            Kernel_selfforce_allboots_allcells, Kernel_partnerforce_allboots_allcells, 
            Kernel_coefs_allboots_allcells_shf, Kernel_spikehist_allboots_allcells_shf, 
            Kernel_selfforce_allboots_allcells_shf, Kernel_partnerforce_allboots_allcells_shf)  = neuralGLM_fitting(animal1, animal2, data_summary_names, data_summary, spiketrain_summary, 
                                                               bhvvaris_toGLM, nbootstraps, traintestperc, trig_twin, dospikehist, spikehist_twin,
                                                               doplots, date_tgt, savefig, save_path, dostrategies, donullshuffle, doforcelevel)

            Kernel_coefs_all_dates[date_tgt] = Kernel_coefs_allboots_allcells
            Kernel_spikehist_all_dates[date_tgt] = Kernel_spikehist_allboots_allcells
            Kernel_selfforce_all_dates[date_tgt] = Kernel_selfforce_allboots_allcells
            Kernel_partnerforce_all_dates[date_tgt] = Kernel_partnerforce_allboots_allcells
            Kernel_coefs_all_shuffled_dates[date_tgt] = Kernel_coefs_allboots_allcells_shf
            Kernel_spikehist_all_shuffled_dates[date_tgt] = Kernel_spikehist_allboots_allcells_shf
            Kernel_selfforce_all_shuffled_dates[date_tgt] = Kernel_selfforce_allboots_allcells_shf
            Kernel_partnerforce_all_shuffled_dates[date_tgt] = Kernel_partnerforce_allboots_allcells_shf
            
            # save the data for each date
            if 1:

                data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'+date_tgt+'/'
                if not os.path.exists(data_saved_subfolder):
                    os.makedirs(data_saved_subfolder)

                # GLM to behavioral events (actions)
                with open(data_saved_subfolder+'/Kernel_coefs_allboots_allcells.pkl', 'wb') as f:
                    pickle.dump(Kernel_coefs_allboots_allcells, f)    
                with open(data_saved_subfolder+'/Kernel_spikehist_allboots_allcells.pkl', 'wb') as f:
                    pickle.dump(Kernel_spikehist_allboots_allcells, f)    
                with open(data_saved_subfolder+'/Kernel_coefs_allboots_allcells_shf.pkl', 'wb') as f:
                    pickle.dump(Kernel_coefs_allboots_allcells_shf, f)    
                with open(data_saved_subfolder+'/Kernel_spikehist_allboots_allcells_shf.pkl', 'wb') as f:
                    pickle.dump(Kernel_spikehist_allboots_allcells_shf, f) 

                with open(data_saved_subfolder+'/Kernel_selfforce_allboots_allcells.pkl', 'wb') as f:
                    pickle.dump(Kernel_selfforce_allboots_allcells, f)  
                with open(data_saved_subfolder+'/Kernel_partnerforce_allboots_allcells.pkl', 'wb') as f:
                    pickle.dump(Kernel_partnerforce_allboots_allcells, f) 
                with open(data_saved_subfolder+'/Kernel_selfforce_allboots_allcells_shf.pkl', 'wb') as f:
                    pickle.dump(Kernel_selfforce_allboots_allcells_shf, f)  
                with open(data_saved_subfolder+'/Kernel_partnerforce_allboots_allcells_shf.pkl', 'wb') as f:
                    pickle.dump(Kernel_partnerforce_allboots_allcells_shf, f) 

        

        # GLM to strategies (pairs of actions)
        if 1:
            print('do single camera GLM fitting (strategies) for '+neural_record_condition)

            nbootstraps = 20 # cannot be 1, will introduce error
            traintestperc = 0.6

            # select the behavioral variables that want to be in the GLM
            dostrategies = 1
            bhvvaris_toGLM = ['self sync_pull_prob', 'self gaze_lead_pull_prob', 'self social_attention_prob', 
                             ]

            # the time window for behavioral variables, 0 means the spike time
            trig_twin = [-4,4] # in the unit of second

            # if consider the spike history
            dospikehist = 1
            spikehist_twin = 3 # the length the spike history to consider, in the unit of second; # has to smallerthan trig_twin

            # if consider the force
            doconsiderforce = 1
            
            # if do a spike time shuffle to generate null distribution 
            donullshuffle = 1

            doplots = 0 # plot the kernel for each variables and each cell clusters
            savefig = 0 # save the plotted kernel function
            #
            save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents_GLMfitting_singlecam/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            (Kernel_coefs_stretagy_allboots_allcells, Kernel_spikehist_stretagy_allboots_allcells,
            Kernel_selfforce_stretagy_allboots_allcells, Kernel_partnerforce_stretagy_allboots_allcells,
            Kernel_coefs_stretagy_allboots_allcells_shf, Kernel_spikehist_stretagy_allboots_allcells_shf,
            Kernel_selfforce_stretagy_allboots_allcells_shf, Kernel_partnerforce_stretagy_allboots_allcells_shf) = neuralGLM_fitting(animal1, animal2, data_summary_names, data_summary, spiketrain_summary, 
                                                               bhvvaris_toGLM, nbootstraps, traintestperc, trig_twin, dospikehist, spikehist_twin,
                                                               doplots, date_tgt, savefig, save_path, dostrategies, donullshuffle, doforcelevel)

            Kernel_coefs_stretagy_all_dates[date_tgt] = Kernel_coefs_stretagy_allboots_allcells
            Kernel_spikehist_stretagy_all_dates[date_tgt] = Kernel_spikehist_stretagy_allboots_allcells
            Kernel_selfforce_stretagy_all_dates[date_tgt] = Kernel_selfforce_stretagy_allboots_allcells
            Kernel_partnerforce_stretagy_all_dates[date_tgt] = Kernel_partnerforce_stretagy_allboots_allcells
            
            Kernel_coefs_stretagy_all_shuffled_dates[date_tgt] = Kernel_coefs_stretagy_allboots_allcells_shf
            Kernel_spikehist_stretagy_all_shuffled_dates[date_tgt] = Kernel_spikehist_stretagy_allboots_allcells_shf
            Kernel_selfforce_stretagy_all_shuffled_dates[date_tgt] = Kernel_selfforce_stretagy_allboots_allcells_shf
            Kernel_partnerforce_stretagy_all_shuffled_dates[date_tgt] = Kernel_partnerforce_stretagy_allboots_allcells_shf        

            # save the data for each date
            if 1:

                data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'+date_tgt+'/'
                if not os.path.exists(data_saved_subfolder):
                    os.makedirs(data_saved_subfolder)
         
                # GLM to strategies (pairs of actions)
                with open(data_saved_subfolder+'/Kernel_coefs_stretagy_allboots_allcells.pkl', 'wb') as f:
                    pickle.dump(Kernel_coefs_stretagy_allboots_allcells, f)    
                with open(data_saved_subfolder+'/Kernel_spikehist_stretagy_allboots_allcells.pkl', 'wb') as f:
                    pickle.dump(Kernel_spikehist_stretagy_allboots_allcells, f)    
                with open(data_saved_subfolder+'/Kernel_coefs_stretagy_allboots_allcells_shf.pkl', 'wb') as f:
                    pickle.dump(Kernel_coefs_stretagy_allboots_allcells_shf, f)    
                with open(data_saved_subfolder+'/Kernel_spikehist_stretagy_allboots_allcells_shf.pkl', 'wb') as f:
                    pickle.dump(Kernel_spikehist_stretagy_allboots_allcells_shf, f) 

                with open(data_saved_subfolder+'/Kernel_selfforce_stretagy_allboots_allcells.pkl', 'wb') as f:
                    pickle.dump(Kernel_selfforce_stretagy_allboots_allcells, f)  
                with open(data_saved_subfolder+'/Kernel_partnerforce_stretagy_allboots_allcells.pkl', 'wb') as f:
                    pickle.dump(Kernel_partnerforce_stretagy_allboots_allcells, f) 
                with open(data_saved_subfolder+'/Kernel_selfforce_stretagy_allboots_allcells_shf.pkl', 'wb') as f:
                    pickle.dump(Kernel_selfforce_stretagy_allboots_allcells_shf, f)  
                with open(data_saved_subfolder+'/Kernel_partnerforce_stretagy_allboots_allcells_shf.pkl', 'wb') as f:
                    pickle.dump(Kernel_partnerforce_stretagy_allboots_allcells_shf, f) 

            
            
    # save the final data
    if 0:
        
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
                
        # GLM to behavioral events (actions)
        if 1:
            with open(data_saved_subfolder+'/Kernel_coefs_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_coefs_all_dates, f)    
            with open(data_saved_subfolder+'/Kernel_spikehist_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_spikehist_all_dates, f)    
            with open(data_saved_subfolder+'/Kernel_coefs_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_coefs_all_shuffled_dates, f)    
            with open(data_saved_subfolder+'/Kernel_spikehist_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_spikehist_all_shuffled_dates, f) 
                
            with open(data_saved_subfolder+'/Kernel_selfforce_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_selfforce_all_dates, f)  
            with open(data_saved_subfolder+'/Kernel_partnerforce_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_partnerforce_all_dates, f) 
            with open(data_saved_subfolder+'/Kernel_selfforce_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_selfforce_all_shuffled_dates, f)  
            with open(data_saved_subfolder+'/Kernel_partnerforce_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_partnerforce_all_shuffled_dates, f) 
                
           
        # GLM to strategies (pairs of actions)
        if 1:
            with open(data_saved_subfolder+'/Kernel_coefs_stretagy_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_coefs_stretagy_all_dates, f)    
            with open(data_saved_subfolder+'/Kernel_spikehist_stretagy_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_spikehist_stretagy_all_dates, f)    
            with open(data_saved_subfolder+'/Kernel_coefs_stretagy_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_coefs_stretagy_all_shuffled_dates, f)    
            with open(data_saved_subfolder+'/Kernel_spikehist_stretagy_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_spikehist_stretagy_all_shuffled_dates, f) 
                
            with open(data_saved_subfolder+'/Kernel_selfforce_stretagy_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_selfforce_stretagy_all_dates, f)  
            with open(data_saved_subfolder+'/Kernel_partnerforce_stretagy_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_partnerforce_stretagy_all_dates, f) 
            with open(data_saved_subfolder+'/Kernel_selfforce_stretagy_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_selfforce_stretagy_all_shuffled_dates, f)  
            with open(data_saved_subfolder+'/Kernel_partnerforce_stretagy_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_partnerforce_stretagy_all_shuffled_dates, f) 
    
    
    
    


# In[31]:


Kernel_selfforce_stretagy_all_shuffled_dates


# In[30]:


Kernel_partnerforce_stretagy_allboots_allcells


# In[21]:


Kernel_stretagy_all_dates


# In[ ]:





# In[16]:


Kernel_coefs_stretagy_all_dates_df


# In[18]:


np.shape(Kernel_coefs_stretagy_all_dates[date_tgt][iclusterID])


# In[ ]:





# In[ ]:





# ## plot - for individual animal
# ### prepare the summarizing data set and run population level analysis such as PCA
# ### plot the kernel defined based on the stretagy (pair of action)

# In[13]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.gridspec import GridSpec

doPCA = 1
doTSNE = 0

Kernel_coefs_stretagy_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                          'Kernel_average'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
       
    # make sure to be the same as the bhvvaris_toGLM
    bhv_types = ['self sync_pull_prob', 'self gaze_lead_pull_prob', 'self social_attention_prob',]
    nbhv_types = np.shape(bhv_types)[0]

    for ibhv_type in np.arange(0,nbhv_types,1):
        
        bhv_type = bhv_types[ibhv_type]

        clusterIDs = Kernel_coefs_stretagy_all_dates[date_tgt].keys()

        for iclusterID in clusterIDs:

            kernel_ibhv = Kernel_coefs_stretagy_all_dates[date_tgt][iclusterID][:,ibhv_type,:]
            
            kernel_ibhv_average = np.nanmean(kernel_ibhv,axis = 0)

            Kernel_coefs_stretagy_all_dates_df = Kernel_coefs_stretagy_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':bhv_type.split()[0],
                                                                                    'bhv_name': bhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'Kernel_average':kernel_ibhv_average,
                                                                                   }, ignore_index=True)

            
            
# only focus on the certain act animal and certain bhv_name
# act_animals_all = np.unique(Kernel_coefs_stretagy_all_dates_df['act_animal'])
act_animals_all = ['self']
bhv_names_all = np.unique(Kernel_coefs_stretagy_all_dates_df['bhv_name'])
# bhv_names_all = ['sync_pull_prob']
conditions_all = np.unique(Kernel_coefs_stretagy_all_dates_df['condition'])

nact_animals = np.shape(act_animals_all)[0]
nbhv_names = np.shape(bhv_names_all)[0]
nconditions = np.shape(conditions_all)[0]


# run PCA and plot
for ianimal in np.arange(0,nact_animals,1):
    
    act_animal = act_animals_all[ianimal]
    
    for icondition in np.arange(0,nconditions,1):
        
        task_condition = conditions_all[icondition]
        
        # set up for plotting
        nPC_toplot = 4

        # Create a figure with GridSpec, specifying height_ratios
        fig = plt.figure(figsize=(nPC_toplot*2,6*nbhv_names))

        # Define a grid with 4*nbhv_names rows in the left and nbhv_names rows in the right, 
        # but scale the right column's height by 3
        gs = GridSpec(nPC_toplot*nbhv_names, 2, height_ratios=[1] * nPC_toplot*nbhv_names)

        # Left column (4*nbhv_names rows, 1 column) for PC 1 to 4 traces
        ax_left = [fig.add_subplot(gs[i, 0]) for i in range(nPC_toplot*nbhv_names)]  # Access all 4*nbhv_names rows in the left column

        # Right column (nbhv_names rows, 1 column, scaling the height by using multiple rows for each plot)
        # for the variance explanation
        ax_right = [fig.add_subplot(gs[nPC_toplot * i:nPC_toplot * i + nPC_toplot, 1]) for i in range(nbhv_names)]  # Group 4 rows for each of the 3 subplots on the right



        for ibhvname in np.arange(0,nbhv_names,1):

            bhv_name = bhv_names_all[ibhvname]

            ind = (Kernel_coefs_stretagy_all_dates_df['act_animal']==act_animal)&(Kernel_coefs_stretagy_all_dates_df['bhv_name']==bhv_name)&(Kernel_coefs_stretagy_all_dates_df['condition']==task_condition)

            Kernel_coefs_stretagy_tgt = np.vstack(list(Kernel_coefs_stretagy_all_dates_df[ind]['Kernel_average']))

            ind_nan = np.isnan(np.sum(Kernel_coefs_stretagy_tgt,axis=1)) # exist because of failed pull in SR
            Kernel_coefs_stretagy_tgt = Kernel_coefs_stretagy_tgt[~ind_nan,:]

            # k means clustering
            # run clustering on the 15 or 2 dimension PC space (for doPCA), or the whole dataset or 2 dimension (for doTSNE)
            pca = PCA(n_components=10)
            Kernel_coefs_stretagy_pca = pca.fit_transform(Kernel_coefs_stretagy_tgt.transpose())

            # Get the explained variance ratio
            explained_variance = pca.explained_variance_ratio_

            # Calculate the cumulative explained variance
            cumulative_variance = np.cumsum(explained_variance)

            # Plot the cumulative explained variance
            ax_right[ibhvname].plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker='o', linestyle='--', color='blue', alpha=0.7)
            ax_right[ibhvname].set_xlabel('Number of the principle component')
            ax_right[ibhvname].set_ylabel('Cumulative Percentage of Variance Explained')
            ax_right[ibhvname].set_title(bhv_name)
            ax_right[ibhvname].set_xticks(np.arange(1, len(cumulative_variance) + 1, 1))
            ax_right[ibhvname].grid(True)
            # 
            # if ibhvname == nbhv_names - 1:
            #     ax_right[ibhvname].set_xlabel('Number of the principle component')

            # plot the PCs
            for iPC_toplot in np.arange(0,nPC_toplot,1):

                PCtoplot = Kernel_coefs_stretagy_pca[:,iPC_toplot]

                trig_twin = [-4,4] # in the unit of second
                xxx = np.arange(trig_twin[0]*fps,trig_twin[1]*fps,1)

                ax_left[iPC_toplot+ibhvname*nPC_toplot].plot(xxx, PCtoplot, 'k')
                ax_left[iPC_toplot+ibhvname*nPC_toplot].plot([0,0],[np.nanmin(PCtoplot)*1.1,np.nanmax(PCtoplot)*1.1],'k--')

                ax_left[iPC_toplot+ibhvname*nPC_toplot].set_title(bhv_name+' kernel PC'+str(iPC_toplot+1))

                if iPC_toplot == nPC_toplot - 1:
                    ax_left[iPC_toplot+ibhvname*nPC_toplot].set_xlabel('time (s)')

        
        # Adjust layout
        plt.tight_layout()
        plt.show()        
        
        if (animal1_filenames[0] == 'Kanga') | (animal2_filenames[0] == 'Kanga'):
            recordedAnimal = 'Kanga'
        elif (animal1_filenames[0] == 'Dodson') | (animal2_filenames[0] == 'Dodson'):
            recordedAnimal = 'Dodson'

        savefig = 1
        if savefig:
            figsavefolder = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents_GLMfitting_singlecam/"+cameraID+"/"+recordedAnimal+"_neuralGLM/"

            if not os.path.exists(figsavefolder):
                os.makedirs(figsavefolder)

            fig.savefig(figsavefolder+'stretagy_kernel_coefs_pca_patterns_all_dates'+savefile_sufix+'_'+act_animal+'action_in'+task_condition+'.pdf')
        
        


# ### prepare the summarizing data set and run population level analysis such as PCA
# ### plot the kernel defined based on the single actions

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.gridspec import GridSpec

doPCA = 1
doTSNE = 0

Kernel_coefs_action_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                          'Kernel_average'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
       
    # make sure to be the same as the bhvvaris_toGLM
    bhv_types = ['self leverpull_prob', 'self socialgaze_prob', 'self juice_prob', 
                 'other leverpull_prob', 'other socialgaze_prob', 'other juice_prob', ]
    nbhv_types = np.shape(bhv_types)[0]

    for ibhv_type in np.arange(0,nbhv_types,1):
        
        bhv_type = bhv_types[ibhv_type]

        clusterIDs = Kernel_coefs_all_dates[date_tgt].keys()

        for iclusterID in clusterIDs:

            kernel_ibhv = Kernel_coefs_all_dates[date_tgt][iclusterID][:,ibhv_type,:]
            
            kernel_ibhv_average = np.nanmean(kernel_ibhv,axis = 0)

            Kernel_coefs_action_all_dates_df = Kernel_coefs_action_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':bhv_type.split()[0],
                                                                                    'bhv_name': bhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'Kernel_average':kernel_ibhv_average,
                                                                                   }, ignore_index=True)

            
            
# only focus on the certain act animal and certain bhv_name
act_animals_all = np.unique(Kernel_coefs_action_all_dates_df['act_animal'])
# act_animals_all = ['self']
bhv_names_all = np.unique(Kernel_coefs_action_all_dates_df['bhv_name'])
# bhv_names_all = ['leverpull_prob']
conditions_all = np.unique(Kernel_coefs_action_all_dates_df['condition'])

nact_animals = np.shape(act_animals_all)[0]
nbhv_names = np.shape(bhv_names_all)[0]
nconditions = np.shape(conditions_all)[0]


# run PCA and plot
for ianimal in np.arange(0,nact_animals,1):
    
    act_animal = act_animals_all[ianimal]
    
    for icondition in np.arange(0,nconditions,1):
        
        task_condition = conditions_all[icondition]
        
        # set up for plotting
        nPC_toplot = 4

        # Create a figure with GridSpec, specifying height_ratios
        fig = plt.figure(figsize=(nPC_toplot*2,6*nbhv_names))

        # Define a grid with 4*nbhv_names rows in the left and nbhv_names rows in the right, 
        # but scale the right column's height by 3
        gs = GridSpec(nPC_toplot*nbhv_names, 2, height_ratios=[1] * nPC_toplot*nbhv_names)

        # Left column (4*nbhv_names rows, 1 column) for PC 1 to 4 traces
        ax_left = [fig.add_subplot(gs[i, 0]) for i in range(nPC_toplot*nbhv_names)]  # Access all 4*nbhv_names rows in the left column

        # Right column (nbhv_names rows, 1 column, scaling the height by using multiple rows for each plot)
        # for the variance explanation
        ax_right = [fig.add_subplot(gs[nPC_toplot * i:nPC_toplot * i + nPC_toplot, 1]) for i in range(nbhv_names)]  # Group 4 rows for each of the 3 subplots on the right



        for ibhvname in np.arange(0,nbhv_names,1):

            bhv_name = bhv_names_all[ibhvname]

            ind = (Kernel_coefs_action_all_dates_df['act_animal']==act_animal)&(Kernel_coefs_action_all_dates_df['bhv_name']==bhv_name)&(Kernel_coefs_action_all_dates_df['condition']==task_condition)

            Kernel_coefs_action_tgt = np.vstack(list(Kernel_coefs_action_all_dates_df[ind]['Kernel_average']))

            ind_nan = np.isnan(np.sum(Kernel_coefs_action_tgt,axis=1)) # exist because of failed pull in SR
            Kernel_coefs_action_tgt = Kernel_coefs_action_tgt[~ind_nan,:]

            # k means clustering
            # run clustering on the 15 or 2 dimension PC space (for doPCA), or the whole dataset or 2 dimension (for doTSNE)
            pca = PCA(n_components=10)
            Kernel_coefs_action_pca = pca.fit_transform(Kernel_coefs_action_tgt.transpose())

            # Get the explained variance ratio
            explained_variance = pca.explained_variance_ratio_

            # Calculate the cumulative explained variance
            cumulative_variance = np.cumsum(explained_variance)

            # Plot the cumulative explained variance
            ax_right[ibhvname].plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker='o', linestyle='--', color='blue', alpha=0.7)
            ax_right[ibhvname].set_xlabel('Number of the principle component')
            ax_right[ibhvname].set_ylabel('Cumulative Percentage of Variance Explained')
            ax_right[ibhvname].set_title(bhv_name)
            ax_right[ibhvname].set_xticks(np.arange(1, len(cumulative_variance) + 1, 1))
            ax_right[ibhvname].grid(True)
            # 
            # if ibhvname == nbhv_names - 1:
            #     ax_right[ibhvname].set_xlabel('Number of the principle component')

            # plot the PCs
            for iPC_toplot in np.arange(0,nPC_toplot,1):

                PCtoplot = Kernel_coefs_action_pca[:,iPC_toplot]

                trig_twin = [-4,4] # in the unit of second
                xxx = np.arange(trig_twin[0]*fps,trig_twin[1]*fps,1)

                ax_left[iPC_toplot+ibhvname*nPC_toplot].plot(xxx, PCtoplot, 'k')
                ax_left[iPC_toplot+ibhvname*nPC_toplot].plot([0,0],[np.nanmin(PCtoplot)*1.1,np.nanmax(PCtoplot)*1.1],'k--')

                ax_left[iPC_toplot+ibhvname*nPC_toplot].set_title(bhv_name+' kernel PC'+str(iPC_toplot+1))

                if iPC_toplot == nPC_toplot - 1:
                    ax_left[iPC_toplot+ibhvname*nPC_toplot].set_xlabel('time (s)')

        
        # Adjust layout
        plt.tight_layout()
        plt.show()        
        
        if (animal1_filenames[0] == 'Kanga') | (animal2_filenames[0] == 'Kanga'):
            recordedAnimal = 'Kanga'
        elif (animal1_filenames[0] == 'Dodson') | (animal2_filenames[0] == 'Dodson'):
            recordedAnimal = 'Dodson'

        savefig = 1
        if savefig:
            figsavefolder = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents_GLMfitting_singlecam/"+cameraID+"/"+recordedAnimal+"_neuralGLM/"

            if not os.path.exists(figsavefolder):
                os.makedirs(figsavefolder)

            fig.savefig(figsavefolder+'action_kernel_coefs_pca_patterns_all_dates'+savefile_sufix+'_'+act_animal+'action_in'+task_condition+'.pdf')
        
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





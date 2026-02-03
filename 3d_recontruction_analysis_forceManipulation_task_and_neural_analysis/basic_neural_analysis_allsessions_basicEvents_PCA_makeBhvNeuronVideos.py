#!/usr/bin/env python
# coding: utf-8

# ### Basic neural activity analysis with single camera tracking
# #### analyze the firing rate PC1,2,3
# #### making the demo videos
# #### analyze the spike triggered pull and gaze ditribution

# In[ ]:


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
from dPCA import dPCA
import string
import warnings
import pickle
import json

from scipy.ndimage import gaussian_filter1d

import os
import glob
import random
from time import time

from pgmpy.models import BayesianModel
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch,BicScore
from pgmpy.base import DAG
import networkx as nx


# ### function - get body part location for each pair of cameras

# In[ ]:


from ana_functions.body_part_locs_eachpair import body_part_locs_eachpair
from ana_functions.body_part_locs_singlecam import body_part_locs_singlecam


# ### function - align the two cameras

# In[ ]:


from ana_functions.camera_align import camera_align       


# ### function - merge the two pairs of cameras

# In[ ]:


from ana_functions.camera_merge import camera_merge


# ### function - find social gaze time point

# In[ ]:


from ana_functions.find_socialgaze_timepoint import find_socialgaze_timepoint
from ana_functions.find_socialgaze_timepoint_singlecam import find_socialgaze_timepoint_singlecam
from ana_functions.find_socialgaze_timepoint_singlecam_wholebody import find_socialgaze_timepoint_singlecam_wholebody


# ### function - define time point of behavioral events

# In[ ]:


from ana_functions.bhv_events_timepoint import bhv_events_timepoint
from ana_functions.bhv_events_timepoint_singlecam import bhv_events_timepoint_singlecam


# ### function - plot behavioral events

# In[ ]:


from ana_functions.plot_bhv_events import plot_bhv_events
from ana_functions.plot_bhv_events_levertube import plot_bhv_events_levertube
from ana_functions.draw_self_loop import draw_self_loop
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection


# ### function - plot inter-pull interval

# In[ ]:


from ana_functions.plot_interpull_interval import plot_interpull_interval


# ### function - make demo videos with skeleton and inportant vectors

# In[ ]:


from ana_functions.tracking_video_singlecam_demo import tracking_video_singlecam_demo
from ana_functions.tracking_video_singlecam_wholebody_demo import tracking_video_singlecam_wholebody_demo
from ana_functions.tracking_video_singlecam_wholebody_withNeuron_demo import tracking_video_singlecam_wholebody_withNeuron_demo
from ana_functions.tracking_video_singlecam_wholebody_withNeuron_sepbhv_demo import tracking_video_singlecam_wholebody_withNeuron_sepbhv_demo
from ana_functions.tracking_frame_singlecam_wholebody_withNeuron_sepbhv_demo import tracking_frame_singlecam_wholebody_withNeuron_sepbhv_demo


# ### function - interval between all behavioral events

# In[ ]:


from ana_functions.bhv_events_interval import bhv_events_interval


# ### function - spike analysis

# In[ ]:


from ana_functions.spike_analysis_FR_calculation import spike_analysis_FR_calculation
from ana_functions.plot_spike_triggered_singlecam_bhvevent import plot_spike_triggered_singlecam_bhvevent
from ana_functions.plot_bhv_events_aligned_FR import plot_bhv_events_aligned_FR
from ana_functions.plot_strategy_aligned_FR import plot_strategy_aligned_FR


# ### function - PCA projection

# In[ ]:


from ana_functions.PCA_around_bhv_events import PCA_around_bhv_events
from ana_functions.PCA_around_bhv_events_video import PCA_around_bhv_events_video
from ana_functions.confidence_ellipse import confidence_ellipse


# ### function - train the dynamic bayesian network - multi time lag (3 lags)

# In[ ]:


from ana_functions.train_DBN_multiLag_withNeuron import train_DBN_multiLag
from ana_functions.train_DBN_multiLag_withNeuron import train_DBN_multiLag_create_df_only
from ana_functions.train_DBN_multiLag_withNeuron import train_DBN_multiLag_training_only
from ana_functions.train_DBN_multiLag_withNeuron import graph_to_matrix
from ana_functions.train_DBN_multiLag_withNeuron import get_weighted_dags
from ana_functions.train_DBN_multiLag_withNeuron import get_significant_edges
from ana_functions.train_DBN_multiLag_withNeuron import threshold_edges
from ana_functions.train_DBN_multiLag_withNeuron import Modulation_Index
from ana_functions.EfficientTimeShuffling import EfficientShuffle
from ana_functions.AicScore import AicScore


# ## Analyze each session

# ### prepare the basic behavioral data (especially the time stamps for each bhv events)

# In[ ]:


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

# force manipulation type
# SR_bothchange: self reward, both forces changed
# CO_bothchange: 1s cooperation, both forces changed
# CO_A1change: 1s cooperation, animal 1 forces changed
# CO_A2change: 1s cooperation, animal 2 forces changed

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

    
# define bhv events and spike aligned summarizing variables     
spike_trig_events_all_dates = dict.fromkeys(dates_list, [])

bhvevents_aligned_FR_all_dates = dict.fromkeys(dates_list, [])
bhvevents_aligned_FR_allevents_all_dates = dict.fromkeys(dates_list, [])

strategy_aligned_FR_all_dates = dict.fromkeys(dates_list, [])
strategy_aligned_FR_allevents_all_dates = dict.fromkeys(dates_list, [])

# where to save the summarizing data
data_saved_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/3d_recontruction_analysis_forceManipulation_task_data_saved/'

# neural data folder
neural_data_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/Marmoset_neural_recording/'

    


# In[ ]:


# basic behavior analysis (define time stamps for each bhv events, etc)

try:
    if redo_anystep:
        dummy
    
    # load saved data
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'

    with open(data_saved_subfolder+'/spike_trig_events_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        spike_trig_events_all_dates = pickle.load(f) 
        
    with open(data_saved_subfolder+'/bhvevents_aligned_FR_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        bhvevents_aligned_FR_all_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/bhvevents_aligned_FR_allevents_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        bhvevents_aligned_FR_allevents_all_dates = pickle.load(f) 
        
    with open(data_saved_subfolder+'/strategy_aligned_FR_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        strategy_aligned_FR_all_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/strategy_aligned_FR_allevents_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        strategy_aligned_FR_allevents_all_dates = pickle.load(f) 
        
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
           
            
        # calculate the firing rate
        # FR_kernel = 0.20 # in the unit of second
        FR_kernel = 1/30 # in the unit of second # 1/30 same resolution as the video recording
        # FR_kernel is sent to to be this if want to explore it's relationship with continuous trackng data

        totalsess_time_forFR = np.floor(np.shape(output_look_ornot['look_at_lever_or_not_merge']['dodson'])[0]/30)  # to match the total time of the video recording
        _,FR_timepoint_allch,FR_allch,FR_zscore_allch = spike_analysis_FR_calculation(fps, FR_kernel, totalsess_time_forFR,
                                                                                      spike_clusters_data, spike_time_data)
        # _,FR_timepoint_allch,FR_allch,FR_zscore_allch = spike_analysis_FR_calculation(fps,FR_kernel,totalsess_time_forFR,
        #                                                                              spike_channels_data, spike_time_data)
        
        
        
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
        
        force12_uniques_names =  [f"{force12_uniques[0][i]}&{force12_uniques[1][i]}" for i in range(force12_uniques.shape[1])]

        # 
        # initialize some variables
        bhvevents_aligned_FR_all_dates[date_tgt] = dict.fromkeys(force12_uniques_names)
        bhvevents_aligned_FR_allevents_all_dates[date_tgt] = dict.fromkeys(force12_uniques_names)
        #
        strategy_aligned_FR_all_dates[date_tgt] = dict.fromkeys(force12_uniques_names)
        strategy_aligned_FR_allevents_all_dates[date_tgt] = dict.fromkeys(force12_uniques_names)
        #
        spike_trig_events_all_dates[date_tgt] = dict.fromkeys(force12_uniques_names)
            
        #    
        for itrialtype in np.arange(0,ntrialtypes,1):
            force1_unique = force12_uniques[0,itrialtype]
            force2_unique = force12_uniques[1,itrialtype]
            
            force12_unique_name = str(force1_unique)+'&'+str(force2_unique)
            
            ind = np.isin(lever1force_list,force1_unique) & np.isin(lever2force_list,force2_unique)
            
            trialID_itrialtype = trialID_list[ind]
            
            tasktype_itrialtype = np.unique(tasktype_list[ind])
            coop_thres_itrialtype = np.unique(coop_thres_list[ind])
            
            # analyze behavior results
            bhv_data_itrialtype = bhv_data[np.isin(bhv_data['trial_number'],trialID_itrialtype)]
            #
            # block time
            block_starttime = bhv_data_itrialtype[bhv_data_itrialtype['behavior_events']==0]['time_points'].iloc[0]
            block_endtime = bhv_data_itrialtype[bhv_data_itrialtype['behavior_events']==9]['time_points'].iloc[-1]
    
            print(block_starttime)
            print(block_endtime)
            
            # only pick time in each block
            # for behavioral variables
            time_point_pull1_iblock = time_point_pull1[(time_point_pull1<=block_endtime)&(time_point_pull1>=block_starttime)]
            time_point_pull2_iblock = time_point_pull2[(time_point_pull2<=block_endtime)&(time_point_pull2>=block_starttime)]
            oneway_gaze1_iblock = oneway_gaze1[(oneway_gaze1<=block_endtime)&(oneway_gaze1>=block_starttime)]
            oneway_gaze2_iblock = oneway_gaze2[(oneway_gaze2<=block_endtime)&(oneway_gaze2>=block_starttime)]
            mutual_gaze1_iblock = mutual_gaze1[(mutual_gaze1<=block_endtime)&(mutual_gaze1>=block_starttime)]
            mutual_gaze2_iblock = mutual_gaze2[(mutual_gaze2<=block_endtime)&(mutual_gaze2>=block_starttime)]
            #
            time_point_pulls_succfail_iblock = time_point_pulls_succfail.copy()
            time_point_pulls_succfail_iblock['pull1_succ'] = time_point_pulls_succfail_iblock['pull1_succ'][(time_point_pulls_succfail_iblock['pull1_succ']<=block_endtime)&(time_point_pulls_succfail_iblock['pull1_succ']>=block_starttime)]
            time_point_pulls_succfail_iblock['pull2_succ'] = time_point_pulls_succfail_iblock['pull2_succ'][(time_point_pulls_succfail_iblock['pull2_succ']<=block_endtime)&(time_point_pulls_succfail_iblock['pull2_succ']>=block_starttime)]
            time_point_pulls_succfail_iblock['pull1_fail'] = time_point_pulls_succfail_iblock['pull1_fail'][(time_point_pulls_succfail_iblock['pull1_fail']<=block_endtime)&(time_point_pulls_succfail_iblock['pull1_fail']>=block_starttime)]
            time_point_pulls_succfail_iblock['pull2_fail'] = time_point_pulls_succfail_iblock['pull2_fail'][(time_point_pulls_succfail_iblock['pull2_fail']<=block_endtime)&(time_point_pulls_succfail_iblock['pull2_fail']>=block_starttime)]

            # for neural data
            spike_clusters_data_iblock = spike_clusters_data[(spike_time_data<=block_endtime*fps)&(spike_time_data>=block_starttime*fps)]
            spike_time_data_iblock = spike_time_data[(spike_time_data<=block_endtime*fps)&(spike_time_data>=block_starttime*fps)]
            spike_channels_data_iblock = spike_channels_data[(spike_time_data<=block_endtime*fps)&(spike_time_data>=block_starttime*fps)]
            
            # remove block that does not have enough variables
            if (
                (np.shape(time_point_pull1_iblock)[0] < 3) or
                (np.shape(time_point_pull2_iblock)[0] < 3) or
                (np.shape(oneway_gaze1_iblock)[0] < 3) or
                (np.shape(oneway_gaze2_iblock)[0] < 3) or
                (np.shape(mutual_gaze1_iblock)[0] < 3) or
                (np.shape(mutual_gaze2_iblock)[0] < 3) or 
                (np.shape(spike_time_data_iblock)[0] < 3)
                ):
                continue

        
            # behavioral events aligned firing rate for each unit
            if 1: 
                print('plot event aligned firing rate')
                #
                savefig = 1
                save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt+"/"+force12_unique_name+"/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                #
                aligntwins = 4 # 5 second
                gaze_thresold = 0.2 # min length threshold to define if a gaze is real gaze or noise, in the unit of second 
                #
                bhvevents_aligned_FR_average_all,bhvevents_aligned_FR_allevents_all = plot_bhv_events_aligned_FR(date_tgt,savefig,save_path, animal1, animal2,
                                           time_point_pull1_iblock,time_point_pull2_iblock,time_point_pulls_succfail_iblock,
                                           oneway_gaze1_iblock,oneway_gaze2_iblock,mutual_gaze1_iblock,mutual_gaze2_iblock,
                                           gaze_thresold,totalsess_time_forFR,
                                           aligntwins,fps,FR_timepoint_allch,FR_zscore_allch,clusters_info_data)

                bhvevents_aligned_FR_all_dates[date_tgt][force12_unique_name] = bhvevents_aligned_FR_average_all
                bhvevents_aligned_FR_allevents_all_dates[date_tgt][force12_unique_name] = bhvevents_aligned_FR_allevents_all


            # the three strategy aligned firing rate for each unit
            if 1: 
                print('plot strategy aligned firing rate')
                #
                savefig = 1
                save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt+"/"+force12_unique_name+"/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                #
                stg_twins = 3 # 3s, the behavioral event interval used to define strategy, consistent with DBN 3s time lags
                aligntwins = 4 # 5 second
                gaze_thresold = 0.2 # min length threshold to define if a gaze is real gaze or noise, in the unit of second 
                #
                strategy_aligned_FR_average_all,strategy_aligned_FR_allevents_all = plot_strategy_aligned_FR(date_tgt,savefig,save_path, animal1, animal2,
                                           time_point_pull1_iblock,time_point_pull2_iblock,time_point_pulls_succfail_iblock,
                                           oneway_gaze1_iblock,oneway_gaze2_iblock,mutual_gaze1_iblock,mutual_gaze2_iblock,
                                           gaze_thresold,totalsess_time_forFR,
                                           aligntwins,stg_twins,fps,FR_timepoint_allch,FR_zscore_allch,clusters_info_data)

                strategy_aligned_FR_all_dates[date_tgt][force12_unique_name] = strategy_aligned_FR_average_all
                strategy_aligned_FR_allevents_all_dates[date_tgt][force12_unique_name] = strategy_aligned_FR_allevents_all


            #
            # Run PCA analysis
            FR_zscore_allch_np_merged = np.array(pd.DataFrame(FR_zscore_allch).T)
            FR_zscore_allch_np_merged = FR_zscore_allch_np_merged[~np.isnan(np.sum(FR_zscore_allch_np_merged,axis=1)),:]
            # # run PCA on the entire session
            pca = PCA(n_components=3)
            FR_zscore_allch_PCs = pca.fit_transform(FR_zscore_allch_np_merged.T)
            #
            # # run PCA around the -PCAtwins to PCAtwins for each behavioral events
            PCAtwins = 4 # 5 second
            gaze_thresold = 0.5 # min length threshold to define if a gaze is real gaze or noise, in the unit of second 
            savefigs = 0 
            if 0:
                PCA_around_bhv_events(FR_timepoint_allch,FR_zscore_allch_np_merged,time_point_pull1,time_point_pull2,time_point_pulls_succfail, 
                              oneway_gaze1,oneway_gaze2,mutual_gaze1,mutual_gaze2,gaze_thresold,totalsess_time_forFR,PCAtwins,fps,
                              savefigs,data_saved_folder,cameraID,animal1_filename,animal2_filename,date_tgt)
            if 0:
                if np.isin(animal1, ['dodson','dannon']):
                    PCA_around_bhv_events_video(FR_timepoint_allch,FR_zscore_allch_np_merged,time_point_pull1_iblock,time_point_pull2_iblock,time_point_pulls_succfail_iblock,
                                      oneway_gaze1_iblock,oneway_gaze2_iblock,mutual_gaze1_iblock,mutual_gaze2_iblock,gaze_thresold,totalsess_time_forFR,PCAtwins,fps,
                                      data_saved_folder,cameraID,animal1_filename,animal2_filename,date_tgt)
                elif np.isin(animal2, ['dodson','dannon']):
                    time_point_pulls_succfail_rev_iblock = time_point_pulls_succfail_iblock.copy()
                    time_point_pulls_succfail_rev_iblock['pull1_succ'] = time_point_pulls_succfail_iblock['pull2_succ']
                    time_point_pulls_succfail_rev_iblock['pull1_fail'] = time_point_pulls_succfail_iblock['pull2_fail']
                    time_point_pulls_succfail_rev_iblock['pull2_succ'] = time_point_pulls_succfail_iblock['pull1_succ']
                    time_point_pulls_succfail_rev_iblock['pull2_fail'] = time_point_pulls_succfail_iblock['pull1_fail']
                    PCA_around_bhv_events_video(FR_timepoint_allch,FR_zscore_allch_np_merged,time_point_pull2_iblock,time_point_pull1_iblock,time_point_pulls_succfail_rev_iblock, 
                                      oneway_gaze2_iblock,oneway_gaze1_iblock,mutual_gaze2_iblock,mutual_gaze1_iblock,gaze_thresold,totalsess_time_forFR,PCAtwins,fps,
                                      data_saved_folder,cameraID,animal1_filename,animal2_filename,date_tgt)



            # do the spike triggered average of different bhv variables, for the single camera tracking, look at the pulling and social gaze actions
            # the goal is to get a sense for glm
            if 1: 
                print('plot spike triggered bhv variables')

                savefig = 1
                save_path = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt+"/"+force12_unique_name+"/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                #
                do_shuffle = 0
                #
                min_length = np.shape(look_at_other_or_not_merge['dodson'])[0] # frame numbers of the video recording
                #
                trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s

                gaze_thresold = 0.2

                stg_twins = 3 # 3s, the behavioral event interval used to define strategy, consistent with DBN 3s time lags
                #
                spike_trig_average_all =  plot_spike_triggered_singlecam_bhvevent(date_tgt,savefig,save_path, animal1, animal2, session_start_time,min_length, trig_twins,
                                                                              stg_twins, time_point_pull1_iblock, time_point_pull2_iblock, time_point_pulls_succfail_iblock,
                                                                              oneway_gaze1_iblock,oneway_gaze2_iblock,mutual_gaze1_iblock,mutual_gaze2_iblock,gaze_thresold,animalnames_videotrack,
                                                                              spike_clusters_data_iblock, spike_time_data_iblock,spike_channels_data_iblock,do_shuffle)

                spike_trig_events_all_dates[date_tgt][force12_unique_name] = spike_trig_average_all

            


        

        
        
        
        
        
        
        
    # save data
    if 1:
        
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
                
        # with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
        #     pickle.dump(DBN_input_data_alltypes, f)

            
        with open(data_saved_subfolder+'/spike_trig_events_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(spike_trig_events_all_dates, f)  
    
        with open(data_saved_subfolder+'/bhvevents_aligned_FR_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(bhvevents_aligned_FR_all_dates, f) 
        with open(data_saved_subfolder+'/bhvevents_aligned_FR_allevents_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(bhvevents_aligned_FR_allevents_all_dates, f) 
            
        with open(data_saved_subfolder+'/strategy_aligned_FR_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(strategy_aligned_FR_all_dates, f) 
        with open(data_saved_subfolder+'/strategy_aligned_FR_allevents_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(strategy_aligned_FR_allevents_all_dates, f) 
    
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### plot 
# #### plot the PCs

# In[ ]:


if 0:
    fig, axs = plt.subplots(12,1)
    fig.set_figheight(20*1)
    fig.set_figwidth(5*3)

    x_lims = [0,totalsess_time_forFR]
    PC1min = np.min(FR_zscore_allch_PCs[:,0])
    PC1max = np.max(FR_zscore_allch_PCs[:,0])
    PC2min = np.min(FR_zscore_allch_PCs[:,1])
    PC2max = np.max(FR_zscore_allch_PCs[:,1])
    PC3min = np.min(FR_zscore_allch_PCs[:,2])
    PC3max = np.max(FR_zscore_allch_PCs[:,2])

    for iplotype in np.arange(0,4,1):

        if iplotype == 0:
            eventplot  = np.array(time_point_pull1)
            eventplotname = 'animal1_pull'
        elif iplotype == 1:
            eventplot  = np.array(time_point_pull2)
            eventplotname = 'animal2_pull'
        elif iplotype == 2:
            eventplot  = np.hstack([oneway_gaze1,mutual_gaze1])
            eventplotname = 'animal1_gaze'
        elif iplotype == 3:
            eventplot  = np.hstack([oneway_gaze2,mutual_gaze2])
            eventplotname = 'animal2_gaze'

        # plot 1
        nevents = np.shape(eventplot)[0]
        for ievent in np.arange(0,nevents,1):
            axs[0+3*iplotype].plot([eventplot[ievent],eventplot[ievent]],[PC1min,PC1max],'k-')
        axs[0+3*iplotype].set_xlim(x_lims[0],x_lims[1])
        axs[0+3*iplotype].set_ylim(PC1min,PC1max)
        #
        axs[0+3*iplotype].plot(FR_timepoint_allch,FR_zscore_allch_PCs[:,0])
        axs[0+3*iplotype].set_xlim(x_lims[0],x_lims[1])
        axs[0+3*iplotype].set_ylabel('PC1\n'+eventplotname)

        # plot 2
        nevents = np.shape(eventplot)[0]
        for ievent in np.arange(0,nevents,1):
            axs[1+3*iplotype].plot([eventplot[ievent],eventplot[ievent]],[PC2min,PC2max],'k-')
        axs[1+3*iplotype].set_xlim(x_lims[0],x_lims[1])
        axs[1+3*iplotype].set_ylim(PC2min,PC2max)
        #
        axs[1+3*iplotype].plot(FR_timepoint_allch,FR_zscore_allch_PCs[:,1])
        axs[1+3*iplotype].set_xlim(x_lims[0],x_lims[1])
        axs[1+3*iplotype].set_ylabel('PC2\n'+eventplotname)

        # plot 3
        nevents = np.shape(eventplot)[0]
        for ievent in np.arange(0,nevents,1):
            axs[2+3*iplotype].plot([eventplot[ievent],eventplot[ievent]],[PC3min,PC3max],'k-')
        axs[2+3*iplotype].set_xlim(x_lims[0],x_lims[1])
        axs[2+3*iplotype].set_ylim(PC3min,PC3max)
        #
        axs[2+3*iplotype].plot(FR_timepoint_allch,FR_zscore_allch_PCs[:,2])
        axs[2+3*iplotype].set_xlim(x_lims[0],x_lims[1])
        axs[2+3*iplotype].set_ylabel('PC3\n'+eventplotname)


# #### analyze the bhv aligned firing rate across all dates
# #### plot the tsne or PCA clusters

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

doPCA = 1
doTSNE = 0

bhvevents_aligned_FR_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_average'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
       
    bhv_types = list(bhvevents_aligned_FR_all_dates[date_tgt].keys())

    for ibhv_type in bhv_types:

        clusterIDs = list(bhvevents_aligned_FR_all_dates[date_tgt][ibhv_type].keys())

        for iclusterID in clusterIDs:

            ichannelID = bhvevents_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = bhvevents_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['FR_average']

            bhvevents_aligned_FR_all_dates_df = bhvevents_aligned_FR_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_average':iFR_average,
                                                                                   }, ignore_index=True)
            
if 0:
    # normalize FR_average for each unit
    nspikeunits = np.shape(bhvevents_aligned_FR_all_dates_df)[0]
    for ispikeunit in np.arange(0,nspikeunits,1):
        stevent = bhvevents_aligned_FR_all_dates_df['FR_average'][ispikeunit]
        stevent_norm = (stevent-np.nanmin(stevent))/(np.nanmax(stevent)-np.nanmin(stevent))
        bhvevents_aligned_FR_all_dates_df['FR_average'][ispikeunit] = stevent_norm            
        
# only focus on the certain act animal and certain bhv_name
# act_animals_all = ['kanga']
# bhv_names_all = ['leverpull_prob']
act_animals_all = np.unique(bhvevents_aligned_FR_all_dates_df['act_animal'])
bhv_names_all = np.unique(bhvevents_aligned_FR_all_dates_df['bhv_name'])
#
nact_animals = np.shape(act_animals_all)[0]
nbhv_names = np.shape(bhv_names_all)[0]

# set for plot
# plot all units
fig1, axs1 = plt.subplots(nact_animals,nbhv_names)
fig1.set_figheight(6*nact_animals)
fig1.set_figwidth(6*nbhv_names)

# plot all units but separate different days
fig2, axs2 = plt.subplots(nact_animals,nbhv_names)
fig2.set_figheight(6*nact_animals)
fig2.set_figwidth(6*nbhv_names)

# plot all units but seprate different channels
fig3, axs3 = plt.subplots(nact_animals,nbhv_names)
fig3.set_figheight(4*nact_animals)
fig3.set_figwidth(4*nbhv_names)

# plot all units but separate different conditions
fig4, axs4 = plt.subplots(nact_animals,nbhv_names)
fig4.set_figheight(6*nact_animals)
fig4.set_figwidth(6*nbhv_names)

# spike triggered average for different task conditions
# # to be save, prepare for five conditions
fig6, axs6 = plt.subplots(nact_animals*5,nbhv_names)
fig6.set_figheight(6*nact_animals*5)
fig6.set_figwidth(6*nbhv_names)
# fig6, axs6 = plt.subplots(nact_animals,nbhv_names)
# fig6.set_figheight(6*nact_animals)
# fig6.set_figwidth(6*nbhv_names)

# plot all units but separate different k-mean cluster
fig5, axs5 = plt.subplots(nact_animals,nbhv_names)
fig5.set_figheight(6*nact_animals)
fig5.set_figwidth(6*nbhv_names)

# spike triggered average for different k-mean cluster
# to be save, prepare for 14 clusters
fig7, axs7 = plt.subplots(nact_animals*14,nbhv_names)
fig7.set_figheight(6*nact_animals*14)
fig7.set_figwidth(6*nbhv_names)

# stacked bar plot to show the cluster distribution of each conditions
fig8, axs8 = plt.subplots(nact_animals,nbhv_names)
fig8.set_figheight(6*nact_animals)
fig8.set_figwidth(6*nbhv_names)

#
for ianimal in np.arange(0,nact_animals,1):
    
    act_animal = act_animals_all[ianimal]
    
    for ibhvname in np.arange(0,nbhv_names,1):
        
        bhv_name = bhv_names_all[ibhvname]
        
        ind = (bhvevents_aligned_FR_all_dates_df['act_animal']==act_animal)&(bhvevents_aligned_FR_all_dates_df['bhv_name']==bhv_name)
        
        bhvevents_aligned_FR_tgt = np.vstack(list(bhvevents_aligned_FR_all_dates_df[ind]['FR_average']))
        
        ind_nan = np.isnan(np.sum(bhvevents_aligned_FR_tgt,axis=1)) # exist because of failed pull in SR
        bhvevents_aligned_FR_tgt = bhvevents_aligned_FR_tgt[~ind_nan,:]
        
        # k means clustering
        # run clustering on the 15 or 2 dimension PC space (for doPCA), or the whole dataset or 2 dimension (for doTSNE)
        pca = PCA(n_components=10)
        bhvevents_aligned_FR_pca = pca.fit_transform(bhvevents_aligned_FR_tgt)
        tsne = TSNE(n_components=2, random_state=0)
        bhvevents_aligned_FR_tsne = tsne.fit_transform(bhvevents_aligned_FR_tgt)
        #
        range_n_clusters = np.arange(2,8,1)
        silhouette_avg_all = np.ones(np.shape(range_n_clusters))*np.nan
        nkmeancls = np.shape(range_n_clusters)[0]
        #
        for ikmeancl in np.arange(0,nkmeancls,1):
            n_clusters = range_n_clusters[ikmeancl]
            #
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            # cluster_labels = clusterer.fit_predict(bhvevents_aligned_FR_tgt)
            if doPCA:
                cluster_labels = clusterer.fit_predict(bhvevents_aligned_FR_pca)
            if doTSNE:
                cluster_labels = clusterer.fit_predict(bhvevents_aligned_FR_tgt)
                # cluster_labels = clusterer.fit_predict(bhvevents_aligned_FR_tsne)
            #
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            # silhouette_avg = silhouette_score(bhvevents_aligned_FR_tgt, cluster_labels)
            if doPCA:
                silhouette_avg = silhouette_score(bhvevents_aligned_FR_pca, cluster_labels)
            if doTSNE:
                silhouette_avg = silhouette_score(bhvevents_aligned_FR_tgt, cluster_labels)
                # silhouette_avg = silhouette_score(bhvevents_aligned_FR_tsne, cluster_labels)
            #
            silhouette_avg_all[ikmeancl] = silhouette_avg
        #
        best_k_num = range_n_clusters[silhouette_avg_all==np.nanmax(silhouette_avg_all)][0]
        #
        clusterer = KMeans(n_clusters=best_k_num, random_state=0)
        # kmean_cluster_labels = clusterer.fit_predict(bhvevents_aligned_FR_tgt)
        if doPCA:
            kmean_cluster_labels = clusterer.fit_predict(bhvevents_aligned_FR_pca)
        if doTSNE:
            kmean_cluster_labels = clusterer.fit_predict(bhvevents_aligned_FR_tgt)
            # kmean_cluster_labels = clusterer.fit_predict(bhvevents_aligned_FR_tsne)
    
    
        # run PCA and TSNE     
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, random_state=0)
        #
        bhvevents_aligned_FR_pca = pca.fit_transform(bhvevents_aligned_FR_tgt)
        bhvevents_aligned_FR_tsne = tsne.fit_transform(bhvevents_aligned_FR_tgt)
        
        # plot all units
        # plot the tsne
        if doTSNE:
            axs1[ianimal,ibhvname].plot(bhvevents_aligned_FR_tsne[:,0],bhvevents_aligned_FR_tsne[:,1],'.')
        # plot the pca
        if doPCA:
            axs1[ianimal,ibhvname].plot(bhvevents_aligned_FR_pca[:,0],bhvevents_aligned_FR_pca[:,1],'.')
        
        axs1[ianimal,ibhvname].set_xticklabels([])
        axs1[ianimal,ibhvname].set_yticklabels([])
        axs1[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        
        
        # plot all units, but seprate different dates
        dates_forplot = np.unique(bhvevents_aligned_FR_all_dates_df[ind]['dates'])
        for idate_forplot in dates_forplot:
            ind_idate = list(bhvevents_aligned_FR_all_dates_df[ind]['dates']==idate_forplot)
            ind_idate = list(np.array(ind_idate)[~ind_nan])
            #
            # plot the tsne
            if doTSNE:
                axs2[ianimal,ibhvname].plot(bhvevents_aligned_FR_tsne[ind_idate,0],bhvevents_aligned_FR_tsne[ind_idate,1],
                                        '.',label=idate_forplot)
            # plot the pca
            if doPCA:
                axs2[ianimal,ibhvname].plot(bhvevents_aligned_FR_pca[ind_idate,0],bhvevents_aligned_FR_pca[ind_idate,1],
                                        '.',label=idate_forplot)
            #
        axs2[ianimal,ibhvname].set_xticklabels([])
        axs2[ianimal,ibhvname].set_yticklabels([])
        axs2[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        axs2[ianimal,ibhvname].legend()
        
        
        # plot all units, but seprate different channels
        chs_forplot = np.unique(bhvevents_aligned_FR_all_dates_df[ind]['channelID'])
        for ich_forplot in chs_forplot:
            ind_ich = list(bhvevents_aligned_FR_all_dates_df[ind]['channelID']==ich_forplot)
            ind_ich = list(np.array(ind_ich)[~ind_nan])
            #
            # plot the tsne
            if doTSNE:
                axs3[ianimal,ibhvname].plot(bhvevents_aligned_FR_tsne[ind_ich,0],bhvevents_aligned_FR_tsne[ind_ich,1],
                                        '.',label=str(ich_forplot))
            # plot the pca
            if doPCA:
                axs3[ianimal,ibhvname].plot(bhvevents_aligned_FR_pca[ind_ich,0],bhvevents_aligned_FR_pca[ind_ich,1],
                                        '.',label=str(ich_forplot))
            #
        axs3[ianimal,ibhvname].set_xticklabels([])
        axs3[ianimal,ibhvname].set_yticklabels([])
        axs3[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        axs3[ianimal,ibhvname].legend()
        
        
        # plot all units, but seprate different task conditions
        cons_forplot = np.unique(bhvevents_aligned_FR_all_dates_df[ind]['condition'])
        for icon_forplot in cons_forplot:
            ind_icon = list(bhvevents_aligned_FR_all_dates_df[ind]['condition']==icon_forplot)
            ind_icon = list(np.array(ind_icon)[~ind_nan])
            #
            # plot the tsne
            if doTSNE:
                axs4[ianimal,ibhvname].plot(bhvevents_aligned_FR_tsne[ind_icon,0],bhvevents_aligned_FR_tsne[ind_icon,1],
                                        '.',label=icon_forplot)
            # plot the pca
            if doPCA:
                axs4[ianimal,ibhvname].plot(bhvevents_aligned_FR_pca[ind_icon,0],bhvevents_aligned_FR_pca[ind_icon,1],
                                        '.',label=icon_forplot)
            #
        axs4[ianimal,ibhvname].set_xticklabels([])
        axs4[ianimal,ibhvname].set_yticklabels([])
        axs4[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        axs4[ianimal,ibhvname].legend()
    
        # plot the mean spike trigger average trace across neurons in each condition
        trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
        xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)
        #
        cons_forplot = np.unique(bhvevents_aligned_FR_all_dates_df[ind]['condition'])
        icon_ind = 0
        for icon_forplot in cons_forplot:
            ind_icon = list(bhvevents_aligned_FR_all_dates_df[ind]['condition']==icon_forplot)
            ind_icon = list(np.array(ind_icon)[~ind_nan])
            #
            mean_trig_trace_icon = np.nanmean(bhvevents_aligned_FR_tgt[ind_icon,:],axis=0)
            std_trig_trace_icon = np.nanstd(bhvevents_aligned_FR_tgt[ind_icon,:],axis=0)
            sem_trig_trace_icon = np.nanstd(bhvevents_aligned_FR_tgt[ind_icon,:],axis=0)/np.sqrt(np.shape(bhvevents_aligned_FR_tgt[ind_icon,:])[0])
            itv95_trig_trace_icon = 1.96*sem_trig_trace_icon
            #
            if 1:
            # plot each trace in a seperate traces
                axs6[ianimal*5+icon_ind,ibhvname].errorbar(xxx_forplot,mean_trig_trace_icon,yerr=itv95_trig_trace_icon,
                                                           color='#E0E0E0',ecolor='#EEEEEE',label=icon_forplot)
                axs6[ianimal*5+icon_ind,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_icon-itv95_trig_trace_icon),
                                                              np.nanmax(mean_trig_trace_icon+itv95_trig_trace_icon)],'--k')
                axs6[ianimal*5+icon_ind,ibhvname].set_xlabel('time (s)')
                axs6[ianimal*5+icon_ind,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
                axs6[ianimal*5+icon_ind,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
                axs6[ianimal*5+icon_ind,ibhvname].set_title(act_animal+'; '+bhv_name)
                axs6[ianimal*5+icon_ind,ibhvname].legend()
            if 0:
                axs6[ianimal,ibhvname].errorbar(xxx_forplot,mean_trig_trace_icon,yerr=itv95_trig_trace_icon,
                                                label=icon_forplot)
                # axs6[ianimal,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_icon-itv95_trig_trace_icon),
                #                                               np.nanmax(mean_trig_trace_icon+itv95_trig_trace_icon)],'--k')
                axs6[ianimal,ibhvname].plot([0,0],[0,0.1],'--k') 
                axs6[ianimal,ibhvname].set_xlabel('time (s)')
                axs6[ianimal,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
                axs6[ianimal,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
                axs6[ianimal,ibhvname].set_title(act_animal+'; '+bhv_name)
                axs6[ianimal,ibhvname].legend()
            #
            icon_ind = icon_ind + 1
    
    
        # plot all units, but seprate different k-mean clusters
        kms_forplot = np.unique(kmean_cluster_labels)
        for ikm_forplot in kms_forplot:
            ind_ikm = list(kmean_cluster_labels==ikm_forplot)
            #
            # plot the tsne
            if doTSNE:
                axs5[ianimal,ibhvname].plot(bhvevents_aligned_FR_tsne[ind_ikm,0],bhvevents_aligned_FR_tsne[ind_ikm,1],
                                        '.',label=str(ikm_forplot))
            # plot the pca
            if doPCA:
                axs5[ianimal,ibhvname].plot(bhvevents_aligned_FR_pca[ind_ikm,0],bhvevents_aligned_FR_pca[ind_ikm,1],
                                        '.',label=str(ikm_forplot))
            #
        axs5[ianimal,ibhvname].set_xticklabels([])
        axs5[ianimal,ibhvname].set_yticklabels([])
        axs5[ianimal,ibhvname].set_title(act_animal+'; '+bhv_name)
        axs5[ianimal,ibhvname].legend()
        
        # plot the mean spike trigger average trace across neurons in each cluster
        trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
        xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)
        #
        kms_forplot = np.unique(kmean_cluster_labels)
        for ikm_forplot in kms_forplot:
            ind_ikm = list(kmean_cluster_labels==ikm_forplot)
            #
            mean_trig_trace_ikm = np.nanmean(bhvevents_aligned_FR_tgt[ind_ikm,:],axis=0)
            std_trig_trace_ikm = np.nanstd(bhvevents_aligned_FR_tgt[ind_ikm,:],axis=0)
            sem_trig_trace_ikm = np.nanstd(bhvevents_aligned_FR_tgt[ind_ikm,:],axis=0)/np.sqrt(np.shape(bhvevents_aligned_FR_tgt[ind_ikm,:])[0])
            itv95_trig_trace_ikm = 1.96*sem_trig_trace_ikm
            #
            axs7[ianimal*14+ikm_forplot,ibhvname].errorbar(xxx_forplot,mean_trig_trace_ikm,yerr=itv95_trig_trace_ikm,
                                                          color='#E0E0E0',ecolor='#EEEEEE',label='cluster#'+str(ikm_forplot))
            axs7[ianimal*14+ikm_forplot,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_ikm-itv95_trig_trace_ikm),
                                                             np.nanmax(mean_trig_trace_ikm+itv95_trig_trace_ikm)],'--k')
            axs7[ianimal*14+ikm_forplot,ibhvname].set_xlabel('time (s)')
            axs7[ianimal*14+ikm_forplot,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
            axs7[ianimal*14+ikm_forplot,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
            axs7[ianimal*14+ikm_forplot,ibhvname].set_title(act_animal+'; '+bhv_name)
            axs7[ianimal*14+ikm_forplot,ibhvname].legend()
    
    
        # stacked bar plot to show the cluster distribution of each conditions
        df = pd.DataFrame({'cond':np.array(bhvevents_aligned_FR_all_dates_df[ind]['condition'])[~ind_nan],
                           'cluID':kmean_cluster_labels})
        (df.groupby('cond')['cluID'].value_counts(normalize=True)
           .unstack('cluID').plot.bar(stacked=True, ax=axs8[ianimal,ibhvname]))
        axs8[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        
        
    
    
savefig = 1
if savefig:
    figsavefolder = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filenames[0]+"_"+animal2_filenames[0]+"/bhvAlignedFRAver_fig/"

    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    if doTSNE:
        fig1.savefig(figsavefolder+'bhv_aligned_FR_tsne_clusters_all_dates'+savefile_sufix+'.pdf')
        fig2.savefig(figsavefolder+'bhv_aligned_FR_tsne_clusters_all_dates_separated_dates'+savefile_sufix+'.pdf')
        fig3.savefig(figsavefolder+'bhv_aligned_FR_tsne_clusters_all_dates_separated_channels'+savefile_sufix+'.pdf')
        fig4.savefig(figsavefolder+'bhv_aligned_FR_tsne_clusters_all_dates_separated_conditions'+savefile_sufix+'.pdf')
        fig5.savefig(figsavefolder+'bhv_aligned_FR_tsne_clusters_all_dates_separated_kmeanclusters'+savefile_sufix+'.pdf')
        fig6.savefig(figsavefolder+'bhv_aligned_FR_tsne_clusters_all_dates_sttraces_for_conditions'+savefile_sufix+'.pdf')        
        fig7.savefig(figsavefolder+'bhv_aligned_FR_tsne_clusters_all_dates_sttraces_for_kmeanclusters'+savefile_sufix+'.pdf')
        fig8.savefig(figsavefolder+'bhv_aligned_FR_tsne_clusters_kmeanclusters_propotion_each_condition'+savefile_sufix+'.pdf')
        
    if doPCA:
        fig1.savefig(figsavefolder+'bhv_aligned_FR_pca_clusters_all_dates'+savefile_sufix+'.pdf')
        fig2.savefig(figsavefolder+'bhv_aligned_FR_pca_clusters_all_dates_separated_dates'+savefile_sufix+'.pdf')
        fig3.savefig(figsavefolder+'bhv_aligned_FR_pca_clusters_all_dates_separated_channels'+savefile_sufix+'.pdf')
        fig4.savefig(figsavefolder+'bhv_aligned_FR_pca_clusters_all_dates_separated_conditions'+savefile_sufix+'.pdf')
        fig5.savefig(figsavefolder+'bhv_aligned_FR_pca_clusters_all_dates_separated_kmeanclusters'+savefile_sufix+'.pdf')
        fig6.savefig(figsavefolder+'bhv_aligned_FR_pca_clusters_all_dates_sttraces_for_conditions'+savefile_sufix+'.pdf')                           
        fig7.savefig(figsavefolder+'bhv_aligned_FR_pca_clusters_all_dates_sttraces_for_kmeanclusters'+savefile_sufix+'.pdf')
        fig8.savefig(figsavefolder+'bhv_aligned_FR_pca_clusters_kmeanclusters_propotion_each_condition'+savefile_sufix+'.pdf')


# #### analyze the spike triggered behavioral variables across all dates
# #### plot the tsne or PCA clusters

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

doPCA = 1
doTSNE = 0

spike_trig_events_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','st_average'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
    
    act_animals = list(spike_trig_events_all_dates[date_tgt].keys())
    
    for iact_animal in act_animals:
        
        bhv_types = list(spike_trig_events_all_dates[date_tgt][iact_animal].keys())
        
        for ibhv_type in bhv_types:
            
            clusterIDs = list(spike_trig_events_all_dates[date_tgt][iact_animal][ibhv_type].keys())
    
            for iclusterID in clusterIDs:
            
                ichannelID = spike_trig_events_all_dates[date_tgt][iact_animal][ibhv_type][iclusterID]['ch']
                ist_average = spike_trig_events_all_dates[date_tgt][iact_animal][ibhv_type][iclusterID]['st_average']

                spike_trig_events_all_dates_df = spike_trig_events_all_dates_df.append({'dates': date_tgt, 
                                                                                        'condition':task_condition,
                                                                                        'act_animal': iact_animal, 
                                                                                        'bhv_name': ibhv_type,
                                                                                        'clusterID':iclusterID,
                                                                                        'channelID':ichannelID,
                                                                                        'st_average':ist_average,
                                                                                       }, ignore_index=True)
if 0:
    # normalize st_average for each unit
    nspikeunits = np.shape(spike_trig_events_all_dates_df)[0]
    for ispikeunit in np.arange(0,nspikeunits,1):
        stevent = spike_trig_events_all_dates_df['st_average'][ispikeunit]
        stevent_norm = (stevent-np.nanmin(stevent))/(np.nanmax(stevent)-np.nanmin(stevent))
        spike_trig_events_all_dates_df['st_average'][ispikeunit] = stevent_norm            
        
# only focus on the certain act animal and certain bhv_name
# act_animals_all = ['kanga']
# bhv_names_all = ['leverpull_prob']
act_animals_all = np.unique(spike_trig_events_all_dates_df['act_animal'])
bhv_names_all = np.unique(spike_trig_events_all_dates_df['bhv_name'])
#
nact_animals = np.shape(act_animals_all)[0]
nbhv_names = np.shape(bhv_names_all)[0]

# set for plot
# plot all units
fig1, axs1 = plt.subplots(nact_animals,nbhv_names)
fig1.set_figheight(6*nact_animals)
fig1.set_figwidth(6*nbhv_names)

# plot all units but separate different days
fig2, axs2 = plt.subplots(nact_animals,nbhv_names)
fig2.set_figheight(6*nact_animals)
fig2.set_figwidth(6*nbhv_names)

# plot all units but seprate different channels
fig3, axs3 = plt.subplots(nact_animals,nbhv_names)
fig3.set_figheight(4*nact_animals)
fig3.set_figwidth(4*nbhv_names)

# plot all units but separate different conditions
fig4, axs4 = plt.subplots(nact_animals,nbhv_names)
fig4.set_figheight(6*nact_animals)
fig4.set_figwidth(6*nbhv_names)

# spike triggered average for different task conditions
# # to be save, prepare for five conditions
fig6, axs6 = plt.subplots(nact_animals*5,nbhv_names)
fig6.set_figheight(6*nact_animals*5)
fig6.set_figwidth(6*nbhv_names)
# fig6, axs6 = plt.subplots(nact_animals,nbhv_names)
# fig6.set_figheight(6*nact_animals)
# fig6.set_figwidth(6*nbhv_names)

# plot all units but separate different k-mean cluster
fig5, axs5 = plt.subplots(nact_animals,nbhv_names)
fig5.set_figheight(6*nact_animals)
fig5.set_figwidth(6*nbhv_names)

# spike triggered average for different k-mean cluster
# to be save, prepare for 14 clusters
fig7, axs7 = plt.subplots(nact_animals*14,nbhv_names)
fig7.set_figheight(6*nact_animals*14)
fig7.set_figwidth(6*nbhv_names)

# stacked bar plot to show the cluster distribution of each conditions
fig8, axs8 = plt.subplots(nact_animals,nbhv_names)
fig8.set_figheight(6*nact_animals)
fig8.set_figwidth(6*nbhv_names)

#
for ianimal in np.arange(0,nact_animals,1):
    
    act_animal = act_animals_all[ianimal]
    
    for ibhvname in np.arange(0,nbhv_names,1):
        
        bhv_name = bhv_names_all[ibhvname]
        
        ind = (spike_trig_events_all_dates_df['act_animal']==act_animal)&(spike_trig_events_all_dates_df['bhv_name']==bhv_name)
        
        spike_trig_events_tgt = np.vstack(list(spike_trig_events_all_dates_df[ind]['st_average']))
        
        ind_nan = np.isnan(np.sum(spike_trig_events_tgt,axis=1)) # exist because of failed pull in SR
        spike_trig_events_tgt = spike_trig_events_tgt[~ind_nan,:]
        
        # k means clustering
        # run clustering on the 15 or 2 dimension PC space (for doPCA), or the whole dataset or 2 dimension (for doTSNE)
        pca = PCA(n_components=10)
        spike_trig_events_pca = pca.fit_transform(spike_trig_events_tgt)
        tsne = TSNE(n_components=2, random_state=0)
        spike_trig_events_tsne = tsne.fit_transform(spike_trig_events_tgt)
        #
        range_n_clusters = np.arange(2,8,1)
        silhouette_avg_all = np.ones(np.shape(range_n_clusters))*np.nan
        nkmeancls = np.shape(range_n_clusters)[0]
        #
        for ikmeancl in np.arange(0,nkmeancls,1):
            n_clusters = range_n_clusters[ikmeancl]
            #
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            # cluster_labels = clusterer.fit_predict(spike_trig_events_tgt)
            if doPCA:
                cluster_labels = clusterer.fit_predict(spike_trig_events_pca)
            if doTSNE:
                cluster_labels = clusterer.fit_predict(spike_trig_events_tgt)
                # cluster_labels = clusterer.fit_predict(spike_trig_events_tsne)
            #
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            # silhouette_avg = silhouette_score(spike_trig_events_tgt, cluster_labels)
            if doPCA:
                silhouette_avg = silhouette_score(spike_trig_events_pca, cluster_labels)
            if doTSNE:
                silhouette_avg = silhouette_score(spike_trig_events_tgt, cluster_labels)
                # silhouette_avg = silhouette_score(spike_trig_events_tsne, cluster_labels)
            #
            silhouette_avg_all[ikmeancl] = silhouette_avg
        #
        best_k_num = range_n_clusters[silhouette_avg_all==np.nanmax(silhouette_avg_all)][0]
        #
        clusterer = KMeans(n_clusters=best_k_num, random_state=0)
        # kmean_cluster_labels = clusterer.fit_predict(spike_trig_events_tgt)
        if doPCA:
            kmean_cluster_labels = clusterer.fit_predict(spike_trig_events_pca)
        if doTSNE:
            kmean_cluster_labels = clusterer.fit_predict(spike_trig_events_tgt)
            # kmean_cluster_labels = clusterer.fit_predict(spike_trig_events_tsne)
    
    
        # run PCA and TSNE     
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, random_state=0)
        #
        spike_trig_events_pca = pca.fit_transform(spike_trig_events_tgt)
        spike_trig_events_tsne = tsne.fit_transform(spike_trig_events_tgt)
        
        # plot all units
        # plot the tsne
        if doTSNE:
            axs1[ianimal,ibhvname].plot(spike_trig_events_tsne[:,0],spike_trig_events_tsne[:,1],'.')
        # plot the pca
        if doPCA:
            axs1[ianimal,ibhvname].plot(spike_trig_events_pca[:,0],spike_trig_events_pca[:,1],'.')
        
        axs1[ianimal,ibhvname].set_xticklabels([])
        axs1[ianimal,ibhvname].set_yticklabels([])
        axs1[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        
        
        # plot all units, but seprate different dates
        dates_forplot = np.unique(spike_trig_events_all_dates_df[ind]['dates'])
        for idate_forplot in dates_forplot:
            ind_idate = list(spike_trig_events_all_dates_df[ind]['dates']==idate_forplot)
            ind_idate = list(np.array(ind_idate)[~ind_nan])
            #
            # plot the tsne
            if doTSNE:
                axs2[ianimal,ibhvname].plot(spike_trig_events_tsne[ind_idate,0],spike_trig_events_tsne[ind_idate,1],
                                        '.',label=idate_forplot)
            # plot the pca
            if doPCA:
                axs2[ianimal,ibhvname].plot(spike_trig_events_pca[ind_idate,0],spike_trig_events_pca[ind_idate,1],
                                        '.',label=idate_forplot)
            #
        axs2[ianimal,ibhvname].set_xticklabels([])
        axs2[ianimal,ibhvname].set_yticklabels([])
        axs2[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        axs2[ianimal,ibhvname].legend()
        
        
        # plot all units, but seprate different channels
        chs_forplot = np.unique(spike_trig_events_all_dates_df[ind]['channelID'])
        for ich_forplot in chs_forplot:
            ind_ich = list(spike_trig_events_all_dates_df[ind]['channelID']==ich_forplot)
            ind_ich = list(np.array(ind_ich)[~ind_nan])
            #
            # plot the tsne
            if doTSNE:
                axs3[ianimal,ibhvname].plot(spike_trig_events_tsne[ind_ich,0],spike_trig_events_tsne[ind_ich,1],
                                        '.',label=str(ich_forplot))
            # plot the pca
            if doPCA:
                axs3[ianimal,ibhvname].plot(spike_trig_events_pca[ind_ich,0],spike_trig_events_pca[ind_ich,1],
                                        '.',label=str(ich_forplot))
            #
        axs3[ianimal,ibhvname].set_xticklabels([])
        axs3[ianimal,ibhvname].set_yticklabels([])
        axs3[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        axs3[ianimal,ibhvname].legend()
        
        
        # plot all units, but seprate different task conditions
        cons_forplot = np.unique(spike_trig_events_all_dates_df[ind]['condition'])
        for icon_forplot in cons_forplot:
            ind_icon = list(spike_trig_events_all_dates_df[ind]['condition']==icon_forplot)
            ind_icon = list(np.array(ind_icon)[~ind_nan])
            #
            # plot the tsne
            if doTSNE:
                axs4[ianimal,ibhvname].plot(spike_trig_events_tsne[ind_icon,0],spike_trig_events_tsne[ind_icon,1],
                                        '.',label=icon_forplot)
            # plot the pca
            if doPCA:
                axs4[ianimal,ibhvname].plot(spike_trig_events_pca[ind_icon,0],spike_trig_events_pca[ind_icon,1],
                                        '.',label=icon_forplot)
            #
        axs4[ianimal,ibhvname].set_xticklabels([])
        axs4[ianimal,ibhvname].set_yticklabels([])
        axs4[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        axs4[ianimal,ibhvname].legend()
    
        # plot the mean spike trigger average trace across neurons in each condition
        trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
        xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)
        #
        cons_forplot = np.unique(spike_trig_events_all_dates_df[ind]['condition'])
        icon_ind = 0
        for icon_forplot in cons_forplot:
            ind_icon = list(spike_trig_events_all_dates_df[ind]['condition']==icon_forplot)
            ind_icon = list(np.array(ind_icon)[~ind_nan])
            #
            mean_trig_trace_icon = np.nanmean(spike_trig_events_tgt[ind_icon,:],axis=0)
            std_trig_trace_icon = np.nanstd(spike_trig_events_tgt[ind_icon,:],axis=0)
            sem_trig_trace_icon = np.nanstd(spike_trig_events_tgt[ind_icon,:],axis=0)/np.sqrt(np.shape(spike_trig_events_tgt[ind_icon,:])[0])
            itv95_trig_trace_icon = 1.96*sem_trig_trace_icon
            #
            if 1:
            # plot each trace in a seperate traces
                axs6[ianimal*5+icon_ind,ibhvname].errorbar(xxx_forplot,mean_trig_trace_icon,yerr=itv95_trig_trace_icon,
                                                           color='#E0E0E0',ecolor='#EEEEEE',label=icon_forplot)
                axs6[ianimal*5+icon_ind,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_icon-itv95_trig_trace_icon),
                                                              np.nanmax(mean_trig_trace_icon+itv95_trig_trace_icon)],'--k')
                axs6[ianimal*5+icon_ind,ibhvname].set_xlabel('time (s)')
                axs6[ianimal*5+icon_ind,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
                axs6[ianimal*5+icon_ind,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
                axs6[ianimal*5+icon_ind,ibhvname].set_title(act_animal+'; '+bhv_name)
                axs6[ianimal*5+icon_ind,ibhvname].legend()
            if 0:
                axs6[ianimal,ibhvname].errorbar(xxx_forplot,mean_trig_trace_icon,yerr=itv95_trig_trace_icon,
                                                label=icon_forplot)
                # axs6[ianimal,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_icon-itv95_trig_trace_icon),
                #                                               np.nanmax(mean_trig_trace_icon+itv95_trig_trace_icon)],'--k')
                axs6[ianimal,ibhvname].plot([0,0],[0,0.1],'--k') 
                axs6[ianimal,ibhvname].set_xlabel('time (s)')
                axs6[ianimal,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
                axs6[ianimal,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
                axs6[ianimal,ibhvname].set_title(act_animal+'; '+bhv_name)
                axs6[ianimal,ibhvname].legend()
            #
            icon_ind = icon_ind + 1
    
    
        # plot all units, but seprate different k-mean clusters
        kms_forplot = np.unique(kmean_cluster_labels)
        for ikm_forplot in kms_forplot:
            ind_ikm = list(kmean_cluster_labels==ikm_forplot)
            #
            # plot the tsne
            if doTSNE:
                axs5[ianimal,ibhvname].plot(spike_trig_events_tsne[ind_ikm,0],spike_trig_events_tsne[ind_ikm,1],
                                        '.',label=str(ikm_forplot))
            # plot the pca
            if doPCA:
                axs5[ianimal,ibhvname].plot(spike_trig_events_pca[ind_ikm,0],spike_trig_events_pca[ind_ikm,1],
                                        '.',label=str(ikm_forplot))
            #
        axs5[ianimal,ibhvname].set_xticklabels([])
        axs5[ianimal,ibhvname].set_yticklabels([])
        axs5[ianimal,ibhvname].set_title(act_animal+'; '+bhv_name)
        axs5[ianimal,ibhvname].legend()
        
        # plot the mean spike trigger average trace across neurons in each cluster
        trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
        xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)
        #
        kms_forplot = np.unique(kmean_cluster_labels)
        for ikm_forplot in kms_forplot:
            ind_ikm = list(kmean_cluster_labels==ikm_forplot)
            #
            mean_trig_trace_ikm = np.nanmean(spike_trig_events_tgt[ind_ikm,:],axis=0)
            std_trig_trace_ikm = np.nanstd(spike_trig_events_tgt[ind_ikm,:],axis=0)
            sem_trig_trace_ikm = np.nanstd(spike_trig_events_tgt[ind_ikm,:],axis=0)/np.sqrt(np.shape(spike_trig_events_tgt[ind_ikm,:])[0])
            itv95_trig_trace_ikm = 1.96*sem_trig_trace_ikm
            #
            axs7[ianimal*14+ikm_forplot,ibhvname].errorbar(xxx_forplot,mean_trig_trace_ikm,yerr=itv95_trig_trace_ikm,
                                                          color='#E0E0E0',ecolor='#EEEEEE',label='cluster#'+str(ikm_forplot))
            axs7[ianimal*14+ikm_forplot,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_ikm-itv95_trig_trace_ikm),
                                                             np.nanmax(mean_trig_trace_ikm+itv95_trig_trace_ikm)],'--k')
            axs7[ianimal*14+ikm_forplot,ibhvname].set_xlabel('time (s)')
            axs7[ianimal*14+ikm_forplot,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
            axs7[ianimal*14+ikm_forplot,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
            axs7[ianimal*14+ikm_forplot,ibhvname].set_title(act_animal+'; '+bhv_name)
            axs7[ianimal*14+ikm_forplot,ibhvname].legend()
    
    
        # stacked bar plot to show the cluster distribution of each conditions
        df = pd.DataFrame({'cond':np.array(spike_trig_events_all_dates_df[ind]['condition'])[~ind_nan],
                           'cluID':kmean_cluster_labels})
        (df.groupby('cond')['cluID'].value_counts(normalize=True)
           .unstack('cluID').plot.bar(stacked=True, ax=axs8[ianimal,ibhvname]))
        axs8[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        
        
    
    
savefig = 1
if savefig:
    figsavefolder = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filenames[0]+"_"+animal2_filenames[0]+"/spikeTrigAver_fig/"

    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    if doTSNE:
        fig1.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates'+savefile_sufix+'.pdf')
        fig2.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_separated_dates'+savefile_sufix+'.pdf')
        fig3.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_separated_channels'+savefile_sufix+'.pdf')
        fig4.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_separated_conditions'+savefile_sufix+'.pdf')
        fig5.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_separated_kmeanclusters'+savefile_sufix+'.pdf')
        fig6.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_sttraces_for_conditions'+savefile_sufix+'.pdf')        
        fig7.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_sttraces_for_kmeanclusters'+savefile_sufix+'.pdf')
        fig8.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_kmeanclusters_propotion_each_condition'+savefile_sufix+'.pdf')
        
    if doPCA:
        fig1.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates'+savefile_sufix+'.pdf')
        fig2.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_separated_dates'+savefile_sufix+'.pdf')
        fig3.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_separated_channels'+savefile_sufix+'.pdf')
        fig4.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_separated_conditions'+savefile_sufix+'.pdf')
        fig5.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_separated_kmeanclusters'+savefile_sufix+'.pdf')
        fig6.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_sttraces_for_conditions'+savefile_sufix+'.pdf')                           
        fig7.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_sttraces_for_kmeanclusters'+savefile_sufix+'.pdf')
        fig8.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_kmeanclusters_propotion_each_condition'+savefile_sufix+'.pdf')


# #### analyze the stretagy aligned firing rate across all dates
# #### plot the tsne or PCA clusters

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

doPCA = 1
doTSNE = 0

strategy_aligned_FR_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_average'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
       
    bhv_types = list(strategy_aligned_FR_all_dates[date_tgt].keys())

    for ibhv_type in bhv_types:

        clusterIDs = list(strategy_aligned_FR_all_dates[date_tgt][ibhv_type].keys())

        for iclusterID in clusterIDs:

            ichannelID = strategy_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = strategy_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['FR_average']

            strategy_aligned_FR_all_dates_df = strategy_aligned_FR_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_average':iFR_average,
                                                                                   }, ignore_index=True)
            
if 0:
    # normalize FR_average for each unit
    nspikeunits = np.shape(strategy_aligned_FR_all_dates_df)[0]
    for ispikeunit in np.arange(0,nspikeunits,1):
        stevent = strategy_aligned_FR_all_dates_df['FR_average'][ispikeunit]
        stevent_norm = (stevent-np.nanmin(stevent))/(np.nanmax(stevent)-np.nanmin(stevent))
        strategy_aligned_FR_all_dates_df['FR_average'][ispikeunit] = stevent_norm            
        
# only focus on the certain act animal and certain bhv_name
# act_animals_all = ['kanga']
# bhv_names_all = ['leverpull_prob']
act_animals_all = np.unique(strategy_aligned_FR_all_dates_df['act_animal'])
bhv_names_all = np.unique(strategy_aligned_FR_all_dates_df['bhv_name'])
#
nact_animals = np.shape(act_animals_all)[0]
nbhv_names = np.shape(bhv_names_all)[0]

# set for plot
# plot all units
fig1, axs1 = plt.subplots(nact_animals,nbhv_names)
fig1.set_figheight(6*nact_animals)
fig1.set_figwidth(6*nbhv_names)

# plot all units but separate different days
fig2, axs2 = plt.subplots(nact_animals,nbhv_names)
fig2.set_figheight(6*nact_animals)
fig2.set_figwidth(6*nbhv_names)

# plot all units but seprate different channels
fig3, axs3 = plt.subplots(nact_animals,nbhv_names)
fig3.set_figheight(4*nact_animals)
fig3.set_figwidth(4*nbhv_names)

# plot all units but separate different conditions
fig4, axs4 = plt.subplots(nact_animals,nbhv_names)
fig4.set_figheight(6*nact_animals)
fig4.set_figwidth(6*nbhv_names)

# spike triggered average for different task conditions
# # to be save, prepare for five conditions
fig6, axs6 = plt.subplots(nact_animals*5,nbhv_names)
fig6.set_figheight(6*nact_animals*5)
fig6.set_figwidth(6*nbhv_names)
# fig6, axs6 = plt.subplots(nact_animals,nbhv_names)
# fig6.set_figheight(6*nact_animals)
# fig6.set_figwidth(6*nbhv_names)

# plot all units but separate different k-mean cluster
fig5, axs5 = plt.subplots(nact_animals,nbhv_names)
fig5.set_figheight(6*nact_animals)
fig5.set_figwidth(6*nbhv_names)

# spike triggered average for different k-mean cluster
# to be save, prepare for 14 clusters
fig7, axs7 = plt.subplots(nact_animals*14,nbhv_names)
fig7.set_figheight(6*nact_animals*14)
fig7.set_figwidth(6*nbhv_names)

# stacked bar plot to show the cluster distribution of each conditions
fig8, axs8 = plt.subplots(nact_animals,nbhv_names)
fig8.set_figheight(6*nact_animals)
fig8.set_figwidth(6*nbhv_names)

#
for ianimal in np.arange(0,nact_animals,1):
    
    act_animal = act_animals_all[ianimal]
    
    for ibhvname in np.arange(0,nbhv_names,1):
        
        bhv_name = bhv_names_all[ibhvname]
        
        ind = (strategy_aligned_FR_all_dates_df['act_animal']==act_animal)&(strategy_aligned_FR_all_dates_df['bhv_name']==bhv_name)
        
        strategy_aligned_FR_tgt = np.vstack(list(strategy_aligned_FR_all_dates_df[ind]['FR_average']))
        
        ind_nan = np.isnan(np.sum(strategy_aligned_FR_tgt,axis=1)) # exist because of failed pull in SR
        strategy_aligned_FR_tgt = strategy_aligned_FR_tgt[~ind_nan,:]
        
        # k means clustering
        # run clustering on the 15 or 2 dimension PC space (for doPCA), or the whole dataset or 2 dimension (for doTSNE)
        pca = PCA(n_components=10)
        strategy_aligned_FR_pca = pca.fit_transform(strategy_aligned_FR_tgt)
        tsne = TSNE(n_components=2, random_state=0)
        strategy_aligned_FR_tsne = tsne.fit_transform(strategy_aligned_FR_tgt)
        #
        range_n_clusters = np.arange(2,15,1)
        silhouette_avg_all = np.ones(np.shape(range_n_clusters))*np.nan
        nkmeancls = np.shape(range_n_clusters)[0]
        #
        for ikmeancl in np.arange(0,nkmeancls,1):
            n_clusters = range_n_clusters[ikmeancl]
            #
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            # cluster_labels = clusterer.fit_predict(strategy_aligned_FR_tgt)
            if doPCA:
                cluster_labels = clusterer.fit_predict(strategy_aligned_FR_pca)
            if doTSNE:
                cluster_labels = clusterer.fit_predict(strategy_aligned_FR_tgt)
                # cluster_labels = clusterer.fit_predict(strategy_aligned_FR_tsne)
            #
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            # silhouette_avg = silhouette_score(strategy_aligned_FR_tgt, cluster_labels)
            if doPCA:
                silhouette_avg = silhouette_score(strategy_aligned_FR_pca, cluster_labels)
            if doTSNE:
                silhouette_avg = silhouette_score(strategy_aligned_FR_tgt, cluster_labels)
                # silhouette_avg = silhouette_score(strategy_aligned_FR_tsne, cluster_labels)
            #
            silhouette_avg_all[ikmeancl] = silhouette_avg
        #
        best_k_num = range_n_clusters[silhouette_avg_all==np.nanmax(silhouette_avg_all)][0]
        #
        clusterer = KMeans(n_clusters=best_k_num, random_state=0)
        # kmean_cluster_labels = clusterer.fit_predict(strategy_aligned_FR_tgt)
        if doPCA:
            kmean_cluster_labels = clusterer.fit_predict(strategy_aligned_FR_pca)
        if doTSNE:
            kmean_cluster_labels = clusterer.fit_predict(strategy_aligned_FR_tgt)
            # kmean_cluster_labels = clusterer.fit_predict(strategy_aligned_FR_tsne)
    
    
        # run PCA and TSNE     
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, random_state=0)
        #
        strategy_aligned_FR_pca = pca.fit_transform(strategy_aligned_FR_tgt)
        strategy_aligned_FR_tsne = tsne.fit_transform(strategy_aligned_FR_tgt)
        
        # plot all units
        # plot the tsne
        if doTSNE:
            axs1[ianimal,ibhvname].plot(strategy_aligned_FR_tsne[:,0],strategy_aligned_FR_tsne[:,1],'.')
        # plot the pca
        if doPCA:
            axs1[ianimal,ibhvname].plot(strategy_aligned_FR_pca[:,0],strategy_aligned_FR_pca[:,1],'.')
        
        axs1[ianimal,ibhvname].set_xticklabels([])
        axs1[ianimal,ibhvname].set_yticklabels([])
        axs1[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        
        
        # plot all units, but seprate different dates
        dates_forplot = np.unique(strategy_aligned_FR_all_dates_df[ind]['dates'])
        for idate_forplot in dates_forplot:
            ind_idate = list(strategy_aligned_FR_all_dates_df[ind]['dates']==idate_forplot)
            ind_idate = list(np.array(ind_idate)[~ind_nan])
            #
            # plot the tsne
            if doTSNE:
                axs2[ianimal,ibhvname].plot(strategy_aligned_FR_tsne[ind_idate,0],strategy_aligned_FR_tsne[ind_idate,1],
                                        '.',label=idate_forplot)
            # plot the pca
            if doPCA:
                axs2[ianimal,ibhvname].plot(strategy_aligned_FR_pca[ind_idate,0],strategy_aligned_FR_pca[ind_idate,1],
                                        '.',label=idate_forplot)
            #
        axs2[ianimal,ibhvname].set_xticklabels([])
        axs2[ianimal,ibhvname].set_yticklabels([])
        axs2[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        axs2[ianimal,ibhvname].legend()
        
        
        # plot all units, but seprate different channels
        chs_forplot = np.unique(strategy_aligned_FR_all_dates_df[ind]['channelID'])
        for ich_forplot in chs_forplot:
            ind_ich = list(strategy_aligned_FR_all_dates_df[ind]['channelID']==ich_forplot)
            ind_ich = list(np.array(ind_ich)[~ind_nan])
            #
            # plot the tsne
            if doTSNE:
                axs3[ianimal,ibhvname].plot(strategy_aligned_FR_tsne[ind_ich,0],strategy_aligned_FR_tsne[ind_ich,1],
                                        '.',label=str(ich_forplot))
            # plot the pca
            if doPCA:
                axs3[ianimal,ibhvname].plot(strategy_aligned_FR_pca[ind_ich,0],strategy_aligned_FR_pca[ind_ich,1],
                                        '.',label=str(ich_forplot))
            #
        axs3[ianimal,ibhvname].set_xticklabels([])
        axs3[ianimal,ibhvname].set_yticklabels([])
        axs3[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        axs3[ianimal,ibhvname].legend()
        
        
        # plot all units, but seprate different task conditions
        cons_forplot = np.unique(strategy_aligned_FR_all_dates_df[ind]['condition'])
        for icon_forplot in cons_forplot:
            ind_icon = list(strategy_aligned_FR_all_dates_df[ind]['condition']==icon_forplot)
            ind_icon = list(np.array(ind_icon)[~ind_nan])
            #
            # plot the tsne
            if doTSNE:
                axs4[ianimal,ibhvname].plot(strategy_aligned_FR_tsne[ind_icon,0],strategy_aligned_FR_tsne[ind_icon,1],
                                        '.',label=icon_forplot)
            # plot the pca
            if doPCA:
                axs4[ianimal,ibhvname].plot(strategy_aligned_FR_pca[ind_icon,0],strategy_aligned_FR_pca[ind_icon,1],
                                        '.',label=icon_forplot)
            #
        axs4[ianimal,ibhvname].set_xticklabels([])
        axs4[ianimal,ibhvname].set_yticklabels([])
        axs4[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        axs4[ianimal,ibhvname].legend()
    
        # plot the mean spike trigger average trace across neurons in each condition
        trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
        xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)
        #
        cons_forplot = np.unique(strategy_aligned_FR_all_dates_df[ind]['condition'])
        icon_ind = 0
        for icon_forplot in cons_forplot:
            ind_icon = list(strategy_aligned_FR_all_dates_df[ind]['condition']==icon_forplot)
            ind_icon = list(np.array(ind_icon)[~ind_nan])
            #
            mean_trig_trace_icon = np.nanmean(strategy_aligned_FR_tgt[ind_icon,:],axis=0)
            std_trig_trace_icon = np.nanstd(strategy_aligned_FR_tgt[ind_icon,:],axis=0)
            sem_trig_trace_icon = np.nanstd(strategy_aligned_FR_tgt[ind_icon,:],axis=0)/np.sqrt(np.shape(strategy_aligned_FR_tgt[ind_icon,:])[0])
            itv95_trig_trace_icon = 1.96*sem_trig_trace_icon
            #
            if 1:
            # plot each trace in a seperate traces
                axs6[ianimal*5+icon_ind,ibhvname].errorbar(xxx_forplot,mean_trig_trace_icon,yerr=itv95_trig_trace_icon,
                                                           color='#E0E0E0',ecolor='#EEEEEE',label=icon_forplot)
                axs6[ianimal*5+icon_ind,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_icon-itv95_trig_trace_icon),
                                                              np.nanmax(mean_trig_trace_icon+itv95_trig_trace_icon)],'--k')
                axs6[ianimal*5+icon_ind,ibhvname].set_xlabel('time (s)')
                axs6[ianimal*5+icon_ind,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
                axs6[ianimal*5+icon_ind,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
                axs6[ianimal*5+icon_ind,ibhvname].set_title(act_animal+'; '+bhv_name)
                axs6[ianimal*5+icon_ind,ibhvname].legend()
            if 0:
                axs6[ianimal,ibhvname].errorbar(xxx_forplot,mean_trig_trace_icon,yerr=itv95_trig_trace_icon,
                                                label=icon_forplot)
                # axs6[ianimal,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_icon-itv95_trig_trace_icon),
                #                                               np.nanmax(mean_trig_trace_icon+itv95_trig_trace_icon)],'--k')
                axs6[ianimal,ibhvname].plot([0,0],[0,0.1],'--k') 
                axs6[ianimal,ibhvname].set_xlabel('time (s)')
                axs6[ianimal,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
                axs6[ianimal,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
                axs6[ianimal,ibhvname].set_title(act_animal+'; '+bhv_name)
                axs6[ianimal,ibhvname].legend()
            #
            icon_ind = icon_ind + 1
    
    
        # plot all units, but seprate different k-mean clusters
        kms_forplot = np.unique(kmean_cluster_labels)
        for ikm_forplot in kms_forplot:
            ind_ikm = list(kmean_cluster_labels==ikm_forplot)
            #
            # plot the tsne
            if doTSNE:
                axs5[ianimal,ibhvname].plot(strategy_aligned_FR_tsne[ind_ikm,0],strategy_aligned_FR_tsne[ind_ikm,1],
                                        '.',label=str(ikm_forplot))
            # plot the pca
            if doPCA:
                axs5[ianimal,ibhvname].plot(strategy_aligned_FR_pca[ind_ikm,0],strategy_aligned_FR_pca[ind_ikm,1],
                                        '.',label=str(ikm_forplot))
            #
        axs5[ianimal,ibhvname].set_xticklabels([])
        axs5[ianimal,ibhvname].set_yticklabels([])
        axs5[ianimal,ibhvname].set_title(act_animal+'; '+bhv_name)
        axs5[ianimal,ibhvname].legend()
        
        # plot the mean spike trigger average trace across neurons in each cluster
        trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
        xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)
        #
        kms_forplot = np.unique(kmean_cluster_labels)
        for ikm_forplot in kms_forplot:
            ind_ikm = list(kmean_cluster_labels==ikm_forplot)
            #
            mean_trig_trace_ikm = np.nanmean(strategy_aligned_FR_tgt[ind_ikm,:],axis=0)
            std_trig_trace_ikm = np.nanstd(strategy_aligned_FR_tgt[ind_ikm,:],axis=0)
            sem_trig_trace_ikm = np.nanstd(strategy_aligned_FR_tgt[ind_ikm,:],axis=0)/np.sqrt(np.shape(strategy_aligned_FR_tgt[ind_ikm,:])[0])
            itv95_trig_trace_ikm = 1.96*sem_trig_trace_ikm
            #
            axs7[ianimal*14+ikm_forplot,ibhvname].errorbar(xxx_forplot,mean_trig_trace_ikm,yerr=itv95_trig_trace_ikm,
                                                          color='#E0E0E0',ecolor='#EEEEEE',label='cluster#'+str(ikm_forplot))
            axs7[ianimal*14+ikm_forplot,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_ikm-itv95_trig_trace_ikm),
                                                             np.nanmax(mean_trig_trace_ikm+itv95_trig_trace_ikm)],'--k')
            axs7[ianimal*14+ikm_forplot,ibhvname].set_xlabel('time (s)')
            axs7[ianimal*14+ikm_forplot,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
            axs7[ianimal*14+ikm_forplot,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
            axs7[ianimal*14+ikm_forplot,ibhvname].set_title(act_animal+'; '+bhv_name)
            axs7[ianimal*14+ikm_forplot,ibhvname].legend()
    
    
        # stacked bar plot to show the cluster distribution of each conditions
        df = pd.DataFrame({'cond':np.array(strategy_aligned_FR_all_dates_df[ind]['condition'])[~ind_nan],
                           'cluID':kmean_cluster_labels})
        (df.groupby('cond')['cluID'].value_counts(normalize=True)
           .unstack('cluID').plot.bar(stacked=True, ax=axs8[ianimal,ibhvname]))
        axs8[ianimal,ibhvname].set_title(act_animal+';'+bhv_name)
        
        
    
    
savefig = 1
if savefig:
    figsavefolder = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filenames[0]+"_"+animal2_filenames[0]+"/bhvAlignedFRAver_fig/"

    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    if doTSNE:
        fig1.savefig(figsavefolder+'stretagy_aligned_FR_tsne_clusters_all_dates'+savefile_sufix+'.pdf')
        fig2.savefig(figsavefolder+'stretagy_aligned_FR_tsne_clusters_all_dates_separated_dates'+savefile_sufix+'.pdf')
        fig3.savefig(figsavefolder+'stretagy_aligned_FR_tsne_clusters_all_dates_separated_channels'+savefile_sufix+'.pdf')
        fig4.savefig(figsavefolder+'stretagy_aligned_FR_tsne_clusters_all_dates_separated_conditions'+savefile_sufix+'.pdf')
        fig5.savefig(figsavefolder+'stretagy_aligned_FR_tsne_clusters_all_dates_separated_kmeanclusters'+savefile_sufix+'.pdf')
        fig6.savefig(figsavefolder+'stretagy_aligned_FR_tsne_clusters_all_dates_sttraces_for_conditions'+savefile_sufix+'.pdf')        
        fig7.savefig(figsavefolder+'stretagy_aligned_FR_tsne_clusters_all_dates_sttraces_for_kmeanclusters'+savefile_sufix+'.pdf')
        fig8.savefig(figsavefolder+'stretagy_aligned_FR_tsne_clusters_kmeanclusters_propotion_each_condition'+savefile_sufix+'.pdf')
        
    if doPCA:
        fig1.savefig(figsavefolder+'stretagy_aligned_FR_pca_clusters_all_dates'+savefile_sufix+'.pdf')
        fig2.savefig(figsavefolder+'stretagy_aligned_FR_pca_clusters_all_dates_separated_dates'+savefile_sufix+'.pdf')
        fig3.savefig(figsavefolder+'stretagy_aligned_FR_pca_clusters_all_dates_separated_channels'+savefile_sufix+'.pdf')
        fig4.savefig(figsavefolder+'stretagy_aligned_FR_pca_clusters_all_dates_separated_conditions'+savefile_sufix+'.pdf')
        fig5.savefig(figsavefolder+'stretagy_aligned_FR_pca_clusters_all_dates_separated_kmeanclusters'+savefile_sufix+'.pdf')
        fig6.savefig(figsavefolder+'stretagy_aligned_FR_pca_clusters_all_dates_sttraces_for_conditions'+savefile_sufix+'.pdf')                           
        fig7.savefig(figsavefolder+'stretagy_aligned_FR_pca_clusters_all_dates_sttraces_for_kmeanclusters'+savefile_sufix+'.pdf')
        fig8.savefig(figsavefolder+'stretagy_aligned_FR_pca_clusters_kmeanclusters_propotion_each_condition'+savefile_sufix+'.pdf')


# #### run PCA on the neuron space, pool sessions from the same condition together
# #### for the activity aligned at the different bhv events

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

doPCA = 1
doTSNE = 0

bhvevents_aligned_FR_allevents_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_allevents'])
bhvevents_aligned_FR_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_average'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
       
    bhv_types = list(bhvevents_aligned_FR_allevents_all_dates[date_tgt].keys())

    for ibhv_type in bhv_types:

        clusterIDs = list(bhvevents_aligned_FR_allevents_all_dates[date_tgt][ibhv_type].keys())

        for iclusterID in clusterIDs:

            ichannelID = bhvevents_aligned_FR_allevents_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = bhvevents_aligned_FR_allevents_all_dates[date_tgt][ibhv_type][iclusterID]['FR_allevents']

            bhvevents_aligned_FR_allevents_all_dates_df = bhvevents_aligned_FR_allevents_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_allevents':iFR_average,
                                                                                   }, ignore_index=True)
            
            #
            ichannelID = bhvevents_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = bhvevents_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['FR_average']

            bhvevents_aligned_FR_all_dates_df = bhvevents_aligned_FR_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_average':iFR_average,
                                                                                   }, ignore_index=True)
            
# act_animals_to_ana = np.unique(bhvevents_aligned_FR_allevents_all_dates_df['act_animal'])
# act_animals_to_ana = ['kanga']
act_animals_to_ana = ['dodson']
nanimal_to_ana = np.shape(act_animals_to_ana)[0]
#
# bhv_names_to_ana = np.unique(bhvevents_aligned_FR_allevents_all_dates_df['bhv_name'])
bhv_names_to_ana = ['pull','gaze']
nbhvnames_to_ana = np.shape(bhv_names_to_ana)[0]
bhvname_clrs = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493','#A9A9A9','#8B4513']
#
conditions_to_ana = np.unique(bhvevents_aligned_FR_allevents_all_dates_df['condition'])
nconds_to_ana = np.shape(conditions_to_ana)[0]

# figures
fig1, axs1 = plt.subplots(3,nconds_to_ana)
fig1.set_figheight(6*3)
fig1.set_figwidth(6*nconds_to_ana)

#
# 3d figure
fig2 = plt.figure(figsize=(6*nconds_to_ana,6))

for icond_ana in np.arange(0,nconds_to_ana,1):
    cond_ana = conditions_to_ana[icond_ana]
    # ind_cond = bhvevents_aligned_FR_allevents_all_dates_df['condition']==cond_ana
    ind_cond = bhvevents_aligned_FR_all_dates_df['condition']==cond_ana    
    
    ax2 = fig2.add_subplot(1,nconds_to_ana,icond_ana+1,projection = '3d')
    
    for ianimal_ana in np.arange(0,nanimal_to_ana,1):
        act_animal_ana = act_animals_to_ana[ianimal_ana]
        # ind_animal = bhvevents_aligned_FR_allevents_all_dates_df['act_animal']==act_animal_ana
        ind_animal = bhvevents_aligned_FR_all_dates_df['act_animal']==act_animal_ana
       
        for ibhvname_ana in np.arange(0,nbhvnames_to_ana,1):
            bhvname_ana = bhv_names_to_ana[ibhvname_ana]
            # ind_bhv = bhvevents_aligned_FR_allevents_all_dates_df['bhv_name']==bhvname_ana
            ind_bhv = bhvevents_aligned_FR_all_dates_df['bhv_name']==bhvname_ana
                
            ind_ana = ind_animal & ind_bhv & ind_cond
            
            # bhvevents_aligned_FR_allevents_tgt = bhvevents_aligned_FR_allevents_all_dates_df[ind_ana]
            bhvevents_aligned_FR_tgt = bhvevents_aligned_FR_all_dates_df[ind_ana]

            # PCA_dataset = np.hstack(list(bhvevents_aligned_FR_allevents_tgt['FR_allevents']))
            PCA_dataset = np.array(list(bhvevents_aligned_FR_tgt['FR_average']))
            
            # remove nan raw from the data set
            # ind_nan = np.isnan(np.sum(PCA_dataset,axis=0))
            # PCA_dataset = PCA_dataset_test[:,~ind_nan]
            ind_nan = np.isnan(np.sum(PCA_dataset,axis=1))
            PCA_dataset = PCA_dataset[~ind_nan,:]
            PCA_dataset = np.transpose(PCA_dataset)
            
            # run PCA
            pca = PCA(n_components=3)
            pca.fit(PCA_dataset)
            PCA_dataset_proj = pca.transform(PCA_dataset)
            
            trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
            xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)
        
            # plot PC1
            axs1[0,icond_ana].plot( xxx_forplot,gaussian_filter1d(PCA_dataset_proj[:,0], 6),
                                   label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
            axs1[1,icond_ana].plot( xxx_forplot,gaussian_filter1d(PCA_dataset_proj[:,1], 6),
                                   label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
            axs1[2,icond_ana].plot( xxx_forplot,gaussian_filter1d(PCA_dataset_proj[:,2], 6),
                                   label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
            
            # plot the 3d trojactory
            ax2.plot(gaussian_filter1d(PCA_dataset_proj[:,0], 6),
                     gaussian_filter1d(PCA_dataset_proj[:,1], 6),
                     gaussian_filter1d(PCA_dataset_proj[:,2], 6),
                     label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
            # start of time window
            ax2.plot(gaussian_filter1d(PCA_dataset_proj[:,0], 6)[0],
                     gaussian_filter1d(PCA_dataset_proj[:,1], 6)[0],
                     gaussian_filter1d(PCA_dataset_proj[:,2], 6)[0],
                     'o',markersize = 9, color=bhvname_clrs[ibhvname_ana])
            # action time
            ax2.plot(gaussian_filter1d(PCA_dataset_proj[:,0], 6)[np.where(xxx_forplot==0)[0][0]],
                     gaussian_filter1d(PCA_dataset_proj[:,1], 6)[np.where(xxx_forplot==0)[0][0]],
                     gaussian_filter1d(PCA_dataset_proj[:,2], 6)[np.where(xxx_forplot==0)[0][0]],
                     '>',markersize = 9, color=bhvname_clrs[ibhvname_ana])
            # end of time window
            ax2.plot(gaussian_filter1d(PCA_dataset_proj[:,0], 6)[-1],
                     gaussian_filter1d(PCA_dataset_proj[:,1], 6)[-1],
                     gaussian_filter1d(PCA_dataset_proj[:,2], 6)[-1],
                     's',markersize = 9, color=bhvname_clrs[ibhvname_ana])
            
            
    axs1[0,icond_ana].set_xlabel('time (s)')
    axs1[0,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[0,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[0,icond_ana].set_title('PC1 '+cond_ana)
    axs1[0,icond_ana].legend()      
    
    axs1[1,icond_ana].set_xlabel('time (s)')
    axs1[1,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[1,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[1,icond_ana].set_title('PC2 '+cond_ana)
    axs1[1,icond_ana].legend()    
    
    axs1[2,icond_ana].set_xlabel('time (s)')
    axs1[2,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[2,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[2,icond_ana].set_title('PC3 '+cond_ana)
    axs1[2,icond_ana].legend()    
    
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2') 
    ax2.set_zlabel('PC3')    
    ax2.set_title(cond_ana)
    ax2.legend()    
    ax2.view_init(elev=30, azim=-30) 
    
savefig = 1
if savefig:
    figsavefolder = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filenames[0]+"_"+animal2_filenames[0]+"/FRsPCA_fig/"

    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)

    fig1.savefig(figsavefolder+'bhvevent_aligned_PCspace_trajectory_allconditions'+savefile_sufix+'_PC123separate.pdf')
    fig2.savefig(figsavefolder+'bhvevent_aligned_PCspace_trajectory_allconditions'+savefile_sufix+'.pdf')
    


# In[ ]:





# #### run PCA on the neuron space, run different days separately each condition
# #### for the activity aligned at the different bhv events

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score



bhvevents_aligned_FR_allevents_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_allevents'])
bhvevents_aligned_FR_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_average'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
       
    bhv_types = list(bhvevents_aligned_FR_allevents_all_dates[date_tgt].keys())

    for ibhv_type in bhv_types:

        clusterIDs = list(bhvevents_aligned_FR_allevents_all_dates[date_tgt][ibhv_type].keys())

        for iclusterID in clusterIDs:

            ichannelID = bhvevents_aligned_FR_allevents_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = bhvevents_aligned_FR_allevents_all_dates[date_tgt][ibhv_type][iclusterID]['FR_allevents']

            bhvevents_aligned_FR_allevents_all_dates_df = bhvevents_aligned_FR_allevents_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_allevents':iFR_average,
                                                                                   }, ignore_index=True)
            
            #
            ichannelID = bhvevents_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = bhvevents_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['FR_average']

            bhvevents_aligned_FR_all_dates_df = bhvevents_aligned_FR_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_average':iFR_average,
                                                                                   }, ignore_index=True)
            
# act_animals_to_ana = np.unique(bhvevents_aligned_FR_allevents_all_dates_df['act_animal'])
# act_animals_to_ana = ['kanga']
act_animals_to_ana = ['dodson']
nanimal_to_ana = np.shape(act_animals_to_ana)[0]
#
# bhv_names_to_ana = np.unique(bhvevents_aligned_FR_allevents_all_dates_df['bhv_name'])
bhv_names_to_ana = ['pull','gaze']
nbhvnames_to_ana = np.shape(bhv_names_to_ana)[0]
bhvname_clrs = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493','#A9A9A9','#8B4513']
#
conditions_to_ana = np.unique(bhvevents_aligned_FR_allevents_all_dates_df['condition'])
nconds_to_ana = np.shape(conditions_to_ana)[0]
# 

# figures
fig1, axs1 = plt.subplots(3,nconds_to_ana)
fig1.set_figheight(6*3)
fig1.set_figwidth(6*nconds_to_ana)

for icond_ana in np.arange(0,nconds_to_ana,1):
    cond_ana = conditions_to_ana[icond_ana]
    ind_cond_allevents = bhvevents_aligned_FR_allevents_all_dates_df['condition']==cond_ana
    ind_cond = bhvevents_aligned_FR_all_dates_df['condition']==cond_ana    
    
    for ianimal_ana in np.arange(0,nanimal_to_ana,1):
        act_animal_ana = act_animals_to_ana[ianimal_ana]
        ind_animal_allevents = bhvevents_aligned_FR_allevents_all_dates_df['act_animal']==act_animal_ana
        ind_animal = bhvevents_aligned_FR_all_dates_df['act_animal']==act_animal_ana
       
        for ibhvname_ana in np.arange(0,nbhvnames_to_ana,1):
            bhvname_ana = bhv_names_to_ana[ibhvname_ana]
            ind_bhv_allevents = bhvevents_aligned_FR_allevents_all_dates_df['bhv_name']==bhvname_ana
            ind_bhv = bhvevents_aligned_FR_all_dates_df['bhv_name']==bhvname_ana
                
            ind_ana_allevents = ind_animal_allevents & ind_bhv_allevents & ind_cond_allevents
            ind_ana = ind_animal & ind_bhv & ind_cond
            
            bhvevents_aligned_FR_allevents_tgt = bhvevents_aligned_FR_allevents_all_dates_df[ind_ana_allevents]
            bhvevents_aligned_FR_tgt = bhvevents_aligned_FR_all_dates_df[ind_ana]

            # separate for each dates
            dates_to_ana = np.unique(bhvevents_aligned_FR_tgt['dates'])
            ndates_ana = np.shape(dates_to_ana)[0]
            
            for idate_ana in np.arange(0,ndates_ana,1):
                date_ana = dates_to_ana[idate_ana]
                ind_date_allevents = bhvevents_aligned_FR_allevents_tgt['dates']==date_ana
                ind_date = bhvevents_aligned_FR_tgt['dates']==date_ana
                
                # get the PCA training data set
                PCA_dataset = np.hstack(list(bhvevents_aligned_FR_allevents_tgt[ind_date_allevents]['FR_allevents']))
                #
                ncells = np.shape(bhvevents_aligned_FR_allevents_tgt[ind_date_allevents])[0]
                PCA_dataset_train_pre_df = pd.DataFrame(columns=['clusterID','channelID','FR_pooled'])
                PCA_dataset_train_pre_df['clusterID'] = bhvevents_aligned_FR_allevents_tgt[ind_date_allevents]['clusterID']
                PCA_dataset_train_pre_df['channelID'] = bhvevents_aligned_FR_allevents_tgt[ind_date_allevents]['channelID']
                for icell in np.arange(0,ncells,1):
                    FR_ravel = np.ravel(bhvevents_aligned_FR_allevents_tgt[ind_date_allevents]['FR_allevents'].iloc[icell])
                    PCA_dataset_train_pre_df['FR_pooled'].iloc[icell] = FR_ravel
                PCA_dataset_train = np.array(list(PCA_dataset_train_pre_df['FR_pooled']))
                # remove nan raw from the data set
                ind_nan = np.isnan(np.sum(PCA_dataset_train,axis=0))
                PCA_dataset_train = PCA_dataset_train[:,~ind_nan]
                
                # get the PCA test dataset
                PCA_dataset_test = np.array(list(bhvevents_aligned_FR_tgt[ind_date]['FR_average']))
                # remove nan raw from the data set
                ind_nan = np.isnan(np.sum(PCA_dataset_test,axis=0))
                PCA_dataset_test = PCA_dataset_test[:,~ind_nan]
               
                # run PCA
                pca = PCA(n_components=3)
                pca.fit(PCA_dataset_train.transpose())
                PCA_dataset_train_proj = pca.transform(PCA_dataset_train.transpose())
                PCA_dataset_proj = pca.transform(PCA_dataset_test.transpose())

                trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
                xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)

                # plot PC1
                axs1[0,icond_ana].plot( xxx_forplot,PCA_dataset_proj[:,0],label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
                axs1[1,icond_ana].plot( xxx_forplot,PCA_dataset_proj[:,1],label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
                axs1[2,icond_ana].plot( xxx_forplot,PCA_dataset_proj[:,2],label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
            
    axs1[0,icond_ana].set_xlabel('time (s)')
    axs1[0,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[0,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[0,icond_ana].set_title('PC1 '+cond_ana)
    axs1[0,icond_ana].legend()      
    
    axs1[1,icond_ana].set_xlabel('time (s)')
    axs1[1,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[1,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[1,icond_ana].set_title('PC2 '+cond_ana)
    axs1[1,icond_ana].legend()    
    
    axs1[2,icond_ana].set_xlabel('time (s)')
    axs1[2,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[2,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[2,icond_ana].set_title('PC3 '+cond_ana)
    axs1[2,icond_ana].legend()    
    


# #### run PCA on the neuron space, pool sessions from the same condition together
# #### for the activity aligned at the different strategies

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


strategy_aligned_FR_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_average'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
       
    bhv_types = list(strategy_aligned_FR_allevents_all_dates[date_tgt].keys())

    for ibhv_type in bhv_types:

        clusterIDs = list(strategy_aligned_FR_allevents_all_dates[date_tgt][ibhv_type].keys())

        ibhv_type_split = ibhv_type.split()
        if np.shape(ibhv_type_split)[0]==3:
            ibhv_type_split[1] = ibhv_type_split[1]+'_'+ibhv_type_split[2]

        for iclusterID in clusterIDs:

            #
            ichannelID = strategy_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = strategy_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['FR_average']

            strategy_aligned_FR_all_dates_df = strategy_aligned_FR_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type_split[0],
                                                                                    'bhv_name': ibhv_type_split[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_average':iFR_average,
                                                                                   }, ignore_index=True)
            
# act_animals_to_ana = np.unique(strategy_aligned_FR_all_dates_df['act_animal'])
# act_animals_to_ana = ['kanga']
act_animals_to_ana = ['dodson']
nanimal_to_ana = np.shape(act_animals_to_ana)[0]
#
# bhv_names_to_ana = np.unique(strategy_aligned_FR_all_dates_df['bhv_name'])
bhv_names_to_ana = ['gaze_lead_pull', 'synced_pull','social_attention', 'not_social_attention']
nbhvnames_to_ana = np.shape(bhv_names_to_ana)[0]
bhvname_clrs = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493','#A9A9A9','#8B4513']
#
conditions_to_ana = np.unique(strategy_aligned_FR_all_dates_df['condition'])
nconds_to_ana = np.shape(conditions_to_ana)[0]

# figures
fig1, axs1 = plt.subplots(3,nconds_to_ana)
fig1.set_figheight(6*3)
fig1.set_figwidth(6*nconds_to_ana)
#
# 3d figure
fig2 = plt.figure(figsize=(6*nconds_to_ana,6))


for icond_ana in np.arange(0,nconds_to_ana,1):
    cond_ana = conditions_to_ana[icond_ana]
    # ind_cond = strategy_aligned_FR_allevents_all_dates_df['condition']==cond_ana
    ind_cond = strategy_aligned_FR_all_dates_df['condition']==cond_ana    
    
    ax2 = fig2.add_subplot(1,nconds_to_ana,icond_ana+1,projection = '3d')
    
    for ianimal_ana in np.arange(0,nanimal_to_ana,1):
        act_animal_ana = act_animals_to_ana[ianimal_ana]
        # ind_animal = strategy_aligned_FR_allevents_all_dates_df['act_animal']==act_animal_ana
        ind_animal = strategy_aligned_FR_all_dates_df['act_animal']==act_animal_ana
       
        for ibhvname_ana in np.arange(0,nbhvnames_to_ana,1):
            bhvname_ana = bhv_names_to_ana[ibhvname_ana]
            # ind_bhv = strategy_aligned_FR_allevents_all_dates_df['bhv_name']==bhvname_ana
            ind_bhv = strategy_aligned_FR_all_dates_df['bhv_name']==bhvname_ana
                
            ind_ana = ind_animal & ind_bhv & ind_cond
            
            # strategy_aligned_FR_allevents_tgt = strategy_aligned_FR_allevents_all_dates_df[ind_ana]
            strategy_aligned_FR_tgt = strategy_aligned_FR_all_dates_df[ind_ana]

            # PCA_dataset = np.hstack(list(strategy_aligned_FR_allevents_tgt['FR_allevents']))
            PCA_dataset = np.array(list(strategy_aligned_FR_tgt['FR_average']))
            
            # remove nan raw from the data set
            # ind_nan = np.isnan(np.sum(PCA_dataset,axis=0))
            # PCA_dataset = PCA_dataset_test[:,~ind_nan]
            ind_nan = np.isnan(np.sum(PCA_dataset,axis=1))
            PCA_dataset = PCA_dataset[~ind_nan,:]
            PCA_dataset = np.transpose(PCA_dataset)
                        
            # run PCA
            # newly added, randomly sample 100 "neuron" units and run PCA for 100 (niters) iterations
            niters = 100
            unitsamplesizes = 100
            #
            nunits = np.shape(PCA_dataset)[1]
            ntimesteps = np.shape(PCA_dataset)[0]
            #
            PCA_dataset_proj_allsamples = np.ones((niters,ntimesteps,3))*np.nan
            #
            for iiter in np.arange(0,niters,1):
                PCA_dataset_sample = PCA_dataset[:,np.random.choice(range(nunits),niters)]
                #
                pca = PCA(n_components=3)
                pca.fit(PCA_dataset_sample)
                PCA_dataset_proj_allsamples[iiter,:,:] = pca.transform(PCA_dataset_sample)
            #
            PCA_dataset_proj = np.nanmean(PCA_dataset_proj_allsamples,axis=0)
            
            
            trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
            xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)
        
            # plot PC1
            axs1[0,icond_ana].plot(xxx_forplot,gaussian_filter1d(PCA_dataset_proj[:,0], 6),
                                   label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
            axs1[1,icond_ana].plot(xxx_forplot,gaussian_filter1d(PCA_dataset_proj[:,1], 6),
                                   label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
            axs1[2,icond_ana].plot(xxx_forplot,gaussian_filter1d(PCA_dataset_proj[:,2], 6),
                                   label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
    
            # plot the 3d trojactory
            ax2.plot(gaussian_filter1d(PCA_dataset_proj[:,0], 6),
                     gaussian_filter1d(PCA_dataset_proj[:,1], 6),
                     gaussian_filter1d(PCA_dataset_proj[:,2], 6),
                     label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
            # start of time window
            ax2.plot(gaussian_filter1d(PCA_dataset_proj[:,0], 6)[0],
                     gaussian_filter1d(PCA_dataset_proj[:,1], 6)[0],
                     gaussian_filter1d(PCA_dataset_proj[:,2], 6)[0],
                     'o',markersize = 9, color=bhvname_clrs[ibhvname_ana])
            # action time
            ax2.plot(gaussian_filter1d(PCA_dataset_proj[:,0], 6)[np.where(xxx_forplot==0)[0][0]],
                     gaussian_filter1d(PCA_dataset_proj[:,1], 6)[np.where(xxx_forplot==0)[0][0]],
                     gaussian_filter1d(PCA_dataset_proj[:,2], 6)[np.where(xxx_forplot==0)[0][0]],
                     '>',markersize = 9, color=bhvname_clrs[ibhvname_ana])
            # end of time window
            ax2.plot(gaussian_filter1d(PCA_dataset_proj[:,0], 6)[-1],
                     gaussian_filter1d(PCA_dataset_proj[:,1], 6)[-1],
                     gaussian_filter1d(PCA_dataset_proj[:,2], 6)[-1],
                     's',markersize = 9, color=bhvname_clrs[ibhvname_ana])
    
    
    axs1[0,icond_ana].set_xlabel('time (s)')
    axs1[0,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[0,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[0,icond_ana].set_title('PC1 '+cond_ana)
    axs1[0,icond_ana].legend()      
    
    axs1[1,icond_ana].set_xlabel('time (s)')
    axs1[1,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[1,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[1,icond_ana].set_title('PC2 '+cond_ana)
    axs1[1,icond_ana].legend()    
    
    axs1[2,icond_ana].set_xlabel('time (s)')
    axs1[2,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[2,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[2,icond_ana].set_title('PC3 '+cond_ana)
    axs1[2,icond_ana].legend()    
    
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2') 
    ax2.set_zlabel('PC3')    
    ax2.set_title(cond_ana)
    ax2.legend()    
    ax2.view_init(elev=30, azim=-30) 
    

savefig = 1
if savefig:
    figsavefolder = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filenames[0]+"_"+animal2_filenames[0]+"/FRsPCA_fig/"

    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)

    fig1.savefig(figsavefolder+'stretagy_aligned_PCspace_trajectory_allconditions'+savefile_sufix+'_PC123separate.pdf')
    fig2.savefig(figsavefolder+'stretagy_aligned_PCspace_trajectory_allconditions'+savefile_sufix+'.pdf')


# In[ ]:





# #### run PCA on the neuron space, run different days separately each condition
# #### for the activity aligned at the different strategies

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score



strategy_aligned_FR_allevents_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_allevents'])
strategy_aligned_FR_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_average'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
       
    bhv_types = list(strategy_aligned_FR_allevents_all_dates[date_tgt].keys())

    for ibhv_type in bhv_types:

        clusterIDs = list(strategy_aligned_FR_allevents_all_dates[date_tgt][ibhv_type].keys())

        for iclusterID in clusterIDs:

            ichannelID = strategy_aligned_FR_allevents_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = strategy_aligned_FR_allevents_all_dates[date_tgt][ibhv_type][iclusterID]['FR_allevents']

            strategy_aligned_FR_allevents_all_dates_df = strategy_aligned_FR_allevents_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_allevents':iFR_average,
                                                                                   }, ignore_index=True)
            
            #
            ichannelID = strategy_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = strategy_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['FR_average']

            strategy_aligned_FR_all_dates_df = strategy_aligned_FR_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_average':iFR_average,
                                                                                   }, ignore_index=True)
            
# act_animals_to_ana = np.unique(strategy_aligned_FR_allevents_all_dates_df['act_animal'])
# act_animals_to_ana = ['kanga']
act_animals_to_ana = ['dodson']
nanimal_to_ana = np.shape(act_animals_to_ana)[0]
#
# bhv_names_to_ana = np.unique(strategy_aligned_FR_allevents_all_dates_df['bhv_name'])
bhv_names_to_ana = ['gaze_lead_pull', 'synced_pull','social_attention', ]
# bhv_names_to_ana = ['gaze_lead_pull', 'synced_pull',]
nbhvnames_to_ana = np.shape(bhv_names_to_ana)[0]
bhvname_clrs = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493','#A9A9A9','#8B4513']
#
conditions_to_ana = np.unique(strategy_aligned_FR_allevents_all_dates_df['condition'])
nconds_to_ana = np.shape(conditions_to_ana)[0]
# 

# figures
fig1, axs1 = plt.subplots(3,nconds_to_ana)
fig1.set_figheight(6*3)
fig1.set_figwidth(6*nconds_to_ana)
#
# 3d figure trace
fig2 = plt.figure(figsize=(6*nconds_to_ana,6))
#
# 3d figure around the action, for the averaged in one session
fig3 = plt.figure(figsize=(6*nconds_to_ana,6))
#
# 3d figure around the action, for each action
fig4 = plt.figure(figsize=(6*nconds_to_ana,6))

for icond_ana in np.arange(0,nconds_to_ana,1):
    cond_ana = conditions_to_ana[icond_ana]
    ind_cond_allevents = strategy_aligned_FR_allevents_all_dates_df['condition']==cond_ana
    ind_cond = strategy_aligned_FR_all_dates_df['condition']==cond_ana    
    
    ax2 = fig2.add_subplot(1,nconds_to_ana,icond_ana+1,projection = '3d')
    ax3 = fig3.add_subplot(1,nconds_to_ana,icond_ana+1,projection = '3d')
    ax4 = fig4.add_subplot(1,nconds_to_ana,icond_ana+1,projection = '3d')
    
    for ianimal_ana in np.arange(0,nanimal_to_ana,1):
        act_animal_ana = act_animals_to_ana[ianimal_ana]
        ind_animal_allevents = strategy_aligned_FR_allevents_all_dates_df['act_animal']==act_animal_ana
        ind_animal = strategy_aligned_FR_all_dates_df['act_animal']==act_animal_ana
       
        for ibhvname_ana in np.arange(0,nbhvnames_to_ana,1):
            bhvname_ana = bhv_names_to_ana[ibhvname_ana]
            ind_bhv_allevents = strategy_aligned_FR_allevents_all_dates_df['bhv_name']==bhvname_ana
            ind_bhv = strategy_aligned_FR_all_dates_df['bhv_name']==bhvname_ana
                
            ind_ana_allevents = ind_animal_allevents & ind_bhv_allevents & ind_cond_allevents
            ind_ana = ind_animal & ind_bhv & ind_cond
            
            strategy_aligned_FR_allevents_tgt = strategy_aligned_FR_allevents_all_dates_df[ind_ana_allevents]
            strategy_aligned_FR_tgt = strategy_aligned_FR_all_dates_df[ind_ana]

            # separate for each dates
            dates_to_ana = np.unique(strategy_aligned_FR_tgt['dates'])
            ndates_ana = np.shape(dates_to_ana)[0]
            
            for idate_ana in np.arange(0,ndates_ana,1):
                date_ana = dates_to_ana[idate_ana]
                ind_date_allevents = strategy_aligned_FR_allevents_tgt['dates']==date_ana
                ind_date = strategy_aligned_FR_tgt['dates']==date_ana
                
                try:
                    # get the PCA training data set
                    #
                    ncells = np.shape(strategy_aligned_FR_allevents_tgt[ind_date_allevents])[0]
                    PCA_dataset_train_pre_df = pd.DataFrame(columns=['clusterID','channelID','FR_pooled','FR_allevents'])
                    PCA_dataset_train_pre_df['clusterID'] = strategy_aligned_FR_allevents_tgt[ind_date_allevents]['clusterID']
                    PCA_dataset_train_pre_df['channelID'] = strategy_aligned_FR_allevents_tgt[ind_date_allevents]['channelID']
                    PCA_dataset_train_pre_df['FR_allevents'] = strategy_aligned_FR_allevents_tgt[ind_date_allevents]['FR_allevents']
                    #
                    for icell in np.arange(0,ncells,1):
                        FR_ravel = np.ravel(strategy_aligned_FR_allevents_tgt[ind_date_allevents]['FR_allevents'].iloc[icell])
                        PCA_dataset_train_pre_df['FR_pooled'].iloc[icell] = FR_ravel
                    PCA_dataset_train = np.array(list(PCA_dataset_train_pre_df['FR_pooled']))
                    # remove nan raw from the data set
                    ind_nan = np.isnan(np.sum(PCA_dataset_train,axis=0))
                    PCA_dataset_train = PCA_dataset_train[:,~ind_nan]

                    # get the PCA test dataset
                    PCA_dataset_test = np.array(list(strategy_aligned_FR_tgt[ind_date]['FR_average']))
                    # remove nan raw from the data set
                    ind_nan = np.isnan(np.sum(PCA_dataset_test,axis=0))
                    PCA_dataset_test = PCA_dataset_test[:,~ind_nan]

                    # run PCA
                    pca = PCA(n_components=3)
                    pca.fit(PCA_dataset_train.transpose())
                    PCA_dataset_train_proj = pca.transform(PCA_dataset_train.transpose())
                    PCA_dataset_proj = pca.transform(PCA_dataset_test.transpose())

                    trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
                    xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)

                    # plot PC1
                    axs1[0,icond_ana].plot(xxx_forplot,gaussian_filter1d(PCA_dataset_proj[:,0], 6),
                                           label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
                    axs1[1,icond_ana].plot(xxx_forplot,gaussian_filter1d(PCA_dataset_proj[:,1], 6),
                                           label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
                    axs1[2,icond_ana].plot(xxx_forplot,gaussian_filter1d(PCA_dataset_proj[:,2], 6),
                                           label=act_animal_ana+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
                
                    # plot 3d PC1,2,3 trace
                    ax2.plot(gaussian_filter1d(PCA_dataset_proj[:,0], 6),
                             gaussian_filter1d(PCA_dataset_proj[:,1], 6),
                             gaussian_filter1d(PCA_dataset_proj[:,2], 6),
                             label=act_animal_ana+' '+bhvname_ana,
                             color=bhvname_clrs[ibhvname_ana])

                    # plot 3d PC1,2,3 datapoint around action
                    ind_twin = (xxx_forplot<=0.05*fps)&(xxx_forplot>=-0.2*fps)
                    xpoint = np.nanmean(PCA_dataset_proj[ind_twin,0])
                    ypoint = np.nanmean(PCA_dataset_proj[ind_twin,1])
                    zpoint = np.nanmean(PCA_dataset_proj[ind_twin,2])
                    ax3.plot(xpoint,ypoint,zpoint,'o',
                             label=act_animal_ana+' '+bhvname_ana,
                             color=bhvname_clrs[ibhvname_ana])

                    
                    # run PCA on the individual action in each session
                    if date_ana == '20240509':
                        PCA_indivacts_df = np.array(PCA_dataset_train_pre_df['FR_allevents'])
                        ncells = np.shape(PCA_indivacts_df)[0]
                        ntsteps = np.shape(PCA_indivacts_df[0])[0]
                        nacts = np.shape(PCA_indivacts_df[0])[1]
                        PCA_indivacts = np.empty((ncells,ntsteps,nacts))
                        for icell in np.arange(0,ncells,1):
                            PCA_indivacts[icell,:,:] = PCA_indivacts_df[icell]
                        #
                        for iact in np.arange(0,nacts,6):
                            PCA_dataset_test_iact = PCA_indivacts[:,:,iact]

                            if ~np.isnan(np.sum(PCA_dataset_test_iact)):
                                PCA_dataset_proj_iact = pca.transform(PCA_dataset_test_iact.transpose())
                                ind_twin = (xxx_forplot<=0.05*fps)&(xxx_forplot>=-0.2*fps)
                                xpoint = np.nanmean(PCA_dataset_proj_iact[ind_twin,0])
                                ypoint = np.nanmean(PCA_dataset_proj_iact[ind_twin,1])
                                zpoint = np.nanmean(PCA_dataset_proj_iact[ind_twin,2])
                                if iact == 0:
                                    ax4.plot(xpoint,ypoint,zpoint,'o',
                                             label=act_animal_ana+' '+bhvname_ana,
                                             color=bhvname_clrs[ibhvname_ana])
                                else:
                                    ax4.plot(xpoint,ypoint,zpoint,'o',
                                             color=bhvname_clrs[ibhvname_ana])
      
                
                except:
                    continue
     
                    
    axs1[0,icond_ana].set_xlabel('time (s)')
    axs1[0,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[0,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[0,icond_ana].set_title('PC1 '+cond_ana)
    axs1[0,icond_ana].legend()      
    
    axs1[1,icond_ana].set_xlabel('time (s)')
    axs1[1,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[1,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[1,icond_ana].set_title('PC2 '+cond_ana)
    axs1[1,icond_ana].legend()    
    
    axs1[2,icond_ana].set_xlabel('time (s)')
    axs1[2,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[2,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[2,icond_ana].set_title('PC3 '+cond_ana)
    axs1[2,icond_ana].legend()    
    
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2') 
    ax2.set_zlabel('PC3')   
    ax2.view_init(elev=30, azim=-30) 
    # ax2.legend() 
    
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2') 
    ax3.set_zlabel('PC3')    
    ax3.view_init(elev=30, azim=-30) 
    # ax3.view_init(elev=90, azim=-90) # PC1 and PC2
    # ax3.view_init(elev= 0, azim=-90) # PC1 and PC3
    # ax3.view_init(elev= 0, azim=  0) # PC2 and PC3
    # ax3.legend() 
    
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2') 
    ax4.set_zlabel('PC3')    
    ax4.view_init(elev=30, azim=-30) 
    # ax4.view_init(elev=90, azim=-90) # PC1 and PC2
    # ax4.view_init(elev= 0, azim=-90) # PC1 and PC3
    # ax4.view_init(elev= 0, azim=  0) # PC2 and PC3
    # ax4.legend() 
    

savefig = 1
if savefig:
    figsavefolder = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filenames[0]+"_"+animal2_filenames[0]+"/FRsPCA_fig/"

    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)

    fig1.savefig(figsavefolder+'stretagy_aligned_PCspace_trajectory_eachsession'+savefile_sufix+'_PC123separate.pdf')
    fig2.savefig(figsavefolder+'stretagy_aligned_PCspace_trajectory_eachsession'+savefile_sufix+'.pdf')
    fig4.savefig(figsavefolder+'stretagy_aligned_PCspace_eventtimestamp_examplesession'+savefile_sufix+'.pdf')



    


# In[ ]:





# In[ ]:





# #### run PCA on the neuron space, run different days separately each condition
# #### for the activity aligned at the different strategies
# #### to make all conditions in the "same" neural space, use psedo population with the "same" number of neurons and concantanate neural activity across all conditions

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score



strategy_aligned_FR_allevents_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_allevents'])
strategy_aligned_FR_all_dates_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','FR_average'])
# strategy_aligned_FR_allevents_all_dates_sepevents_df = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
#                                                        'channelID','eventID','FR_ievent'])

# reorganize to a dataframes
for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    task_condition = task_conditions[idate]
       
    bhv_types = list(strategy_aligned_FR_allevents_all_dates[date_tgt].keys())

    for ibhv_type in bhv_types:

        clusterIDs = list(strategy_aligned_FR_allevents_all_dates[date_tgt][ibhv_type].keys())

        for iclusterID in clusterIDs:

            # averaged FR across events
            ichannelID = strategy_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = strategy_aligned_FR_all_dates[date_tgt][ibhv_type][iclusterID]['FR_average']

            strategy_aligned_FR_all_dates_df = strategy_aligned_FR_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_average':iFR_average,
                                                                                   }, ignore_index=True)
            
            # FR for individual events
            ichannelID = strategy_aligned_FR_allevents_all_dates[date_tgt][ibhv_type][iclusterID]['ch']
            iFR_average = strategy_aligned_FR_allevents_all_dates[date_tgt][ibhv_type][iclusterID]['FR_allevents']

            strategy_aligned_FR_allevents_all_dates_df = strategy_aligned_FR_allevents_all_dates_df.append({'dates': date_tgt, 
                                                                                    'condition':task_condition,
                                                                                    'act_animal':ibhv_type.split()[0],
                                                                                    'bhv_name': ibhv_type.split()[1],
                                                                                    'clusterID':iclusterID,
                                                                                    'channelID':ichannelID,
                                                                                    'FR_allevents':iFR_average,
                                                                                   }, ignore_index=True)
            
            # nevents = np.shape(iFR_average)[1]
            # 
            # for ievent in np.arange(0,nevents,1):
            #     strategy_aligned_FR_allevents_all_dates_sepevents_df = strategy_aligned_FR_allevents_all_dates_sepevents_df.append({'dates': date_tgt, 
            #                                                                         'condition':task_condition,
            #                                                                         'act_animal':ibhv_type.split()[0],
            #                                                                         'bhv_name': ibhv_type.split()[1],
            #                                                                         'clusterID':iclusterID,
            #                                                                         'channelID':ichannelID,
            #                                                                         'eventID':ievent,                
            #                                                                         'FR_ievent':iFR_average[:,ievent],
            #                                                                        }, ignore_index=True)
            
            
            
# act_animals_to_ana = np.unique(strategy_aligned_FR_allevents_all_dates_df['act_animal'])
# act_animals_to_ana = ['kanga']
act_animals_to_ana = ['dodson']
nanimal_to_ana = np.shape(act_animals_to_ana)[0]
#
# bhv_names_to_ana = np.unique(strategy_aligned_FR_allevents_all_dates_df['bhv_name'])
bhv_names_to_ana = ['gaze_lead_pull', 'synced_pull','social_attention', ]
# bhv_names_to_ana = ['gaze_lead_pull', 'synced_pull',]
nbhvnames_to_ana = np.shape(bhv_names_to_ana)[0]
bhvname_clrs = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493','#A9A9A9','#8B4513']
#
conditions_to_ana = np.unique(strategy_aligned_FR_allevents_all_dates_df['condition'])
nconds_to_ana = np.shape(conditions_to_ana)[0]
# 



# concatanate firing rate for individal events
# random sampling 500 time to create a new pseudo neural population
nsamples = 300
strategy_aligned_FR_sepevents_tgt = pd.DataFrame(columns=['dates','condition','act_animal','bhv_name','clusterID',
                                                       'channelID','eventID','FR_ievent'])

for icond_ana in np.arange(0,nconds_to_ana,1):
    cond_ana = conditions_to_ana[icond_ana]
    ind_cond_allevents = strategy_aligned_FR_allevents_all_dates_df['condition']==cond_ana
    
    
    
    for ianimal_ana in np.arange(0,nanimal_to_ana,1):
        act_animal_ana = act_animals_to_ana[ianimal_ana]
        ind_animal_allevents = strategy_aligned_FR_allevents_all_dates_df['act_animal']==act_animal_ana
       
        for ibhvname_ana in np.arange(0,nbhvnames_to_ana,1):
            bhvname_ana = bhv_names_to_ana[ibhvname_ana]
            ind_bhv_allevents = strategy_aligned_FR_allevents_all_dates_df['bhv_name']==bhvname_ana
                   
            ind_ana_allevents = ind_animal_allevents & ind_bhv_allevents & ind_cond_allevents
            ind_ana = ind_animal & ind_bhv & ind_cond
            
            strategy_aligned_FR_allevents_tgt = strategy_aligned_FR_allevents_all_dates_df[ind_ana_allevents]

            nentries = np.shape(strategy_aligned_FR_allevents_tgt)[0]
            
            isample = 0
            
            # randomly sample
            while isample < nsamples:
                
                try:
                    # randomly pick the entry
                    ientry = random.randint(0, nentries-1)

                    strategy_aligned_FR_ientry = strategy_aligned_FR_allevents_tgt.iloc[ientry]

                    nevents = np.shape(strategy_aligned_FR_ientry['FR_allevents'])[1]

                    # randomly pick the event
                    ievent = random.randint(0, nevents-1)
                    
                    if ~np.isnan(np.sum(strategy_aligned_FR_ientry['FR_allevents'][:,ievent])):
                        strategy_aligned_FR_sepevents_tgt = strategy_aligned_FR_sepevents_tgt.append({'dates': strategy_aligned_FR_ientry['dates'], 
                                                                                                'condition':strategy_aligned_FR_ientry['condition'],
                                                                                                'act_animal':strategy_aligned_FR_ientry['act_animal'],
                                                                                                'bhv_name': strategy_aligned_FR_ientry['bhv_name'],
                                                                                                'clusterID':strategy_aligned_FR_ientry['clusterID'],
                                                                                                'channelID':strategy_aligned_FR_ientry['channelID'],
                                                                                                'eventID':ievent,                
                                                                                                'FR_ievent':strategy_aligned_FR_ientry['FR_allevents'][:,ievent],
                                                                                                }, ignore_index=True)
                        isample = isample + 1
                
                except:
                    continue
            

            

# work on the sampled new data
strategy_aligned_PC123_cond_concat = pd.DataFrame(columns=['condition','act_animal','bhv_name',
                                                                   'trainOrtest','PC123'])
for ianimal_ana in np.arange(0,nanimal_to_ana,1):
    act_animal_ana = act_animals_to_ana[ianimal_ana]
    ind_animal_allevents = strategy_aligned_FR_sepevents_tgt['act_animal']==act_animal_ana

    for ibhvname_ana in np.arange(0,nbhvnames_to_ana,1):
        bhvname_ana = bhv_names_to_ana[ibhvname_ana]
        ind_bhv_allevents = strategy_aligned_FR_sepevents_tgt['bhv_name']==bhvname_ana
                   
        ind_ana_allevents = ind_animal_allevents & ind_bhv_allevents

        strategy_aligned_FR_sepevents_forPCA = strategy_aligned_FR_sepevents_tgt[ind_ana_allevents]
        
        # Concatenate across conditions
        for icond_ana in np.arange(0,nconds_to_ana,1):
            cond_ana = conditions_to_ana[icond_ana]
            ind_cond_allevents = strategy_aligned_FR_sepevents_forPCA['condition']==cond_ana
    
            if icond_ana == 0:
                strategy_aligned_FR_conct = np.vstack(strategy_aligned_FR_sepevents_forPCA[ind_cond_allevents]['FR_ievent'])
            else:
                strategy_aligned_FR_conct_new = np.vstack(strategy_aligned_FR_sepevents_forPCA[ind_cond_allevents]['FR_ievent'])
                strategy_aligned_FR_conct = np.hstack([strategy_aligned_FR_conct, strategy_aligned_FR_conct_new])

        # run PCA 
        pca = PCA(n_components=3)
        pca.fit(strategy_aligned_FR_conct.transpose())
        PCA_dataset_train_proj = pca.transform(strategy_aligned_FR_conct.transpose())
        PCA_dataset_proj = pca.transform(strategy_aligned_FR_conct.transpose())
        
        # seperate the result for each condition
        twinlength = int(np.shape(PCA_dataset_proj)[0]/nconds_to_ana)
        for icond_ana in np.arange(0,nconds_to_ana,1):
            cond_ana = conditions_to_ana[icond_ana]
            
            PCA_dataset_train_proj_icond = PCA_dataset_train_proj[icond_ana*twinlength:(icond_ana+1)*twinlength,:]
            PCA_dataset_proj_icond = PCA_dataset_proj[icond_ana*twinlength:(icond_ana+1)*twinlength,:]
            
            strategy_aligned_PC123_cond_concat = strategy_aligned_PC123_cond_concat.append({'condition':cond_ana,
                                                                                   'act_animal':act_animal_ana,
                                                                                   'bhv_name': bhvname_ana,
                                                                                   'trainOrtest':'training',
                                                                                   'PC123':PCA_dataset_train_proj_icond,
                                                                                    }, ignore_index=True)
            
            strategy_aligned_PC123_cond_concat = strategy_aligned_PC123_cond_concat.append({'condition':cond_ana,
                                                                                   'act_animal':act_animal_ana,
                                                                                   'bhv_name': bhvname_ana,
                                                                                   'trainOrtest':'testing',
                                                                                   'PC123':PCA_dataset_proj_icond,
                                                                                    }, ignore_index=True)


# figures
# animal_toplot = 'kanga'
animal_toplot = 'dodson'
trainortest_toplot = 'testing'

fig1, axs1 = plt.subplots(3,nconds_to_ana)
fig1.set_figheight(6*3)
fig1.set_figwidth(6*nconds_to_ana)
#
# 3d figure trace
fig2 = plt.figure(figsize=(6*nconds_to_ana,6))

for icond_ana in np.arange(0,nconds_to_ana,1):
    
    ax2 = fig2.add_subplot(1,nconds_to_ana,icond_ana+1,projection = '3d')
        
    cond_ana = conditions_to_ana[icond_ana]
    ind_cond_toplot = strategy_aligned_PC123_cond_concat['condition']==cond_ana
    
    ind_animal_toplot = strategy_aligned_PC123_cond_concat['act_animal']==animal_toplot
    ind_traintest_toplot = strategy_aligned_PC123_cond_concat['trainOrtest']==trainortest_toplot
    
    ind_toplot = ind_cond_toplot & ind_animal_toplot & ind_traintest_toplot
    
    strategy_aligned_PC123_toplot = strategy_aligned_PC123_cond_concat[ind_toplot]
    
    for ibhvname_ana in np.arange(0,nbhvnames_to_ana,1):
        bhvname_ana = bhv_names_to_ana[ibhvname_ana]
        ind_bhv_toplot = strategy_aligned_PC123_toplot['bhv_name']==bhvname_ana
        
        PCA_toplot = strategy_aligned_PC123_toplot[ind_bhv_toplot]['PC123']
        PCA_toplot = np.array(PCA_toplot)[0]
        
        trig_twins = [-4,4] # the time window to examine the spike triggered average, in the unit of s
        xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)

        # plot PC1
        axs1[0,icond_ana].plot(xxx_forplot,PCA_toplot[:,0],label=animal_toplot+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
        axs1[1,icond_ana].plot(xxx_forplot,PCA_toplot[:,1],label=animal_toplot+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])
        axs1[2,icond_ana].plot(xxx_forplot,PCA_toplot[:,2],label=animal_toplot+' '+bhvname_ana,color=bhvname_clrs[ibhvname_ana])

        # plot 3d PC1,2,3 trace
        ax2.plot(gaussian_filter1d(PCA_toplot[:,0], 6),
                 gaussian_filter1d(PCA_toplot[:,1], 6),
                 gaussian_filter1d(PCA_toplot[:,2], 6),
                 label=animal_toplot+' '+bhvname_ana,
                 color=bhvname_clrs[ibhvname_ana])
    
    axs1[0,icond_ana].set_xlabel('time (s)')
    axs1[0,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[0,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[0,icond_ana].set_title('PC1 '+cond_ana)
    axs1[0,icond_ana].legend()      
    
    axs1[1,icond_ana].set_xlabel('time (s)')
    axs1[1,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[1,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[1,icond_ana].set_title('PC2 '+cond_ana)
    axs1[1,icond_ana].legend()    
    
    axs1[2,icond_ana].set_xlabel('time (s)')
    axs1[2,icond_ana].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
    axs1[2,icond_ana].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
    axs1[2,icond_ana].set_title('PC3 '+cond_ana)
    axs1[2,icond_ana].legend()    
    
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2') 
    ax2.set_zlabel('PC3')   
    ax2.view_init(elev=30, azim=-30) 
    ax2.set_title(cond_ana)
    
    
savefig = 1
if savefig:
    figsavefolder = data_saved_folder+"fig_for_basic_neural_analysis_allsessions_basicEvents/"+cameraID+"/"+animal1_filenames[0]+"_"+animal2_filenames[0]+"/FRsPCA_fig/"

    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)

    fig1.savefig(figsavefolder+'stretagy_aligned_PCspace_trajectory_samplingForCommonSpaceAcrossConditions_'+savefile_sufix+'_PC123separate.pdf')
    fig2.savefig(figsavefolder+'stretagy_aligned_PCspace_trajectory_samplingForCommonSpaceAcrossConditions_'+savefile_sufix+'.pdf')
    
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





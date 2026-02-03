#!/usr/bin/env python
# coding: utf-8

# ### In this script, DBN is run on the all the sessions
# ### In this script, DBN is run with 1s time bin, 3 time lag 
# ### In this script, the animal tracking is done with only one camera - camera 2 (middle) 
# ### In this script, in order to run DBN, will separate trials into high force or low force

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import scipy
import scipy.stats as st
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
import string
import warnings
import pickle

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


# ### function - plot inter-pull interval

# In[8]:


from ana_functions.plot_interpull_interval import plot_interpull_interval


# ### function - make demo videos with skeleton and inportant vectors

# In[9]:


from ana_functions.tracking_video_singlecam_demo import tracking_video_singlecam_demo
from ana_functions.tracking_video_singlecam_wholebody_demo import tracking_video_singlecam_wholebody_demo


# ### function - interval between all behavioral events

# In[10]:


from ana_functions.bhv_events_interval import bhv_events_interval


# ### function - train the dynamic bayesian network - multi time lag (3 lags)

# In[11]:


from ana_functions.train_DBN_multiLag import train_DBN_multiLag
from ana_functions.train_DBN_multiLag import train_DBN_multiLag_create_df_only
from ana_functions.train_DBN_multiLag import train_DBN_multiLag_training_only
from ana_functions.train_DBN_multiLag import graph_to_matrix
from ana_functions.train_DBN_multiLag import get_weighted_dags
from ana_functions.train_DBN_multiLag import get_significant_edges
from ana_functions.train_DBN_multiLag import threshold_edges
from ana_functions.train_DBN_multiLag import Modulation_Index
from ana_functions.EfficientTimeShuffling import EfficientShuffle
from ana_functions.AicScore import AicScore


# ## Analyze each session

# ### prepare the basic behavioral data (especially the time stamps for each bhv events)
# ### separate each session based on trial types (different force levels)

# In[12]:


# instead of using gaze angle threshold, use the target rectagon to deside gaze info
# ...need to update
sqr_thres_tubelever = 75 # draw the square around tube and lever
sqr_thres_face = 1.15 # a ratio for defining face boundary
sqr_thres_body = 4 # how many times to enlongate the face box boundry to the body


# get the fps of the analyzed video
fps = 30

# frame number of the demo video
# nframes = 0.5*30 # second*30fps
nframes = 1*30 # second*30fps

# re-analyze the video or not
reanalyze_video = 0
redo_anystep = 0

# only analyze the best (five) sessions for each conditions
do_bestsession = 1
if do_bestsession:
    savefile_sufix = '_bestsessions'
else:
    savefile_sufix = '_allsessions'
    
# all the videos (no misaligned ones)
# aligned with the audio
# get the session start time from "videosound_bhv_sync.py/.ipynb"
# currently the session_start_time will be manually typed in. It can be updated after a better method is used

# force manipulation type
# SR_bothchange: self reward, both forces changed
# CO_bothchange: 1s cooperation, both forces changed
# CO_A1change: 1s cooperation, animal 1 forces changed
# CO_A2change: 1s cooperation, animal 2 forces changed
forceManiType = 'CO_A2change'


    # Koala Vermelho
if 0:
    if do_bestsession:      
        # both animals' lever force were changed - Self reward
        if forceManiType == 'SR_bothchange':
            dates_list = [ "20240228","20240229","20240409","20240411",
                           "20240412","20240416","20240419",] 
            session_start_times = [ 64.5,  73.5,  0.00,  0.00,  
                                    0.00,  0.00,  0.00,  ] # in second
        # both animals' lever force were changed - cooperation
        elif forceManiType == 'CO_bothchange':
            dates_list = [ "20240304", ]
            session_start_times = [ 0.00, ] # in second
        # Koala's lever force were changed
        if forceManiType == 'CO_A1change':
            dates_list = [ "20240305","20240306","20240313","20240318","20240321",
                           "20240426","20240429","20240430",]
            session_start_times = [ 62.0,  55.2,  0.00,  0.00,  0.00, 
                                    0.00,  0.00,  0.00,  ] # in second
        # Verm's lever force were changed
        if forceManiType == 'CO_A2change':
            dates_list = [ "20240307","20240308","20240311","20240319",
                           "20240320","20240422","20240423","20240425",
                           "20240621", ]
            session_start_times = [ 72.2,  0.00,  60.8,  0.00,  
                                    0.00,  53.0,  0.00,  0.00, 
                                    0.00, ] # in second       
    
    elif not do_bestsession:
        # pick only five sessions for each conditions
        dates_list = [
                    
                     ]
        session_start_times = [ 
                               
                              ] # in second
    
    animal1_fixedorder = ['koala']
    animal2_fixedorder = ['vermelho']

    animal1_filename = "Koala"
    animal2_filename = "Vermelho"
    

# Dannon Kanga
if 1:
    if do_bestsession:      
        # both animals' lever force were changed - Self reward
        if forceManiType == 'SR_bothchange':
            dates_list = [ "20240912","20240913","20240917","20241101","20241104",
                           "20241105",
                           ] 
            session_start_times = [ 0.00,  0.00, 0.00, 0.00, 0.00,
                                    0.00,
                                   ] # in second
        # both animals' lever force were changed - cooperation
        elif forceManiType == 'CO_bothchange':
            dates_list = [  ]
            session_start_times = [  ] # in second
        # Dannon's lever force were changed
        if forceManiType == 'CO_A1change':
            dates_list = [ "20241009","20241011","20241016","20241018","20241022", 
                           "20241025", ]
            session_start_times = [ 0.00, 0.00, 0.00, 0.00, 0.00, 
                                    0.00, ] # in second
        # Kanga's lever force were changed
        if forceManiType == 'CO_A2change':
            dates_list = [ "20240910","20240911","20240916","20240918","20240919" ,
                           "20241008","20241010","20241014","20241017"]
            session_start_times = [ 0.00, 0.00, 0.00, 43.5, 0.00, 
                                    59.6, 66.0, 0.00, 0.00, ] # in second       
    
    elif not do_bestsession:
        # pick only five sessions for each conditions
        dates_list = [
                      
                     ]
        session_start_times = [ 
                               
                              ] # in second
    
    animal1_fixedorder = ['dannon']
    animal2_fixedorder = ['kanga']

    animal1_filename = "Dannon"
    animal2_filename = "Kanga"
    
    
    
#    
# dates_list = ["20240430"]
# session_start_times = [0.00] # in second
ndates = np.shape(dates_list)[0]

session_start_frames = session_start_times * fps # fps is 30Hz

totalsess_time = 600

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
# align the animal1 and animal2 across the sessions to the right animal1 and 2 fixed_order
animal1_name_all_dates = np.empty(shape=(0,), dtype=str)
animal2_name_all_dates = np.empty(shape=(0,), dtype=str)
trialdates_all_dates = np.empty(shape=(0,), dtype=str)
tasktypes_all_dates = np.zeros((0,))
coopthres_all_dates = np.zeros((0,))
force1_all_dates = np.zeros((0,)) 
force2_all_dates = np.zeros((0,)) 

subblockID_all_dates = np.zeros((0,))

succ_rate_all_dates = np.zeros((0,))
trialnum_all_dates = np.zeros((0,))
blocktime_all_dates = np.zeros((0,))

interpullintv_all_dates = np.zeros((0,))
pull1_IPI_all_dates = np.zeros((0,))
pull2_IPI_all_dates = np.zeros((0,))
pull1_IPI_std_all_dates = np.zeros((0,))
pull2_IPI_std_all_dates = np.zeros((0,))

owgaze1_num_all_dates = np.zeros((0,))
owgaze2_num_all_dates = np.zeros((0,))
mtgaze1_num_all_dates = np.zeros((0,))
mtgaze2_num_all_dates = np.zeros((0,))
pull1_num_all_dates = np.zeros((0,))
pull2_num_all_dates = np.zeros((0,))

lever1_holdtime_all_dates = np.zeros((0,))
lever2_holdtime_all_dates = np.zeros((0,))
lever1_holdtime_std_all_dates = np.zeros((0,))
lever2_holdtime_std_all_dates = np.zeros((0,))

lever1_gauge_all_dates = np.zeros((0,))
lever2_gauge_all_dates = np.zeros((0,))
lever1_gauge_std_all_dates = np.zeros((0,))
lever2_gauge_std_all_dates = np.zeros((0,))

bhv_intv_all_dates = dict.fromkeys(dates_list, [])


# where to save the summarizing data
data_saved_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/3d_recontruction_analysis_forceManipulation_task_data_saved/'


    


# In[13]:


# basic behavior analysis (define time stamps for each bhv events, etc)

try:
    
    print('load basic data for '+forceManiType)
    
    if redo_anystep:
        dummy
    
    # load saved data
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
   
    with open(data_saved_subfolder+'/animal1_name_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        animal1_name_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/animal2_name_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        animal2_name_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/trialdates_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        trialdates_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/tasktypes_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        tasktypes_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/coopthres_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        coopthres_all_dates = pickle.load(f)

    with open(data_saved_subfolder+'/force1_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        force1_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/force2_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        force2_all_dates = pickle.load(f)

    with open(data_saved_subfolder+'/subblockID_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        subblockID_all_dates = pickle.load(f)

    with open(data_saved_subfolder+'/succ_rate_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        succ_rate_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/trialnum_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        trialnum_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/blocktime_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
        blocktime_all_dates = pickle.load(f)

    print('all data from all dates are loaded')

except:

    print('analyze all dates for '+forceManiType)

    for idate in np.arange(0,ndates,1):
        date_tgt = dates_list[idate]
        session_start_time = session_start_times[idate]
        
        # load behavioral results
        try:
            bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_forceManipulation_task/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
            trial_record_json = glob.glob(bhv_data_path +date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_TrialRecord_" + "*.json")
            bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_bhv_data_" + "*.json")
            session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_session_info_" + "*.json")
            lever_reading_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_lever_reading_" + "*.json") 
            #
            trial_record = pd.read_json(trial_record_json[0])
            bhv_data = pd.read_json(bhv_data_json[0])
            session_info = pd.read_json(session_info_json[0])
            lever_reading = pd.read_json(lever_reading_json[0])
        except:
            bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_forceManipulation_task/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
            trial_record_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_TrialRecord_" + "*.json")
            bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_bhv_data_" + "*.json")
            session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_session_info_" + "*.json")
            lever_reading_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_lever_reading_" + "*.json")             
            #
            trial_record = pd.read_json(trial_record_json[0])
            bhv_data = pd.read_json(bhv_data_json[0])
            session_info = pd.read_json(session_info_json[0])
            lever_reading = pd.read_json(lever_reading_json[0])

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
        
        
        # load behavioral event results from the tracking analysis
        if 1:
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
                                                                                                                       considerlevertube,considertubeonly,sqr_thres_tubelever,
                                                                                                                       sqr_thres_face,sqr_thres_body)
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
            oneway_gaze1 = output_time_points_socialgaze['oneway_gaze1']
            oneway_gaze2 = output_time_points_socialgaze['oneway_gaze2']
            mutual_gaze1 = output_time_points_socialgaze['mutual_gaze1']
            mutual_gaze2 = output_time_points_socialgaze['mutual_gaze2']
             
            
        # after all the analysis, separate them based on different subblock    
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
        for itrialtype in np.arange(0,ntrialtypes,1):
            force1_unique = force12_uniques[0,itrialtype]
            force2_unique = force12_uniques[1,itrialtype]

            ind = np.isin(lever1force_list,force1_unique) & np.isin(lever2force_list,force2_unique)
            
            trialID_itrialtype = trialID_list[ind]
            
            tasktype_itrialtype = np.unique(tasktype_list[ind])
            coop_thres_itrialtype = np.unique(coop_thres_list[ind])
            
            # save some simple measures
            animal1_name_all_dates = np.append(animal1_name_all_dates,animal1)
            animal2_name_all_dates = np.append(animal2_name_all_dates,animal2)
            trialdates_all_dates = np.append(trialdates_all_dates,date_tgt)
            tasktypes_all_dates = np.append(tasktypes_all_dates,tasktype_itrialtype)
            coopthres_all_dates = np.append(coopthres_all_dates,coop_thres_itrialtype)
            #
            if np.isin(animal1,animal1_fixedorder):
                force1_all_dates = np.append(force1_all_dates,force1_unique)
                force2_all_dates = np.append(force2_all_dates,force2_unique)
            else:
                force1_all_dates = np.append(force1_all_dates,force2_unique)
                force2_all_dates = np.append(force2_all_dates,force1_unique)
            #
            trialnum_all_dates = np.append(trialnum_all_dates,np.sum(ind))
            subblockID_all_dates = np.append(subblockID_all_dates,itrialtype)
            
            # analyze behavior results
            bhv_data_itrialtype = bhv_data[np.isin(bhv_data['trial_number'],trialID_itrialtype)]
            #
            # successful rates
            succ_rate_itrialtype = np.sum((bhv_data_itrialtype['behavior_events']==3)|(bhv_data_itrialtype['behavior_events']==4))/np.sum((bhv_data_itrialtype['behavior_events']==1)|(bhv_data_itrialtype['behavior_events']==2))
            succ_rate_all_dates = np.append(succ_rate_all_dates,succ_rate_itrialtype)
            #
            # block time
            block_starttime = bhv_data_itrialtype[bhv_data_itrialtype['behavior_events']==0]['time_points'].iloc[0]
            block_endtime = bhv_data_itrialtype[bhv_data_itrialtype['behavior_events']==9]['time_points'].iloc[-1]
            blocktime_all_dates = np.append(blocktime_all_dates,block_endtime-block_starttime)
            #
            
                
    # save data
    if 1:        
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
                
        with open(data_saved_subfolder+'/animal1_name_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(animal1_name_all_dates, f)
        with open(data_saved_subfolder+'/animal2_name_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(animal2_name_all_dates, f)
        with open(data_saved_subfolder+'/trialdates_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(trialdates_all_dates, f)
        with open(data_saved_subfolder+'/tasktypes_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(tasktypes_all_dates, f)
        with open(data_saved_subfolder+'/coopthres_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(coopthres_all_dates, f)
            
        with open(data_saved_subfolder+'/force1_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(force1_all_dates, f)
        with open(data_saved_subfolder+'/force2_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(force2_all_dates, f)
            
        with open(data_saved_subfolder+'/subblockID_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(subblockID_all_dates, f)
            
        with open(data_saved_subfolder+'/succ_rate_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(succ_rate_all_dates, f)
        with open(data_saved_subfolder+'/trialnum_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(trialnum_all_dates, f)
        with open(data_saved_subfolder+'/blocktime_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
            pickle.dump(blocktime_all_dates, f)
        
                
    


# ### prepare the input data for DBN
# #### distribution of gaze before and after pulls

# In[23]:


# define DBN related summarizing variables
DBN_group_typenames = ['lowforce','highforce']

nDBN_groups = np.shape(DBN_group_typenames)[0]

DBN_input_data_alltypes = dict.fromkeys(DBN_group_typenames, [])

prepare_input_data = 1

# DBN resolutions (make sure they are the same as in the later part of the code)
totalsess_time = 600 # total session time in s
# temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
temp_resolus = [1] # temporal resolution in the DBN model, eg: 0.5 means 500ms

mergetempRos = 0

if mergetempRos:
    temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    # use bhv event to decide temporal resolution
    #
    #low_lim,up_lim,_ = bhv_events_interval(totalsess_time, session_start_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2)
    #temp_resolus = temp_resolus = np.arange(low_lim,up_lim,0.1)

#
ntemp_reses = np.shape(temp_resolus)[0]        

# # train the dynamic bayesian network - Alec's model 
#   prepare the multi-session table; one time lag; multi time steps (temporal resolution) as separate files

# prepare the DBN input data
if prepare_input_data:
   
    print('prepare DBN input data for '+forceManiType)
    
    # try different temporal resolutions
    for temp_resolu in temp_resolus:
        
        # bhv_df = [] # combine all dates
        
        for idate in np.arange(0,ndates,1):
            
            # bhv_df = [] # combine all block in one day
            
            date_tgt = dates_list[idate]
            session_start_time = session_start_times[idate]

            # load behavioral results
            try:
                bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_forceManipulation_task/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
                trial_record_json = glob.glob(bhv_data_path +date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_TrialRecord_" + "*.json")
                bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_bhv_data_" + "*.json")
                session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_session_info_" + "*.json")
                lever_reading_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_lever_reading_" + "*.json") 
                #
                trial_record = pd.read_json(trial_record_json[0])
                bhv_data = pd.read_json(bhv_data_json[0])
                session_info = pd.read_json(session_info_json[0])
                lever_reading = pd.read_json(lever_reading_json[0])
            except:
                bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_forceManipulation_task/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
                trial_record_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_TrialRecord_" + "*.json")
                bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_bhv_data_" + "*.json")
                session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_session_info_" + "*.json")
                lever_reading_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_lever_reading_" + "*.json")             
                #
                trial_record = pd.read_json(trial_record_json[0])
                bhv_data = pd.read_json(bhv_data_json[0])
                session_info = pd.read_json(session_info_json[0])
                lever_reading = pd.read_json(lever_reading_json[0])

            # get animal info
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


            # load behavioral event results
            print('load social gaze with '+cameraID+' only of '+date_tgt)
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_look_ornot.pkl', 'rb') as f:
                output_look_ornot = pickle.load(f)
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_allvectors.pkl', 'rb') as f:
                output_allvectors = pickle.load(f)
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_allangles.pkl', 'rb') as f:
                output_allangles = pickle.load(f)  
            #
            look_at_other_or_not_merge = output_look_ornot['look_at_other_or_not_merge']
            look_at_tube_or_not_merge = output_look_ornot['look_at_tube_or_not_merge']
            look_at_lever_or_not_merge = output_look_ornot['look_at_lever_or_not_merge']
            # change the unit to second
            # align to the session start time    
            look_at_other_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_other_or_not_merge['dodson'])[0],1)/fps - session_start_time
            look_at_lever_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_lever_or_not_merge['dodson'])[0],1)/fps - session_start_time
            look_at_tube_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_tube_or_not_merge['dodson'])[0],1)/fps - session_start_time 

            # find time point of behavioral events
            output_time_points_socialgaze ,output_time_points_levertube = bhv_events_timepoint_singlecam(bhv_data,look_at_other_or_not_merge,look_at_lever_or_not_merge,look_at_tube_or_not_merge)
            time_point_pull1 = output_time_points_socialgaze['time_point_pull1']
            time_point_pull2 = output_time_points_socialgaze['time_point_pull2']
            oneway_gaze1 = output_time_points_socialgaze['oneway_gaze1']
            oneway_gaze2 = output_time_points_socialgaze['oneway_gaze2']
            mutual_gaze1 = output_time_points_socialgaze['mutual_gaze1']
            mutual_gaze2 = output_time_points_socialgaze['mutual_gaze2']     


            # after all the analysis, separate them based on different subblock    
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
            if np.isin(animal1,animal1_fixedorder):
                force12names = [str(force12_uniques[0][i])+'&'+str(force12_uniques[1][i]) for i in np.arange(0,ntrialtypes,1)]
            else:
                force12names = [str(force12_uniques[1][i])+'&'+str(force12_uniques[0][i]) for i in np.arange(0,ntrialtypes,1)]                
            #
            # get the id of high or low force level
            yyy = np.sum(force12_uniques,axis=0)
            # only one kind of force
            if np.shape(np.unique(yyy))[0] == 1:
                yyy_quant = np.ones(np.shape(yyy))*2
            # two kinds of force
            elif np.shape(np.unique(yyy))[0] == 2:
                ranks = st.rankdata(yyy, method='average')  # Average ranks for ties
                # yyy_quant = (np.ceil(ranks / len(yyy) * 2)-1)*2+1 # separate into three quantiles
                yyy_quant = (np.ceil(ranks / len(yyy) * 2)) # separate into two quantiles         
            # more than two kinds of force,
            else:
                ranks = st.rankdata(yyy, method='average')  # Average ranks for ties
                # yyy_quant = np.ceil(ranks / len(yyy) * 3) # separate into three quantiles
                yyy_quant = (np.ceil(ranks / len(yyy) * 2)) # separate into two quantiles

            
            # 
            for itrialtype in np.arange(0,ntrialtypes,1):
                
                forcehighlowID = yyy_quant[itrialtype]
                
                bhv_df = []
                
                force1_unique = force12_uniques[0,itrialtype]
                force2_unique = force12_uniques[1,itrialtype]

                ind = np.isin(lever1force_list,force1_unique) & np.isin(lever2force_list,force2_unique)

                trialID_itrialtype = trialID_list[ind]

                tasktype_itrialtype = np.unique(tasktype_list[ind])
                coop_thres_itrialtype = np.unique(coop_thres_list[ind])

                # analyze behavior results
                bhv_data_itrialtype = bhv_data[np.isin(bhv_data['trial_number'],trialID_itrialtype)]

                # block time
                block_starttime = bhv_data_itrialtype[bhv_data_itrialtype['behavior_events']==0]['time_points'].iloc[0]
                block_endtime = bhv_data_itrialtype[bhv_data_itrialtype['behavior_events']==9]['time_points'].iloc[-1]


                #
                # prepare the DBN input data
                #
                totalsess_time_ittype = block_endtime - block_starttime
                session_start_time_ittype = 0
                #
                time_point_pull1_ittype = time_point_pull1[(time_point_pull1<block_endtime)&(time_point_pull1>block_starttime)]-block_starttime
                time_point_pull2_ittype = time_point_pull2[(time_point_pull2<block_endtime)&(time_point_pull2>block_starttime)]-block_starttime
                oneway_gaze1_ittype = oneway_gaze1[(oneway_gaze1<block_endtime)&(oneway_gaze1>block_starttime)]-block_starttime
                oneway_gaze2_ittype = oneway_gaze2[(oneway_gaze2<block_endtime)&(oneway_gaze2>block_starttime)]-block_starttime
                mutual_gaze1_ittype = mutual_gaze1[(mutual_gaze1<block_endtime)&(mutual_gaze1>block_starttime)]-block_starttime
                mutual_gaze2_ittype = mutual_gaze2[(mutual_gaze2<block_endtime)&(mutual_gaze2>block_starttime)]-block_starttime
                
                if np.isin(animal1,animal1_fixedorder):
                    bhv_df_itr,_,_ = train_DBN_multiLag_create_df_only(totalsess_time_ittype, 
                                                                       session_start_time_ittype, 
                                                                       temp_resolu, 
                                                                       time_point_pull1_ittype, time_point_pull2_ittype, 
                                                                       oneway_gaze1_ittype, oneway_gaze2_ittype, 
                                                                       mutual_gaze1_ittype, mutual_gaze2_ittype)
                else:
                    bhv_df_itr,_,_ = train_DBN_multiLag_create_df_only(totalsess_time_ittype, 
                                                                       session_start_time_ittype, 
                                                                       temp_resolu, 
                                                                       time_point_pull2_ittype, time_point_pull1_ittype, 
                                                                       oneway_gaze2_ittype, oneway_gaze1_ittype, 
                                                                       mutual_gaze2_ittype, mutual_gaze1_ittype)     

                # save data separately for high and low force
                if forcehighlowID == 1: # low force          
                    if len(DBN_input_data_alltypes['lowforce'])==0:
                        DBN_input_data_alltypes['lowforce'] = bhv_df_itr
                    else:
                        DBN_input_data_alltypes['lowforce'] = pd.concat([DBN_input_data_alltypes['lowforce'],
                                                                         bhv_df_itr])                   
                        DBN_input_data_alltypes['lowforce'] = DBN_input_data_alltypes['lowforce'].reset_index(drop=True)        
                #
                elif forcehighlowID == 2: # high force
                    if len(DBN_input_data_alltypes['highforce'])==0:
                        DBN_input_data_alltypes['highforce'] = bhv_df_itr
                    else:
                        DBN_input_data_alltypes['highforce'] = pd.concat([DBN_input_data_alltypes['highforce'],
                                                                         bhv_df_itr])                   
                        DBN_input_data_alltypes['highforce'] = DBN_input_data_alltypes['highforce'].reset_index(drop=True)        
                
                                
            
        # save data
        if 1:
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_combinedsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
            if not os.path.exists(data_saved_subfolder):
                os.makedirs(data_saved_subfolder)
            if not mergetempRos:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_'+str(temp_resolu)+'sReSo.pkl', 'wb') as f:
                    pickle.dump(DBN_input_data_alltypes, f)
            else:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_mergeTempsReSo.pkl', 'wb') as f:
                    pickle.dump(DBN_input_data_alltypes, f)     
                
                    


# In[ ]:





# ### run the DBN model on the combined session data set

# #### a test run

# In[24]:


# run DBN on the large table with merged sessions

mergetempRos = 0 # 1: merge different time bins

moreSampSize = 0 # 1: use more sample size (more than just minimal row number and max row number)

num_starting_points = 1 # number of random starting points/graphs
nbootstraps = 1

if 0:

    if moreSampSize:
        # different data (down/re)sampling numbers
        samplingsizes = np.arange(1100,3000,100)
        # samplingsizes = [100,500,1000,1500,2000,2500,3000]        
        # samplingsizes = [100,500]
        # samplingsizes_name = ['100','500','1000','1500','2000','2500','3000']
        samplingsizes_name = list(map(str, samplingsizes))
        nsamplings = np.shape(samplingsizes)[0]

    weighted_graphs_diffTempRo_diffSampSize = {}
    weighted_graphs_shuffled_diffTempRo_diffSampSize = {}
    sig_edges_diffTempRo_diffSampSize = {}
    DAGscores_diffTempRo_diffSampSize = {}
    DAGscores_shuffled_diffTempRo_diffSampSize = {}

    totalsess_time = 600 # total session time in s
    # temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    temp_resolus = [1] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    ntemp_reses = np.shape(temp_resolus)[0]

    # try different temporal resolutions, remember to use the same settings as in the previous ones
    for temp_resolu in temp_resolus:

        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_combinedsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not mergetempRos:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_mergeTempsReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)

                
        # only try two sample sizes - minimal row number (require data downsample) and maximal row number (require data upsample)
       
        if not moreSampSize:
            key_to_value_lengths = {k:len(v) for k, v in DBN_input_data_alltypes.items()}
            key_to_value_lengths_array = np.fromiter(key_to_value_lengths.values(),dtype=float)
            key_to_value_lengths_array[key_to_value_lengths_array==0]=np.nan
            min_samplesize = np.nanmin(key_to_value_lengths_array)
            min_samplesize = int(min_samplesize/100)*100
            max_samplesize = np.nanmax(key_to_value_lengths_array)
            max_samplesize = int(max_samplesize/100)*100
            # samplingsizes = [min_samplesize,max_samplesize]
            # samplingsizes_name = ['min_row_number','max_row_number'] 
            samplingsizes = [min_samplesize,]
            samplingsizes_name = ['min_row_number',] 
            nsamplings = np.shape(samplingsizes)[0]
            print(samplingsizes)
                
        # try different down/re-sampling size
        # for jj in np.arange(0,nsamplings,1):
        for jj in np.arange(0,1,1):
            
            isamplingsize = samplingsizes[jj]
            
            DAGs_alltypes = dict.fromkeys(DBN_group_typenames, [])
            DAGs_shuffle_alltypes = dict.fromkeys(DBN_group_typenames, [])
            DAGs_scores_alltypes = dict.fromkeys(DBN_group_typenames, [])
            DAGs_shuffle_scores_alltypes = dict.fromkeys(DBN_group_typenames, [])

            weighted_graphs_alltypes = dict.fromkeys(DBN_group_typenames, [])
            weighted_graphs_shuffled_alltypes = dict.fromkeys(DBN_group_typenames, [])
            sig_edges_alltypes = dict.fromkeys(DBN_group_typenames, [])

            # different session conditions (aka DBN groups)
            # for iDBN_group in np.arange(0,nDBN_groups,1):
            for iDBN_group in np.arange(0,1,1):
                iDBN_group_typename = DBN_group_typenames[iDBN_group] 

                # try:
                bhv_df_all = DBN_input_data_alltypes[iDBN_group_typename]
                # bhv_df = bhv_df_all.sample(30*100,replace = True, random_state = round(time())) # take the subset for DBN training

                #Anirban(Alec) shuffle, slow
                # bhv_df_shuffle, df_shufflekeys = EfficientShuffle(bhv_df,round(time()))


                # define DBN graph structures; make sure they are the same as in the train_DBN_multiLag
                colnames = list(bhv_df_all.columns)
                eventnames = ["pull1","pull2","owgaze1","owgaze2"]
                nevents = np.size(eventnames)

                all_pops = list(bhv_df_all.columns)
                from_pops = [pop for pop in all_pops if not pop.endswith('t3')]
                to_pops = [pop for pop in all_pops if pop.endswith('t3')]
                causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]

                nFromNodes = np.shape(from_pops)[0]
                nToNodes = np.shape(to_pops)[0]

                DAGs_randstart = np.zeros((num_starting_points, nFromNodes, nToNodes))
                DAGs_randstart_shuffle = np.zeros((num_starting_points, nFromNodes, nToNodes))
                score_randstart = np.zeros((num_starting_points))
                score_randstart_shuffle = np.zeros((num_starting_points))

                # step 1: randomize the starting point for num_starting_points times
                for istarting_points in np.arange(0,num_starting_points,1):

                    # try different down/re-sampling size
                    bhv_df = bhv_df_all.sample(isamplingsize,replace = True, random_state = istarting_points) # take the subset for DBN training
                    aic = AicScore(bhv_df)

                    #Anirban(Alec) shuffle, slow
                    bhv_df_shuffle, df_shufflekeys = EfficientShuffle(bhv_df,round(time()))
                    aic_shuffle = AicScore(bhv_df_shuffle)

                    np.random.seed(istarting_points)
                    random.seed(istarting_points)
                    starting_edges = random.sample(causal_whitelist, np.random.randint(1,len(causal_whitelist)))
                    starting_graph = DAG()
                    starting_graph.add_nodes_from(nodes=all_pops)
                    starting_graph.add_edges_from(ebunch=starting_edges)

                    best_model,edges,DAGs = train_DBN_multiLag_training_only(bhv_df,starting_graph,colnames,eventnames,from_pops,to_pops)           
                    DAGs[0][np.isnan(DAGs[0])]=0

                    DAGs_randstart[istarting_points,:,:] = DAGs[0]
                    score_randstart[istarting_points] = aic.score(best_model)

                    # step 2: add the shffled data results
                    # shuffled bhv_df
                    best_model,edges,DAGs = train_DBN_multiLag_training_only(bhv_df_shuffle,starting_graph,colnames,eventnames,from_pops,to_pops)           
                    DAGs[0][np.isnan(DAGs[0])]=0

                    DAGs_randstart_shuffle[istarting_points,:,:] = DAGs[0]
                    score_randstart_shuffle[istarting_points] = aic_shuffle.score(best_model)

                DAGs_alltypes[iDBN_group_typename] = DAGs_randstart 
                DAGs_shuffle_alltypes[iDBN_group_typename] = DAGs_randstart_shuffle

                DAGs_scores_alltypes[iDBN_group_typename] = score_randstart
                DAGs_shuffle_scores_alltypes[iDBN_group_typename] = score_randstart_shuffle

                weighted_graphs = get_weighted_dags(DAGs_alltypes[iDBN_group_typename],nbootstraps)
                weighted_graphs_shuffled = get_weighted_dags(DAGs_shuffle_alltypes[iDBN_group_typename],nbootstraps)
                sig_edges = get_significant_edges(weighted_graphs,weighted_graphs_shuffled)

                weighted_graphs_alltypes[iDBN_group_typename] = weighted_graphs
                weighted_graphs_shuffled_alltypes[iDBN_group_typename] = weighted_graphs_shuffled
                sig_edges_alltypes[iDBN_group_typename] = sig_edges
                    
                # except:
                #     DAGs_alltypes[iDBN_group_typename] = [] 
                 #    DAGs_shuffle_alltypes[iDBN_group_typename] = []

                #     DAGs_scores_alltypes[iDBN_group_typename] = []
                #     DAGs_shuffle_scores_alltypes[iDBN_group_typename] = []

                #     weighted_graphs_alltypes[iDBN_group_typename] = []
                #     weighted_graphs_shuffled_alltypes[iDBN_group_typename] = []
                #     sig_edges_alltypes[iDBN_group_typename] = []
                
            DAGscores_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = DAGs_scores_alltypes
            DAGscores_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = DAGs_shuffle_scores_alltypes

            weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_alltypes
            weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_shuffled_alltypes
            sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = sig_edges_alltypes

    print(weighted_graphs_diffTempRo_diffSampSize)
            
   


# In[ ]:





# #### run on the entire population

# In[25]:


# run DBN on the large table with merged sessions

mergetempRos = 0 # 1: merge different time bins

moreSampSize = 0 # 1: use more sample size (more than just minimal row number and max row number)

num_starting_points = 100 # number of random starting points/graphs
nbootstraps = 95

try:
    dumpy
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_combinedsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'

    if moreSampSize:
        with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'rb') as f:
            DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
        with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'rb') as f:
            DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
        with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'rb') as f:
            weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
        with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'rb') as f:
            weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
        with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'rb') as f:
            sig_edges_diffTempRo_diffSampSize = pickle.load(f)

    else:
        with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
            DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
        with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
            DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
        with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
            weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
        with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
            weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
        with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'rb') as f:
            sig_edges_diffTempRo_diffSampSize = pickle.load(f)
    
    print('load DBN trained data')

except:
    if moreSampSize:
        # different data (down/re)sampling numbers
        samplingsizes = np.arange(1100,3000,100)
        # samplingsizes = [100,500,1000,1500,2000,2500,3000]        
        # samplingsizes = [100,500]
        # samplingsizes_name = ['100','500','1000','1500','2000','2500','3000']
        samplingsizes_name = list(map(str, samplingsizes))
        nsamplings = np.shape(samplingsizes)[0]

    weighted_graphs_diffTempRo_diffSampSize = {}
    weighted_graphs_shuffled_diffTempRo_diffSampSize = {}
    sig_edges_diffTempRo_diffSampSize = {}
    DAGscores_diffTempRo_diffSampSize = {}
    DAGscores_shuffled_diffTempRo_diffSampSize = {}

    totalsess_time = 600 # total session time in s
    # temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    temp_resolus = [1] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    ntemp_reses = np.shape(temp_resolus)[0]

    # try different temporal resolutions, remember to use the same settings as in the previous ones
    for temp_resolu in temp_resolus:

        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_combinedsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not mergetempRos:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_mergeTempsReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)

                
        # only try two sample sizes - minimal row number (require data downsample) and maximal row number (require data upsample)
       
        if not moreSampSize:
            key_to_value_lengths = {k:len(v) for k, v in DBN_input_data_alltypes.items()}
            key_to_value_lengths_array = np.fromiter(key_to_value_lengths.values(),dtype=float)
            key_to_value_lengths_array[key_to_value_lengths_array==0]=np.nan
            min_samplesize = np.nanmin(key_to_value_lengths_array)
            min_samplesize = int(min_samplesize/100)*100
            max_samplesize = np.nanmax(key_to_value_lengths_array)
            max_samplesize = int(max_samplesize/100)*100
            # samplingsizes = [min_samplesize,max_samplesize]
            # samplingsizes_name = ['min_row_number','max_row_number']   
            samplingsizes = [min_samplesize]
            samplingsizes_name = ['min_row_number']   
            nsamplings = np.shape(samplingsizes)[0]
            print(samplingsizes)
                
        # try different down/re-sampling size
        for jj in np.arange(0,nsamplings,1):
            
            isamplingsize = samplingsizes[jj]
            
            DAGs_alltypes = dict.fromkeys(DBN_group_typenames, [])
            DAGs_shuffle_alltypes = dict.fromkeys(DBN_group_typenames, [])
            DAGs_scores_alltypes = dict.fromkeys(DBN_group_typenames, [])
            DAGs_shuffle_scores_alltypes = dict.fromkeys(DBN_group_typenames, [])

            weighted_graphs_alltypes = dict.fromkeys(DBN_group_typenames, [])
            weighted_graphs_shuffled_alltypes = dict.fromkeys(DBN_group_typenames, [])
            sig_edges_alltypes = dict.fromkeys(DBN_group_typenames, [])

            # different session conditions (aka DBN groups)
            for iDBN_group in np.arange(0,nDBN_groups,1):
                iDBN_group_typename = DBN_group_typenames[iDBN_group] 
                
                try:
                    bhv_df_all = DBN_input_data_alltypes[iDBN_group_typename]
                    # bhv_df = bhv_df_all.sample(30*100,replace = True, random_state = round(time())) # take the subset for DBN training

                    #Anirban(Alec) shuffle, slow
                    # bhv_df_shuffle, df_shufflekeys = EfficientShuffle(bhv_df,round(time()))


                    # define DBN graph structures; make sure they are the same as in the train_DBN_multiLag
                    colnames = list(bhv_df_all.columns)
                    eventnames = ["pull1","pull2","owgaze1","owgaze2"]
                    nevents = np.size(eventnames)

                    all_pops = list(bhv_df_all.columns)
                    from_pops = [pop for pop in all_pops if not pop.endswith('t3')]
                    to_pops = [pop for pop in all_pops if pop.endswith('t3')]
                    causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]

                    nFromNodes = np.shape(from_pops)[0]
                    nToNodes = np.shape(to_pops)[0]

                    DAGs_randstart = np.zeros((num_starting_points, nFromNodes, nToNodes))
                    DAGs_randstart_shuffle = np.zeros((num_starting_points, nFromNodes, nToNodes))
                    score_randstart = np.zeros((num_starting_points))
                    score_randstart_shuffle = np.zeros((num_starting_points))

                    # step 1: randomize the starting point for num_starting_points times
                    for istarting_points in np.arange(0,num_starting_points,1):

                        # try different down/re-sampling size
                        bhv_df = bhv_df_all.sample(isamplingsize,replace = True, random_state = istarting_points) # take the subset for DBN training
                        aic = AicScore(bhv_df)

                        #Anirban(Alec) shuffle, slow
                        bhv_df_shuffle, df_shufflekeys = EfficientShuffle(bhv_df,round(time()))
                        aic_shuffle = AicScore(bhv_df_shuffle)

                        np.random.seed(istarting_points)
                        random.seed(istarting_points)
                        starting_edges = random.sample(causal_whitelist, np.random.randint(1,len(causal_whitelist)))
                        starting_graph = DAG()
                        starting_graph.add_nodes_from(nodes=all_pops)
                        starting_graph.add_edges_from(ebunch=starting_edges)

                        best_model,edges,DAGs = train_DBN_multiLag_training_only(bhv_df,starting_graph,colnames,eventnames,from_pops,to_pops)           
                        DAGs[0][np.isnan(DAGs[0])]=0

                        DAGs_randstart[istarting_points,:,:] = DAGs[0]
                        score_randstart[istarting_points] = aic.score(best_model)

                        # step 2: add the shffled data results
                        # shuffled bhv_df
                        best_model,edges,DAGs = train_DBN_multiLag_training_only(bhv_df_shuffle,starting_graph,colnames,eventnames,from_pops,to_pops)           
                        DAGs[0][np.isnan(DAGs[0])]=0

                        DAGs_randstart_shuffle[istarting_points,:,:] = DAGs[0]
                        score_randstart_shuffle[istarting_points] = aic_shuffle.score(best_model)

                    DAGs_alltypes[iDBN_group_typename] = DAGs_randstart 
                    DAGs_shuffle_alltypes[iDBN_group_typename] = DAGs_randstart_shuffle

                    DAGs_scores_alltypes[iDBN_group_typename] = score_randstart
                    DAGs_shuffle_scores_alltypes[iDBN_group_typename] = score_randstart_shuffle

                    weighted_graphs = get_weighted_dags(DAGs_alltypes[iDBN_group_typename],nbootstraps)
                    weighted_graphs_shuffled = get_weighted_dags(DAGs_shuffle_alltypes[iDBN_group_typename],nbootstraps)
                    sig_edges = get_significant_edges(weighted_graphs,weighted_graphs_shuffled)

                    weighted_graphs_alltypes[iDBN_group_typename] = weighted_graphs
                    weighted_graphs_shuffled_alltypes[iDBN_group_typename] = weighted_graphs_shuffled
                    sig_edges_alltypes[iDBN_group_typename] = sig_edges
                    
                except:
                    DAGs_alltypes[iDBN_group_typename] = [] 
                    DAGs_shuffle_alltypes[iDBN_group_typename] = []

                    DAGs_scores_alltypes[iDBN_group_typename] = []
                    DAGs_shuffle_scores_alltypes[iDBN_group_typename] = []

                    weighted_graphs_alltypes[iDBN_group_typename] = []
                    weighted_graphs_shuffled_alltypes[iDBN_group_typename] = []
                    sig_edges_alltypes[iDBN_group_typename] = []
                
            DAGscores_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = DAGs_scores_alltypes
            DAGscores_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = DAGs_shuffle_scores_alltypes

            weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_alltypes
            weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_shuffled_alltypes
            sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = sig_edges_alltypes

            
    # save data
    savedata = 1
    if savedata:
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_combinedsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
        if moreSampSize:  
            with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'wb') as f:
                pickle.dump(DAGscores_diffTempRo_diffSampSize, f)
            with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'wb') as f:
                pickle.dump(DAGscores_shuffled_diffTempRo_diffSampSize, f)
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'wb') as f:
                pickle.dump(weighted_graphs_diffTempRo_diffSampSize, f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'wb') as f:
                pickle.dump(weighted_graphs_shuffled_diffTempRo_diffSampSize, f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'_moreSampSize.pkl', 'wb') as f:
                pickle.dump(sig_edges_diffTempRo_diffSampSize, f)

        else:
            with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
                pickle.dump(DAGscores_diffTempRo_diffSampSize, f)
            with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
                pickle.dump(DAGscores_shuffled_diffTempRo_diffSampSize, f)
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
                pickle.dump(weighted_graphs_diffTempRo_diffSampSize, f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
                pickle.dump(weighted_graphs_shuffled_diffTempRo_diffSampSize, f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+forceManiType+'.pkl', 'wb') as f:
                pickle.dump(sig_edges_diffTempRo_diffSampSize, f)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





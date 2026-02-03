#!/usr/bin/env python
# coding: utf-8

# ### This script runs some basic bhv analysis, and detailed analysis focus on the continuous behavioral variables

# #### The output of this script will also be used by the DBN scripts

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

import os
import glob
import random
from time import time


# ### function - get body part location for each pair of cameras

# In[2]:


from ana_functions.body_part_locs_eachpair import body_part_locs_eachpair


# ### function - find social gaze time point

# In[3]:


from ana_functions.find_socialgaze_timepoint import find_socialgaze_timepoint
from ana_functions.find_socialgaze_timepoint_Anipose import find_socialgaze_timepoint_Anipose
from ana_functions.find_socialgaze_timepoint_Anipose_2 import find_socialgaze_timepoint_Anipose_2


# ### function - define time point of behavioral events

# In[4]:


from ana_functions.bhv_events_timepoint import bhv_events_timepoint
from ana_functions.bhv_events_timepoint_Anipose import bhv_events_timepoint_Anipose


# ### function - plot behavioral events

# In[5]:


from ana_functions.plot_bhv_events import plot_bhv_events
from ana_functions.plot_bhv_events_levertube import plot_bhv_events_levertube
from ana_functions.tracking_video_Anipose_events_demo import tracking_video_Anipose_events_demo
from ana_functions.plot_continuous_bhv_var import plot_continuous_bhv_var
from ana_functions.draw_self_loop import draw_self_loop
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection


# ### function - interval between all behavioral events

# In[6]:


from ana_functions.bhv_events_interval import bhv_events_interval


# ### function - spike analysis

# In[7]:


from ana_functions.spike_analysis_FR_calculation import spike_analysis_FR_calculation
from ana_functions.plot_spike_triggered_continuous_bhv import plot_spike_triggered_continuous_bhv


# ### function - PCA projection

# In[8]:


from ana_functions.PCA_around_bhv_events import PCA_around_bhv_events
from ana_functions.PCA_around_bhv_events_video import PCA_around_bhv_events_video
from ana_functions.confidence_ellipse import confidence_ellipse

from ana_functions.plot_continuous_bhv_var_and_neuralFR import plot_continuous_bhv_var_and_neuralFR


# ## Analyze each session

# ### prepare the basic behavioral data (especially the time stamps for each bhv events)

# In[9]:


# gaze angle threshold
# angle_thres = np.pi/36 # 5 degree
# angle_thres = np.pi/18 # 10 degree
angle_thres = np.pi/12 # 15 degree
# angle_thres = np.pi/4 # 45 degree
# angle_thres = np.pi/6 # 30 degree
angle_thres_name = '15'

merge_campairs = ['_Anipose'] # "_Anipose": this script is only for Anipose 3d reconstruction of camera 1,2,3 

with_tubelever = 1 # 1: consider the location of tubes and levers, only works if using Anipose 3d (or single camera)

# get the fps of the analyzed video
fps = 30

# get the fs for neural recording
fs_spikes = 20000
fs_lfp = 1000

# frame number of the demo video
# nframes = 1*30
nframes = 1

# re-analyze the video or not
reanalyze_video = 0
redo_anystep = 1

# do OFC sessions or DLPFC sessions
do_OFC = 0
do_DLPFC = 1
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


# dodson ginger
if 0:
    if do_DLPFC:
        neural_record_conditions = [
                                     # # '20231204_Dodson_withGinger_SR', 
                                     # '20231204_Dodson_withGinger_MC',
                                    # '20240610_Dodson_MC',
                                    '20240531_Dodson_MC_and_SR',
                                   ]
        dates_list = [
                      # # "20231204_SR","20231204_MC",
                      # '20240610_MC',
                      '20240531',
                     ]
        task_conditions = [
                            'MCandSR',    
                          ]
        session_start_times = [ 
                                # # 0.00,  107.50, 
                                # 0.00, 
                                0.00,
                              ] # in second
        kilosortvers = [ 
                         # # 2, 2,
                         # 4,
                         4,
                       ]
    elif do_OFC:
        # pick only five sessions for each conditions
        neural_record_conditions = [
                                     '20231101_Dodson_withGinger_MC',
                                     # '20231107_Dodson_withGinger_MC',
                                     # '20231122_Dodson_withGinger_MC',
                                     # '20231129_Dodson_withGinger_MC',
                                     # '20231101_Dodson_withGinger_SR',
                                   ]
        dates_list = [
                      "20231101_MC",
                      # "20231107_MC",
                      # "20231122_MC",
                      # "20231129_MC",
                      # "20231101_SR",
                     ]
        task_conditions = [
                            'MC',    
                          ]
        session_start_times = [ 
                                 0.00,   
                                #  0.00,  
                                #  0.00,  
                                #  0.00, 
                              ] # in second
        kilosortvers = [ 2, # 2, 2, 2,
                       ]
    
    animal1_fixedorder = ['dodson']
    animal2_fixedorder = ['ginger']

    animal1_filename = "Dodson"
    animal2_filename = "Ginger"

    
# dannon kanga
if 1:
    if do_DLPFC:
        neural_record_conditions = [
                                     '20240508_Kanga_SR',
                                     '20240509_Kanga_MC',
                                     '20240513_Kanga_MC',
                                     '20240514_Kanga_SR',
                                     '20240523_Kanga_MC',
                                     '20240524_Kanga_SR',
                                     '20240606_Kanga_MC',
                                     '20240613_Kanga_MC_DannonAuto',
                                     '20240614_Kanga_MC_DannonAuto',
                                     '20240617_Kanga_MC_DannonAuto',
                                     '20240618_Kanga_MC_KangaAuto',
                                     '20240619_Kanga_MC_KangaAuto',
                                     '20240620_Kanga_MC_KangaAuto',
                                     '20240621_1_Kanga_NoVis',
                                     '20240624_Kanga_NoVis',
                                     '20240626_Kanga_NoVis',
                                   ]
        dates_list = [
                      "20240508",
                      "20240509",
                      "20240513",
                      "20240514",
                      "20240523",
                      "20240524",
                      "20240606",
                      "20240613",
                      "20240614",
                      "20240617",
                      "20240618",
                      "20240619",
                      "20240620",
                      "20240621_1",
                      "20240624",
                      "20240626",
                     ]
        task_conditions = [
                             'SR',
                             'MC',
                             'MC',
                             'SR',
                             'MC',
                             'SR',
                             'MC',
                             'MC_DannonAuto',
                             'MC_DannonAuto',
                             'MC_DannonAuto',
                             'MC_KangaAuto',
                             'MC_KangaAuto',
                             'MC_KangaAuto',
                             'NV',
                             'NV',
                             'NV',    
                          ]
        session_start_times = [ 
                                 0.00,
                                 36.0,
                                 69.5,
                                 0.00,
                                 62.0,
                                 0.00,
                                 89.0,
                                 0.00,
                                 0.00,
                                 0.00,
                                 165.8,
                                 96.0, 
                                 0.00,
                                 0.00,
                                 0.00,
                                 48.0,
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
                        ]
    elif do_OFC:
        # pick only five sessions for each conditions
        neural_record_conditions = [
                                     
                                   ]
        dates_list = [
                      
                     ]
        task_conditions = [
                                
                          ]
        session_start_times = [ 
                                
                              ] # in second
        kilosortvers = [
     
                        ] # 2 or 4
    
    animal1_fixedorder = ['dannon']
    animal2_fixedorder = ['kanga']

    animal1_filename = "Dannon"
    animal2_filename = "Kanga"
    


# a test case
if 0:
    neural_record_conditions = ['20240523_Kanga_MC']
    dates_list = ["20240523"]
    task_conditions = ['MC']
    session_start_times = [62.0] # in second
    kilosortvers = [4]
    animal1_fixedorder = ['dannon']
    animal2_fixedorder = ['kanga']
    animal1_filename = "Dannon"
    animal2_filename = "Kanga"

ndates = np.shape(dates_list)[0]

session_start_frames = session_start_times * fps # fps is 30Hz

totalsess_time = 600

if np.shape(session_start_times)[0] != np.shape(dates_list)[0]:
    exit()    
    
# define bhv events summarizing variables     
tasktypes_all_dates = np.zeros((ndates,1))
coopthres_all_dates = np.zeros((ndates,1))

succ_rate_all_dates = np.zeros((ndates,1))
interpullintv_all_dates = np.zeros((ndates,1))
trialnum_all_dates = np.zeros((ndates,1))

# switch animals to make animal 1 and 2 consistent
owgaze1_num_all_dates = np.zeros((ndates,1))
owgaze2_num_all_dates = np.zeros((ndates,1))
mtgaze1_num_all_dates = np.zeros((ndates,1))
mtgaze2_num_all_dates = np.zeros((ndates,1))
pull1_num_all_dates = np.zeros((ndates,1))
pull2_num_all_dates = np.zeros((ndates,1))

bhv_intv_all_dates = dict.fromkeys(dates_list, [])
pull_trig_events_all_dates = dict.fromkeys(dates_list, [])
pull_trig_events_succtrials_all_dates = dict.fromkeys(dates_list, [])
pull_trig_events_errtrials_all_dates = dict.fromkeys(dates_list, [])

spike_trig_events_all_dates = dict.fromkeys(dates_list, [])

# video tracking results info
animalnames_videotrack = ['dodson','scorch'] # does not really mean dodson and scorch, instead, indicate animal1 and animal2
bodypartnames_videotrack = ['rightTuft','whiteBlaze','leftTuft','rightEye','leftEye','mouth']

# where to save the summarizing data
data_saved_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/3d_recontruction_analysis_self_and_coop_task_data_saved/'

# neural data folder
neural_data_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/Marmoset_neural_recording/'

# where to save the demo video
withboxCorner = 1
video_file_dir = data_saved_folder+'/example_videos_Anipose_bhv_demo/'+animal1_filename+'_'+animal2_filename
if not os.path.exists(video_file_dir):
    os.makedirs(video_file_dir)

    


# In[ ]:


# basic behavior analysis (define time stamps for each bhv events, etc)
# NOTE: THIS STEP will save the data to the combinedsession_Anipose folder, since they are the same
try:
    if redo_anystep:
        dummy
    # dummy
    
    data_saved_subfolder = data_saved_folder+'data_saved_combinedsessions_Anipose'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    
    with open(data_saved_subfolder+'/owgaze1_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        owgaze1_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/owgaze2_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        owgaze2_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/mtgaze1_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        mtgaze1_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/mtgaze2_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        mtgaze2_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/pull1_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        pull1_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/pull2_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        pull2_num_all_dates = pickle.load(f)

    with open(data_saved_subfolder+'/tasktypes_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        tasktypes_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/coopthres_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        coopthres_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/succ_rate_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        succ_rate_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/interpullintv_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        interpullintv_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/trialnum_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        trialnum_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/bhv_intv_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        bhv_intv_all_dates = pickle.load(f)
        
    with open(data_saved_subfolder+'/pull_trig_events_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        pull_trig_events_all_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/pull_trig_events_succtrials_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        pull_trig_events_succtrials_all_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/pull_trig_events_errtrials_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        pull_trig_events_errtrials_all_dates = pickle.load(f) 
    
    with open(data_saved_subfolder+'/spike_trig_events_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'rb') as f:
        spike_trig_events_all_dates = pickle.load(f) 
    
    
    print('all data saved, loading them')

        
except:

    for idate in np.arange(0,ndates,1):
        date_tgt = dates_list[idate]
        session_start_time = session_start_times[idate]
        neural_record_condition = neural_record_conditions[idate]
        kilosortver = kilosortvers[idate]

        # folder path
        camera12_analyzed_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera12/"
        camera23_analyzed_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera23/"
        Anipose_analyzed_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera12/anipose_cam123_3d_h5_files/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"

        for imergepair in np.arange(0,np.shape(merge_campairs)[0],1):
            
            # should be only one merge type - "Anipose"
            merge_campair = merge_campairs[imergepair]

            # load camera tracking results
            try:
                # dummy
                if reanalyze_video:
                    print("re-analyze the data ",date_tgt)
                    dummy
                ## read
                with open(Anipose_analyzed_path + 'body_part_locs_Anipose.pkl', 'rb') as f:
                    body_part_locs_Anipose = pickle.load(f)                 
            except:
                print("did not save data for Anipose - body part tracking "+date_tgt)
                # analyze and save
                Anipose_h5_file = Anipose_analyzed_path +date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_anipose.h5"
                Anipose_h5_data = pd.read_hdf(Anipose_h5_file)
                body_part_locs_Anipose = body_part_locs_eachpair(Anipose_h5_data)
                with open(Anipose_analyzed_path + 'body_part_locs_Anipose.pkl', 'wb') as f:
                    pickle.dump(body_part_locs_Anipose, f)            
            
            min_length = np.min(list(body_part_locs_Anipose.values())[0].shape[0])
                    
            # load behavioral results
            try:
                bhv_data_path = "/home/ws523/marmoset_tracking_bhv_data_from_task_code/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
                trial_record_json = glob.glob(bhv_data_path +date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_TrialRecord_" + "*.json")
                bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_bhv_data_" + "*.json")
                session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_session_info_" + "*.json")
                ni_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_ni_data_" + "*.json")
                #
                trial_record = pd.read_json(trial_record_json[0])
                bhv_data = pd.read_json(bhv_data_json[0])
                session_info = pd.read_json(session_info_json[0])
                # 
                with open(ni_data_json[0]) as f:
                    for line in f:
                        ni_data=json.loads(line)   
            except:
                bhv_data_path = "/home/ws523/marmoset_tracking_bhv_data_from_task_code/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
                trial_record_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_TrialRecord_" + "*.json")
                bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_bhv_data_" + "*.json")
                session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_session_info_" + "*.json")
                ni_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_ni_data_" + "*.json")
                #
                trial_record = pd.read_json(trial_record_json[0])
                bhv_data = pd.read_json(bhv_data_json[0])
                session_info = pd.read_json(session_info_json[0])
                #
                with open(ni_data_json[0]) as f:
                    for line in f:
                        ni_data=json.loads(line)

            # get animal info
            animal1 = session_info['lever1_animal'][0].lower()
            animal2 = session_info['lever2_animal'][0].lower()

            # get task type and cooperation threshold
            try:
                coop_thres = session_info["pulltime_thres"][0]
                tasktype = session_info["task_type"][0]
            except:
                coop_thres = 0
                tasktype = 1
            tasktypes_all_dates[idate] = tasktype
            coopthres_all_dates[idate] = coop_thres   

            # successful trial or not
            succtrial_ornot = np.array((trial_record['rewarded']>0).astype(int))
            succpull1_ornot = np.array((np.isin(bhv_data[bhv_data['behavior_events']==1]['trial_number'],trial_record[trial_record['rewarded']>0]['trial_number'])).astype(int))
            succpull2_ornot = np.array((np.isin(bhv_data[bhv_data['behavior_events']==2]['trial_number'],trial_record[trial_record['rewarded']>0]['trial_number'])).astype(int))
            succpulls_ornot = [succpull1_ornot,succpull2_ornot]
            
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


            # analyze behavior results
            # succ_rate_all_dates[idate] = np.sum(trial_record_clean["rewarded"]>0)/np.shape(trial_record_clean)[0]
            succ_rate_all_dates[idate] = np.sum((bhv_data['behavior_events']==3)|(bhv_data['behavior_events']==4))/np.sum((bhv_data['behavior_events']==1)|(bhv_data['behavior_events']==2))

            trialnum_all_dates[idate] = np.shape(trial_record_clean)[0]
            #
            pullid = np.array(bhv_data[(bhv_data['behavior_events']==1) | (bhv_data['behavior_events']==2)]["behavior_events"])
            pulltime = np.array(bhv_data[(bhv_data['behavior_events']==1) | (bhv_data['behavior_events']==2)]["time_points"])
            pullid_diff = np.abs(pullid[1:] - pullid[0:-1])
            pulltime_diff = pulltime[1:] - pulltime[0:-1]
            interpull_intv = pulltime_diff[pullid_diff==1]
            interpull_intv = interpull_intv[interpull_intv<10]
            mean_interpull_intv = np.nanmean(interpull_intv)
            std_interpull_intv = np.nanstd(interpull_intv)
            #
            interpullintv_all_dates[idate] = mean_interpull_intv

            if np.isin(animal1,animal1_fixedorder):
                pull1_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==1) 
                pull2_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==2)
            else:
                pull1_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==2)
                pull2_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==1) 
                
            # load behavioral event results
            try:
                # dummy
                print('load social gaze with Anipose 3d of '+date_tgt)
                with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_look_ornot.pkl', 'rb') as f:
                    output_look_ornot = pickle.load(f)
                with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_allvectors.pkl', 'rb') as f:
                    output_allvectors = pickle.load(f)
                with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_allangles.pkl', 'rb') as f:
                    output_allangles = pickle.load(f)  
                with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_key_locations.pkl', 'rb') as f:
                    output_key_locations = pickle.load(f)
            except:
                print('analyze social gaze with Anipose 3d only of '+date_tgt)
                # get social gaze information 
                output_look_ornot, output_allvectors, output_allangles = find_socialgaze_timepoint_Anipose(body_part_locs_Anipose,min_length,angle_thres,with_tubelever)
                output_key_locations = find_socialgaze_timepoint_Anipose_2(body_part_locs_Anipose,min_length,angle_thres,with_tubelever)
               
                # save data
                current_dir = data_saved_folder+'/bhv_events_Anipose/'+animal1_fixedorder[0]+animal2_fixedorder[0]
                add_date_dir = os.path.join(current_dir+'/'+date_tgt)
                if not os.path.exists(add_date_dir):
                    os.makedirs(add_date_dir)
                #
                with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_look_ornot.pkl', 'wb') as f:
                    pickle.dump(output_look_ornot, f)
                with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_allvectors.pkl', 'wb') as f:
                    pickle.dump(output_allvectors, f)
                with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_allangles.pkl', 'wb') as f:
                    pickle.dump(output_allangles, f)
                with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_key_locations.pkl', 'wb') as f:
                    pickle.dump(output_key_locations, f)
                
             
            look_at_face_or_not_Anipose = output_look_ornot['look_at_face_or_not_Anipose']
            look_at_selftube_or_not_Anipose = output_look_ornot['look_at_selftube_or_not_Anipose']
            look_at_selflever_or_not_Anipose = output_look_ornot['look_at_selflever_or_not_Anipose']
            look_at_othertube_or_not_Anipose = output_look_ornot['look_at_othertube_or_not_Anipose']
            look_at_otherlever_or_not_Anipose = output_look_ornot['look_at_otherlever_or_not_Anipose']
            # change the unit to second, and aligned to session start
            session_start_time = session_start_times[idate]
            look_at_face_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_face_or_not_Anipose['dodson'])[0],1)/fps - session_start_time
            look_at_selflever_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_selflever_or_not_Anipose['dodson'])[0],1)/fps - session_start_time
            look_at_selftube_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_selftube_or_not_Anipose['dodson'])[0],1)/fps - session_start_time 
            look_at_otherlever_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_otherlever_or_not_Anipose['dodson'])[0],1)/fps - session_start_time
            look_at_othertube_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_othertube_or_not_Anipose['dodson'])[0],1)/fps - session_start_time 

            look_at_Anipose = {"face":look_at_face_or_not_Anipose,"selflever":look_at_selflever_or_not_Anipose,
                               "selftube":look_at_selftube_or_not_Anipose,"otherlever":look_at_otherlever_or_not_Anipose,
                               "othertube":look_at_othertube_or_not_Anipose} 
            
            # find time point of behavioral events
            output_time_points_socialgaze ,output_time_points_levertube = bhv_events_timepoint_Anipose(bhv_data,look_at_Anipose)
            time_point_pull1 = output_time_points_socialgaze['time_point_pull1']
            time_point_pull2 = output_time_points_socialgaze['time_point_pull2']
            oneway_gaze1 = output_time_points_socialgaze['oneway_gaze1']
            oneway_gaze2 = output_time_points_socialgaze['oneway_gaze2']
            mutual_gaze1 = output_time_points_socialgaze['mutual_gaze1']
            mutual_gaze2 = output_time_points_socialgaze['mutual_gaze2']
            timepoint_lever1 = output_time_points_levertube['time_point_lookatlever1']   
            timepoint_lever2 = output_time_points_levertube['time_point_lookatlever2']   
            timepoint_tube1 = output_time_points_levertube['time_point_lookattube1']   
            timepoint_tube2 = output_time_points_levertube['time_point_lookattube2']   
                
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
            
            # # plot behavioral events
            if 0:
                if np.isin(animal1,animal1_fixedorder):
                    plot_bhv_events_levertube(date_tgt+merge_campair,animal1, animal2, session_start_time, totalsess_time, 
                                              time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2,
                                              timepoint_lever1,timepoint_lever2,timepoint_tube1,timepoint_tube2)
                else:
                    plot_bhv_events_levertube(date_tgt+merge_campair,animal2, animal1, session_start_time, totalsess_time, 
                                              time_point_pull2, time_point_pull1, oneway_gaze2, oneway_gaze1, mutual_gaze2, mutual_gaze1,
                                              timepoint_lever2,timepoint_lever1,timepoint_tube2,timepoint_tube1)
            #
            # save behavioral events plot
            if 0:
                current_dir = data_saved_folder+'/bhv_events_Anipose/'+animal1_fixedorder[0]+animal2_fixedorder[0]
                add_date_dir = os.path.join(current_dir+'/'+date_tgt)
                if not os.path.exists(add_date_dir):
                    os.makedirs(add_date_dir)
                plt.savefig(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/'+date_tgt+"_Anipose.pdf")
  
            #
            # # old definition
            # if np.isin(animal1,animal1_fixedorder):
            #     owgaze1_num_all_dates[idate] = np.shape(oneway_gaze1)[0]
            #     owgaze2_num_all_dates[idate] = np.shape(oneway_gaze2)[0]
            #     mtgaze1_num_all_dates[idate] = np.shape(mutual_gaze1)[0]
            #     mtgaze2_num_all_dates[idate] = np.shape(mutual_gaze2)[0]
            # else:
            #     owgaze1_num_all_dates[idate] = np.shape(oneway_gaze2)[0]
            #     owgaze2_num_all_dates[idate] = np.shape(oneway_gaze1)[0]
            #     mtgaze1_num_all_dates[idate] = np.shape(mutual_gaze2)[0]
            #     mtgaze2_num_all_dates[idate] = np.shape(mutual_gaze1)[0]
            #
            # new defnition
            # <500ms counts as one gaze, gaze number per second
            if np.isin(animal1,animal1_fixedorder):
                owgaze1_num_all_dates[idate] = np.sum(oneway_gaze1[1:]-oneway_gaze1[:-1]>=0.5)/(min_length/fps)
                owgaze2_num_all_dates[idate] = np.sum(oneway_gaze2[1:]-oneway_gaze2[:-1]>=0.5)/(min_length/fps)
                mtgaze1_num_all_dates[idate] = np.sum(mutual_gaze1[1:]-mutual_gaze1[:-1]>=0.5)/(min_length/fps)
                mtgaze2_num_all_dates[idate] = np.sum(mutual_gaze2[1:]-mutual_gaze2[:-1]>=0.5)/(min_length/fps)
            else:
                owgaze1_num_all_dates[idate] = np.sum(oneway_gaze2[1:]-oneway_gaze2[:-1]>=0.5)/(min_length/fps)
                owgaze2_num_all_dates[idate] = np.sum(oneway_gaze1[1:]-oneway_gaze1[:-1]>=0.5)/(min_length/fps)
                mtgaze1_num_all_dates[idate] = np.sum(mutual_gaze2[1:]-mutual_gaze2[:-1]>=0.5)/(min_length/fps)
                mtgaze2_num_all_dates[idate] = np.sum(mutual_gaze1[1:]-mutual_gaze1[:-1]>=0.5)/(min_length/fps)
            
            
            # plot key continuous behavioral variables
            if 1:
                filepath_cont_var = data_saved_folder+'bhv_events_continuous_variables_Anipose/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'+date_tgt+'/'
                if not os.path.exists(filepath_cont_var):
                    os.makedirs(filepath_cont_var)
                
                savefig = 0
                pull_trig_events_summary, pull_trig_events_succtrial_summary, pull_trig_events_errtrial_summary = plot_continuous_bhv_var(filepath_cont_var+date_tgt+merge_campair,savefig, animal1, animal2, 
                                        session_start_time, min_length,succpulls_ornot,
                                        time_point_pull1, time_point_pull2,animalnames_videotrack,
                                        output_look_ornot, output_allvectors, output_allangles,output_key_locations)
                pull_trig_events_all_dates[date_tgt] = pull_trig_events_summary
                pull_trig_events_succtrials_all_dates[date_tgt] = pull_trig_events_succtrial_summary
                pull_trig_events_errtrials_all_dates[date_tgt] = pull_trig_events_errtrial_summary
                    
                
            # analyze the events interval, especially for the pull to other and other to pull interval
            # could be used for define time bin for DBN
            if 1:
                _,_,_,pullTOother_itv, otherTOpull_itv = bhv_events_interval(totalsess_time, session_start_time, time_point_pull1, time_point_pull2, 
                                                                             oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2)
                #
                pull_other_pool_itv = np.concatenate((pullTOother_itv,otherTOpull_itv))
                bhv_intv_all_dates[date_tgt] = {'pull_to_other':pullTOother_itv,'other_to_pull':otherTOpull_itv,
                                'pull_other_pooled': pull_other_pool_itv}
        
        
            
            # plot the tracking demo video
            if 0:      
                video_file = video_file_dir+'/'+date_tgt+'_'+animal1_filename+'_'+animal2_filename+'_anipose_bhv_demo.mp4'
                tracking_video_Anipose_events_demo(body_part_locs_Anipose,output_look_ornot,output_allvectors,output_allangles,
                                                   time_point_pull1,time_point_pull2,animalnames_videotrack,bodypartnames_videotrack,
                                                   date_tgt,animal1_filename,animal2_filename,animal1,animal2,
                                                   session_start_time,fps,nframes,video_file,withboxCorner)

                
                
            # get the neural data
            
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
            FR_kernel = 1/30 # in the unit of second # 1/30 same resolution as the video recording
            # FR_kernel is sent to to be this if want to explore it's relationship with continuous trackng data
            
            totalsess_time_forFR = np.floor(np.shape(output_look_ornot['look_at_face_or_not_Anipose']['dodson'])[0]/30)  # to match the total time of the video recording
            # _,FR_timepoint_allclusters,FR_allclusters,FR_zscore_allclusters = spike_analysis_FR_calculation(fps, FR_kernel, totalsess_time_forFR,
            #                                                                                                spike_clusters_data, spike_time_data)
            _,FR_timepoint_allch,FR_allch,FR_zscore_allch = spike_analysis_FR_calculation(fps,FR_kernel,totalsess_time_forFR,
                                                                                         spike_channels_data, spike_time_data)
            #
            # Run PCA analysis
            FR_zscore_allch_np_merged = np.array(pd.DataFrame(FR_zscore_allch).T)
            FR_zscore_allch_np_merged = FR_zscore_allch_np_merged[~np.isnan(np.sum(FR_zscore_allch_np_merged,axis=1)),:]
            # # run PCA on the entire session
            pca = PCA(n_components=3)
            FR_zscore_allch_PCs = pca.fit_transform(FR_zscore_allch_np_merged.T)
            #
            # # run PCA around the -PCAtwins to PCAtwins for each behavioral events
            PCAtwins = 5 # 5 second
            gaze_thresold = 0.5 # min length threshold to define if a gaze is real gaze or noise, in the unit of second 
            
            if 0:
                savefigs = 0 
                PCA_around_bhv_events(FR_timepoint_allch,FR_zscore_allch_np_merged,time_point_pull1,time_point_pull2,time_point_pulls_succfail, 
                              oneway_gaze1,oneway_gaze2,mutual_gaze1,mutual_gaze2,gaze_thresold,totalsess_time_forFR,PCAtwins,fps,
                              savefigs,data_saved_folder,cameraID,animal1_filename,animal2_filename,date_tgt)

                    
            # plot and PC1,2,3 traces and the continous variable traces
            if 0:
                print('plot the PC1,2,3 and continuous variable traces')
                
                savefig = 1
                save_path = data_saved_folder+"fig_for_neural_and_BasicBhvAna_and_ContVariAna_Aniposelib3d_allsessions_basicEvents/"+merge_campair+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                #
                mintime_forplot = 50 # in the unit of second
                maxtime_forplot = 550 # in the unit of second
                plot_continuous_bhv_var_and_neuralFR(date_tgt,savefig,save_path, animal1, animal2, session_start_time,
                                                     min_length, mintime_forplot, maxtime_forplot, succpulls_ornot, 
                                                     time_point_pull1, time_point_pull2, animalnames_videotrack, 
                                                     output_look_ornot, output_allvectors, output_allangles, output_key_locations, 
                                                     FR_timepoint_allch, FR_zscore_allch_PCs)
                
           
            # do the spike triggered average of different bhv variables, both continuous tracking result, or the pulling actions
            # the goal is to get a sense for glm
            if 1: 
                print('plot spike triggered bhv variables')
                
                savefig = 1
                save_path = data_saved_folder+"fig_for_neural_and_BasicBhvAna_and_ContVariAna_Aniposelib3d_allsessions_basicEvents/"+merge_campair+"/"+animal1_filename+"_"+animal2_filename+"/"+date_tgt
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                #
                do_shuffle = 0
                #
                spike_trig_average_all =  plot_spike_triggered_continuous_bhv(date_tgt,savefig,save_path, animal1, animal2, session_start_time, min_length, 
                                                                              time_point_pull1, time_point_pull2, time_point_pulls_succfail,
                                                                              oneway_gaze1,oneway_gaze2,mutual_gaze1,mutual_gaze2,gaze_thresold,animalnames_videotrack, 
                                                                              output_look_ornot, output_allvectors, output_allangles, output_key_locations, 
                                                                              spike_clusters_data, spike_time_data,spike_channels_data,do_shuffle)
                
                spike_trig_events_all_dates[date_tgt] = spike_trig_average_all
            
                
                
                
                
    # save data
    if 1:
        data_saved_subfolder = data_saved_folder+'data_saved_combinedsessions_Anipose'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)

        with open(data_saved_subfolder+'/owgaze1_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(owgaze1_num_all_dates, f)
        with open(data_saved_subfolder+'/owgaze2_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(owgaze2_num_all_dates, f)
        with open(data_saved_subfolder+'/mtgaze1_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(mtgaze1_num_all_dates, f)
        with open(data_saved_subfolder+'/mtgaze2_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(mtgaze2_num_all_dates, f)
        with open(data_saved_subfolder+'/pull1_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(pull1_num_all_dates, f)
        with open(data_saved_subfolder+'/pull2_num_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(pull2_num_all_dates, f)

        with open(data_saved_subfolder+'/tasktypes_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(tasktypes_all_dates, f)
        with open(data_saved_subfolder+'/coopthres_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(coopthres_all_dates, f)
        with open(data_saved_subfolder+'/succ_rate_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(succ_rate_all_dates, f)
        with open(data_saved_subfolder+'/interpullintv_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(interpullintv_all_dates, f)
        with open(data_saved_subfolder+'/trialnum_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(trialnum_all_dates, f)
        with open(data_saved_subfolder+'/bhv_intv_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(bhv_intv_all_dates, f)
            
        with open(data_saved_subfolder+'/pull_trig_events_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(pull_trig_events_all_dates, f)
        with open(data_saved_subfolder+'/pull_trig_events_succtrials_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(pull_trig_events_succtrials_all_dates, f)
        with open(data_saved_subfolder+'/pull_trig_events_errtrials_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(pull_trig_events_errtrials_all_dates, f)
            
        with open(data_saved_subfolder+'/spike_trig_events_all_dates_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
            pickle.dump(spike_trig_events_all_dates, f)    
            


# In[ ]:





# In[ ]:





# ### analyze the spike triggered behavioral variables across all dates
# ### plot the tsne or PCA clusters

# In[21]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

doPCA = 0
doTSNE = 1

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
fig1.set_figheight(8*nact_animals)
fig1.set_figwidth(8*nbhv_names)

# plot all units but separate different days
fig2, axs2 = plt.subplots(nact_animals,nbhv_names)
fig2.set_figheight(8*nact_animals)
fig2.set_figwidth(8*nbhv_names)

# plot all units but seprate different channels
fig3, axs3 = plt.subplots(nact_animals,nbhv_names)
fig3.set_figheight(8*nact_animals)
fig3.set_figwidth(8*nbhv_names)

# plot all units but separate different conditions
fig4, axs4 = plt.subplots(nact_animals,nbhv_names)
fig4.set_figheight(8*nact_animals)
fig4.set_figwidth(8*nbhv_names)

# spike triggered average for different task conditions
# to be save, prepare for five conditions
fig6, axs6 = plt.subplots(nact_animals*5,nbhv_names)
fig6.set_figheight(8*nact_animals*5)
fig6.set_figwidth(8*nbhv_names)

# plot all units but separate different k-mean cluster
fig5, axs5 = plt.subplots(nact_animals,nbhv_names)
fig5.set_figheight(8*nact_animals)
fig5.set_figwidth(8*nbhv_names)

# spike triggered average for different k-mean cluster
# to be save, prepare for 14 clusters
fig7, axs7 = plt.subplots(nact_animals*14,nbhv_names)
fig7.set_figheight(8*nact_animals*14)
fig7.set_figwidth(8*nbhv_names)

#
for ianimal in np.arange(0,nact_animals,1):
    
    act_animal = act_animals_all[ianimal]
    
    for ibhvname in np.arange(0,nbhv_names,1):
        
        bhv_name = bhv_names_all[ibhvname]
        
        ind = (spike_trig_events_all_dates_df['act_animal']==act_animal)&(spike_trig_events_all_dates_df['bhv_name']==bhv_name)
        
        spike_trig_events_tgt = np.vstack(list(spike_trig_events_all_dates_df[ind]['st_average']))
        
        
        # k means clustering
        # run clustering on the 15 dimension PC space (for doPCA), or the whole dataset (for doTSNE)
        pca = PCA(n_components=15)
        spike_trig_events_pca = pca.fit_transform(spike_trig_events_tgt)
        tsne = TSNE(n_components=3, random_state=0)
        spike_trig_events_tsne = tsne.fit_transform(spike_trig_events_tgt)
        #
        range_n_clusters = np.arange(2,15,1)
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
        axs1[ianimal,ibhvname].set_title(act_animal+'\n'+bhv_name)
        
        
        # plot all units, but seprate different dates
        dates_forplot = np.unique(spike_trig_events_all_dates_df[ind]['dates'])
        for idate_forplot in dates_forplot:
            ind_idate = list(spike_trig_events_all_dates_df[ind]['dates']==idate_forplot)
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
        axs2[ianimal,ibhvname].set_title(act_animal+'\n'+bhv_name)
        axs2[ianimal,ibhvname].legend()
        
        
        # plot all units, but seprate different channels
        chs_forplot = np.unique(spike_trig_events_all_dates_df[ind]['channelID'])
        for ich_forplot in chs_forplot:
            ind_ich = list(spike_trig_events_all_dates_df[ind]['channelID']==ich_forplot)
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
        axs3[ianimal,ibhvname].set_title(act_animal+'\n'+bhv_name)
        axs3[ianimal,ibhvname].legend()
        
        
        # plot all units, but seprate different task conditions
        cons_forplot = np.unique(spike_trig_events_all_dates_df[ind]['condition'])
        for icon_forplot in cons_forplot:
            ind_icon = list(spike_trig_events_all_dates_df[ind]['condition']==icon_forplot)
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
        axs4[ianimal,ibhvname].set_title(act_animal+'\n'+bhv_name)
        axs4[ianimal,ibhvname].legend()
    
        # plot the mean spike trigger average trace across neurons in each condition
        trig_twins = [-6,6] # the time window to examine the spike triggered average, in the unit of s
        xxx_forplot = np.arange(trig_twins[0]*fps,trig_twins[1]*fps,1)
        #
        cons_forplot = np.unique(spike_trig_events_all_dates_df[ind]['condition'])
        icon_ind = 0
        for icon_forplot in cons_forplot:
            ind_icon = list(spike_trig_events_all_dates_df[ind]['condition']==icon_forplot)
            #
            mean_trig_trace_icon = np.nanmean(spike_trig_events_tgt[ind_icon,:],axis=0)
            std_trig_trace_icon = np.nanstd(spike_trig_events_tgt[ind_icon,:],axis=0)
            sem_trig_trace_icon = np.nanstd(spike_trig_events_tgt[ind_icon,:],axis=0)/np.sqrt(np.shape(spike_trig_events_tgt[ind_icon,:])[0])
            itv95_trig_trace_icon = 1.96*sem_trig_trace_icon
            #
            axs6[ianimal*5+icon_ind,ibhvname].errorbar(xxx_forplot,mean_trig_trace_icon,yerr=itv95_trig_trace_icon,
                                                       color='#E0E0E0',ecolor='#EEEEEE',label=icon_forplot)
            axs6[ianimal*5+icon_ind,ibhvname].plot([0,0],[np.nanmin(mean_trig_trace_icon-itv95_trig_trace_icon),
                                                          np.nanmax(mean_trig_trace_icon+itv95_trig_trace_icon)],'--k')
            axs6[ianimal*5+icon_ind,ibhvname].set_xlabel('time (s)')
            axs6[ianimal*5+icon_ind,ibhvname].set_xticks(np.arange(trig_twins[0]*fps,trig_twins[1]*fps,60))
            axs6[ianimal*5+icon_ind,ibhvname].set_xticklabels(list(map(str,np.arange(trig_twins[0],trig_twins[1],2))))
            axs6[ianimal*5+icon_ind,ibhvname].set_title(act_animal+'\n'+bhv_name)
            axs6[ianimal*5+icon_ind,ibhvname].legend()
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
        axs5[ianimal,ibhvname].set_title(act_animal+'\n'+bhv_name)
        axs5[ianimal,ibhvname].legend()
        
        # plot the mean spike trigger average trace across neurons in each cluster
        trig_twins = [-6,6] # the time window to examine the spike triggered average, in the unit of s
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
            axs7[ianimal*14+ikm_forplot,ibhvname].set_title(act_animal+'\n'+bhv_name)
            axs7[ianimal*14+ikm_forplot,ibhvname].legend()
    
    
savefig = 1
if savefig:
    figsavefolder = data_saved_folder+"fig_for_neural_and_BasicBhvAna_and_ContVariAna_Aniposelib3d_allsessions_basicEvents/"+merge_campairs[0]+"/"+animal1_filename+"_"+animal2_filename+"/"
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    if doTSNE:
        fig1.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates.pdf')
        fig2.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_separated_dates.pdf')
        fig3.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_separated_channels.pdf')
        fig4.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_separated_conditions.pdf')
        fig5.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_separated_kmeanclusters.pdf')
        fig6.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_sttraces_for_conditions.pdf')        
        fig7.savefig(figsavefolder+'spike_triggered_bhv_variables_tsne_clusters_all_dates_sttraces_for_kmeanclusters.pdf')

    if doPCA:
        fig1.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates.pdf')
        fig2.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_separated_dates.pdf')
        fig3.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_separated_channels.pdf')
        fig4.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_separated_conditions.pdf')
        fig5.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_separated_kmeanclusters.pdf')
        fig6.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_sttraces_for_conditions.pdf')                           
        fig7.savefig(figsavefolder+'spike_triggered_bhv_variables_pca_clusters_all_dates_sttraces_for_kmeanclusters.pdf')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


spike_trig_events_all_dates_df['st_average'][0]


# In[ ]:


mean_trig_trace_ikm = np.nanmean(spike_trig_events_tgt[ind_ikm,:],axis=1)
std_trig_trace_ikm = np.nanstd(spike_trig_events_tgt[ind_ikm,:],axis=1)
sem_trig_trace_ikm = np.nanstd(spike_trig_events_tgt[ind_ikm,:],axis=1)/np.sqrt(np.shape(spike_trig_events_tgt[ind_ikm,:])[1])
itv95_trig_trace_ikm = 1.96*sem_trig_trace_ikm


# In[ ]:


itv95_trig_trace_ikm


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





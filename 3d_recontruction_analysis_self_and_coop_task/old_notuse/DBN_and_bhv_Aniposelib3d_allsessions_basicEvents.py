#!/usr/bin/env python
# coding: utf-8

# ### In this script, DBN is run on each session, regardless of the conditions

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import scipy
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


# ### function - align the two cameras

# In[3]:


from ana_functions.camera_align import camera_align       


# ### function - merge the two pairs of cameras

# In[4]:


from ana_functions.camera_merge import camera_merge


# ### function - find social gaze time point

# In[5]:


from ana_functions.find_socialgaze_timepoint import find_socialgaze_timepoint
from ana_functions.find_socialgaze_timepoint_Anipose import find_socialgaze_timepoint_Anipose
from ana_functions.find_socialgaze_timepoint_Anipose_2 import find_socialgaze_timepoint_Anipose_2


# ### function - define time point of behavioral events

# In[6]:


from ana_functions.bhv_events_timepoint import bhv_events_timepoint
from ana_functions.bhv_events_timepoint_Anipose import bhv_events_timepoint_Anipose


# ### function - plot behavioral events

# In[7]:


from ana_functions.plot_bhv_events import plot_bhv_events
from ana_functions.plot_bhv_events_levertube import plot_bhv_events_levertube
from ana_functions.tracking_video_Anipose_events_demo import tracking_video_Anipose_events_demo
from ana_functions.plot_continuous_bhv_var import plot_continuous_bhv_var
from ana_functions.draw_self_loop import draw_self_loop
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection


# ### function - interval between all behavioral events

# In[8]:


from ana_functions.bhv_events_interval import bhv_events_interval


# ### function - train the dynamic bayesian network

# In[9]:


from ana_functions.train_DBN import train_DBN


# ### function - train the dynamic bayesian network - Alec's methods

# In[10]:


from ana_functions.train_DBN_alec import train_DBN_alec
from ana_functions.train_DBN_alec import train_DBN_alec_create_df_only
from ana_functions.train_DBN_alec import train_DBN_alec_training_only
from ana_functions.train_DBN_alec import graph_to_matrix
from ana_functions.train_DBN_alec import get_weighted_dags
from ana_functions.train_DBN_alec import get_significant_edges
from ana_functions.train_DBN_alec import threshold_edges
from ana_functions.EfficientTimeShuffling import EfficientShuffle
from ana_functions.AicScore import AicScore


# ### methods used by Alec - separate into different "trials"

# In[11]:


from ana_functions.train_DBN_alec import train_DBN_alec_eachtrial


# ## Analyze each session

# ### prepare the basic behavioral data (especially the time stamps for each bhv events)

# In[12]:


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

# frame number of the demo video
nframes = 1*30

# re-analyze the video or not
reanalyze_video = 0
redo_anystep = 0

# only analyze the best (five) sessions for each conditions
do_bestsession = 1
if do_bestsession:
    savefile_sufix = '_bestsessions'
else:
    savefile_sufix = ''
    
# all the videos (no misaligned ones)
# aligned with the audio
# get the session start time from "videosound_bhv_sync.py/.ipynb"
# currently the session_start_time will be manually typed in. It can be updated after a better method is used

# dodson scorch
if 0:
    if not do_bestsession:
        dates_list = [
                      "20220909","20220912","20220915","20220920","20220922","20220923","20221010",
                      "20221011","20221013","20221014","20221015","20221017","20230215",     
                      "20221018","20221019","20221020","20221021","20221022","20221026","20221028","20221030",
                      "20221107","20221108","20221109","20221110","20221111","20221114","20221115","20221116",
                      "20221117","20221118","20221121","20221122","20221123","20221125","20221128","20221129",              
                      "20221205","20221206","20221209","20221212","20221214","20221216","20221219","20221220",
                      "20221221","20230208","20230209","20230213","20230214","20230111","20230112","20230201",

                     ]
        session_start_times = [ 
                                 6.50, 18.10, 0,      33.03, 549.0, 116.80, 6.50,
                                 2.80, 27.80, 272.50, 27.90, 27.00,  33.00,
                                28.70, 45.30, 21.10,  27.10, 51.90,  21.00, 30.80, 17.50,                      
                                15.70,  2.65, 27.30,   0.00,  0.00,  71.80,  0.00,  0.00, 
                                75.50, 20.20,  0.00,  24.20, 36.70,  26.40, 22.50, 28.50,                       
                                 0.00,  0.00, 21.70,  84.70, 17.00,  19.80, 23.50, 25.20,  
                                 0.00,  0.00,  0.00,   0.00,  0.00, 130.00, 14.20, 24.20, 
                              ] # in second
    elif do_bestsession:
        # pick only five sessions for each conditions
        dates_list = [
                      "20220912","20220915","20220920","20221010","20230208",
                      "20221011","20221013","20221015","20221017",
                      "20221022","20221026","20221028","20221030","20230209",
                      "20221125","20221128","20221129","20230214","20230215",                  
                      "20221205","20221206","20221209","20221214","20230112",
                     ]
        session_start_times = [ 
                                18.10,  0.00, 33.03,  6.50,  0.00, 
                                 2.80, 27.80, 27.90, 27.00,  
                                51.90, 21.00, 30.80, 17.50,  0.00,                    
                                26.40, 22.50, 28.50,  0.00, 33.00,                     
                                 0.00,  0.00, 21.70, 17.00, 14.20, 
                              ] # in second
    
    animal1_fixedorder = ['dodson']
    animal2_fixedorder = ['scorch']

    animal1_filename = "Dodson"
    animal2_filename = "Scorch"
    
# eddie sparkle
if 1:
    if not do_bestsession:
        dates_list = [
                      "20221122","20221125","20221128","20221129","20221130","20221202","20221206",
                      "20221207","20221208","20221209","20230126","20230127","20230130","20230201","20230203-1",
                      "20230206","20230207","20230208-1","20230209","20230222","20230223-1","20230227-1",
                      "20230228-1","20230302-1","20230307-2","20230313","20230315","20230316","20230317",
                      "20230321","20230322","20230324","20230327","20230328",
                      "20230330","20230331","20230403","20230404","20230405","20230406","20230407",
                      
                   ]
        session_start_times = [ 
                                 8.00,38.00,1.00,3.00,5.00,9.50,1.00,
                                 4.50,4.50,5.00,38.00,166.00,4.20,3.80,3.60,
                                 7.50,9.00,7.50,8.50,14.50,7.80,8.00,7.50,
                                 8.00,8.00,4.00,123.00,14.00,8.80,
                                 7.00,7.50,5.50,11.00,9.00,
                                 17.00,4.50,9.30,25.50,20.40,21.30,24.80,
                                 
                              ] # in second
    elif do_bestsession:   
        dates_list = [
                      "20221122",  "20221125",  
                      "20221202",  "20221206",  "20230126",  "20230130",  "20230201",
                      "20230207",  "20230208-1","20230209",  "20230222",  "20230223-1",
                      "20230227-1","20230228-1","20230302-1","20230307-2","20230313",
                      "20230321",  "20230322",  "20230324",  "20230327",  "20230328",
                      "20230331",  "20230403",  "20230404",  "20230405",  "20230406"
                   ]
        session_start_times = [ 
                                  8.00,  38.00, 
                                  9.50,   1.00, 38.00,  4.20,  3.80,
                                  9.00,   7.50,  8.50, 14.50,  7.80,
                                  8.00,   7.50,  8.00,  8.00,  4.00,
                                  7.00,   7.50,  5.50, 11.00,  9.00,
                                  4.50,   9.30, 25.50, 20.40, 21.30,
                              ] # in second
    
    animal1_fixedorder = ['eddie']
    animal2_fixedorder = ['sparkle']

    animal1_filename = "Eddie"
    animal2_filename = "Sparkle"
    
# ginger kanga
if 0:
    if not do_bestsession:
        dates_list = [
                      "20230209","20230213","20230214","20230216","20230222","20230223","20230228","20230302",
                      "20230303","20230307","20230314","20230315","20230316","20230317"         
                   ]
        session_start_times = [ 
                                 0.00,  0.00,  0.00, 48.00, 26.20, 18.00, 23.00, 28.50,
                                34.00, 25.50, 25.50, 31.50, 28.00, 30.50
                              ] # in second 
    elif do_bestsession:   
        dates_list = [
                      "20230213","20230214","20230216",
                      "20230228","20230302","20230303",
                      "20230307",          
                      "20230314","20230315","20230316","20230317",
                      "20230301","20230320","20230321","20230322",
                      "20230323","20230412","20230413","20230517","20230614","20230615"
                   ]
        session_start_times = [ 
                                 0.00,  0.00, 48.00, 
                                23.00, 28.50, 34.00, 
                                25.50, 
                                25.50, 31.50, 28.00, 30.50,
                                33.50, 22.20, 50.00,  0.00, 
                                33.00, 18.20, 22.80, 31.00, 24.00, 21.00
                              ] # in second 
    
    animal1_fixedorder = ['ginger']
    animal2_fixedorder = ['kanga']

    animal1_filename = "Ginger"
    animal2_filename = "Kanga"
    
#    
#dates_list = ["20221128"]
#session_start_times = [1.00] # in second
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

owgaze1_num_all_dates = np.zeros((ndates,1))
owgaze2_num_all_dates = np.zeros((ndates,1))
mtgaze1_num_all_dates = np.zeros((ndates,1))
mtgaze2_num_all_dates = np.zeros((ndates,1))
pull1_num_all_dates = np.zeros((ndates,1))
pull2_num_all_dates = np.zeros((ndates,1))

bhv_intv_all_dates = dict.fromkeys(dates_list, [])


# video tracking results info
animalnames_videotrack = ['dodson','scorch'] # does not really mean dodson and scorch, instead, indicate animal1 and animal2
bodypartnames_videotrack = ['rightTuft','whiteBlaze','leftTuft','rightEye','leftEye','mouth']

# where to save the summarizing data
data_saved_folder = '/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/3d_recontruction_analysis_self_and_coop_task_data_saved/'

# where to save the demo video
withboxCorner = 1
video_file_dir = data_saved_folder+'/example_videos_Anipose_bhv_demo/'+animal1_filename+'_'+animal2_filename
if not os.path.exists(video_file_dir):
    os.makedirs(video_file_dir)

    


# In[13]:


# basic behavior analysis (define time stamps for each bhv events, etc)
# NOTE: THIS STEP will save the data to the combinedsession_Anipose folder, since they are the same
try:
    if redo_anystep:
        dummy
        
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

except:

    for idate in np.arange(0,ndates,1):
        date_tgt = dates_list[idate]
        session_start_time = session_start_times[idate]

        # folder path
        camera12_analyzed_path = "/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera12/"
        camera23_analyzed_path = "/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera23/"
        Anipose_analyzed_path = "/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera12/anipose_cam123_3d_h5_files/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"

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
                #
                trial_record = pd.read_json(trial_record_json[0])
                bhv_data = pd.read_json(bhv_data_json[0])
                session_info = pd.read_json(session_info_json[0])
            except:
                bhv_data_path = "/home/ws523/marmoset_tracking_bhv_data_from_task_code/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
                trial_record_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_TrialRecord_" + "*.json")
                bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_bhv_data_" + "*.json")
                session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_session_info_" + "*.json")
                #
                trial_record = pd.read_json(trial_record_json[0])
                bhv_data = pd.read_json(bhv_data_json[0])
                session_info = pd.read_json(session_info_json[0])

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

            pull1_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==1) 
            pull2_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==2)

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
            # change the unit to second
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
            owgaze1_num_all_dates[idate] = np.shape(oneway_gaze1)[0]
            owgaze2_num_all_dates[idate] = np.shape(oneway_gaze2)[0]
            mtgaze1_num_all_dates[idate] = np.shape(mutual_gaze1)[0]
            mtgaze2_num_all_dates[idate] = np.shape(mutual_gaze2)[0]

            
            # plot key continuous behavioral variables
            if 0:
                filepath_cont_var = data_saved_folder+'bhv_events_continuous_variables_Anipose/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'+date_tgt+'/'
                if not os.path.exists(filepath_cont_var):
                    os.makedirs(filepath_cont_var)
                plot_continuous_bhv_var(filepath_cont_var+date_tgt+merge_campair,animal1, animal2, session_start_time, min_length, 
                                        time_point_pull1, time_point_pull2,animalnames_videotrack,
                                        output_look_ornot, output_allvectors, output_allangles,output_key_locations)
                
                
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

    # save data
    if 1:
        data_saved_subfolder = data_saved_folder+'data_saved_combinedsessions_Anipose'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
                
        # with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pkl', 'wb') as f:
        #     pickle.dump(DBN_input_data_alltypes, f)

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
            
            


# #### redefine the tasktype and cooperation threshold to merge them together

# In[14]:


# 100: self; 3: 3s coop; 2: 2s coop; 1.5: 1.5s coop; 1: 1s coop; -1: no-vision

tasktypes_all_dates[tasktypes_all_dates==5] = -1 # change the task type code for no-vision
coopthres_forsort = (tasktypes_all_dates-1)*coopthres_all_dates/2
coopthres_forsort[coopthres_forsort==0] = 100 # get the cooperation threshold for sorting


# ### plot behavioral events interval to get a sense about time bin
# #### only focus on pull_to_other_bhv_interval and other_bhv_to_pull_interval

# In[15]:


fig, ax1 = plt.subplots(figsize=(10, 5))
#
# sort the data based on task type and dates
sorting_df = pd.DataFrame({'dates': dates_list, 'coopthres': coopthres_forsort.ravel()}, columns=['dates', 'coopthres'])
sorting_df = sorting_df.sort_values(by=['coopthres','dates'], ascending = [False, True])
dates_list_sorted = np.array(dates_list)[sorting_df.index]
ndates_sorted = np.shape(dates_list_sorted)[0]

pull_other_intv_forplots = {}
pull_other_intv_mean = np.zeros((1,ndates_sorted))[0]
pull_other_intv_ii = []
for ii in np.arange(0,ndates_sorted,1):
    pull_other_intv_ii = pd.Series(bhv_intv_all_dates[dates_list_sorted[ii]]['pull_other_pooled'])
    # remove the interval that is too large
    pull_other_intv_ii[pull_other_intv_ii>(np.nanmean(pull_other_intv_ii)+2*np.nanstd(pull_other_intv_ii))]= np.nan    
    # pull_other_intv_ii[pull_other_intv_ii>10]= np.nan
    pull_other_intv_forplots[ii] = pull_other_intv_ii
    pull_other_intv_mean[ii] = np.nanmean(pull_other_intv_ii)
    
    
#
pull_other_intv_forplots = pd.DataFrame(pull_other_intv_forplots)

#
# plot
pull_other_intv_forplots.plot(kind = 'box',ax=ax1, positions=np.arange(0,ndates_sorted,1))
# plt.boxplot(pull_other_intv_forplots)
plt.plot(np.arange(0,ndates_sorted,1),pull_other_intv_mean,'r*',markersize=10)
#
ax1.set_ylabel("bhv event interval(around pulls)",fontsize=13)
ax1.set_ylim([-2,16])
#
plt.xticks(np.arange(0,ndates_sorted,1),dates_list_sorted, rotation=90,fontsize=10)
plt.yticks(fontsize=10)
#
tasktypes = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
taskswitches = np.where(np.array(sorting_df['coopthres'])[1:]-np.array(sorting_df['coopthres'])[:-1]!=0)[0]+0.5
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.plot([taskswitch,taskswitch],[-2,15],'k--')
taskswitches = np.concatenate(([0],taskswitches))
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.text(taskswitch+0.25,-1,tasktypes[itaskswitch],fontsize=10)
ax1.text(taskswitch-5,15,'mean Inteval = '+str(np.nanmean(pull_other_intv_forplots)),fontsize=10)
    
print(pull_other_intv_mean)
print(np.nanmean(pull_other_intv_forplots))

savefigs = 1
if savefigs:
    figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    plt.savefig(figsavefolder+"bhvInterval_hist_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'.jpg')
    


# ### plot some other basis behavioral measures
# #### successful rate

# In[16]:


fig, ax1 = plt.subplots(figsize=(10, 5))
#
# sort the data based on task type and dates
sorting_df = pd.DataFrame({'dates': dates_list, 'coopthres': coopthres_forsort.ravel()}, columns=['dates', 'coopthres'])
sorting_df = sorting_df.sort_values(by=['coopthres','dates'], ascending = [False, True])
dates_list_sorted = np.array(dates_list)[sorting_df.index]
ndates_sorted = np.shape(dates_list_sorted)[0]


ax1.plot(np.arange(0,ndates_sorted,1),succ_rate_all_dates[sorting_df.index],'o',markersize=10)
#
ax1.set_ylabel("successful rate",fontsize=13)
ax1.set_ylim([-0.1,1.1])
ax1.set_xlim([-0.5,ndates_sorted-0.5])
#
plt.xticks(np.arange(0,ndates_sorted,1),dates_list_sorted, rotation=90,fontsize=10)
plt.yticks(fontsize=10)
#
tasktypes = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
taskswitches = np.where(np.array(sorting_df['coopthres'])[1:]-np.array(sorting_df['coopthres'])[:-1]!=0)[0]+0.5
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.plot([taskswitch,taskswitch],[-0.1,1.1],'k--')
taskswitches = np.concatenate(([0],taskswitches))
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.text(taskswitch+0.25,-0.05,tasktypes[itaskswitch],fontsize=10)
    
savefigs = 1
if savefigs:
    figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    plt.savefig(figsavefolder+"successfulrate_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'.jpg')


# #### animal pull numbers

# In[17]:


fig, ax1 = plt.subplots(figsize=(10, 5))
#
# sort the data based on task type and dates
sorting_df = pd.DataFrame({'dates': dates_list, 'coopthres': coopthres_forsort.ravel()}, columns=['dates', 'coopthres'])
sorting_df = sorting_df.sort_values(by=['coopthres','dates'], ascending = [False, True])
dates_list_sorted = np.array(dates_list)[sorting_df.index]
ndates_sorted = np.shape(dates_list_sorted)[0]

pullmean_num_all_dates = (pull1_num_all_dates+pull2_num_all_dates)/2
ax1.plot(np.arange(0,ndates_sorted,1),pull1_num_all_dates[sorting_df.index],'bv',markersize=5,label='animal1 pull #')
ax1.plot(np.arange(0,ndates_sorted,1),pull2_num_all_dates[sorting_df.index],'rv',markersize=5,label='animal2 pull #')
ax1.plot(np.arange(0,ndates_sorted,1),pullmean_num_all_dates[sorting_df.index],'kv',markersize=8,label='mean pull #')
ax1.legend()


#
ax1.set_ylabel("pull numbers",fontsize=13)
ax1.set_ylim([-20,240])
ax1.set_xlim([-0.5,ndates_sorted-0.5])

#
plt.xticks(np.arange(0,ndates_sorted,1),dates_list_sorted, rotation=90,fontsize=10)
plt.yticks(fontsize=10)
#
tasktypes = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
taskswitches = np.where(np.array(sorting_df['coopthres'])[1:]-np.array(sorting_df['coopthres'])[:-1]!=0)[0]+0.5
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.plot([taskswitch,taskswitch],[-20,240],'k--')
taskswitches = np.concatenate(([0],taskswitches))
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.text(taskswitch+0.25,-10,tasktypes[itaskswitch],fontsize=10)
    
savefigs = 1
if savefigs:
    figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    plt.savefig(figsavefolder+"pullnumbers_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'.jpg')


# #### gaze number

# In[18]:


gaze1_num_all_dates = owgaze1_num_all_dates + mtgaze1_num_all_dates
gaze2_num_all_dates = owgaze2_num_all_dates + mtgaze2_num_all_dates
gazemean_num_all_dates = (gaze1_num_all_dates+gaze2_num_all_dates)/2

print(np.nanmax(gaze1_num_all_dates))
print(np.nanmax(gaze2_num_all_dates))


# In[19]:


fig, ax1 = plt.subplots(figsize=(10, 5))
#
# sort the data based on task type and dates
sorting_df = pd.DataFrame({'dates': dates_list, 'coopthres': coopthres_forsort.ravel()}, columns=['dates', 'coopthres'])
sorting_df = sorting_df.sort_values(by=['coopthres','dates'], ascending = [False, True])
dates_list_sorted = np.array(dates_list)[sorting_df.index]
ndates_sorted = np.shape(dates_list_sorted)[0]



ax1.plot(np.arange(0,ndates_sorted,1),gaze1_num_all_dates[sorting_df.index],'b^',markersize=5,label='animal1 gaze #')
ax1.plot(np.arange(0,ndates_sorted,1),gaze2_num_all_dates[sorting_df.index],'r^',markersize=5,label='animal2 gaze #')
ax1.plot(np.arange(0,ndates_sorted,1),gazemean_num_all_dates[sorting_df.index],'k^',markersize=8,label='mean gaze #')
ax1.legend()


#
ax1.set_ylabel("social gaze number",fontsize=13)
ax1.set_ylim([-20,1500])
ax1.set_xlim([-0.5,ndates_sorted-0.5])

#
plt.xticks(np.arange(0,ndates_sorted,1),dates_list_sorted, rotation=90,fontsize=10)
plt.yticks(fontsize=10)
#
tasktypes = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
taskswitches = np.where(np.array(sorting_df['coopthres'])[1:]-np.array(sorting_df['coopthres'])[:-1]!=0)[0]+0.5
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.plot([taskswitch,taskswitch],[-20,1500],'k--')
taskswitches = np.concatenate(([0],taskswitches))
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.text(taskswitch+0.25,-10,tasktypes[itaskswitch],fontsize=10)
    
savefigs = 1
if savefigs:
    figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    plt.savefig(figsavefolder+"gazenumbers_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'.jpg')


# In[20]:


gaze_numbers = (owgaze1_num_all_dates+owgaze2_num_all_dates+mtgaze1_num_all_dates+mtgaze2_num_all_dates)
gaze_pull_ratios = (owgaze1_num_all_dates+owgaze2_num_all_dates+mtgaze1_num_all_dates+mtgaze2_num_all_dates)/(pull1_num_all_dates+pull2_num_all_dates)

fig, ax1 = plt.subplots(figsize=(10, 5))

grouptypes = ['self reward','3s threshold','2s threshold','1.5s threshold','1s threshold','novision']

gaze_numbers_groups = [np.transpose(gaze_numbers[np.transpose(coopthres_forsort==100)[0]])[0],
                       np.transpose(gaze_numbers[np.transpose(coopthres_forsort==3)[0]])[0],
                       np.transpose(gaze_numbers[np.transpose(coopthres_forsort==2)[0]])[0],
                       np.transpose(gaze_numbers[np.transpose(coopthres_forsort==1.5)[0]])[0],
                       np.transpose(gaze_numbers[np.transpose(coopthres_forsort==1)[0]])[0],
                       np.transpose(gaze_numbers[np.transpose(coopthres_forsort==-1)[0]])[0]]

gaze_numbers_plot = plt.boxplot(gaze_numbers_groups)

plt.xticks(np.arange(1, len(grouptypes)+1, 1), grouptypes, fontsize = 12);
ax1.set_ylim([-20,3000])
ax1.set_ylabel("average social gaze numbers",fontsize=13)

savefigs = 1
if savefigs:
    figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    plt.savefig(figsavefolder+"averaged_gazenumbers_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'.jpg')


# ### prepare the input data for DBN
# #### for each session

# In[21]:


# define DBN related summarizing variables
DBN_input_data_allsessions = dict.fromkeys(dates_list, [])

doBhvitv_timebin = 0 # 1: if use the mean bhv event interval for time bin

# DBN resolutions (make sure they are the same as in the later part of the code)
totalsess_time = 600 # total session time in s
## use the same time bins for all the sessions; Alternative, use the mean bhv event interval 
if not doBhvitv_timebin:    
    # temp_resolus = [0.5,1,1.5,2.5] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    temp_resolus = [1] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    ntemp_reses = np.shape(temp_resolus)[0]

#
mergetempRos = 0 # 1: if merge different time bin


# # train the dynamic bayesian network - Alec's model 
#   prepare the multi-session table; one time lag; multi time steps (temporal resolution) as separate files

# prepare the DBN input data
if 0:
    
    for idate in np.arange(0,ndates,1):
        date_tgt = dates_list[idate]
        session_start_time = session_start_times[idate]

        # load behavioral results
        try:
            bhv_data_path = "/home/ws523/marmoset_tracking_bhv_data_from_task_code/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
            trial_record_json = glob.glob(bhv_data_path +date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_TrialRecord_" + "*.json")
            bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_bhv_data_" + "*.json")
            session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal2_filename+"_"+animal1_filename+"_session_info_" + "*.json")
            #
            trial_record = pd.read_json(trial_record_json[0])
            bhv_data = pd.read_json(bhv_data_json[0])
            session_info = pd.read_json(session_info_json[0])
        except:
            bhv_data_path = "/home/ws523/marmoset_tracking_bhv_data_from_task_code/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
            trial_record_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_TrialRecord_" + "*.json")
            bhv_data_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_bhv_data_" + "*.json")
            session_info_json = glob.glob(bhv_data_path + date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_session_info_" + "*.json")
            #
            trial_record = pd.read_json(trial_record_json[0])
            bhv_data = pd.read_json(bhv_data_json[0])
            session_info = pd.read_json(session_info_json[0])
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
        
        # get task type and cooperation threshold
        try:
            coop_thres = session_info["pulltime_thres"][0]
            tasktype = session_info["task_type"][0]
        except:
            coop_thres = 0
            tasktype = 1

        # load behavioral event results
        print('load social gaze with Anipose 3d of '+date_tgt)
        with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_look_ornot.pkl', 'rb') as f:
            output_look_ornot = pickle.load(f)
        with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_allvectors.pkl', 'rb') as f:
            output_allvectors = pickle.load(f)
        with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_allangles.pkl', 'rb') as f:
            output_allangles = pickle.load(f)  
        with open(data_saved_folder+"bhv_events_Anipose/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+date_tgt+'/output_key_locations.pkl', 'rb') as f:
            output_key_locations = pickle.load(f)     
        look_at_face_or_not_Anipose = output_look_ornot['look_at_face_or_not_Anipose']
        look_at_selftube_or_not_Anipose = output_look_ornot['look_at_selftube_or_not_Anipose']
        look_at_selflever_or_not_Anipose = output_look_ornot['look_at_selflever_or_not_Anipose']
        look_at_othertube_or_not_Anipose = output_look_ornot['look_at_othertube_or_not_Anipose']
        look_at_otherlever_or_not_Anipose = output_look_ornot['look_at_otherlever_or_not_Anipose']
        # change the unit to second
        session_start_time = session_start_times[idate]
        look_at_face_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_face_or_not_Anipose['dodson'])[0],1)/fps - session_start_time
        look_at_selflever_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_selflever_or_not_Anipose['dodson'])[0],1)/fps - session_start_time
        look_at_selftube_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_selftube_or_not_Anipose['dodson'])[0],1)/fps - session_start_time 
        look_at_otherlever_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_otherlever_or_not_Anipose['dodson'])[0],1)/fps - session_start_time
        look_at_othertube_or_not_Anipose['time_in_second'] = np.arange(0,np.shape(look_at_othertube_or_not_Anipose['dodson'])[0],1)/fps - session_start_time 
        # 
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


        if mergetempRos:
            temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
            # use bhv event to decide temporal resolution
            #
            #low_lim,up_lim,_ = bhv_events_interval(totalsess_time, session_start_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2)
            #temp_resolus = temp_resolus = np.arange(low_lim,up_lim,0.1)

        if doBhvitv_timebin:
            pull_other_intv_ii = pd.Series(bhv_intv_all_dates[date_tgt]['pull_other_pooled'])
            # remove the interval that is too large
            pull_other_intv_ii[pull_other_intv_ii>(np.nanmean(pull_other_intv_ii)+2*np.nanstd(pull_other_intv_ii))]= np.nan    
            # pull_other_intv_ii[pull_other_intv_ii>10]= np.nan
            temp_resolus = [np.nanmean(pull_other_intv_ii)]
            
        ntemp_reses = np.shape(temp_resolus)[0]           

        # try different temporal resolutions
        for temp_resolu in temp_resolus:
            bhv_df = []

            if np.isin(animal1,animal1_fixedorder):
                bhv_df_itr = train_DBN_alec_create_df_only(totalsess_time, session_start_time, temp_resolu, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2)
            else:
                bhv_df_itr = train_DBN_alec_create_df_only(totalsess_time, session_start_time, temp_resolu, time_point_pull2, time_point_pull1, oneway_gaze2, oneway_gaze1, mutual_gaze2, mutual_gaze1)     

            if len(bhv_df)==0:
                bhv_df = bhv_df_itr
            else:
                bhv_df = pd.concat([bhv_df,bhv_df_itr])                   
                #bhv_df = bhv_df.reset_index(drop=True)        

            DBN_input_data_allsessions[date_tgt] = bhv_df
            
            
    # save data
    if 1:
        data_saved_subfolder = data_saved_folder+'data_saved_allsessions_Anipose'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
        if not mergetempRos:
            if doBhvitv_timebin:
                with open(data_saved_subfolder+'/DBN_input_data_allsessions_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_bhvItvTempReSo.pkl', 'wb') as f:
                    pickle.dump(DBN_input_data_allsessions, f)
            else:
                with open(data_saved_subfolder+'/DBN_input_data_allsessions_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'sReSo.pkl', 'wb') as f:
                    pickle.dump(DBN_input_data_allsessions, f)
        if mergetempRos:
            with open(data_saved_subfolder+'/DBN_input_data_allsessions_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_mergeTempsReSo.pkl', 'wb') as f:
                pickle.dump(DBN_input_data_allsessions, f)     



# ### run the DBN model on the combined session data set

# In[ ]:


# run DBN on the large table with merged sessions

mergetempRos = 0 # 1: merge different time bins

minmaxfullSampSize = 1 # 1: use the  min row number and max row number, or the full row for each session

moreSampSize = 0 # 1: if use more fixed sample size

num_starting_points = 100 # number of random starting points/graphs
nbootstraps = 95

try:
    #dumpy
    data_saved_subfolder = data_saved_folder+'data_saved_allsessions_Anipose'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(data_saved_subfolder):
        os.makedirs(data_saved_subfolder)
    if moreSampSize:
        with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'rb') as f:
            DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
        with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'rb') as f:
            DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
        with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'rb') as f:
            weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
        with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'rb') as f:
            weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
        with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'rb') as f:
            sig_edges_diffTempRo_diffSampSize = pickle.load(f)

    if minmaxfullSampSize:
        with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'rb') as f:
            DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
        with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'rb') as f:
            DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
        with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'rb') as f:
            weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
        with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'rb') as f:
            weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
        with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'rb') as f:
            sig_edges_diffTempRo_diffSampSize = pickle.load(f)

except:
    if moreSampSize:
        # different data (down/re)sampling numbers
        # samplingsizes = np.arange(100,3100,300)
        # samplingsizes = [100,500,1000,1500,2000,2500,3000]
        samplingsizes = np.arange(300,1100,100)
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
    
    if not doBhvitv_timebin:
        # temp_resolus = [0.5,1,1.5,2,2.5] # temporal resolution in the DBN model, eg: 0.5 means 500ms
        temp_resolus = [1] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    else:
        temp_resolus = [0] # use the bhv interval for each session, so they are different for each session
    ntemp_reses = np.shape(temp_resolus)[0]

    # try different temporal resolutions, remember to use the same settings as in the previous ones
    for temp_resolu in temp_resolus:

        data_saved_subfolder = data_saved_folder+'data_saved_allsessions_Anipose'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not mergetempRos:
            if doBhvitv_timebin:
                with open(data_saved_subfolder+'/DBN_input_data_allsessions_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_bhvItvTempReSo.pkl', 'rb') as f:
                    DBN_input_data_allsessions = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/DBN_input_data_allsessions_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                    DBN_input_data_allsessions = pickle.load(f)
        if mergetempRos:
            with open(data_saved_subfolder+'//DBN_input_data_allsessions_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_mergeTempsReSo.pkl', 'rb') as f:
                DBN_input_data_allsessions = pickle.load(f)     
                


                
        # only try three sample sizes
        #- minimal row number (require data downsample) and maximal row number (require data upsample)
        #- full row number of each session
       
        if minmaxfullSampSize:
            key_to_value_lengths = {k:len(v) for k, v in DBN_input_data_allsessions.items()}
            key_to_value_lengths_array = np.fromiter(key_to_value_lengths.values(),dtype=float)
            key_to_value_lengths_array[key_to_value_lengths_array==0]=np.nan
            min_samplesize = np.nanmin(key_to_value_lengths_array)
            min_samplesize = int(min_samplesize/100)*100
            max_samplesize = np.nanmax(key_to_value_lengths_array)
            max_samplesize = int(max_samplesize/100)*100
            samplingsizes = [min_samplesize,max_samplesize,np.nan]
            samplingsizes_name = ['min_row_number','max_row_number','full_row_number']   
            nsamplings = np.shape(samplingsizes)[0]
        
        
        # try different down/re-sampling size
        for jj in np.arange(0,nsamplings,1):
            
            isamplingsize = samplingsizes[jj]
            
            DAGs_alltypes = dict.fromkeys(dates_list, [])
            DAGs_shuffle_alltypes = dict.fromkeys(dates_list, [])
            DAGs_scores_alltypes = dict.fromkeys(dates_list, [])
            DAGs_shuffle_scores_alltypes = dict.fromkeys(dates_list, [])

            weighted_graphs_alltypes = dict.fromkeys(dates_list, [])
            weighted_graphs_shuffled_alltypes = dict.fromkeys(dates_list, [])
            sig_edges_alltypes = dict.fromkeys(dates_list, [])

            # different individual sessions
            ndates = np.shape(dates_list)[0]
            for idate in np.arange(0,ndates,1):
                date_tgt = dates_list[idate]
                
                if samplingsizes_name[jj]=='full_row_number':
                    isamplingsize = np.shape(DBN_input_data_allsessions[date_tgt])[0]

                try:
                    bhv_df_all = DBN_input_data_allsessions[date_tgt]
                    # bhv_df = bhv_df_all.sample(30*100,replace = True, random_state = round(time())) # take the subset for DBN training

                    #Anirban(Alec) shuffle, slow
                    # bhv_df_shuffle, df_shufflekeys = EfficientShuffle(bhv_df,round(time()))


                    # define DBN graph structures; make sure they are the same as in the train_DBN_alec
                    colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1"]
                    eventnames = ["pull1","pull2","owgaze1","owgaze2"]
                    nevents = np.size(eventnames)

                    all_pops = list(bhv_df_all.columns)
                    from_pops = [pop for pop in all_pops if not pop.endswith('t1')]
                    to_pops = [pop for pop in all_pops if pop.endswith('t1')]
                    causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]

                    nFromNodes = nevents
                    nToNodes = nevents

                    DAGs_randstart = np.zeros((num_starting_points, nFromNodes, nToNodes))
                    DAGs_randstart_shuffle = np.zeros((num_starting_points, nFromNodes, nToNodes))
                    score_randstart = np.zeros((num_starting_points))
                    score_randstart_shuffle = np.zeros((num_starting_points))

                    # step 1: randomize the starting point for num_starting_points times
                    for istarting_points in np.arange(0,num_starting_points,1):

                        # try different down/re-sampling size
                        bhv_df = bhv_df_all.sample(isamplingsize,replace = True, random_state = num_starting_points) # take the subset for DBN training
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

                        best_model,edges,DAGs,eventnames,from_pops,to_pops = train_DBN_alec_training_only(bhv_df,starting_graph)           
                        DAGs[0][np.isnan(DAGs[0])]=0

                        DAGs_randstart[istarting_points,:,:] = DAGs[0]
                        score_randstart[istarting_points] = aic.score(best_model)

                        # step 2: add the shffled data results
                        # shuffled bhv_df
                        best_model,edges,DAGs,eventnames,from_pops,to_pops = train_DBN_alec_training_only(bhv_df_shuffle,starting_graph)           
                        DAGs[0][np.isnan(DAGs[0])]=0

                        DAGs_randstart_shuffle[istarting_points,:,:] = DAGs[0]
                        score_randstart_shuffle[istarting_points] = aic_shuffle.score(best_model)

                    DAGs_alltypes[date_tgt] = DAGs_randstart 
                    DAGs_shuffle_alltypes[date_tgt] = DAGs_randstart_shuffle

                    DAGs_scores_alltypes[date_tgt] = score_randstart
                    DAGs_shuffle_scores_alltypes[date_tgt] = score_randstart_shuffle

                    weighted_graphs = get_weighted_dags(DAGs_alltypes[date_tgt],nbootstraps)
                    weighted_graphs_shuffled = get_weighted_dags(DAGs_shuffle_alltypes[date_tgt],nbootstraps)
                    sig_edges = get_significant_edges(weighted_graphs,weighted_graphs_shuffled)

                    weighted_graphs_alltypes[date_tgt] = weighted_graphs
                    weighted_graphs_shuffled_alltypes[date_tgt] = weighted_graphs_shuffled
                    sig_edges_alltypes[date_tgt] = sig_edges
                    
                except:
                    DAGs_alltypes[date_tgt] = [] 
                    DAGs_shuffle_alltypes[date_tgt] = []

                    DAGs_scores_alltypes[date_tgt] = []
                    DAGs_shuffle_scores_alltypes[date_tgt] = []

                    weighted_graphs_alltypes[date_tgt] = []
                    weighted_graphs_shuffled_alltypes[date_tgt] = []
                    sig_edges_alltypes[date_tgt] = []
                
            DAGscores_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = DAGs_scores_alltypes
            DAGscores_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = DAGs_shuffle_scores_alltypes

            weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_alltypes
            weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_shuffled_alltypes
            sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),samplingsizes_name[jj])] = sig_edges_alltypes

            
    # save data
    data_saved_subfolder = data_saved_folder+'data_saved_allsessions_Anipose'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(data_saved_subfolder):
        os.makedirs(data_saved_subfolder)
    if moreSampSize:  
        with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(DAGscores_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(DAGscores_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(sig_edges_diffTempRo_diffSampSize, f)

    if minmaxfullSampSize:
        with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(DAGscores_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(DAGscores_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(sig_edges_diffTempRo_diffSampSize, f)


# ### plot graphs - show the edge with arrows; show the best time bin and row number

# In[ ]:


# make sure these variables are the same as in the previous steps

if not doBhvitv_timebin:
    temp_resolus = [1] # best temporal resolution in the DBN model, eg: 0.5 means 500ms
else:
    temp_resolus = [0]
ntemp_reses = np.shape(temp_resolus)[0]
#
# best sampling size
if moreSampSize:
    samplingsizes_name = ['600']
if minmaxfullSampSize:
    samplingsizes_name = ['full_row_number']   
nsamplings = np.shape(samplingsizes_name)[0]

# make sure these variables are consistent with the train_DBN_alec.py settings
eventnames = ["pull1","pull2","gaze1","gaze2"]
#eventnames = ["M1pull","M2pull","M1gazeM2","M2gazeM1"]
eventnode_locations = [[0,1],[1,1],[0,0],[1,0]]
eventname_locations = [[-0.5,1.0],[1.2,1],[-0.6,0],[1.2,0]]
# indicate where edge starts
# for the self edge, it's the center of the self loop
nodearrow_locations = [[[0.00,1.25],[0.25,1.10],[-.10,0.75],[0.15,0.65]],
                       [[0.75,1.00],[1.00,1.25],[0.85,0.65],[1.10,0.75]],
                       [[0.00,0.25],[0.25,0.35],[0.00,-.25],[0.25,-.10]],
                       [[0.75,0.35],[1.00,0.25],[0.75,0.00],[1.00,-.25]]]
# indicate where edge goes
# for the self edge, it's the theta1 and theta2 (with fixed radius)
nodearrow_directions = [[[ -45,-180],[0.50,0.00],[0.00,-.50],[0.50,-.50]],
                        [[-.50,0.00],[ -45,-180],[-.50,-.50],[0.00,-.50]],
                        [[0.00,0.50],[0.50,0.50],[ 180,  45],[0.50,0.00]],
                        [[-.50,0.50],[0.00,0.50],[-.50,0.00],[ 180,  45]]]

nevents = np.size(eventnames)
eventnodes_color = ['b','r','y','g']
nFromNodes = nevents
nToNodes = nevents
    
savefigs = 1

# different session conditions (aka DBN groups)
DBN_group_typenames = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
DBN_group_typeIDs  =  [1,3,3,  3,3,5]
DBN_group_coopthres = [0,3,2,1.5,1,0]
DBN_group_sorting_df_coopthres = [100,3,2,1.5,1,-1]
nDBN_groups = np.shape(DBN_group_typenames)[0]

fig, axs = plt.subplots(2,nDBN_groups)
fig.set_figheight(10)
fig.set_figwidth(30)
    
# prepare the averaged graph across days/sessions in each condition

sig_avg_dags_allsessions = dict.fromkeys(dates_list, [])

for idate in np.arange(0,ndates,1):
    date_tgt = dates_list[idate]
    
    weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolus[0]), samplingsizes_name[0])][date_tgt]
    sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(str(temp_resolus[0]), samplingsizes_name[0])][date_tgt]
    # sig_avg_dags_allsessions[date_tgt] = weighted_graphs_tgt.mean(axis = 0) * sig_edges_tgt
    sig_avg_dags_allsessions[date_tgt] = weighted_graphs_tgt.mean(axis = 0)
    
    
for iDBN_group in np.arange(0,nDBN_groups,1):
    try: 
        iDBN_group_typename = DBN_group_typenames[iDBN_group] 
        iDBN_group_typeID =  DBN_group_typeIDs[iDBN_group] 
        iDBN_group_cothres = DBN_group_coopthres[iDBN_group]
        iDBN_group_sorting_df_coopthres = DBN_group_sorting_df_coopthres[iDBN_group]

        iDBN_group_date_tgt = sorting_df['dates'][sorting_df['coopthres']==iDBN_group_sorting_df_coopthres]

        sig_avg_dags = np.nanmean([sig_avg_dags_allsessions[x] for x in iDBN_group_date_tgt],axis=0)
        
        if np.shape(sig_avg_dags)[0]<=1:
            dummy
    
        # plot
        axs[0,iDBN_group].set_title(iDBN_group_typename,fontsize=15)
        axs[0,iDBN_group].set_xlim([-0.5,1.5])
        axs[0,iDBN_group].set_ylim([-0.5,1.5])
        axs[0,iDBN_group].set_xticks([])
        axs[0,iDBN_group].set_xticklabels([])
        axs[0,iDBN_group].set_yticks([])
        axs[0,iDBN_group].set_yticklabels([])
        axs[0,iDBN_group].spines['top'].set_visible(False)
        axs[0,iDBN_group].spines['right'].set_visible(False)
        axs[0,iDBN_group].spines['bottom'].set_visible(False)
        axs[0,iDBN_group].spines['left'].set_visible(False)
        # axs[0,iDBN_group].axis('equal')

        for ieventnode in np.arange(0,nevents,1):
            # plot the event nodes
            axs[0,iDBN_group].plot(eventnode_locations[ieventnode][0],eventnode_locations[ieventnode][1],
                                   '.',markersize=60,markerfacecolor=eventnodes_color[ieventnode],
                                   markeredgecolor='none')              
            axs[0,iDBN_group].text(eventname_locations[ieventnode][0],eventname_locations[ieventnode][1],
                                   eventnames[ieventnode],fontsize=10)

            # plot the event edges
            for ifromNode in np.arange(0,nFromNodes,1):
                for itoNode in np.arange(0,nToNodes,1):
                    edge_weight_tgt = sig_avg_dags[ifromNode,itoNode]
                    if edge_weight_tgt>0:
                        if not ifromNode == itoNode:
                            #axs[0,iDBN_group].plot(eventnode_locations[ifromNode],eventnode_locations[itoNode],'k-',linewidth=edge_weight_tgt*3)
                            axs[0,iDBN_group].arrow(nodearrow_locations[ifromNode][itoNode][0],
                                                    nodearrow_locations[ifromNode][itoNode][1],
                                                    nodearrow_directions[ifromNode][itoNode][0],
                                                    nodearrow_directions[ifromNode][itoNode][1],
                                                    head_width=0.08*edge_weight_tgt,width=0.04*edge_weight_tgt)
                        if ifromNode == itoNode:
                            ring = mpatches.Wedge(nodearrow_locations[ifromNode][itoNode],
                                                  .1, nodearrow_directions[ifromNode][itoNode][0],
                                                  nodearrow_directions[ifromNode][itoNode][1], 
                                                  0.04*edge_weight_tgt)
                            p = PatchCollection(
                                [ring], 
                                facecolor='#2693de', 
                                edgecolor='#000000'
                            )
                            axs[0,iDBN_group].add_collection(p)
                            # add arrow head
                            if ifromNode < 2:
                                axs[0,iDBN_group].arrow(nodearrow_locations[ifromNode][itoNode][0]-0.1+0.02*edge_weight_tgt,
                                                        nodearrow_locations[ifromNode][itoNode][1],
                                                        0,-0.05,fc='#2693de',
                                                        head_width=0.08*edge_weight_tgt,width=0.04*edge_weight_tgt)
                            else:
                                axs[0,iDBN_group].arrow(nodearrow_locations[ifromNode][itoNode][0]-0.1+0.02*edge_weight_tgt,
                                                        nodearrow_locations[ifromNode][itoNode][1],
                                                        0,0.02,fc='#2693de',
                                                        head_width=0.08*edge_weight_tgt,width=0.04*edge_weight_tgt)

        # heatmap for the weights
        sig_avg_dags_df = pd.DataFrame(sig_avg_dags)
        sig_avg_dags_df.columns = eventnames
        sig_avg_dags_df.index = eventnames
        im = axs[1,iDBN_group].pcolormesh(sig_avg_dags_df,cmap="Blues")
        #
        if iDBN_group == nDBN_groups-1:
            cax = axs[1,iDBN_group].inset_axes([1.04, 0.2, 0.05, 0.8])
            fig.colorbar(im, ax=axs[1,iDBN_group], cax=cax,label='transition probability')

        axs[1,iDBN_group].axis('equal')
        axs[1,iDBN_group].set_xlabel('to Node',fontsize=14)
        axs[1,iDBN_group].set_xticks(np.arange(0.5,4.5,1))
        axs[1,iDBN_group].set_xticklabels(eventnames)
        if iDBN_group == 0:
            axs[1,iDBN_group].set_ylabel('from Node',fontsize=14)
            axs[1,iDBN_group].set_yticks(np.arange(0.5,4.5,1))
            axs[1,iDBN_group].set_yticklabels(eventnames)
        else:
            axs[1,iDBN_group].set_yticks([])
            axs[1,iDBN_group].set_yticklabels([])
                                     
    except:
        continue
    
if savefigs:
    if moreSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"oneTimeLag_DAGs_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_'+str(samplingsizes_name[0])+'_rows_ConditionMerged.jpg')
    if minmaxfullSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"oneTimeLag_DAGs_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_BhvIntBin_'+samplingsizes_name[0]+'_ConditionMerged.jpg')
            
                      
            
    
    


# ## plot some key edges

# ### plot the transition probability from social gaze to pull

# In[ ]:


DAGs_all_dates = np.array(list(sig_avg_dags_allsessions.values()))


# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 5))
#
# sort the data based on task type, cooperation threshold and dates
sorting_df = pd.DataFrame({'dates': dates_list, 'coopthres': coopthres_forsort.ravel()}, columns=['dates', 'coopthres'])
sorting_df = sorting_df.sort_values(by=['coopthres','dates'], ascending = [False, True])
#
ax1.plot(DAGs_all_dates[sorting_df.index,2,0],'o-',label = "socialgaze1->pull1")
ax1.plot(DAGs_all_dates[sorting_df.index,2,1],'o-',label = "socialgaze1->pull2")
ax1.plot(DAGs_all_dates[sorting_df.index,3,0],'o-',label = "socialgaze2->pull1")
ax1.plot(DAGs_all_dates[sorting_df.index,3,1],'o-',label = "socialgaze2->pull2")
ax1.legend(fontsize=12)
ax1.set_ylim(-0.1,1.1)
ax1.set_ylabel("transition probability",fontsize=13)
#
plt.xticks(np.arange(0,ndates,1),np.array(dates_list)[sorting_df.index], rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.title("transition probability from social gaze to pull", fontsize = 14)
#
tasktypes = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','novision']
taskswitches = np.where(np.array(sorting_df['coopthres'])[1:]-np.array(sorting_df['coopthres'])[:-1]!=0)[0]+0.5
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.plot([taskswitch,taskswitch],[-0.15,1.15],'k--')
taskswitches = np.concatenate(([0],taskswitches))
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.text(taskswitch+0.25,-0.075,tasktypes[itaskswitch],fontsize=10)
#
plt.show()


# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 5))

grouptypes = ['self reward','3s threshold','2s threshold','1.5s threshold','1s threshold','no-vision']

gaze1_pull1 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],2,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],2,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],2,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],2,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],2,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],2,0]]
gaze1_pull2 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],2,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],2,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],2,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],2,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],2,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],2,1]]
gaze2_pull1 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],3,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],3,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],3,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],3,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],3,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],3,0]]
gaze2_pull2 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],3,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],3,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],3,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],3,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],3,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],3,1]
               ]

doboxplot = 0
domeanplot = 1
savefigs=1

if doboxplot:
    gaze1_pull1_plot = plt.boxplot(gaze1_pull1,positions=np.array(np.arange(len(gaze1_pull1)))*4.0-1.05,widths=0.6)
    gaze1_pull2_plot = plt.boxplot(gaze1_pull2,positions=np.array(np.arange(len(gaze1_pull2)))*4.0-0.35,widths=0.6)
    gaze2_pull1_plot = plt.boxplot(gaze2_pull1,positions=np.array(np.arange(len(gaze2_pull1)))*4.0+0.35,widths=0.6)
    gaze2_pull2_plot = plt.boxplot(gaze2_pull2,positions=np.array(np.arange(len(gaze2_pull2)))*4.0+1.05,widths=0.6)
    #
    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
        plt.plot([], c=color_code, label=label)
        plt.legend()
    #
    define_box_properties(gaze1_pull1_plot, '#0343DF', 'social_gaze1->pull1')
    define_box_properties(gaze1_pull2_plot, '#FFA500', 'social_gaze1->pull2')
    define_box_properties(gaze2_pull1_plot, '#008000', 'social_gaze2->pull1')
    define_box_properties(gaze2_pull2_plot, '#8C000F', 'social_gaze2->pull2')
    #
    # set the x label values
    plt.xticks(np.arange(0, len(grouptypes)*4, 4), grouptypes);
    
if domeanplot:
    gaze1_pull1_plot = plt.plot(np.array(np.arange(len(gaze1_pull1)))*4.0,np.nanmean(pd.DataFrame(gaze1_pull1),axis=1),'-o',label='M1gazeM2->pull1')
    gaze1_pull2_plot = plt.plot(np.array(np.arange(len(gaze1_pull2)))*4.0,np.nanmean(pd.DataFrame(gaze1_pull2),axis=1),'-o',label='M1gazeM2->pull2')
    gaze2_pull1_plot = plt.plot(np.array(np.arange(len(gaze2_pull1)))*4.0,np.nanmean(pd.DataFrame(gaze2_pull1),axis=1),'-o',label='M2gazeM1->pull1')
    gaze2_pull2_plot = plt.plot(np.array(np.arange(len(gaze2_pull2)))*4.0,np.nanmean(pd.DataFrame(gaze2_pull2),axis=1),'-o',label='M2gazeM1->pull2')
    plt.legend(fontsize=14)
    plt.xticks(np.arange(0, len(grouptypes)*4, 4), grouptypes,fontsize=15,rotation=45);
    plt.ylabel('mean transition probability',fontsize=15)
    
if savefigs:
    if moreSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"gaze_pull_prob_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_'+str(samplingsizes_name[0])+'_rows_ConditionMerged.jpg')
    if minmaxfullSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"gaze_pull_prob_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_BhvIntBin_'+samplingsizes_name[0]+'_ConditionMerged.jpg')
            


# ### plot the transition probability from pull to social gaze

# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 5))
#
# sort the data based on task type and dates
sorting_df = pd.DataFrame({'dates': dates_list, 'coopthres': coopthres_forsort.ravel()}, columns=['dates', 'coopthres'])
sorting_df = sorting_df.sort_values(by=['coopthres','dates'], ascending = [False, True])
#
ax1.plot(DAGs_all_dates[sorting_df.index,0,2],'o-',label = "pull1->socialgaze1")
ax1.plot(DAGs_all_dates[sorting_df.index,1,2],'o-',label = "pull2->socialgaze1")
ax1.plot(DAGs_all_dates[sorting_df.index,0,3],'o-',label = "pull1->socialgaze2")
ax1.plot(DAGs_all_dates[sorting_df.index,1,3],'o-',label = "pull2->socialgaze2")
ax1.legend(fontsize=12)
ax1.set_ylim(-0.1,1.1)
ax1.set_ylabel("transition probability",fontsize=13)
#
plt.xticks(np.arange(0,ndates,1),np.array(dates_list)[sorting_df.index], rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.title("transition probability from pull to social gaze", fontsize = 14)
#
tasktypes = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
taskswitches = np.where(np.array(sorting_df['coopthres'])[1:]-np.array(sorting_df['coopthres'])[:-1]!=0)[0]+0.5
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.plot([taskswitch,taskswitch],[-0.15,1.15],'k--')
taskswitches = np.concatenate(([0],taskswitches))
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.text(taskswitch+0.25,-0.075,tasktypes[itaskswitch],fontsize=10)
#
plt.show()


# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 5))

grouptypes = ['self reward','3s threshold','2s threshold','1.5s threshold','1s threshold','no-vision']

pull1_gaze1 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],0,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],0,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],0,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],0,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],0,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],0,2]]
pull2_gaze1 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],1,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],1,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],1,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],1,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],1,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],1,2]]
pull1_gaze2 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],0,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],0,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],0,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],0,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],0,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],0,3]]
pull2_gaze2 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],1,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],1,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],1,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],1,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],1,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],1,3]]

doboxplot = 0
domeanplot = 1
savefigs=1

if doboxplot:
    pull1_gaze1_plot = plt.boxplot(pull1_gaze1,positions=np.array(np.arange(len(pull1_gaze1)))*4.0-1.05,widths=0.6)
    pull2_gaze1_plot = plt.boxplot(pull2_gaze1,positions=np.array(np.arange(len(pull2_gaze1)))*4.0-0.35,widths=0.6)
    pull1_gaze2_plot = plt.boxplot(pull1_gaze2,positions=np.array(np.arange(len(pull1_gaze2)))*4.0+0.35,widths=0.6)
    pull2_gaze2_plot = plt.boxplot(pull2_gaze2,positions=np.array(np.arange(len(pull2_gaze2)))*4.0+1.05,widths=0.6)
    #
    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
        plt.plot([], c=color_code, label=label)
        plt.legend()
    #
    define_box_properties(pull1_gaze1_plot, '#0343DF', 'pull1->gaze1')
    define_box_properties(pull2_gaze1_plot, '#FFA500', 'pull2->gaze1')
    define_box_properties(pull1_gaze2_plot, '#008000', 'pull1->gaze2')
    define_box_properties(pull2_gaze2_plot, '#8C000F', 'pull2->gaze2')
    #
    # set the x label values
    plt.xticks(np.arange(0, len(grouptypes)*4, 4), grouptypes);
    
if domeanplot:
    pull1_gaze1_plot = plt.plot(np.array(np.arange(len(pull1_gaze1)))*4.0,np.nanmean(pd.DataFrame(pull1_gaze1),axis=1),'-o',label='pull1->M1gazeM2')
    pull1_gaze2_plot = plt.plot(np.array(np.arange(len(pull1_gaze2)))*4.0,np.nanmean(pd.DataFrame(pull1_gaze2),axis=1),'-o',label='pull1->M2gazeM1')
    pull2_gaze1_plot = plt.plot(np.array(np.arange(len(pull2_gaze1)))*4.0,np.nanmean(pd.DataFrame(pull2_gaze1),axis=1),'-o',label='pull2->M1gazeM2')
    pull2_gaze2_plot = plt.plot(np.array(np.arange(len(pull2_gaze2)))*4.0,np.nanmean(pd.DataFrame(pull2_gaze2),axis=1),'-o',label='pull2->M2gazeM1')
    plt.legend(fontsize=14)
    plt.xticks(np.arange(0, len(grouptypes)*4, 4), grouptypes,fontsize=15,rotation=45);
    plt.ylabel('mean transition probability',fontsize=15)
    
if savefigs:
    if moreSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"pull_gaze_prob_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_'+str(samplingsizes_name[0])+'_rows_ConditionMerged.jpg')
    if minmaxfullSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"pull_gaze_prob_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_BhvIntBin_'+samplingsizes_name[0]+'_ConditionMerged.jpg')
            
        


# ### plot the transition probability from pull to pull

# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 5))
#
# sort the data based on task type and dates
sorting_df = pd.DataFrame({'dates': dates_list, 'coopthres': coopthres_forsort.ravel()}, columns=['dates', 'coopthres'])
sorting_df = sorting_df.sort_values(by=['coopthres','dates'], ascending = [False, True])
#
ax1.plot(DAGs_all_dates[sorting_df.index,0,0],'o-',label = "pull1->pull1")
ax1.plot(DAGs_all_dates[sorting_df.index,1,1],'o-',label = "pull2->pull2")
ax1.plot(DAGs_all_dates[sorting_df.index,0,1],'o-',label = "pull1->pull2")
ax1.plot(DAGs_all_dates[sorting_df.index,1,0],'o-',label = "pull2->pull1")
ax1.legend(fontsize=12)
ax1.set_ylim(-0.1,1.1)
ax1.set_ylabel("transition probability",fontsize=13)
#
plt.xticks(np.arange(0,ndates,1),np.array(dates_list)[sorting_df.index], rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.title("transition probability from pull to pull", fontsize = 14)
#
tasktypes = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','novision']
taskswitches = np.where(np.array(sorting_df['coopthres'])[1:]-np.array(sorting_df['coopthres'])[:-1]!=0)[0]+0.5
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.plot([taskswitch,taskswitch],[-0.15,1.15],'k--')
taskswitches = np.concatenate(([0],taskswitches))
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.text(taskswitch+0.25,-0.075,tasktypes[itaskswitch],fontsize=10)
#
plt.show()


# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 5))

grouptypes = ['self reward','3s threshold','2s threshold','1.5s threshold','1s threshold','novision']

pull1_pull1 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],0,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],0,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],0,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],0,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],0,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],0,0]]
pull2_pull2 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],1,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],1,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],1,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],1,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],1,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],1,1]]
pull1_pull2 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],0,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],0,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],0,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],0,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],0,1],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],0,1]]
pull2_pull1 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],1,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],1,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],1,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],1,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],1,0],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],1,0]]

doboxplot = 0
domeanplot = 1
savefigs=1

if doboxplot:
    pull1_pull1_plot = plt.boxplot(pull1_pull1,positions=np.array(np.arange(len(pull1_pull1)))*4.0-1.05,widths=0.6)
    pull2_pull2_plot = plt.boxplot(pull2_pull2,positions=np.array(np.arange(len(pull2_pull2)))*4.0-0.35,widths=0.6)
    pull1_pull2_plot = plt.boxplot(pull1_pull2,positions=np.array(np.arange(len(pull1_pull2)))*4.0+0.35,widths=0.6)
    pull2_pull1_plot = plt.boxplot(pull2_pull1,positions=np.array(np.arange(len(pull2_pull1)))*4.0+1.05,widths=0.6)
    #
    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
        plt.plot([], c=color_code, label=label)
        plt.legend()
    #
    define_box_properties(pull1_pull1_plot, '#0343DF', 'pull1->pull1')
    define_box_properties(pull2_pull2_plot, '#FFA500', 'pull2->pull2')
    define_box_properties(pull1_pull2_plot, '#008000', 'pull1->pull2')
    define_box_properties(pull2_pull1_plot, '#8C000F', 'pull2->pull1')
    #
    # set the x label values
    plt.xticks(np.arange(0, len(grouptypes)*4, 4), grouptypes);
    
if domeanplot:
    pull1_pull1_plot = plt.plot(np.array(np.arange(len(pull1_pull1)))*4.0,np.nanmean(pd.DataFrame(pull1_pull1),axis=1),'-o',label='pull1->pull1')
    pull1_pull2_plot = plt.plot(np.array(np.arange(len(pull1_pull2)))*4.0,np.nanmean(pd.DataFrame(pull1_pull2),axis=1),'-o',label='pull1->pull2')
    pull2_pull1_plot = plt.plot(np.array(np.arange(len(pull2_pull1)))*4.0,np.nanmean(pd.DataFrame(pull2_pull1),axis=1),'-o',label='pull2->pull1')
    pull2_pull2_plot = plt.plot(np.array(np.arange(len(pull2_pull2)))*4.0,np.nanmean(pd.DataFrame(pull2_pull2),axis=1),'-o',label='pull2->pull2')
    plt.legend(fontsize=14)
    plt.xticks(np.arange(0, len(grouptypes)*4, 4), grouptypes,fontsize=15,rotation=45);
    plt.ylabel('mean transition probability',fontsize=15)
    
if savefigs:
    if moreSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"pull_pull_prob_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_'+str(samplingsizes_name[0])+'_rows_ConditionMerged.jpg')
    if minmaxfullSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"pull_pull_prob_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_BhvIntBin_'+samplingsizes_name[0]+'_ConditionMerged.jpg')
            
        


# ### plot the transition probability from social gaze to social gaze

# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 5))
#
# sort the data based on task type and dates
sorting_df = pd.DataFrame({'dates': dates_list, 'coopthres': coopthres_forsort.ravel()}, columns=['dates', 'coopthres'])
sorting_df = sorting_df.sort_values(by=['coopthres','dates'], ascending = [False, True])
#
ax1.plot(DAGs_all_dates[sorting_df.index,2,2],'o-',label = "socialgaze1->socialgaze1")
ax1.plot(DAGs_all_dates[sorting_df.index,3,3],'o-',label = "socialgaze2->socialgaze2")
ax1.plot(DAGs_all_dates[sorting_df.index,2,3],'o-',label = "socialgaze1->socialgaze2")
ax1.plot(DAGs_all_dates[sorting_df.index,3,2],'o-',label = "socialgaze2->socialgaze1")
ax1.legend(fontsize=12)
ax1.set_ylim(-0.1,1.1)
ax1.set_ylabel("transition probability",fontsize=13)
#
plt.xticks(np.arange(0,ndates,1),np.array(dates_list)[sorting_df.index], rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.title("transition probability from social gaze to social gaze", fontsize = 14)
#
tasktypes = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','novision']
taskswitches = np.where(np.array(sorting_df['coopthres'])[1:]-np.array(sorting_df['coopthres'])[:-1]!=0)[0]+0.5
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.plot([taskswitch,taskswitch],[-0.15,1.15],'k--')
taskswitches = np.concatenate(([0],taskswitches))
for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
    taskswitch = taskswitches[itaskswitch]
    ax1.text(taskswitch+0.25,-0.075,tasktypes[itaskswitch],fontsize=10)
#
plt.show()


# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 5))

grouptypes = ['self reward','3s threshold','2s threshold','1.5s threshold','1s threshold','novision']

gaze1_gaze1 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],2,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],2,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],2,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],2,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],2,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],2,2]]
gaze2_gaze2 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],3,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],3,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],3,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],3,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],3,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],3,3]]
gaze1_gaze2 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],2,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],2,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],2,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],2,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],2,3],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],2,3]]
gaze2_gaze1 = [DAGs_all_dates[np.transpose(coopthres_forsort==100)[0],3,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==3)[0],3,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==2)[0],3,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==1.5)[0],3,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==1)[0],3,2],
               DAGs_all_dates[np.transpose(coopthres_forsort==-1)[0],3,2]]

doboxplot = 0
domeanplot = 1
savefigs=1

if doboxplot:
    
    gaze1_gaze1_plot = plt.boxplot(gaze1_gaze1,positions=np.array(np.arange(len(gaze1_gaze1)))*4.0-1.05,widths=0.6)
    gaze2_gaze2_plot = plt.boxplot(gaze2_gaze2,positions=np.array(np.arange(len(gaze2_gaze2)))*4.0-0.35,widths=0.6)
    gaze1_gaze2_plot = plt.boxplot(gaze1_gaze2,positions=np.array(np.arange(len(gaze1_gaze2)))*4.0+0.35,widths=0.6)
    gaze2_gaze1_plot = plt.boxplot(gaze2_gaze1,positions=np.array(np.arange(len(gaze2_gaze1)))*4.0+1.05,widths=0.6)
    #
    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
        plt.plot([], c=color_code, label=label)
        plt.legend()
    #
    define_box_properties(gaze1_gaze1_plot, '#0343DF', 'gaze1->gaze1')
    define_box_properties(gaze2_gaze2_plot, '#FFA500', 'gaze2->gaze2')
    define_box_properties(gaze1_gaze2_plot, '#008000', 'gaze1->gaze2')
    define_box_properties(gaze2_gaze1_plot, '#8C000F', 'gaze2->gaze1')
    #
    # set the x label values
    plt.xticks(np.arange(0, len(grouptypes)*4, 4), grouptypes);
    
if domeanplot:
    gaze1_gaze1_plot = plt.plot(np.array(np.arange(len(gaze1_gaze1)))*4.0,np.nanmean(pd.DataFrame(gaze1_gaze1),axis=1),'-o',label='M1gazeM2->M1gazeM2')
    gaze1_gaze2_plot = plt.plot(np.array(np.arange(len(gaze1_gaze2)))*4.0,np.nanmean(pd.DataFrame(gaze1_gaze2),axis=1),'-o',label='M1gazeM2->M2gazeM1')
    gaze2_gaze1_plot = plt.plot(np.array(np.arange(len(gaze2_gaze1)))*4.0,np.nanmean(pd.DataFrame(gaze2_gaze1),axis=1),'-o',label='M2gazeM1->M1gazeM2')
    gaze2_gaze2_plot = plt.plot(np.array(np.arange(len(gaze2_gaze2)))*4.0,np.nanmean(pd.DataFrame(gaze2_gaze2),axis=1),'-o',label='M2gazeM1->M2gazeM1')
    plt.legend(fontsize=14)
    plt.xticks(np.arange(0, len(grouptypes)*4, 4), grouptypes,fontsize=15,rotation=45);
    plt.ylabel('mean transition probability',fontsize=15)
    
if savefigs:
    if moreSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"gaze_gaze_prob_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_'+str(samplingsizes_name[0])+'_rows_ConditionMerged.jpg')
    if minmaxfullSampSize:
        figsavefolder = data_saved_folder+'figs_for_DBN_and_bhv_Aniposelib3d_ana_allsessions_morevars_task/'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        plt.savefig(figsavefolder+"gaze_gaze_prob_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_BhvIntBin_'+samplingsizes_name[0]+'_ConditionMerged.jpg')
            


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





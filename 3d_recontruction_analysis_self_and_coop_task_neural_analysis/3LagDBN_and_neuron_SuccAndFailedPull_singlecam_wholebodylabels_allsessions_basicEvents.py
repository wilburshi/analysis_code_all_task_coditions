#!/usr/bin/env python
# coding: utf-8

# ### In this script, DBN is run on the all the sessions
# ### In this script, DBN is run with 1s time bin, 3 time lag 
# ### In this script, the animal tracking is done with only one camera - camera 2 (middle)
# ### In this script, DBN is run together with neural data and basic bhv variables

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import scipy
import scipy.stats as st
from sklearn.neighbors import KernelDensity
import string
import warnings
import pickle
import json

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


# ### function - interval between all behavioral events

# In[ ]:


from ana_functions.bhv_events_interval import bhv_events_interval


# ### function - spike analysis

# In[ ]:


from ana_functions.spike_analysis_FR_calculation import spike_analysis_FR_calculation


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
nframes = 5*30 # second*30fps

# re-analyze the video or not
reanalyze_video = 0
redo_anystep = 1

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

# dodson ginger
if 1:
    if not do_bestsession:
        neural_record_conditions = [
                                     # '20231101_Dodson_withGinger_SR', 
                                     '20231101_Dodson_withGinger_MC',
                                   ]
        dates_list = [
                      # "20231101_SR",
                      "20231101_MC",
                     ]
        session_start_times = [ 
                                 # 0.00,  
                                 0.00, 
                              ] # in second
    elif do_bestsession:
        # pick only five sessions for each conditions
        neural_record_conditions = [
                                     # '20231101_Dodson_withGinger_SR', 
                                     '20231101_Dodson_withGinger_MC',
                                   ]
        dates_list = [
                      # "20231101_SR",
                      "20231101_MC"
                     ]
        session_start_times = [ 
                                # 0.00,  
                                0.00,   
                              ] # in second
    
    animal1_fixedorder = ['dodson']
    animal2_fixedorder = ['ginger']

    animal1_filename = "Dodson"
    animal2_filename = "Ginger"
    

    
#    
# dates_list = ["20221128"]
# session_start_times = [1.00] # in second
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


# where to save the summarizing data
data_saved_folder = '/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/3d_recontruction_analysis_self_and_coop_task_data_saved/'

# neural data folder
neural_data_folder = '/gpfs/gibbs/pi/jadi/Marmoset_neural_recording/'

    


# In[ ]:


# basic behavior analysis (define time stamps for each bhv events, etc)

try:
    if redo_anystep:
        dummy
    
    # load saved data
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    
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

    print('all data from all dates are loaded')

except:

    print('analyze all dates')

    for idate in np.arange(0,ndates,1):
        date_tgt = dates_list[idate]
        session_start_time = session_start_times[idate]

        # folder and file path
        camera12_analyzed_path = "/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera12/"
        camera23_analyzed_path = "/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera23/"
        
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

        # get animal info from the session information
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
        # 
        pull1_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==1) 
        pull2_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==2)

        
        # load behavioral event results
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
            
                
        # # plot behavioral events
        if np.isin(animal1,animal1_fixedorder):
                plot_bhv_events(date_tgt,animal1, animal2, session_start_time, 600, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2)
        else:
                plot_bhv_events(date_tgt,animal2, animal1, session_start_time, 600, time_point_pull2, time_point_pull1, oneway_gaze2, oneway_gaze1, mutual_gaze2, mutual_gaze1)
        #
        # save behavioral events plot
        if 0:
            current_dir = data_saved_folder+'/bhv_events_singlecam_wholebody/'+animal1_fixedorder[0]+animal2_fixedorder[0]
            add_date_dir = os.path.join(current_dir,cameraID+'/'+date_tgt)
            if not os.path.exists(add_date_dir):
                os.makedirs(add_date_dir)
            plt.savefig(data_saved_folder+"/bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/'+date_tgt+"_"+cameraID_short+".pdf")

        #
        owgaze1_num_all_dates[idate] = np.shape(oneway_gaze1)[0]
        owgaze2_num_all_dates[idate] = np.shape(oneway_gaze2)[0]
        mtgaze1_num_all_dates[idate] = np.shape(mutual_gaze1)[0]
        mtgaze2_num_all_dates[idate] = np.shape(mutual_gaze2)[0]

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
            tracking_video_singlecam_wholebody_demo(bodyparts_locs_camI,output_look_ornot,output_allvectors,output_allangles,
                                              lever_locs_camI,tube_locs_camI,time_point_pull1,time_point_pull2,
                                              animalnames_videotrack,bodypartnames_videotrack,date_tgt,
                                              animal1_filename,animal2_filename,session_start_time,fps,nframes,cameraID,
                                              video_file_original,sqr_thres_tubelever,sqr_thres_face,sqr_thres_body)         
        

    # save data
    if 1:
        
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
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
    


# In[ ]:


pull2_num_all_dates


# #### redefine the tasktype and cooperation threshold to merge them together

# In[ ]:


# 100: self; 3: 3s coop; 2: 2s coop; 1.5: 1.5s coop; 1: 1s coop; -1: no-vision

tasktypes_all_dates[tasktypes_all_dates==5] = -1 # change the task type code for no-vision
coopthres_forsort = (tasktypes_all_dates-1)*coopthres_all_dates/2
coopthres_forsort[coopthres_forsort==0] = 100 # get the cooperation threshold for sorting


# ### prepare the input data for DBN
# #### for both neural and behavioral events data

# In[ ]:


# define DBN related summarizing variables

doBhvitv_timebin = 0 # 1: if use the mean bhv event interval for time bin

pulltypes = ['succpull','failedpull']
npulltypes = np.shape(pulltypes)[0]

prepare_input_data = 1

# DBN resolutions (make sure they are the same as in the later part of the code)
totalsess_time = 600 # total session time in s
# temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
temp_resolus = [0.5] # temporal resolution in the DBN model, eg: 0.5 means 500ms
ntemp_reses = np.shape(temp_resolus)[0]

mergetempRos = 0

# # train the dynamic bayesian network - Alec's model 
#   prepare the multi-session table; one time lag; multi time steps (temporal resolution) as separate files

# prepare the DBN input data
if prepare_input_data:
    
    DBN_input_data_alltypes = dict.fromkeys(pulltypes, [])
    
    for ipulltype in np.arange(0,npulltypes,1):
        pulltype = pulltypes[ipulltype]
        
        DBN_input_data_alltypes[pulltype] = dict.fromkeys(dates_list, [])
        
        for idate in np.arange(0,ndates,1):

            neural_record_condition = neural_record_conditions[idate]

            date_tgt = dates_list[idate]
            session_start_time = session_start_times[idate]

            # # load spike sorting results
            spike_time_file = neural_data_folder+neural_record_condition+'/Kilosort/spike_times.npy'
            spike_time_data = np.load(spike_time_file)
            #
            spike_clusters_file = neural_data_folder+neural_record_condition+'/Kilosort/spike_clusters.npy'
            spike_clusters_data = np.load(spike_clusters_file)
            spike_clusters_file = neural_data_folder+neural_record_condition+'/Kilosort/cluster_KSLabel.tsv'
            spike_clusters_data_withLabel = pd.read_csv(spike_clusters_file,sep='\t')
            spike_clusters_file = neural_data_folder+neural_record_condition+'/Kilosort/cluster_group.tsv'
            spike_clusters_data_clean = pd.read_csv(spike_clusters_file,sep='\t')
            # 
            good_cluster_id=spike_clusters_data_clean[spike_clusters_data_clean['group']=='good']['cluster_id'].reset_index(drop=True)
            # 
            spike_time_data = spike_time_data[np.isin(spike_clusters_data,good_cluster_id)]

            # load the behavioral data
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

            # separate successful and failed pulls
            if pulltype == 'succpull':
                trialnum_tgtpull = trial_record[trial_record['rewarded']>0]['trial_number'].reset_index(drop=True)
            elif pulltype == 'failedpull':
                trialnum_tgtpull = trial_record[trial_record['rewarded']==0]['trial_number'].reset_index(drop=True)
            bhv_data = bhv_data[bhv_data['trial_number'].isin(trialnum_tgtpull)]
            trial_record = trial_record[trial_record['trial_number'].isin(trialnum_tgtpull)]
            bhv_data = bhv_data.reset_index(drop=True)
            trial_record = trial_record.reset_index(drop=True)
            
            # get animal info
            animal1 = session_info['lever1_animal'][0].lower()
            animal2 = session_info['lever2_animal'][0].lower()

            # clean up the trial_record
            #try:
            warnings.filterwarnings('ignore')
            trial_record_clean = pd.DataFrame(columns=trial_record.columns)
            for itrial in trial_record['trial_number']:
                # trial_record_clean.loc[itrial] = trial_record[trial_record['trial_number']==itrial+1].iloc[[0]]
                trial_record_clean = trial_record_clean.append(trial_record[trial_record['trial_number']==itrial].iloc[[0]])
            trial_record_clean = trial_record_clean.reset_index(drop = True)

            # change bhv_data time to the absolute time
            time_points_new = pd.DataFrame(np.zeros(np.shape(bhv_data)[0]),columns=["time_points_new"])
            for itrial in trial_record_clean['trial_number']:
                ind = bhv_data["trial_number"]==itrial
                new_time_itrial = bhv_data[ind]["time_points"] + float(trial_record_clean[trial_record_clean['trial_number']==itrial]['trial_starttime'].iloc[0])
                time_points_new["time_points_new"][ind] = new_time_itrial
            bhv_data["time_points"] = time_points_new["time_points_new"]
            bhv_data = bhv_data[bhv_data["time_points"] != 0]
            #except:
            #    trial_record_clean = trial_record

            # get task type and cooperation threshold
            try:
                coop_thres = session_info["pulltime_thres"][0]
                tasktype = session_info["task_type"][0]
            except:
                coop_thres = 0
                tasktype = 1
                

            # session starting time compared with the neural recording
            session_start_time_niboard_offset = ni_data['session_t0_offset'] # in the unit of second
            neural_start_time_niboard_offset = ni_data['trigger_ts'][0]['elapsed_time'] # in the unit of second
            neural_start_time_session_start_offset = neural_start_time_niboard_offset-session_start_time_niboard_offset    

            # align the spike time to the session start, change the temporal resolution to 100ms
            spike_time_point = np.round((spike_time_data/fs_spikes+neural_start_time_session_start_offset)*10)/10


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


            if mergetempRos:
                temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
                # use bhv event to decide temporal resolution
                #
                #low_lim,up_lim,_ = bhv_events_interval(totalsess_time, session_start_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2)
                #temp_resolus = temp_resolus = np.arange(low_lim,up_lim,0.1)
            #
            if doBhvitv_timebin:
                pull_other_intv_ii = pd.Series(bhv_intv_all_dates[date_tgt]['pull_other_pooled'])
                # remove the interval that is too large
                pull_other_intv_ii[pull_other_intv_ii>(np.nanmean(pull_other_intv_ii)+2*np.nanstd(pull_other_intv_ii))]= np.nan    
                # pull_other_intv_ii[pull_other_intv_ii>10]= np.nan
                temp_resolus = [np.nanmean(pull_other_intv_ii)]          
            #
            ntemp_reses = np.shape(temp_resolus)[0]           


            # try different temporal resolutions
            for temp_resolu in temp_resolus:
                bhv_df = []

                if np.isin(animal1,animal1_fixedorder):
                    bhv_df_itr,_,_ = train_DBN_multiLag_create_df_only(totalsess_time, session_start_time, temp_resolu, spike_time_point, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2)
                else:
                    bhv_df_itr,_,_ = train_DBN_multiLag_create_df_only(totalsess_time, session_start_time, temp_resolu, spike_time_point, time_point_pull2, time_point_pull1, oneway_gaze2, oneway_gaze1, mutual_gaze2, mutual_gaze1)     

                if len(bhv_df)==0:
                    bhv_df = bhv_df_itr
                else:
                    bhv_df = pd.concat([bhv_df,bhv_df_itr])                   
                    bhv_df = bhv_df.reset_index(drop=True)        

                DBN_input_data_alltypes[pulltype][date_tgt] = bhv_df
            
    # save data
    if 1:
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_withNeuron_SuccAndFailedPull_allsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
        if not mergetempRos:
            if doBhvitv_timebin:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'bhvItvTempReSo.pkl', 'wb') as f:
                    pickle.dump(DBN_input_data_alltypes, f)
            else:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'sReSo.pkl', 'wb') as f:
                    pickle.dump(DBN_input_data_alltypes, f)
        else:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_mergeTempsReSo.pkl', 'wb') as f:
                pickle.dump(DBN_input_data_alltypes, f)     


# In[ ]:


# np.sum(DBN_input_data_alltypes['succpull']['20231101_MC']['pull2_t0'])


# In[ ]:


# np.sum(DBN_input_data_alltypes['failedpull']['20231101_MC']['pull2_t0'])


# ### run the DBN model on the combined session data set

# #### a test run

# In[ ]:


# run DBN on the large table with merged sessions

pulltypes = ['succpull','failedpull']
npulltypes = np.shape(pulltypes)[0]

mergetempRos = 0 # 1: merge different time bins

minmaxfullSampSize = 1 # 1: use the  min row number and max row number, or the full row for each session

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
    temp_resolus = [0.5] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    ntemp_reses = np.shape(temp_resolus)[0]

    # analyze successful pull and failed pull separately
    for pulltype in pulltypes:
        
        # try different temporal resolutions, remember to use the same settings as in the previous ones
        for temp_resolu in temp_resolus:

            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_withNeuron_SuccAndFailedPull_allsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
            if not mergetempRos:
                if doBhvitv_timebin:
                    with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'bhvItvTempReSo.pkl', 'rb') as f:
                        DBN_input_data_alltypes = pickle.load(f)
                else:
                    with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                        DBN_input_data_alltypes = pickle.load(f)
            else:
                with open(data_saved_subfolder+'//DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_mergeTempsReSo.pkl', 'rb') as f:
                    DBN_input_data_alltypes = pickle.load(f)

            DBN_input_data_alltypes = DBN_input_data_alltypes[pulltype]

            # only try three sample sizes
            #- minimal row number (require data downsample) and maximal row number (require data upsample)
            #- full row number of each session
            if minmaxfullSampSize:
                key_to_value_lengths = {k:len(v) for k, v in DBN_input_data_alltypes.items()}
                key_to_value_lengths_array = np.fromiter(key_to_value_lengths.values(),dtype=float)
                key_to_value_lengths_array[key_to_value_lengths_array==0]=np.nan
                min_samplesize = np.nanmin(key_to_value_lengths_array)
                min_samplesize = int(min_samplesize/100)*100
                max_samplesize = np.nanmax(key_to_value_lengths_array)
                max_samplesize = int(max_samplesize/100)*100
                #samplingsizes = [min_samplesize,max_samplesize,np.nan]
                #samplingsizes_name = ['min_row_number','max_row_number','full_row_number']
                samplingsizes = [np.nan]
                samplingsizes_name = ['full_row_number']
                nsamplings = np.shape(samplingsizes)[0]
                print(samplingsizes)

            # try different down/re-sampling size
            # for jj in np.arange(0,nsamplings,1):
            for jj in np.arange(0,1,1):

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
                        isamplingsize = np.shape(DBN_input_data_alltypes[date_tgt])[0]

                    try:
                        bhv_df_all = DBN_input_data_alltypes[date_tgt]

                        # define DBN graph structures; make sure they are the same as in the train_DBN_multiLag
                        colnames = list(bhv_df_all.columns)
                        eventnames = ["pull1","pull2","owgaze1","owgaze2","spikes"]
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

                DAGscores_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = DAGs_scores_alltypes
                DAGscores_shuffled_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = DAGs_shuffle_scores_alltypes

                weighted_graphs_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_alltypes
                weighted_graphs_shuffled_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_shuffled_alltypes
                sig_edges_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = sig_edges_alltypes

        print(weighted_graphs_diffTempRo_diffSampSize)
            
   


# In[ ]:





# #### run on the entire population

# In[ ]:


# run DBN on the large table with merged sessions

pulltypes = ['succpull','failedpull']
npulltypes = np.shape(pulltypes)[0]

mergetempRos = 0 # 1: merge different time bins

minmaxfullSampSize = 1 # 1: use the  min row number and max row number, or the full row for each session

moreSampSize = 0 # 1: use more sample size (more than just minimal row number and max row number)

num_starting_points = 100 # number of random starting points/graphs
nbootstraps = 95

try:
    dumpy
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_withNeuron_SuccAndFailedPull_allsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
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
    temp_resolus = [0.5] # temporal resolution in the DBN model, eg: 0.5 means 500ms
    ntemp_reses = np.shape(temp_resolus)[0]

     # analyze successful pull and failed pull separately
    for pulltype in pulltypes:
        
        # try different temporal resolutions, remember to use the same settings as in the previous ones
        for temp_resolu in temp_resolus:

            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_withNeuron_SuccAndFailedPull_allsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
            if not mergetempRos:
                if doBhvitv_timebin:
                    with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'bhvItvTempReSo.pkl', 'rb') as f:
                        DBN_input_data_allsessions = pickle.load(f)
                else:
                    with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                        DBN_input_data_allsessions = pickle.load(f)
            else:
                with open(data_saved_subfolder+'//DBN_input_data_alltypes_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_mergeTempsReSo.pkl', 'rb') as f:
                    DBN_input_data_allsessions = pickle.load(f)
            #        
            DBN_input_data_allsessions = DBN_input_data_allsessions[pulltype]

            # only try three sample sizes
            #- minimal row number (require data downsample) and maximal row number (require data upsample)
            #- full row number of each session
            if minmaxfullSampSize:
                key_to_value_lengths = {k:len(v) for k, v in DBN_input_data_alltypes.items()}
                key_to_value_lengths_array = np.fromiter(key_to_value_lengths.values(),dtype=float)
                key_to_value_lengths_array[key_to_value_lengths_array==0]=np.nan
                min_samplesize = np.nanmin(key_to_value_lengths_array)
                min_samplesize = int(min_samplesize/100)*100
                max_samplesize = np.nanmax(key_to_value_lengths_array)
                max_samplesize = int(max_samplesize/100)*100
                # samplingsizes = [min_samplesize,max_samplesize,np.nan]
                # samplingsizes_name = ['min_row_number','max_row_number','full_row_number']   
                samplingsizes = [np.nan]
                samplingsizes_name = ['full_row_number']
                nsamplings = np.shape(samplingsizes)[0]
                print(samplingsizes)

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


                        # define DBN graph structures; make sure they are the same as in the train_DBN_multiLag
                        colnames = list(bhv_df_all.columns)
                        eventnames = ["pull1","pull2","owgaze1","owgaze2","spikes"]
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

                DAGscores_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = DAGs_scores_alltypes
                DAGscores_shuffled_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = DAGs_shuffle_scores_alltypes

                weighted_graphs_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_alltypes
                weighted_graphs_shuffled_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = weighted_graphs_shuffled_alltypes
                sig_edges_diffTempRo_diffSampSize[(pulltype,str(temp_resolu),samplingsizes_name[jj])] = sig_edges_alltypes

            
    # save data
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_withNeuron_SuccAndFailedPull_allsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(data_saved_subfolder):
        os.makedirs(data_saved_subfolder)
    if moreSampSize:  
        with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(DAGscores_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(DAGscores_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'wb') as f:
            pickle.dump(sig_edges_diffTempRo_diffSampSize, f)
    elif minmaxfullSampSize:
        with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(DAGscores_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(DAGscores_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'wb') as f:
            pickle.dump(sig_edges_diffTempRo_diffSampSize, f)        
    else:
        with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'wb') as f:
            pickle.dump(DAGscores_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'wb') as f:
            pickle.dump(DAGscores_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'wb') as f:
            pickle.dump(weighted_graphs_shuffled_diffTempRo_diffSampSize, f)
        with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'wb') as f:
            pickle.dump(sig_edges_diffTempRo_diffSampSize, f)


# In[ ]:





# In[ ]:





# ### plot graphs - show the edge with arrows; show the best time bin and row number; show the three time lag separately

# In[ ]:


dates_list_plot = ["20231101_SR","20231101_MC",]
ndates_plot = np.shape(dates_list_plot)[0]


# In[ ]:


pulltype_forplot = 'failedpull' # 'succpull' or 'failedpull'

# make sure these variables are the same as in the previous steps
# temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
temp_resolus = [0.5] # temporal resolution in the DBN model, eg: 0.5 means 500ms
ntemp_reses = np.shape(temp_resolus)[0]
#
if moreSampSize:
    # different data (down/re)sampling numbers
    # samplingsizes = np.arange(1100,3000,100)
    samplingsizes = [1100]
    # samplingsizes = [100,500,1000,1500,2000,2500,3000]        
    # samplingsizes = [100,500]
    # samplingsizes_name = ['100','500','1000','1500','2000','2500','3000']
    samplingsizes_name = list(map(str, samplingsizes))
else:
    samplingsizes_name = ['full_row_number']   
nsamplings = np.shape(samplingsizes_name)[0]

# make sure these variables are consistent with the train_DBN_alec.py settings
eventnames = ["pull1","pull2","gaze1","gaze2",'OFCspikes']
eventnode_locations = [[0,1],[1,1],[0,0],[1,0],[-0.5,0.5],]
eventname_locations = [[0,1],[1,1],[0,0],[1,0],[-0.5,0.5],]
# indicate where edge starts
# for the self edge, it's the center of the self loop
nodearrow_locations = [[[0.00,1.25],[0.25,1.10],[-.10,0.75],[0.15,0.65],[-0.15,1.0],[0.15,0.75]],
                       [[0.75,1.00],[1.00,1.25],[0.85,0.65],[1.10,0.75],[0.85,0.75],[1.15,1.00]],
                       [[0.00,0.25],[0.25,0.35],[0.00,-.25],[0.25,-.10],[-0.15,-.05],[0.15,0.00]],
                       [[0.75,0.35],[1.00,0.25],[0.75,0.00],[1.00,-.25],[0.85,0.00],[1.15,-.05]],
                       [[-0.35,.65],[-0.25,.65],[-0.35,.25],[-0.35,.35],[-0.65,.50],[-0.2,0.45]],
                       ]
# indicate where edge goes
# for the self edge, it's the theta1 and theta2 (with fixed radius)
nodearrow_directions = [[[ -45,-180],[0.50,0.00],[0.00,-.50],[0.50,-.50],[-0.2,-0.2],[1.00,-0.2]],
                        [[-.50,0.00],[ -45,-180],[-.50,-.50],[0.00,-.50],[-1.0,-0.2],[0.20,-0.2]],
                        [[0.00,0.50],[0.50,0.50],[ 180,  45],[0.50,0.00],[-0.2,0.20],[1.00,0.20]],
                        [[-.50,0.50],[0.00,0.50],[-.50,0.00],[ 180,  45],[-1.0,0.20],[0.20,0.20]],
                        [[0.20,0.20],[1.10,0.20],[0.20,-.20],[1.10,-.20],[  45, -90],[1.25,0.00]],
                       ]

nevents = np.size(eventnames)
eventnodes_color = ['#BF3EFF','#FF7F00','#BF3EFF','#FF7F00','#BF3EFF',]
eventnodes_shape = ["o","o","^","^","*",]
    

# different session conditions (aka DBN groups)
# different time lags (t_-3, t_-2 and t_-1)
fig, axs = plt.subplots(6,ndates_plot)
fig.set_figheight(48)
fig.set_figwidth(8*ndates_plot)

time_lags = ['t_-1.5','t_-1','t_-0.5']
# fromRowIDs =[[0,1,2,3],[6,7,8,9],[12,13,14,15]]
# toColID = [0,1,2,3]
fromRowIDs =[[0,1,2,3,4,],[5,6,7,8,9,],[10,11,12,13,14,]]
toColID = [0,1,2,3,4,]
ntime_lags = np.shape(time_lags)[0]

temp_resolu = temp_resolus[0]
j_sampsize_name = samplingsizes_name[0]    

for ilag in np.arange(0,ntime_lags,1):
    
    time_lag_name = time_lags[ilag]
    fromRowID = fromRowIDs[ilag]
    
    for idate in np.arange(0,ndates_plot,1):

        # try:

            idate_typename = dates_list_plot[idate]

            # load data accordingly
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_withNeuron_SuccAndFailedPull_allsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
            #
            if moreSampSize:
                with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)
            elif minmaxfullSampSize:
                with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)


            weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(pulltype_forplot,str(temp_resolu),j_sampsize_name)][idate_typename]
            weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(pulltype_forplot,str(temp_resolu),j_sampsize_name)][idate_typename]
            # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(pulltype_forplot,str(temp_resolu),j_sampsize_name)][idate_typename]
            sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)

            #sig_edges_tgt = sig_edges_tgt*((weighted_graphs_tgt.mean(axis=0)>0.5)*1)

            sig_avg_dags = weighted_graphs_tgt.mean(axis = 0) * sig_edges_tgt
            sig_avg_dags = sig_avg_dags[fromRowID,:]
            sig_avg_dags = sig_avg_dags[:,toColID]

            # plot
            axs[ilag*2+0,idate].set_title(idate_typename,fontsize=14)
            axs[ilag*2+0,idate].set_xlim([-1,2])
            axs[ilag*2+0,idate].set_ylim([-1,2])
            axs[ilag*2+0,idate].set_xticks([])
            axs[ilag*2+0,idate].set_xticklabels([])
            axs[ilag*2+0,idate].set_yticks([])
            axs[ilag*2+0,idate].set_yticklabels([])
            axs[ilag*2+0,idate].spines['top'].set_visible(False)
            axs[ilag*2+0,idate].spines['right'].set_visible(False)
            axs[ilag*2+0,idate].spines['bottom'].set_visible(False)
            axs[ilag*2+0,idate].spines['left'].set_visible(False)
            # axs[ilag*2+0,idate].axis('equal')


            for ieventnode in np.arange(0,nevents,1):
                # plot the event nodes
                axs[ilag*2+0,idate].plot(eventnode_locations[ieventnode][0],eventnode_locations[ieventnode][1],
                                              eventnodes_shape[ieventnode],markersize=40,markerfacecolor=eventnodes_color[ieventnode],
                                              markeredgecolor='none')              
                axs[ilag*2+0,idate].text(eventname_locations[ieventnode][0],eventname_locations[ieventnode][1],
                                       eventnames[ieventnode],fontsize=15)

                clmap = mpl.cm.get_cmap('Greens')

                # plot the event edges
                for ifromNode in np.arange(0,nevents,1):
                    for itoNode in np.arange(0,nevents,1):
                        edge_weight_tgt = sig_avg_dags[ifromNode,itoNode]
                        if edge_weight_tgt>0:
                            if not ifromNode == itoNode:
                                #axs[ilag*2+0,idate].plot(eventnode_locations[ifromNode],eventnode_locations[itoNode],'k-',linewidth=edge_weight_tgt*3)
                                axs[ilag*2+0,idate].arrow(nodearrow_locations[ifromNode][itoNode][0],
                                                        nodearrow_locations[ifromNode][itoNode][1],
                                                        nodearrow_directions[ifromNode][itoNode][0],
                                                        nodearrow_directions[ifromNode][itoNode][1],
                                                        # head_width=0.08*abs(edge_weight_tgt),
                                                        # width=0.04*abs(edge_weight_tgt),
                                                        head_width=0.08,
                                                        width=0.04,   
                                                        color = clmap(edge_weight_tgt))
                            if ifromNode == itoNode:
                                ring = mpatches.Wedge(nodearrow_locations[ifromNode][itoNode],
                                                      .1, nodearrow_directions[ifromNode][itoNode][0],
                                                      nodearrow_directions[ifromNode][itoNode][1], 
                                                      # 0.04*abs(edge_weight_tgt),
                                                      0.04,
                                                      color = clmap(edge_weight_tgt))
                                p = PatchCollection(
                                    [ring], 
                                    facecolor=clmap(edge_weight_tgt), 
                                    edgecolor=clmap(edge_weight_tgt)
                                )
                                axs[ilag*2+0,idate].add_collection(p)
                                # add arrow head
                                if ifromNode < 2:
                                    axs[ilag*2+0,idate].arrow(nodearrow_locations[ifromNode][itoNode][0]-0.1+0.02*edge_weight_tgt,
                                                            nodearrow_locations[ifromNode][itoNode][1],
                                                            0,-0.05,color=clmap(edge_weight_tgt),
                                                            # head_width=0.08*edge_weight_tgt,width=0.04*edge_weight_tgt
                                                            head_width=0.08,width=0.04      
                                                            )
                                elif ifromNode < 4:
                                    axs[ilag*2+0,idate].arrow(nodearrow_locations[ifromNode][itoNode][0]-0.1+0.02*edge_weight_tgt,
                                                            nodearrow_locations[ifromNode][itoNode][1],
                                                            0,0.02,color=clmap(edge_weight_tgt),
                                                            # head_width=0.08*edge_weight_tgt,width=0.04*edge_weight_tgt
                                                            head_width=0.08,width=0.04      
                                                            )

            # heatmap for the weights
            sig_avg_dags_df = pd.DataFrame(sig_avg_dags)
            sig_avg_dags_df.columns = eventnames
            sig_avg_dags_df.index = eventnames
            vmin,vmax = 0,1
            import matplotlib as mpl
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            im = axs[ilag*2+1,idate].pcolormesh(sig_avg_dags_df,cmap="Greens",norm=norm)
            #
            if idate == ndates_plot-1:
                cax = axs[ilag*2+1,idate].inset_axes([1.04, 0.2, 0.05, 0.8])
                fig.colorbar(im, ax=axs[ilag*2+1,idate], cax=cax,label='edge confidence')

            axs[ilag*2+1,idate].axis('equal')
            axs[ilag*2+1,idate].set_xlabel('to Node',fontsize=14)
            axs[ilag*2+1,idate].set_xticks(np.arange(0.5,5.5,1))
            axs[ilag*2+1,idate].set_xticklabels(eventnames)
            if idate == 0:
                axs[ilag*2+1,idate].set_ylabel('from Node',fontsize=14)
                axs[ilag*2+1,idate].set_yticks(np.arange(0.5,5.5,1))
                axs[ilag*2+1,idate].set_yticklabels(eventnames)
                axs[ilag*2+1,idate].text(-1.5,1,time_lag_name+' time lag',rotation=90,fontsize=20)
                axs[ilag*2+0,idate].text(-1.25,0,time_lag_name+' time lag',rotation=90,fontsize=20)
            else:
                axs[ilag*2+1,idate].set_yticks([])
                axs[ilag*2+1,idate].set_yticklabels([])

        # except:
        #      continue
                
savefigs = 1
if savefigs:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_neuron_SuccAndFailedPull_singlecam_wholebodylabels_allsessions_basicEvents/'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'+idate_typename[0:7]+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    if moreSampSize:
        plt.savefig(figsavefolder+"threeTimeLag_DAGs_"+pulltype_forplot+"_"+panimal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_'+str(j_sampsize_name)+'_rows.pdf')
    else:  
        plt.savefig(figsavefolder+"threeTimeLag_DAGs_"+pulltype_forplot+"_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_'+j_sampsize_name+'.pdf')
            
            
            
            


# ### plot graphs - show the edge differences, use one condition as the base

# In[ ]:


dates_list_plot = ["20231101_SR","20231101_MC",]
ndates_plot = np.shape(dates_list_plot)[0]

basecondition = '20231101_SR'


# In[ ]:


pulltype_forplot = 'succpull' # 'succpull' or 'failedpull'

# make sure these variables are the same as in the previous steps
# temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
temp_resolus = [0.5] # temporal resolution in the DBN model, eg: 0.5 means 500ms
ntemp_reses = np.shape(temp_resolus)[0]
#
if moreSampSize:
    # different data (down/re)sampling numbers
    # samplingsizes = np.arange(1100,3000,100)
    samplingsizes = [1100]
    # samplingsizes = [100,500,1000,1500,2000,2500,3000]        
    # samplingsizes = [100,500]
    # samplingsizes_name = ['100','500','1000','1500','2000','2500','3000']
    samplingsizes_name = list(map(str, samplingsizes))
else:
    samplingsizes_name = ['full_row_number']   
nsamplings = np.shape(samplingsizes_name)[0]


# make sure these variables are consistent with the train_DBN_alec.py settings
eventnames = ["pull1","pull2","gaze1","gaze2",'OFCspikes']
eventnode_locations = [[0,1],[1,1],[0,0],[1,0],[-0.5,0.5],]
eventname_locations = [[0,1],[1,1],[0,0],[1,0],[-0.5,0.5],]
# indicate where edge starts
# for the self edge, it's the center of the self loop
nodearrow_locations = [[[0.00,1.25],[0.25,1.10],[-.10,0.75],[0.15,0.65],[-0.15,1.0],[0.15,0.75]],
                       [[0.75,1.00],[1.00,1.25],[0.85,0.65],[1.10,0.75],[0.85,0.75],[1.15,1.00]],
                       [[0.00,0.25],[0.25,0.35],[0.00,-.25],[0.25,-.10],[-0.15,-.05],[0.15,0.00]],
                       [[0.75,0.35],[1.00,0.25],[0.75,0.00],[1.00,-.25],[0.85,0.00],[1.15,-.05]],
                       [[-0.35,.65],[-0.25,.65],[-0.35,.25],[-0.35,.35],[-0.65,.50],[-0.2,0.45]],
                       ]
# indicate where edge goes
# for the self edge, it's the theta1 and theta2 (with fixed radius)
nodearrow_directions = [[[ -45,-180],[0.50,0.00],[0.00,-.50],[0.50,-.50],[-0.2,-0.2],[1.00,-0.2]],
                        [[-.50,0.00],[ -45,-180],[-.50,-.50],[0.00,-.50],[-1.0,-0.2],[0.20,-0.2]],
                        [[0.00,0.50],[0.50,0.50],[ 180,  45],[0.50,0.00],[-0.2,0.20],[1.00,0.20]],
                        [[-.50,0.50],[0.00,0.50],[-.50,0.00],[ 180,  45],[-1.0,0.20],[0.20,0.20]],
                        [[0.20,0.20],[1.10,0.20],[0.20,-.20],[1.10,-.20],[  45, -90],[1.25,0.00]],
                       ]

nevents = np.size(eventnames)
eventnodes_color = ['#BF3EFF','#FF7F00','#BF3EFF','#FF7F00','#BF3EFF',]
eventnodes_shape = ["o","o","^","^","*",]
    

# different session conditions (aka DBN groups)
# different time lags (t_-3, t_-2 and t_-1)
fig, axs = plt.subplots(6,ndates_plot)
fig.set_figheight(48)
fig.set_figwidth(8*ndates_plot)

time_lags = ['t_-1.5','t_-1','t_-0.5']
# fromRowIDs =[[0,1,2,3],[6,7,8,9],[12,13,14,15]]
# toColID = [0,1,2,3]
fromRowIDs =[[0,1,2,3,4,],[5,6,7,8,9,],[10,11,12,13,14,]]
toColID = [0,1,2,3,4,]
ntime_lags = np.shape(time_lags)[0]

temp_resolu = temp_resolus[0]
j_sampsize_name = samplingsizes_name[0]  

# load data accordingly
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_withNeuron_SuccAndFailedPull_allsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
#
if moreSampSize:
    with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
        DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
    with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
        DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
    with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
        weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
    with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
        weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
    with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
        sig_edges_diffTempRo_diffSampSize = pickle.load(f)
elif minmaxfullSampSize:
    with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
        DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
    with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
        DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
    with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
        weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
    with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
        weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
    with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
        sig_edges_diffTempRo_diffSampSize = pickle.load(f)
else:
    with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
        DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
    with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
        DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
    with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
        weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
    with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
        weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
    with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
        sig_edges_diffTempRo_diffSampSize = pickle.load(f)
                    
weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(pulltype_forplot,str(temp_resolu),j_sampsize_name)][basecondition]
weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(pulltype_forplot,str(temp_resolu),j_sampsize_name)][basecondition]
#sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(pulltype_forplot,str(temp_resolu),j_sampsize_name)][basecondition]
sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)
           
# sig_edges_tgt = sig_edges_tgt*((weighted_graphs_tgt.mean(axis=0)>0.5)*1)

weighted_graphs_base = weighted_graphs_tgt

sig_edges_base = sig_edges_tgt

sig_avg_dags_base =  weighted_graphs_base.mean(axis = 0) * sig_edges_base
    
for ilag in np.arange(0,ntime_lags,1):

    time_lag_name = time_lags[ilag]
    fromRowID = fromRowIDs[ilag]

    for idate in np.arange(0,ndates_plot,1):
            
        # try:

            idate_typename = dates_list_plot[idate]

            # load data accordingly
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_withNeuron_SuccAndFailedPull_allsessions'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
            #
            if moreSampSize:
                with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_moreSampSize.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)
            elif minmaxfullSampSize:
                with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_minmaxfullSampSize.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/DAGscores_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    DAGscores_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/DAGscores_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    DAGscores_shuffled_diffTempRo_diffSampSize = pickle.load(f) 
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)

            weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(pulltype_forplot,str(temp_resolu),j_sampsize_name)][idate_typename]
            weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(pulltype_forplot,str(temp_resolu),j_sampsize_name)][idate_typename]
            # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(pulltype_forplot,str(temp_resolu),j_sampsize_name)][idate_typename]
            sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)
           
            #sig_edges_tgt = sig_edges_tgt*((weighted_graphs_tgt.mean(axis=0)>0.5)*1)
            
            if 0:
                weighted_graphs_delta = (weighted_graphs_tgt-weighted_graphs_base)
                weighted_graphs_delta = weighted_graphs_delta.mean(axis=0)
                #
                sig_edges_delta = ((sig_edges_tgt+sig_edges_base)>0)*1
            else:
                weighted_graphs_delta,sig_edges_delta = Modulation_Index(weighted_graphs_base, weighted_graphs_tgt,
                                                                         sig_edges_base, sig_edges_tgt, 8000)
                weighted_graphs_delta = weighted_graphs_delta.mean(axis=0)
                
            sig_avg_dags = weighted_graphs_delta * sig_edges_delta
            sig_avg_dags = sig_avg_dags[fromRowID,:]
            sig_avg_dags = sig_avg_dags[:,toColID]

            # plot
            axs[ilag*2+0,idate].set_title(idate_typename,fontsize=18)
            axs[ilag*2+0,idate].set_xlim([-1,2])
            axs[ilag*2+0,idate].set_ylim([-1,2])
            axs[ilag*2+0,idate].set_xticks([])
            axs[ilag*2+0,idate].set_xticklabels([])
            axs[ilag*2+0,idate].set_yticks([])
            axs[ilag*2+0,idate].set_yticklabels([])
            axs[ilag*2+0,idate].spines['top'].set_visible(False)
            axs[ilag*2+0,idate].spines['right'].set_visible(False)
            axs[ilag*2+0,idate].spines['bottom'].set_visible(False)
            axs[ilag*2+0,idate].spines['left'].set_visible(False)
            # axs[ilag*2+0,idate].axis('equal')

            for ieventnode in np.arange(0,nevents,1):
                # plot the event nodes
                axs[ilag*2+0,idate].plot(eventnode_locations[ieventnode][0],eventnode_locations[ieventnode][1],
                                              eventnodes_shape[ieventnode],markersize=45,markerfacecolor=eventnodes_color[ieventnode],
                                              markeredgecolor='none')              
                
                axs[ilag*2+0,idate].text(eventname_locations[ieventnode][0],eventname_locations[ieventnode][1],
                                       eventnames[ieventnode],fontsize=10)
                
                clmap = mpl.cm.get_cmap('bwr')
                
                # plot the event edges
                for ifromNode in np.arange(0,nevents,1):
                    for itoNode in np.arange(0,nevents,1):
                        edge_weight_tgt = sig_avg_dags[ifromNode,itoNode]
                        if edge_weight_tgt!=0:
                            if not ifromNode == itoNode:
                                #axs[ilag*2+0,idate].plot(eventnode_locations[ifromNode],eventnode_locations[itoNode],'k-',linewidth=edge_weight_tgt*3)
                                axs[ilag*2+0,idate].arrow(nodearrow_locations[ifromNode][itoNode][0],
                                                        nodearrow_locations[ifromNode][itoNode][1],
                                                        nodearrow_directions[ifromNode][itoNode][0],
                                                        nodearrow_directions[ifromNode][itoNode][1],
                                                        # head_width=0.08*abs(edge_weight_tgt),
                                                        # width=0.04*abs(edge_weight_tgt),
                                                        head_width=0.08,
                                                        width=0.04,       
                                                        color = clmap((1+edge_weight_tgt)/2))
                            if ifromNode == itoNode:
                                ring = mpatches.Wedge(nodearrow_locations[ifromNode][itoNode],
                                                      .1, nodearrow_directions[ifromNode][itoNode][0],
                                                      nodearrow_directions[ifromNode][itoNode][1], 
                                                      # 0.04*abs(edge_weight_tgt)
                                                      0.04
                                                     )
                                p = PatchCollection(
                                    [ring], 
                                    facecolor=clmap((1+edge_weight_tgt)/2), 
                                    edgecolor=clmap((1+edge_weight_tgt)/2)
                                )
                                axs[ilag*2+0,idate].add_collection(p)
                                # add arrow head
                                if ifromNode < 2:
                                    axs[ilag*2+0,idate].arrow(nodearrow_locations[ifromNode][itoNode][0]-0.1+0.02*edge_weight_tgt,
                                                            nodearrow_locations[ifromNode][itoNode][1],
                                                            0,-0.05,color=clmap((1+edge_weight_tgt)/2),
                                                            # head_width=0.08*edge_weight_tgt,width=0.04*edge_weight_tgt
                                                            head_width=0.08,width=0.04      
                                                            )
                                elif ifromNode < 4:
                                    axs[ilag*2+0,idate].arrow(nodearrow_locations[ifromNode][itoNode][0]-0.1+0.02*edge_weight_tgt,
                                                            nodearrow_locations[ifromNode][itoNode][1],
                                                            0,0.02,color=clmap((1+edge_weight_tgt)/2),
                                                            # head_width=0.08*edge_weight_tgt,width=0.04*edge_weight_tgt
                                                            head_width=0.08,width=0.04      
                                                            )

            # heatmap for the weights
            sig_avg_dags_df = pd.DataFrame(sig_avg_dags)
            sig_avg_dags_df.columns = eventnames
            sig_avg_dags_df.index = eventnames
            vmin,vmax = -1,1
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            im = axs[ilag*2+1,idate].pcolormesh(sig_avg_dags_df,cmap="bwr",norm=norm)
            #-
            if idate == ndates_plot-1:
                cax = axs[ilag*2+1,idate].inset_axes([1.04, 0.2, 0.05, 0.8])
                fig.colorbar(im, ax=axs[ilag*2+1,idate], cax=cax,label='edge confidence')

            axs[ilag*2+1,idate].axis('equal')
            axs[ilag*2+1,idate].set_xlabel('to Node',fontsize=14)
            axs[ilag*2+1,idate].set_xticks(np.arange(0.5,5.5,1))
            axs[ilag*2+1,idate].set_xticklabels(eventnames)
            if idate == 0:
                axs[ilag*2+1,idate].set_ylabel('from Node',fontsize=14)
                axs[ilag*2+1,idate].set_yticks(np.arange(0.5,5.5,1))
                axs[ilag*2+1,idate].set_yticklabels(eventnames)
                axs[ilag*2+1,idate].text(-1.5,1,time_lag_name+' time lag',rotation=90,fontsize=20)
                axs[ilag*2+0,idate].text(-1.25,0,time_lag_name+' time lag',rotation=90,fontsize=20)
            else:
                axs[ilag*2+1,idate].set_yticks([])
                axs[ilag*2+1,idate].set_yticklabels([])

        # except:
        #     continue
    
    
savefigs = 1    
if savefigs:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_neuron_SuccAndFailedPull_singlecam_wholebodylabels_allsessions_basicEvents/'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'+idate_typename[0:7]+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    if moreSampSize:
        plt.savefig(figsavefolder+"threeTimeLag_DAGs_"+pulltype_forplot+"_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_'+str(j_sampsize_name)+'_rows_EdgeFifferenceFrom_'+basecondition+'AsBase.pdf')
    else:
        plt.savefig(figsavefolder+"threeTimeLag_DAGs_"+pulltype_forplot+"_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+str(temp_resolu)+'_'+j_sampsize_name+'_EdgeFifferenceFrom_'+basecondition+'AsBase.pdf')
            
            
            


# In[ ]:





# ### plot the edges over time (session)
# #### mean edge weights of selected edges

# In[ ]:


# 100: self; 3: 3s coop; 2: 2s coop; 1.5: 1.5s coop; 1: 1s coop; -1: no-vision
tasktypes_all_dates[tasktypes_all_dates==5] = -1 # change the task type code for no-vision
coopthres_forsort = (tasktypes_all_dates-1)*coopthres_all_dates/2
coopthres_forsort[coopthres_forsort==0] = 100 # get the cooperation threshold for sorting

#
# sort the data based on task type and dates
sorting_df = pd.DataFrame({'dates': dates_list, 'coopthres': coopthres_forsort.ravel()}, columns=['dates', 'coopthres'])
sorting_df = sorting_df.sort_values(by=['coopthres','dates'], ascending = [False, True])
dates_list_sorted = np.array(dates_list)[sorting_df.index]
ndates_sorted = np.shape(dates_list_sorted)[0]


# In[ ]:


# make sure these variables are the same as in the previous steps
# temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
temp_resolus = [0.5] # temporal resolution in the DBN model, eg: 0.5 means 500ms
ntemp_reses = np.shape(temp_resolus)[0]
#
if moreSampSize:
    # different data (down/re)sampling numbers
    # samplingsizes = np.arange(1100,3000,100)
    samplingsizes = [1100]
    # samplingsizes = [100,500,1000,1500,2000,2500,3000]        
    # samplingsizes = [100,500]
    # samplingsizes_name = ['100','500','1000','1500','2000','2500','3000']
    samplingsizes_name = list(map(str, samplingsizes))
elif minmaxfullSampSize:
    samplingsizes_name = ['full_row_number']   
nsamplings = np.shape(samplingsizes_name)[0]

temp_resolu = temp_resolus[0]
j_sampsize_name = samplingsizes_name[0]   

# 0.5s time lag
edges_target_names = [['0.5slag_pull2_pull1','0.5slag_pull1_pull2'],
                      ['0.5slag_gaze1_pull1','0.5slag_gaze2_pull2'],
                      ['0.5slag_pull2_gaze1','0.5slag_pull1_gaze2'],]
fromNodesIDs = [[11,10],
                [12,13],
                [ 9, 8],]
toNodesIDs = [[0,1],
              [0,1],
              [2,3]]

n_edges = np.shape(np.array(edges_target_names).flatten())[0]

# figure initiate
fig, axs = plt.subplots(int(np.ceil(n_edges/2)),2)
fig.set_figheight(5*np.ceil(n_edges/2))
fig.set_figwidth(10*2)

#
for i_edge in np.arange(0,n_edges,1):
    #
    edgeweight_mean_forplot_all_dates = np.zeros((ndates_sorted,1))
    edgeweight_shuffled_mean_forplot_all_dates = np.zeros((ndates_sorted,1))
    edgeweight_std_forplot_all_dates = np.zeros((ndates_sorted,1))
    edgeweight_shuffled_std_forplot_all_dates = np.zeros((ndates_sorted,1))
    
    edge_tgt_name = np.array(edges_target_names).flatten()[i_edge]
    fromNodesID = np.array(fromNodesIDs).flatten()[i_edge]
    toNodesID = np.array(toNodesIDs).flatten()[i_edge]
    
    for idate in np.arange(0,ndates_sorted,1):
        idate_name = dates_list_sorted[idate]
        
        weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][idate_name]
        weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][idate_name]
    
        edgeweight_mean_forplot_all_dates[idate] = np.nanmean(weighted_graphs_tgt[:,fromNodesID,toNodesID])
        edgeweight_shuffled_mean_forplot_all_dates[idate] = np.nanmean(weighted_graphs_shuffled_tgt[:,fromNodesID,toNodesID])
        edgeweight_std_forplot_all_dates[idate] = np.nanstd(weighted_graphs_tgt[:,fromNodesID,toNodesID])
        edgeweight_shuffled_std_forplot_all_dates[idate] = np.nanstd(weighted_graphs_shuffled_tgt[:,fromNodesID,toNodesID])
        
      
    # plot 
    axs.flatten()[i_edge].plot(np.arange(0,ndates_sorted,1),edgeweight_mean_forplot_all_dates,'ko',markersize=10)
    #axs.flatten()[i_edge].plot(np.arange(0,ndates_sorted,1),edgeweight_shuffled_mean_forplot_all_dates,'bo',markersize=10)
    #
    axs.flatten()[i_edge].set_title(edge_tgt_name,fontsize=16)
    axs.flatten()[i_edge].set_ylabel('mean edge weight',fontsize=13)
    axs.flatten()[i_edge].set_ylim([-0.1,1.1])
    axs.flatten()[i_edge].set_xlim([-0.5,ndates_sorted-0.5])
    #
    if i_edge > int(n_edges-1):
        axs.flatten()[i_edge].set_xticks(np.arange(0,ndates_sorted,1))
        axs.flatten()[i_edge].set_xticklabels(dates_list_sorted, rotation=90,fontsize=10)
    else:
        axs.flatten()[i_edge].set_xticklabels('')
    #
    tasktypes = ['self','coop(3s)','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
    taskswitches = np.where(np.array(sorting_df['coopthres'])[1:]-np.array(sorting_df['coopthres'])[:-1]!=0)[0]+0.5
    for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
        taskswitch = taskswitches[itaskswitch]
        axs.flatten()[i_edge].plot([taskswitch,taskswitch],[-0.1,1.1],'k--')
    taskswitches = np.concatenate(([0],taskswitches))
    for itaskswitch in np.arange(0,np.shape(taskswitches)[0],1):
        taskswitch = taskswitches[itaskswitch]
        axs.flatten()[i_edge].text(taskswitch+0.25,-0.05,tasktypes[itaskswitch],fontsize=10)


        
savefigs = 0
if savefigs:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_neuron_bhv_singlecam_wholebodylabels_allsessions_basicEvents/'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    plt.savefig(figsavefolder+"edgeweight_acrossAllSessions_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pdf')
    
    


# #### mean edge weights of selected edges v.s. other behavioral measures
# ##### only the cooperation days

# In[ ]:


# only select the targeted dates
sorting_tgt_df = sorting_df[(sorting_df['coopthres']==1)|(sorting_df['coopthres']==2)|(sorting_df['coopthres']==3)]
# sorting_tgt_df = sorting_df[(sorting_df['coopthres']==1)]
dates_list_tgt = sorting_tgt_df['dates']
dates_list_tgt = np.array(dates_list_tgt)
#
ndates_tgt = np.shape(dates_list_tgt)[0]


# In[ ]:


# make sure these variables are the same as in the previous steps
# temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
temp_resolus = [0.5] # temporal resolution in the DBN model, eg: 0.5 means 500ms
ntemp_reses = np.shape(temp_resolus)[0]
#
if moreSampSize:
    # different data (down/re)sampling numbers
    # samplingsizes = np.arange(1100,3000,100)
    samplingsizes = [1100]
    # samplingsizes = [100,500,1000,1500,2000,2500,3000]        
    # samplingsizes = [100,500]
    # samplingsizes_name = ['100','500','1000','1500','2000','2500','3000']
    samplingsizes_name = list(map(str, samplingsizes))
elif minmaxfullSampSize:
    samplingsizes_name = ['full_row_number']   
nsamplings = np.shape(samplingsizes_name)[0]

temp_resolu = temp_resolus[0]
j_sampsize_name = samplingsizes_name[0]   

# 0.5s time lag
edges_target_names = [['0.5slag_pull2_pull1','0.5slag_pull1_pull2'],
                      ['0.5slag_gaze1_pull1','0.5slag_gaze2_pull2'],
                      ['0.5slag_pull2_gaze1','0.5slag_pull1_gaze2'],]
fromNodesIDs = [[11,10],
                [12,13],
                [11,10],]
toNodesIDs = [[0,1],
              [0,1],
              [2,3]]

#
xplottype = 'succrate' # 'succrate', 'meangazenum'
xplotlabel = 'successful rate' # 'successful rate', 'mean gaze number'
# xplottype = 'meangazenum' # 'succrate', 'meangazenum'
# xplotlabel = 'mean gaze number' # 'successful rate', 'mean gaze number'

n_edges = np.shape(np.array(edges_target_names).flatten())[0]

# figure initiate
fig, axs = plt.subplots(int(np.ceil(n_edges/2)),2)
fig.set_figheight(5*np.ceil(n_edges/2))
fig.set_figwidth(5*2)

#
for i_edge in np.arange(0,n_edges,1):
    #
    edgeweight_mean_forplot_all_dates = np.zeros((ndates_tgt,1))
    edgeweight_shuffled_mean_forplot_all_dates = np.zeros((ndates_tgt,1))
    edgeweight_std_forplot_all_dates = np.zeros((ndates_tgt,1))
    edgeweight_shuffled_std_forplot_all_dates = np.zeros((ndates_tgt,1))
    
    edge_tgt_name = np.array(edges_target_names).flatten()[i_edge]
    fromNodesID = np.array(fromNodesIDs).flatten()[i_edge]
    toNodesID = np.array(toNodesIDs).flatten()[i_edge]
    
    for idate in np.arange(0,ndates_tgt,1):
        idate_name = dates_list_tgt[idate]
        
        weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][idate_name]
        weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][idate_name]
    
        edgeweight_mean_forplot_all_dates[idate] = np.nanmean(weighted_graphs_tgt[:,fromNodesID,toNodesID])
        edgeweight_shuffled_mean_forplot_all_dates[idate] = np.nanmean(weighted_graphs_shuffled_tgt[:,fromNodesID,toNodesID])
        edgeweight_std_forplot_all_dates[idate] = np.nanstd(weighted_graphs_tgt[:,fromNodesID,toNodesID])
        edgeweight_shuffled_std_forplot_all_dates[idate] = np.nanstd(weighted_graphs_shuffled_tgt[:,fromNodesID,toNodesID])
        
      
    # plot 
    if xplottype == 'succrate':
        xxx = succ_rate_all_dates[sorting_tgt_df.index]
    elif xplottype == 'meangazenum':   
        xxx = gazemean_num_all_dates[sorting_tgt_df.index]
    #     
    yyy = edgeweight_mean_forplot_all_dates
    #
    rr_spe,pp_spe = scipy.stats.spearmanr(xxx, yyy)
    slope, intercept, rr_reg, pp_reg, std_err = st.linregress(xxx.astype(float).T[0], yyy.astype(float).T[0])
    #
    axs.flatten()[i_edge].plot(xxx,yyy,'bo',markersize=8)
    axs.flatten()[i_edge].plot(np.array([xxx.min(),xxx.max()]),np.array([xxx.min(),xxx.max()])*slope+intercept,'k-')
    #
    axs.flatten()[i_edge].set_title(edge_tgt_name,fontsize=16)
    axs.flatten()[i_edge].set_ylabel('mean edge weight',fontsize=13)
    axs.flatten()[i_edge].set_ylim([-0.1,1.1])
    #
    if i_edge > int(n_edges-3):
        axs.flatten()[i_edge].set_xlabel(xplotlabel,fontsize=13)
    else:
        axs.flatten()[i_edge].set_xticklabels('')
    #
    # axs.flatten()[i_edge].text(xxx.min(),1.0,'spearman r='+"{:.2f}".format(rr_spe),fontsize=10)
    # axs.flatten()[i_edge].text(xxx.min(),0.9,'spearman p='+"{:.2f}".format(pp_spe),fontsize=10)
    axs.flatten()[i_edge].text(xxx.min(),1.0,'regression r='+"{:.2f}".format(rr_reg),fontsize=10)
    axs.flatten()[i_edge].text(xxx.min(),0.9,'regression p='+"{:.2f}".format(pp_reg),fontsize=10)
    


        
savefigs = 0
if savefigs:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_neuron_bhv_singlecam_wholebodylabels_allsessions_basicEvents/'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    plt.savefig(figsavefolder+"edgeweights_vs_"+xplottype+"_"+animal1_fixedorder[0]+animal2_fixedorder[0]+'.pdf')
    
    


# ## Plots that include all pairs
# ####  mean edge weights of selected edges v.s. other behavioral measures
# ##### only the cooperation days

# In[ ]:





# In[ ]:





# ## Plots that include all pairs
# ####  plot the coorelation between pull time, and social gaze time
# #### pull <-> pull; with animal gaze -> pull; across animal pull -> gaze; with animal pull -> gaze

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





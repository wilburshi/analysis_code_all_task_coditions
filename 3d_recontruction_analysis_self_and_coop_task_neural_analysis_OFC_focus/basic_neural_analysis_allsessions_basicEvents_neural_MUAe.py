#!/usr/bin/env python
# coding: utf-8

# ### Basic neural activity analysis with single camera tracking
# #### analyze based on the neural MUAe
# #### cebra and hmm using muae data
# #### right now it only works on OFCs - only OFCs has MUAe defined right now

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
# from dPCA import dPCA
import string
import warnings
import pickle
import json

from scipy.ndimage import gaussian_filter1d

import os
import glob
import random
from time import time

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


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


# ### function - get the continuous behavioral variables

# In[9]:


from ana_functions.plot_continuous_bhv_var_singlecam_PullStartToPull_variedSection import plot_continuous_bhv_var_singlecam_PullStartToPull_variedSection
from ana_functions.plot_continuous_bhv_var_singlecam_PullStartToPull_variedSection_highbhvDimension_to_lowPCspace import plot_continuous_bhv_var_singlecam_PullStartToPull_variedSection_highbhvDimension_to_lowPCspace
from ana_functions.singlecam_conBhv_from_highDimension_to_PCspace import get_data_for_singlecam_conBhv_from_highDimension_to_PCspace


# ### function - make demo videos with skeleton and inportant vectors

# In[10]:


from ana_functions.tracking_video_singlecam_demo import tracking_video_singlecam_demo
from ana_functions.tracking_video_singlecam_wholebody_demo import tracking_video_singlecam_wholebody_demo
from ana_functions.tracking_video_singlecam_wholebody_withNeuron_demo import tracking_video_singlecam_wholebody_withNeuron_demo
from ana_functions.tracking_video_singlecam_wholebody_withNeuron_sepbhv_demo import tracking_video_singlecam_wholebody_withNeuron_sepbhv_demo
from ana_functions.tracking_frame_singlecam_wholebody_withNeuron_sepbhv_demo import tracking_frame_singlecam_wholebody_withNeuron_sepbhv_demo


# ### function - interval between all behavioral events

# In[11]:


from ana_functions.bhv_events_interval import bhv_events_interval


# ### function - spike analysis

# In[12]:


from ana_functions.spike_analysis_FR_calculation import spike_analysis_FR_calculation
from ana_functions.plot_spike_triggered_singlecam_bhvevent import plot_spike_triggered_singlecam_bhvevent
from ana_functions.plot_bhv_events_aligned_FR import plot_bhv_events_aligned_FR
from ana_functions.plot_strategy_aligned_FR import plot_strategy_aligned_FR


# ### function - PCA projection

# In[13]:


from ana_functions.PCA_around_bhv_events import PCA_around_bhv_events
from ana_functions.PCA_around_bhv_events_video import PCA_around_bhv_events_video
from ana_functions.confidence_ellipse import confidence_ellipse


# ### function - other function

# In[14]:


# function related to clean up the muae
def detect_common_artifact(data, mad_threshold=6):
    pop_signal = np.median(data, axis=0)
    median = np.median(pop_signal)
    mad = np.median(np.abs(pop_signal - median))
    robust_std = mad * 1.4826
    if robust_std == 0:
        return np.zeros(len(pop_signal), dtype=bool), pop_signal
    z = (pop_signal - median) / robust_std
    is_artifact = np.abs(z) > mad_threshold
    return is_artifact, pop_signal


def remove_common_artifact(data, pad_samples=30, mad_threshold=6, max_artifact_frac=0.05):
    is_artifact, pop_signal = detect_common_artifact(data, mad_threshold)

    widened = is_artifact.copy()
    idx_artifact = np.where(is_artifact)[0]
    for idx in idx_artifact:
        lo = max(0, idx - pad_samples)
        hi = min(len(widened), idx + pad_samples + 1)
        widened[lo:hi] = True

    frac_flagged = widened.sum() / len(widened) if len(widened) > 0 else 0

    info = {
        'n_artifact_events': len(idx_artifact),
        'artifact_indices': idx_artifact,
        'frac_session_flagged': frac_flagged,
        'skipped_safety_cap': False,
    }

    if frac_flagged > max_artifact_frac:
        info['skipped_safety_cap'] = True
        return data.copy(), widened, info

    cleaned = data.copy().astype(float)
    for ch in range(data.shape[0]):
        trace = cleaned[ch, :].copy()
        trace[widened] = np.nan
        trace = pd.Series(trace).interpolate(method='linear', limit_direction='both').values
        cleaned[ch, :] = trace

    return cleaned, widened, info


def bin_average_to_video_fps(signal_1000hz, target_fps=30, source_fs=1000):
    """
    Bin-average a signal from source_fs down to target_fps by averaging
    within each target-frame window. Equivalent in spirit to your FR_kernel
    binning, and avoids the aliasing risk of naive decimation.
    signal_1000hz: shape (n_channels, n_samples_at_1000hz)
    """
    samples_per_bin = source_fs / target_fps  # e.g. 1000/30 ≈ 33.33
    n_samples = signal_1000hz.shape[1]
    n_bins = int(np.floor(n_samples / samples_per_bin))

    binned = np.zeros((signal_1000hz.shape[0], n_bins))
    for b in range(n_bins):
        start = int(round(b * samples_per_bin))
        end = int(round((b + 1) * samples_per_bin))
        end = min(end, n_samples)
        binned[:, b] = signal_1000hz[:, start:end].mean(axis=1)

    bin_times = (np.arange(n_bins) + 0.5) * samples_per_bin / source_fs  # bin-center time, in seconds
    return binned, bin_times


# ## Analyze each session

# ### prepare the basic behavioral data (especially the time stamps for each bhv events)

# In[15]:


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
nframes = 0.5*30 # second*30fps
# nframes = 45*30 # second*30fps

# re-analyze the video or not
reanalyze_video = 0
redo_anystep = 0

# do OFC sessions or DLPFC sessions
do_OFC = 1
do_DLPFC  = 0   # DLPFC does not have muae yet
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


# dodson ginger for dlpfc (dmpfc)
# dodson selene for ofc
if 1:
    if do_DLPFC:
        neural_record_conditions = [
                        '20240531_Dodson_MC', '20240603_Dodson_MC_and_SR', '20240603_Dodson_MC_and_SR',
                        '20240604_Dodson_MC', '20240605_Dodson_MC_and_SR', '20240605_Dodson_MC_and_SR',

                        '20240606_Dodson_MC_and_SR', '20240606_Dodson_MC_and_SR', '20240607_Dodson_SR',
                        '20240610_Dodson_MC', '20240611_Dodson_SR', '20240612_Dodson_MC',

                        '20240613_Dodson_SR', '20240620_Dodson_SR', '20240719_Dodson_MC',
                        '20250129_Dodson_MC', '20250130_Dodson_SR', '20250131_Dodson_MC',

                        '20250210_Dodson_SR_withKoala', '20250211_Dodson_MC_withKoala',
                        '20250212_Dodson_SR_withKoala', '20250214_Dodson_MC_withKoala',
                        '20250217_Dodson_SR_withKoala', '20250218_Dodson_MC_withKoala',

                        '20250219_Dodson_SR_withKoala', '20250220_Dodson_MC_withKoala',
                        '20250224_Dodson_KoalaAL_withKoala', '20250226_Dodson_MC_withKoala',
                        '20250227_Dodson_KoalaAL_withKoala', '20250228_Dodson_DodsonAL_withKoala',

                        '20250304_Dodson_DodsonAL_withKoala', '20250305_Dodson_MC_withKoala',
                        '20250306_Dodson_KoalaAL_withKoala', '20250307_Dodson_DodsonAL_withKoala',
                        '20250310_Dodson_MC_withKoala', '20250312_Dodson_NV_withKoala',

                        '20250313_Dodson_NV_withKoala', '20250314_Dodson_NV_withKoala',
                        '20250401_Dodson_MC_withKanga', '20250402_Dodson_MC_withKanga',
                        '20250403_Dodson_MC_withKanga', '20250404_Dodson_SR_withKanga',

                        '20250407_Dodson_SR_withKanga', '20250408_Dodson_SR_withKanga',
                        '20250409_Dodson_MC_withKanga', '20250415_Dodson_MC_withKanga',
                        # '20250416_Dodson_SR_withKanga',
                        '20250417_Dodson_MC_withKanga',

                        '20250418_Dodson_SR_withKanga', '20250421_Dodson_SR_withKanga',
                        '20250422_Dodson_MC_withKanga', '20250422_Dodson_SR_withKanga',
                        '20250423_Dodson_MC_withKanga', '20250423_Dodson_SR_withKanga',

                        '20250424_Dodson_NV_withKanga', '20250424_Dodson_MC_withKanga',
                        '20250424_Dodson_SR_withKanga', '20250425_Dodson_NV_withKanga',
                        '20250425_Dodson_SR_withKanga', '20250428_Dodson_NV_withKanga',

                        '20250428_Dodson_MC_withKanga', '20250428_Dodson_SR_withKanga',
                        '20250429_Dodson_NV_withKanga', '20250429_Dodson_MC_withKanga',
                        '20250429_Dodson_SR_withKanga', '20250430_Dodson_NV_withKanga',

                        '20250430_Dodson_MC_withKanga', '20250430_Dodson_SR_withKanga',
                    ]
        task_conditions = [
                        'MC', 'MC', 'SR', 'MC', 'MC', 'SR',
                        'MC', 'SR', 'SR', 'MC', 'SR', 'MC',
                        'SR', 'SR', 'MC', 'MC_withGingerNew', 'SR_withGingerNew', 'MC_withGingerNew',

                        'SR_withKoala', 'MC_withKoala', 'SR_withKoala',
                        'MC_withKoala', 'SR_withKoala', 'MC_withKoala',

                        'SR_withKoala', 'MC_withKoala', 'MC_KoalaAuto_withKoala',
                        'MC_withKoala', 'MC_KoalaAuto_withKoala', 'MC_DodsonAuto_withKoala',

                        'MC_DodsonAuto_withKoala', 'MC_withKoala', 'MC_KoalaAuto_withKoala',
                        'MC_DodsonAuto_withKoala', 'MC_withKoala', 'NV_withKoala',

                        'NV_withKoala', 'NV_withKoala', 'MC_withKanga',
                        'MC_withKanga', 'MC_withKanga', 'SR_withKanga',

                        'SR_withKanga', 'SR_withKanga', 'MC_withKanga',
                        'MC_withKanga',
                        # 'SR_withKanga',
                        'MC_withKanga', 'SR_withKanga', 'SR_withKanga', 'MC_withKanga', 'SR_withKanga',

                        'MC_withKanga', 'SR_withKanga', 'NV_withKanga',
                        'MC_withKanga', 'SR_withKanga', 'NV_withKanga',

                        'SR_withKanga', 'NV_withKanga', 'MC_withKanga',
                        'SR_withKanga', 'NV_withKanga', 'MC_withKanga',

                        'SR_withKanga', 'NV_withKanga', 'MC_withKanga', 'SR_withKanga',
                    ]
        dates_list = [
                        '20240531', '20240603_MC', '20240603_SR', '20240604', '20240605_MC', '20240605_SR',
                        '20240606_MC', '20240606_SR', '20240607', '20240610_MC', '20240611', '20240612',

                        '20240613', '20240620', '20240719',
                        '20250129', '20250130', '20250131',

                        '20250210', '20250211', '20250212', '20250214', '20250217', '20250218',
                        '20250219', '20250220', '20250224', '20250226', '20250227', '20250228',

                        '20250304', '20250305', '20250306', '20250307', '20250310', '20250312',
                        '20250313', '20250314',

                        '20250401', '20250402', '20250403', '20250404', '20250407', '20250408',
                        '20250409',

                        '20250415',
                        # '20250416',
                        '20250417', '20250418', '20250421', '20250422', '20250422_SR',

                        '20250423', '20250423_SR', '20250424', '20250424_MC', '20250424_SR',
                        '20250425', '20250425_SR',

                        '20250428_NV', '20250428_MC', '20250428_SR',
                        '20250429_NV', '20250429_MC', '20250429_SR',

                        '20250430_NV', '20250430_MC', '20250430_SR',
                    ]
        videodates_list = [
                        '20240531', '20240603', '20240603', '20240604', '20240605', '20240605',
                        '20240606', '20240606', '20240607', '20240610_MC', '20240611', '20240612',

                        '20240613', '20240620', '20240719',
                        '20250129', '20250130', '20250131',

                        '20250210', '20250211', '20250212', '20250214', '20250217', '20250218',
                        '20250219', '20250220', '20250224', '20250226', '20250227', '20250228',

                        '20250304', '20250305', '20250306', '20250307', '20250310', '20250312',
                        '20250313', '20250314',

                        '20250401', '20250402', '20250403', '20250404', '20250407', '20250408',
                        '20250409',

                        '20250415',
                        # '20250416',
                        '20250417', '20250418', '20250421', '20250422', '20250422_SR',

                        '20250423', '20250423_SR', '20250424', '20250424_MC', '20250424_SR',
                        '20250425', '20250425_SR',

                        '20250428_NV', '20250428_MC', '20250428_SR',
                        '20250429_NV', '20250429_MC', '20250429_SR',

                        '20250430_NV', '20250430_MC', '20250430_SR',
                    ] # to deal with the sessions that MC and SR were in the same session
        session_start_times = [
                        0.00, 340, 340, 72.0, 60.1, 60.1,
                        82.2, 82.2, 35.8, 0.00, 29.2, 35.8,

                        62.5, 71.5, 54.4,
                        0.00, 0.00, 0.00,

                        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,

                        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                        0.00, 0.00,

                        0.00, 0.00, 73.5, 0.00, 76.1, 81.5,
                        0.00,

                        363,
                        # 0.00,
                        79.0, 162.6, 231.9, 109, 0.00,

                        0.00, 0.00, 0.00, 0.00, 0.00,
                        0.00, 93.0,

                        0.00, 0.00, 0.00,
                        0.00, 0.00, 0.00,

                        0.00, 274.4, 0.00,
                    ]
        
        kilosortvers = list((np.ones(np.shape(dates_list))*4).astype(int))
        
        trig_channelnames = [ 'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0', #'Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                             
                              ]
        animal1_fixedorders = ['dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',# 'dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson',
                              ]
        recordedanimals = animal1_fixedorders 
        animal2_fixedorders = ['ginger','ginger','ginger','ginger','ginger','ginger','ginger','ginger','ginger',
                               'ginger','ginger','ginger','ginger','ginger','ginger','gingerNew','gingerNew','gingerNew',
                               'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala',
                               'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala',
                               'koala', 'koala', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', # 'kanga',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                              ]

        animal1_filenames = ["Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",# "Dodson",
                             'Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson',
                             'Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson',
                             'Dodson','Dodson','Dodson','Dodson','Dodson',
                            ]
        animal2_filenames = ["Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger",
                             "Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger",
                             "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala",
                             "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala",
                             "Koala", "Koala", "Kanga", "Kanga", "Kanga", "Kanga", "Kanga", "Kanga", # "Kanga",
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                            ]
        
    elif do_OFC:
        neural_record_conditions = [
                        '20260219_Dodson_OFC_33turns_SRwithSelene',  '20260303_Dodson_OFC_33turns_1sMCwithSelene',
                        '20260303_Dodson_OFC_33turns_SRwithSelene',  '20260304_Dodson_OFC_33turns_1sMCwithSelene',
                        '20260304_Dodson_OFC_33turns_SRwithSelene',  '20260305_Dodson_OFC_33turns_1sMCwithSelene',
                        '20260309_Dodson_OFC_33turns_1sMCwithKanga', '20260309_Dodson_OFC_33turns_SRwithKanga',
                        '20260310_Dodson_OFC_33turns_1sMCwithKanga', '20260310_Dodson_OFC_33turns_SRwithKanga',
                        '20260311_Dodson_OFC_33turns_1sMCwithKanga', '20260311_Dodson_OFC_33turns_SRwithKanga',
                        '20260312_Dodson_OFC_33turns_1sMCwithKanga', '20260312_Dodson_OFC_33turns_SRwithKanga',
                        '20260313_Dodson_OFC_33turns_1sMCwithKanga', '20260313_Dodson_OFC_33turns_SRwithKanga',
            
                        '20260317_Dodson_OFC_33turns_1sMCwithKanga', '20260317_Dodson_OFC_33turns_SRwithKanga',
                        '20260318_Dodson_OFC_33turns_1sMCwithKanga', # '20260318_Dodson_OFC_33turns_SRwithKanga',
                        '20260319_Dodson_OFC_33turns_1sMCwithKanga', '20260319_Dodson_OFC_33turns_SRwithKanga',
                        '20260323_Dodson_OFC_32turns_1sMCwithKanga', '20260323_Dodson_OFC_32turns_SRwithKanga',
                        '20260324_Dodson_OFC_32turns_1sMCwithKanga', '20260324_Dodson_OFC_32turns_SRwithKanga',
                    ]
        task_conditions = [
                        'SR', 'MC', 'SR', 'MC', 'SR', 'MC', 
                        'MC', 'SR', 'MC', 'SR', 'MC', 'SR',
                        'MC', 'SR', 'MC', 'SR', 'MC', 'SR',
                        'MC',       'MC', 'SR', 'MC', 'SR',
                        'MC', 'SR', 
                    ]
        dates_list = [
                        '20260219', '20260303',    '20260303_SR', '20260304',    '20260304_SR', '20260305', 
                        '20260309', '20260309_SR', '20260310',    '20260310_SR', '20260311',    '20260311_SR',
                        '20260312', '20260312_SR', '20260313',    '20260313_SR', '20260317',    '20260317_SR',
                        '20260318',                '20260319',    '20260319_SR', '20260323',    '20260323_SR',
                        '20260324', '20260324_SR',
                    ]
        videodates_list = [
                        '20260219', '20260303',    '20260303_SR', '20260304',    '20260304_SR', '20260305', 
                        '20260309', '20260309_SR', '20260310',    '20260310_SR', '20260311',    '20260311_SR',
                        '20260312', '20260312_SR', '20260313',    '20260313_SR', '20260317',    '20260317_SR',
                        '20260318',                '20260319',    '20260319_SR', '20260323',    '20260323_SR',
                        '20260324', '20260324_SR',
                    ] 
        
        session_start_times = [
                        188.7, 0.00, 0.00, 0.00, 0.00,  0.00, 
                         0.00, 0.00, 0.00, 0.00, 0.00,  0.00, 
                         0.00, 0.00, 0.00, 0.00, 0.00,  0.00, 
                         0.00,       0.00, 0.00, 0.00, 129.5,
                        116.2, 0.00,
                    ]
        
        kilosortvers = list((np.ones(np.shape(dates_list))*4).astype(int))
        
        trig_channelnames = [ 'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9',
                              'Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9',
                              'Dev1/ai9',           'Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9',
                              'Dev1/ai9','Dev1/ai9',
                              ]
        animal1_fixedorders = ['dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson',         'dodson','dodson','dodson','dodson',
                               'dodson','dodson',
                              ]
        recordedanimals = animal1_fixedorders 
        animal2_fixedorders = ['selene','selene','selene','selene','selene','selene',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga',          'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', 'kanga',
                              ]

        animal1_filenames = ["Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson",         "Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson",
                            ]
        animal2_filenames = ['Selene','Selene','Selene','Selene','Selene','Selene',
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga',          'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga', 'Kanga',
                            ]


    
# dannon kanga
if 1:
    if do_DLPFC:
        neural_record_conditions = [
                        '20240508_Kanga_SR', '20240509_Kanga_MC', '20240513_Kanga_MC',
                        '20240514_Kanga_SR', '20240523_Kanga_MC', '20240524_Kanga_SR',

                        '20240606_Kanga_MC', '20240613_Kanga_MC_DannonAuto',
                        '20240614_Kanga_MC_DannonAuto', '20240617_Kanga_MC_DannonAuto',
                        '20240618_Kanga_MC_KangaAuto', '20240619_Kanga_MC_KangaAuto',

                        '20240620_Kanga_MC_KangaAuto', '20240621_1_Kanga_NoVis',
                        '20240624_Kanga_NoVis', '20240626_Kanga_NoVis',

                        '20240808_Kanga_MC_withGinger', '20240809_Kanga_MC_withGinger',
                        '20240812_Kanga_MC_withGinger', '20240813_Kanga_MC_withKoala',
                        '20240814_Kanga_MC_withKoala', '20240815_Kanga_MC_withKoala',

                        '20240819_Kanga_MC_withVermelho', '20240821_Kanga_MC_withVermelho',
                        '20240822_Kanga_MC_withVermelho',

                        '20250415_Kanga_MC_withDodson', '20250416_Kanga_SR_withDodson',
                        '20250417_Kanga_MC_withDodson', '20250418_Kanga_SR_withDodson',
                        '20250421_Kanga_SR_withDodson', '20250422_Kanga_MC_withDodson',

                        '20250422_Kanga_SR_withDodson', '20250423_Kanga_MC_withDodson',
                        '20250423_Kanga_SR_withDodson',

                        '20250424_Kanga_NV_withDodson', '20250424_Kanga_MC_withDodson',
                        '20250424_Kanga_SR_withDodson', '20250425_Kanga_NV_withDodson',
                        '20250425_Kanga_SR_withDodson',

                        '20250428_Kanga_NV_withDodson', '20250428_Kanga_MC_withDodson',
                        '20250428_Kanga_SR_withDodson', '20250429_Kanga_NV_withDodson',
                        '20250429_Kanga_MC_withDodson', '20250429_Kanga_SR_withDodson',

                        '20250430_Kanga_NV_withDodson', '20250430_Kanga_MC_withDodson',
                        '20250430_Kanga_SR_withDodson',
                    ]
        dates_list = [
                        "20240508", "20240509", "20240513", "20240514", "20240523", "20240524",
                        "20240606", "20240613", "20240614", "20240617", "20240618", "20240619",
                        "20240620", "20240621_1", "20240624", "20240626",

                        "20240808", "20240809", "20240812", "20240813", "20240814", "20240815",
                        "20240819", "20240821", "20240822",

                        "20250415", "20250416", "20250417", "20250418", "20250421", "20250422",
                        "20250422_SR",

                        '20250423', '20250423_SR', '20250424', '20250424_MC', '20250424_SR',
                        '20250425', '20250425_SR',

                        '20250428_NV', '20250428_MC', '20250428_SR',
                        '20250429_NV', '20250429_MC', '20250429_SR',

                        '20250430_NV', '20250430_MC', '20250430_SR',
                    ]
        videodates_list = dates_list
        task_conditions = [
                        'SR', 'MC', 'MC', 'SR', 'MC', 'SR',
                        'MC', 'MC_DannonAuto', 'MC_DannonAuto', 'MC_DannonAuto',
                        'MC_KangaAuto', 'MC_KangaAuto',

                        'MC_KangaAuto', 'NV', 'NV', 'NV',

                        'MC_withGinger', 'MC_withGinger', 'MC_withGinger',
                        'MC_withKoala', 'MC_withKoala', 'MC_withKoala',

                        'MC_withVermelho', 'MC_withVermelho', 'MC_withVermelho',

                        'MC_withDodson', 'SR_withDodson', 'MC_withDodson',
                        'SR_withDodson', 'SR_withDodson', 'MC_withDodson',

                        'SR_withDodson', 'MC_withDodson', 'SR_withDodson',

                        'NV_withDodson', 'MC_withDodson', 'SR_withDodson',
                        'NV_withDodson', 'SR_withDodson',

                        'NV_withDodson', 'MC_withDodson', 'SR_withDodson',
                        'NV_withDodson', 'MC_withDodson', 'SR_withDodson',

                        'NV_withDodson', 'MC_withDodson', 'SR_withDodson',
                    ]
        session_start_times = [
                        0.00, 36.0, 69.5, 0.00, 62.0, 0.00,
                        89.0, 0.00, 0.00, 0.00, 165.8, 96.0,
            
                        0.00, 0.00, 0.00, 48.0,
                        59.2, 49.5, 40.0, 50.0, 0.00, 69.8,
            
                        85.0, 212.9, 68.5,
                        363, 0.00, 79.0, 162.6, 231.9, 109,
            
                        0.00,
                        0.00, 0.00, 0.00, 0.00, 0.00,

                        0.00, 93.0,

                        0.00, 0.00, 0.00, 0.00, 0.00,
                        0.00,

                        0.00, 274.4, 0.00,
                    ]
        
        kilosortvers = list((np.ones(np.shape(dates_list))*4).astype(int))
        
        trig_channelnames = ['Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                             'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                             'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                             'Dev1/ai0','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai0','Dev1/ai0',
                             'Dev1/ai0','Dev1/ai0','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9',
                             'Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9',
                              ]
        
        animal1_fixedorders = ['dannon','dannon','dannon','dannon','dannon','dannon','dannon','dannon',
                               'dannon','dannon','dannon','dannon','dannon','dannon','dannon','dannon',
                               'ginger','ginger','ginger','koala','koala','koala','vermelho','vermelho',
                               'vermelho','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                              ]
        animal2_fixedorders = ['kanga','kanga','kanga','kanga','kanga','kanga','kanga','kanga',
                               'kanga','kanga','kanga','kanga','kanga','kanga','kanga','kanga',
                               'kanga','kanga','kanga','kanga','kanga','kanga','kanga','kanga',
                               'kanga','kanga','kanga','kanga','kanga','kanga','kanga','kanga',
                               'kanga','kanga','kanga','kanga','kanga','kanga','kanga','kanga',
                               'kanga','kanga','kanga','kanga','kanga','kanga','kanga','kanga',
                              ]
        recordedanimals = animal2_fixedorders

        animal1_filenames = ["Dannon","Dannon","Dannon","Dannon","Dannon","Dannon","Dannon","Dannon",
                             "Dannon","Dannon","Dannon","Dannon","Dannon","Dannon","Dannon","Dannon",
                             "Ginger","Ginger","Ginger", "Kanga", "Kanga", "Kanga", "Kanga", "Kanga",
                              "Kanga","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             
                            ]
        animal2_filenames = ["Kanga","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga",
                             "Kanga","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga",
                             "Kanga","Kanga","Kanga","Koala","Koala","Koala","Vermelho","Vermelho",
                             "Vermelho","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga",
                             "Kanga","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga",
                             "Kanga","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga","Kanga",
                            ]
        
    elif do_OFC:
        neural_record_conditions = [
                        '20260309_Kanga_OFC_31turns_1sMCwithDodson', '20260309_Kanga_OFC_31turns_SRwithDodson',
                        '20260310_Kanga_OFC_31turns_1sMCwithDodson', '20260310_Kanga_OFC_31turns_SRwithDodson',
                        '20260311_Kanga_OFC_31turns_1sMCwithDodson', '20260311_Kanga_OFC_31turns_SRwithDodson',
                        '20260312_Kanga_OFC_31turns_1sMCwithDodson', '20260312_Kanga_OFC_31turns_SRwithDodson',
                        '20260313_Kanga_OFC_31turns_1sMCwithDodson', '20260313_Kanga_OFC_31turns_SRwithDodson',
                        '20260317_Kanga_OFC_31turns_1sMCwithDodson', '20260317_Kanga_OFC_31turns_SRwithDodson',
                        '20260318_Kanga_OFC_31turns_1sMCwithDodson', '20260318_Kanga_OFC_31turns_SRwithDodson',
                        '20260319_Kanga_OFC_31turns_1sMCwithDodson', '20260319_Kanga_OFC_31turns_SRwithDodson',
                        '20260323_Kanga_OFC_31turns_1sMCwithDodson', '20260323_Kanga_OFC_31turns_SRwithDodson',
                        '20260324_Kanga_OFC_31turns_1sMCwithDodson', '20260324_Kanga_OFC_31turns_SRwithDodson',
                        '20260326_Kanga_OFC_31turns_1sMCwithDodson', '20260326_Kanga_OFC_31turns_SRwithDodson',
                        '20260330_Kanga_OFC_30dot5turns_SRwithDodson',   '20260331_Kanga_OFC_30dot5turns_1sMCwithDodson',
                        '20260403_Kanga_OFC_30dot5turns_1sMCwithDodson', '20260406_Kanga_OFC_30dot5turns_1sMCwithDodson',
                        '20260406_Kanga_OFC_30dot5turns_SRwithDodson',
            
                        # dannon kanga
                        '20260409_Kanga_OFC_30dot5turns_1sMCwithDannon', '20260410_Kanga_OFC_30dot5turns_1sMCwithDannon',
                        '20260410_Kanga_OFC_30dot5turns_SRwithDannon',   '20260413_Kanga_OFC_30dot5turns_1sMCwithDannon',
                        '20260421_Kanga_OFC_30dot5turns_1sMCwithDannon',
                    ]
        task_conditions = [
                        'MC', 'SR', 'MC', 'SR', 'MC', 'SR',
                        'MC', 'SR', 'MC', 'SR', 'MC', 'SR',
                        'MC', 'SR', 'MC', 'SR', 'MC', 'SR',
                        'MC', 'SR', 'MC', 'SR', 'SR', 'MC',
                        'MC', 'MC', 'SR',
            
                        'MC', 'MC', 'SR', 'MC', 'MC',
                    ]
        dates_list = [
                        '20260309', '20260309_SR', '20260310', '20260310_SR', '20260311',    '20260311_SR',
                        '20260312', '20260312_SR', '20260313', '20260313_SR', '20260317',    '20260317_SR',
                        '20260318', '20260318_SR', '20260319', '20260319_SR', '20260323',    '20260323_SR',
                        '20260324', '20260324_SR', '20260326', '20260326_SR', '20260330_SR', '20260331',
                        '20260403', '20260406', '20260406_SR',
                    
                        '20260409', '20260410', '20260410_SR', '20260413',    '20260421',
                    ]
        videodates_list = [
                        '20260309', '20260309_SR', '20260310', '20260310_SR', '20260311',    '20260311_SR',
                        '20260312', '20260312_SR', '20260313', '20260313_SR', '20260317',    '20260317_SR',
                        '20260318', '20260318_SR', '20260319', '20260319_SR', '20260323',    '20260323_SR',
                        '20260324', '20260324_SR', '20260326', '20260326_SR', '20260330_SR', '20260331',
                        '20260403', '20260406', '20260406_SR', 
            
                        '20260409', '20260410', '20260410_SR', '20260413',    '20260421',
                    ] 
        
        session_start_times = [
                         0.00, 0.00, 0.00, 0.00, 0.00,  0.00, 
                         0.00, 0.00, 0.00, 0.00, 0.00,  0.00, 
                         0.00, 0.00, 0.00, 0.00, 0.00, 129.5,
                        116.2, 0.00, 0.00, 0.00, 0.00, 49.50,
                         0.00, 0.00, 0.00,
            
                         0.00, 0.00, 0.00, 0.00, 0.00,
                    ]
        
        kilosortvers = list((np.ones(np.shape(dates_list))*4).astype(int))
        
        trig_channelnames = [ 'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0', 
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0', 
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0', 
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0',
                             
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              ]
        animal1_fixedorders = [
                               'dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson', 
                               'dodson','dodson','dodson',
             
                               'dannon','dannon','dannon','dannon','dannon',
                              ]
        animal2_fixedorders = [
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 
                               'kanga', 'kanga', 'kanga', 
            
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                              ]
        recordedanimals = animal2_fixedorders 

        animal1_filenames = [
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson",
            
                             "Dannon","Dannon","Dannon","Dannon","Dannon",
                            ]
        animal2_filenames = [
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 
                             'Kanga', 'Kanga', 'Kanga', 
            
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 
                            ]
    

    
# a test case
if 0:
    if do_DLPFC:
        if 0: # kanga example
            neural_record_conditions = ['20240606_Kanga_MC']
            dates_list = ["20240606"]
            videodates_list = dates_list
            task_conditions = ['MC']
            session_start_times = [89] # in second
            kilosortvers = [4]
            trig_channelnames = ['Dev1/ai0']
            animal1_fixedorders = ['dannon']
            animal2_fixedorders = ['kanga']
            recordedanimals = animal2_fixedorders
            animal1_filenames = ["Dannon"]
            animal2_filenames = ["Kanga"]
        if 0: # dodson example 
            neural_record_conditions = ['20250415_Dodson_MC_withKanga']
            dates_list = ["20250415"]
            videodates_list = dates_list
            task_conditions = ['MC_withKanga']
            session_start_times = [363] # in second
            kilosortvers = [4]
            trig_channelnames = ['Dev1/ai0']
            animal1_fixedorders = ['dodson']
            recordedanimals = animal1_fixedorders
            animal2_fixedorders = ['kanga']
            animal1_filenames = ["Dodson"]
            animal2_filenames = ["Kanga"]
    #
    elif do_OFC:
        if 1: # kanga example
            neural_record_conditions = [ '20260309_Kanga_OFC_31turns_1sMCwithDodson',]
            task_conditions = [ 'MC', ]
            dates_list = [ '20260309', ]
            videodates_list = [ '20260309', ] 
            session_start_times = [0.00, ]
            kilosortvers = list((np.ones(np.shape(dates_list))*4).astype(int))
            trig_channelnames = [ 'Dev1/ai0',]
            animal1_fixedorders = ['dodson',]
            animal2_fixedorders = [ 'kanga',]
            recordedanimals = animal2_fixedorders 
            animal1_filenames = [ "Dodson",]
            animal2_filenames = ['Kanga', ]
        if 1: # dodson example
            neural_record_conditions = [ '20260309_Dodson_OFC_33turns_1sMCwithKanga',]
            task_conditions = [ 'MC', ]
            dates_list = [ '20260309', ]
            videodates_list = [ '20260309', ] 
            session_start_times = [0.00, ]
            kilosortvers = list((np.ones(np.shape(dates_list))*4).astype(int))
            trig_channelnames = [ 'Dev1/ai9',]
            animal1_fixedorders = ['dodson',]
            animal2_fixedorders = [ 'kanga',]
            recordedanimals = animal1_fixedorders 
            animal1_filenames = [ "Dodson",]
            animal2_filenames = ['Kanga', ]
    
    

ndates = np.shape(dates_list)[0]

session_start_frames = session_start_times * fps # fps is 30Hz

totalsess_time = 600

# video tracking results info
animalnames_videotrack = ['dodson','scorch'] # does not really mean dodson and scorch, instead, indicate animal1 and animal2
bodypartnames_videotrack = ['rightTuft','whiteBlaze','leftTuft','rightEye','leftEye','mouth']


# which camera to analyzed
cameraID = 'camera-2'
cameraID_short = 'cam2'

considerlevertube = 1
considertubeonly = 0

# location of levers and tubes for camera 2
# # camera 1
# lever_locs_camI = {'dodson':np.array([645,600]),'scorch':np.array([425,435])}
# tube_locs_camI  = {'dodson':np.array([1350,630]),'scorch':np.array([555,345])}
# # camera 2
# # location of the estimiated middle of the box
# lever_locs_camI = {'dodson':np.array([1325,615]),'scorch':np.array([560,615])}
# # location of the estimated lever
lever_locs_camI = {'dodson':np.array([1335,715]),'scorch':np.array([550,715])}
tube_locs_camI  = {'dodson':np.array([1550,515]),'scorch':np.array([350,515])}
# # old
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
totalsessiontime_all_dates = np.zeros((ndates,1))

owgaze1_num_all_dates = np.zeros((ndates,1))
owgaze2_num_all_dates = np.zeros((ndates,1))
mtgaze1_num_all_dates = np.zeros((ndates,1))
mtgaze2_num_all_dates = np.zeros((ndates,1))
pull1_num_all_dates = np.zeros((ndates,1))
pull2_num_all_dates = np.zeros((ndates,1))


# where to save the summarizing data
data_saved_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/3d_recontruction_analysis_self_and_coop_task_data_saved/'

# neural data folder
if not do_OFC:
    neural_data_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/Marmoset_neural_recording/'
elif do_OFC:
    neural_data_folder = '/gpfs/marilyn/pi/nandy/Marmoset_neural_recording/'

    


# In[16]:


print(np.shape(neural_record_conditions))
print(np.shape(task_conditions))
print(np.shape(dates_list))
print(np.shape(videodates_list)) 
print(np.shape(session_start_times))

print(np.shape(kilosortvers))

print(np.shape(trig_channelnames))
print(np.shape(animal1_fixedorders)) 
print(np.shape(recordedanimals))
print(np.shape(animal2_fixedorders))

print(np.shape(animal1_filenames))
print(np.shape(animal2_filenames))  


# In[17]:


# basic behavior analysis (define time stamps for each bhv events, etc)

try:
    if redo_anystep:
        dummy
    
    # load saved data
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'
    
    with open(data_saved_subfolder+'/owgaze1_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        owgaze1_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/owgaze2_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        owgaze2_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/mtgaze1_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        mtgaze1_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/mtgaze2_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        mtgaze2_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/pull1_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        pull1_num_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/pull2_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        pull2_num_all_dates = pickle.load(f)

    with open(data_saved_subfolder+'/tasktypes_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        tasktypes_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/coopthres_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        coopthres_all_dates = pickle.load(f)
    with open(data_saved_subfolder+'/succ_rate_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        succ_rate_all_dates = pickle.load(f)
   
    with open(data_saved_subfolder+'/trialnum_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        trialnum_all_dates = pickle.load(f)
        
    if do_OFC:
        with open(data_saved_subfolder+'/totalsessiontime_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
            totalsessiontime_all_dates = pickle.load(f)
        
    print('all data from all dates are loaded')

except:

    print('analyze all dates')

    for idate in np.arange(0,ndates,1):
    
        date_tgt = dates_list[idate]
        videodate_tgt = videodates_list[idate]
        
        neural_record_condition = neural_record_conditions[idate]
        
        session_start_time = session_start_times[idate]
        
        kilosortver = kilosortvers[idate]
        
        trig_channelname = trig_channelnames[idate]
        
        animal1_filename = animal1_filenames[idate]
        animal2_filename = animal2_filenames[idate]
        
        animal1_fixedorder = [animal1_fixedorders[idate]]
        animal2_fixedorder = [animal2_fixedorders[idate]]
        
        recordedanimal = recordedanimals[idate]

        # folder and file path
        if not do_OFC:
            camera12_analyzed_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/test_video_cooperative_task_DLPFCs_3d/"+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera12/"
            camera23_analyzed_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/test_video_cooperative_task_DLPFCs_3d/"+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera23/"
        elif do_OFC:
            camera12_analyzed_path = "/gpfs/marilyn/pi/nandy/VideoTracker_SocialInter/test_video_cooperative_task_OFCs_3d/"+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera12/"
            camera23_analyzed_path = "/gpfs/marilyn/pi/nandy/VideoTracker_SocialInter/test_video_cooperative_task_OFCs_3d/"+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera23/"
        # 
        try: 
            singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_middle_camera_withHeadchamberFeb28shuffle1_167500"
            bodyparts_camI_camIJ = camera12_analyzed_path+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"
            if not os.path.exists(bodyparts_camI_camIJ):
                singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_middle_camera_withHeadchamberFeb28shuffle1_80000"
                bodyparts_camI_camIJ = camera12_analyzed_path+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"
            if not os.path.exists(bodyparts_camI_camIJ):
                singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000"
                bodyparts_camI_camIJ = camera12_analyzed_path+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"                
            # get the bodypart data from files
            bodyparts_locs_camI = body_part_locs_singlecam(bodyparts_camI_camIJ,singlecam_ana_type,animalnames_videotrack,bodypartnames_videotrack,videodate_tgt)
            video_file_original = camera12_analyzed_path+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+".mp4"
        except:
            singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_middle_camera_withHeadchamberFeb28shuffle1_167500"
            bodyparts_camI_camIJ = camera23_analyzed_path+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"
            if not os.path.exists(bodyparts_camI_camIJ):
                singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_middle_camera_withHeadchamberFeb28shuffle1_80000"
                bodyparts_camI_camIJ = camera23_analyzed_path+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"
            if not os.path.exists(bodyparts_camI_camIJ):
                singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000"
                bodyparts_camI_camIJ = camera23_analyzed_path+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+singlecam_ana_type+"_el_filtered.h5"
            
            # get the bodypart data from files
            bodyparts_locs_camI = body_part_locs_singlecam(bodyparts_camI_camIJ,singlecam_ana_type,animalnames_videotrack,bodypartnames_videotrack,videodate_tgt)
            video_file_original = camera23_analyzed_path+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_"+cameraID+".mp4"        
        
        # load behavioral results
        if not do_OFC:
            try:
                bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_cooperation_task_DLPFCs/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
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
                bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_cooperation_task_DLPFCs/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
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
        
        elif do_OFC:
            try:
                bhv_data_path = "/gpfs/marilyn/pi/nandy/VideoTracker_SocialInter/marmoset_tracking_bhv_data_cooperation_task_OFCs/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
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
                bhv_data_path = "/gpfs/marilyn/pi/nandy/VideoTracker_SocialInter/marmoset_tracking_bhv_data_cooperation_task_OFCs/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
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
        # for itrial in np.arange(0,np.max(trial_record['trial_number']),1):
        for itrial in trial_record['trial_number']:
            # trial_record_clean.loc[itrial] = trial_record[trial_record['trial_number']==itrial+1].iloc[[0]]
            trial_record_clean = trial_record_clean.append(trial_record[trial_record['trial_number']==itrial].iloc[[0]])
        trial_record_clean = trial_record_clean.reset_index(drop = True)

        # change bhv_data time to the absolute time
        time_points_new = pd.DataFrame(np.zeros(np.shape(bhv_data)[0]),columns=["time_points_new"])
        # for itrial in np.arange(0,np.max(trial_record_clean['trial_number']),1):
        for itrial in np.arange(0,np.shape(trial_record_clean)[0],1):
            # ind = bhv_data["trial_number"]==itrial+1
            ind = bhv_data["trial_number"]==trial_record_clean['trial_number'][itrial]
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
        if np.isin(animal1,animal1_fixedorder):
            pull1_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==1) 
            pull2_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==2)
        else:
            pull1_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==2) 
            pull2_num_all_dates[idate] = np.sum(bhv_data['behavior_events']==1)

        
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
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_key_locations.pkl', 'rb') as f:
                output_key_locations = pickle.load(f)
                
        except:   
            print('analyze social gaze with '+cameraID+' only of '+date_tgt)
            # get social gaze information 
            output_look_ornot, output_allvectors, output_allangles = find_socialgaze_timepoint_singlecam_wholebody(bodyparts_locs_camI,lever_locs_camI,tube_locs_camI,
                                                                                                                   considerlevertube,considertubeonly,sqr_thres_tubelever,
                                                                                                                   sqr_thres_face,sqr_thres_body)
            
            output_key_locations = find_socialgaze_timepoint_singlecam_wholebody_2(bodyparts_locs_camI,lever_locs_camI,tube_locs_camI,considerlevertube)
            
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
        # change the unit to second and align to the start of the session
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
        # 
        # mostly just for the sessions in which MC and SR are in the same session 
        firstpulltime = np.nanmin([np.nanmin(time_point_pull1),np.nanmin(time_point_pull2)])
        oneway_gaze1 = oneway_gaze1[oneway_gaze1>(firstpulltime-15)] # 15s before the first pull (animal1 or 2) count as the active period
        oneway_gaze2 = oneway_gaze2[oneway_gaze2>(firstpulltime-15)]
        mutual_gaze1 = mutual_gaze1[mutual_gaze1>(firstpulltime-15)]
        mutual_gaze2 = mutual_gaze2[mutual_gaze2>(firstpulltime-15)]  
        #    
        # newly added condition: only consider gaze during the active pulling time (15s after the last pull)    
        lastpulltime = np.nanmax([np.nanmax(time_point_pull1),np.nanmax(time_point_pull2)])
        oneway_gaze1 = oneway_gaze1[oneway_gaze1<(lastpulltime+15)]    
        oneway_gaze2 = oneway_gaze2[oneway_gaze2<(lastpulltime+15)]
        mutual_gaze1 = mutual_gaze1[mutual_gaze1<(lastpulltime+15)]
        mutual_gaze2 = mutual_gaze2[mutual_gaze2<(lastpulltime+15)] 
            
        # define successful pulls and failed pulls
        if 0: # old definition; not in use
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
        else:
            # a new definition of successful and failed pulls
            # separate successful and failed pulls
            # step 1 all pull and juice
            time_point_pull1 = bhv_data["time_points"][bhv_data["behavior_events"]==1]
            time_point_pull2 = bhv_data["time_points"][bhv_data["behavior_events"]==2]
            time_point_juice1 = bhv_data["time_points"][bhv_data["behavior_events"]==3]
            time_point_juice2 = bhv_data["time_points"][bhv_data["behavior_events"]==4]
            # step 2:
            # pull 1
            # Find the last pull before each juice
            successful_pull1 = [time_point_pull1[time_point_pull1 < juice].max() for juice in time_point_juice1]
            # Convert to Pandas Series
            successful_pull1 = pd.Series(successful_pull1, index=time_point_juice1.index)
            # Find failed pulls (pulls that are not successful)
            failed_pull1 = time_point_pull1[~time_point_pull1.isin(successful_pull1)]
            # pull 2
            # Find the last pull before each juice
            successful_pull2 = [time_point_pull2[time_point_pull2 < juice].max() for juice in time_point_juice2]
            # Convert to Pandas Series
            successful_pull2 = pd.Series(successful_pull2, index=time_point_juice2.index)
            # Find failed pulls (pulls that are not successful)
            failed_pull2 = time_point_pull2[~time_point_pull2.isin(successful_pull2)]
            #
            # step 3:
            time_point_pull1_succ = np.round(successful_pull1,1)
            time_point_pull2_succ = np.round(successful_pull2,1)
            time_point_pull1_fail = np.round(failed_pull1,1)
            time_point_pull2_fail = np.round(failed_pull2,1)
        # 
        time_point_pulls_succfail = { "pull1_succ":time_point_pull1_succ,
                                      "pull2_succ":time_point_pull2_succ,
                                      "pull1_fail":time_point_pull1_fail,
                                      "pull2_fail":time_point_pull2_fail,
                                    }
        
        # define the follow/lead pull
        def classify_leads_and_follows(series_a, series_b):
            leads = []
            follows = []
            #
            for val_a in series_a:
                # Find the nearest time value in series_b
                # .argmin() gets the numeric position of the minimum distance
                nearest_val_b = series_b.iloc[(series_b - val_a).abs().argmin()]
                # If the nearest point in B happens AFTER A, then A "leads"
                if nearest_val_b > val_a:
                    leads.append(val_a)
                # If the nearest point in B happens BEFORE A, then A "follows"
                else:
                    follows.append(val_a)
            # Return as pandas Series for easy viewing/manipulation later
            return pd.Series(leads, name="Lead"), pd.Series(follows, name="Follow")
        # 1. Classify pull1 relative to pull2
        time_point_pull1_lead, time_point_pull1_follow = classify_leads_and_follows(time_point_pull1, time_point_pull2)
        # 2. Classify pull2 relative to pull1
        time_point_pull2_lead, time_point_pull2_follow = classify_leads_and_follows(time_point_pull2, time_point_pull1)
        # 3. Update the lead/follow lists to be rounded to 1 decimal place
        time_point_pull1_lead = np.round(time_point_pull1_lead, 1)
        time_point_pull1_follow = np.round(time_point_pull1_follow, 1)
        time_point_pull2_lead = np.round(time_point_pull2_lead, 1)
        time_point_pull2_follow = np.round(time_point_pull2_follow, 1)
        # 4. 
        # 2. Separate into the 8 final categories using Pandas .isin()
        # --- PULL 1 ---
        pull1_succlead = time_point_pull1_lead[time_point_pull1_lead.isin(time_point_pull1_succ)]
        pull1_succfollow = time_point_pull1_follow[time_point_pull1_follow.isin(time_point_pull1_succ)]
        pull1_faillead = time_point_pull1_lead[time_point_pull1_lead.isin(time_point_pull1_fail)]
        pull1_failfollow = time_point_pull1_follow[time_point_pull1_follow.isin(time_point_pull1_fail)]
        # --- PULL 2 ---
        pull2_succlead = time_point_pull2_lead[time_point_pull2_lead.isin(time_point_pull2_succ)]
        pull2_succfollow = time_point_pull2_follow[time_point_pull2_follow.isin(time_point_pull2_succ)]
        pull2_faillead = time_point_pull2_lead[time_point_pull2_lead.isin(time_point_pull2_fail)]
        pull2_failfollow = time_point_pull2_follow[time_point_pull2_follow.isin(time_point_pull2_fail)]
        
        time_point_pulls_leadfollow = {   "pull1_lead":time_point_pull1_lead,
                                          "pull2_lead":time_point_pull2_lead,
                                          "pull1_follow":time_point_pull1_follow,
                                          "pull2_follow":time_point_pull2_follow,
                                           # 
                                          "pull1_succlead":pull1_succlead,
                                          "pull2_succlead":pull2_succlead,
                                          "pull1_succfollow":pull1_succfollow,
                                          "pull2_succfollow":pull2_succfollow,
                                           #
                                          "pull1_faillead":pull1_faillead,
                                          "pull2_faillead":pull2_faillead,
                                          "pull1_failfollow":pull1_failfollow,
                                          "pull2_failfollow":pull2_failfollow,
                                      }
        
        
            
        # new total session time (instead of 600s) - total time of the video recording
        totalsess_time = np.floor(np.shape(output_look_ornot['look_at_lever_or_not_merge']['dodson'])[0]/30) 
                
        totalsessiontime_all_dates[idate] = totalsess_time - session_start_time    
        
    
    
        #
        if np.isin(animal1,animal1_fixedorder):
            owgaze1_num_all_dates[idate] = np.shape(oneway_gaze1)[0]
            owgaze2_num_all_dates[idate] = np.shape(oneway_gaze2)[0]
            mtgaze1_num_all_dates[idate] = np.shape(mutual_gaze1)[0]
            mtgaze2_num_all_dates[idate] = np.shape(mutual_gaze2)[0]
        else:            
            owgaze1_num_all_dates[idate] = np.shape(oneway_gaze2)[0]
            owgaze2_num_all_dates[idate] = np.shape(oneway_gaze1)[0]
            mtgaze1_num_all_dates[idate] = np.shape(mutual_gaze2)[0]
            mtgaze2_num_all_dates[idate] = np.shape(mutual_gaze1)[0]

     
        # get the continuous variables
        gausKernelsize = 16 # 4 or 16
        #
        data_summary_twoanimals, data_summary_names = get_data_for_singlecam_conBhv_from_highDimension_to_PCspace(gausKernelsize, fps, animal1, animal2, 
                                                    animalnames_videotrack, session_start_time, 
                                                    time_point_pull1, time_point_pull2,
                                                    time_point_juice1, time_point_juice2, 
                                                    oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, 
                                                    output_look_ornot, output_allvectors, 
                                                        output_allangles, output_key_locations)
        
        
        
        # session starting time compared with the neural recording
        session_start_time_niboard_offset = ni_data['session_t0_offset'] # in the unit of second
        try:
            neural_start_time_niboard_offset = ni_data['trigger_ts'][0]['elapsed_time'] # in the unit of second
        except: # for the multi-animal recording setup
            neural_start_time_niboard_offset = next(
                entry['timepoints'][0]['elapsed_time']
                for entry in ni_data['trigger_ts']
                if entry['channel_name'] == f"{trig_channelname}")
        neural_start_time_session_start_offset = neural_start_time_niboard_offset-session_start_time_niboard_offset
    
    
    
        # load channel maps
        channel_map_file = '/home/ws523/kilisort_spikesorting/Channel-Maps/Neuronexus_whitematter_2x32.mat'
        # channel_map_file = '/home/ws523/kilisort_spikesorting/Channel-Maps/Neuronexus_whitematter_2x32_kilosort4_new.mat'
        channel_map_data = scipy.io.loadmat(channel_map_file)
            
        # # load spike sorting results
        if 1:
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
            #
           
            
            # calculate the firing rate
            # FR_kernel = 0.20 # in the unit of second
            FR_kernel = 1/30 # in the unit of second # 1/30 same resolution as the video recording
            # FR_kernel is sent to to be this if want to explore it's relationship with continuous trackng data
            
            # totalsess_time_forFR = np.floor(np.shape(output_look_ornot['look_at_lever_or_not_merge']['dodson'])[0]/30)  # to match the total time of the video recording
            # totalsess_time_forFR = np.ceil(np.nanmax([np.nanmax(time_point_pull1), \
            #                                           np.nanmax(time_point_pull2)])+session_start_time)+10 # only the functioning time (pulling time)
            totalsess_time_forFR = np.ceil(np.nanmax([np.nanmax(time_point_pull1),                                                       np.nanmax(time_point_pull2)]))+10 # only the functioning time (pulling time)
            
            # load FR data
            if 1:
                neuralalFR_save_folder = neural_data_folder+neural_record_condition+'/FR_calculated/'
                FR_timepoint_path = neuralFR_save_folder+'/FR_timepoint_allch.pkl'
                FR_allch_path = neuralFR_save_folder+'/FR_allch.pkl'
                FR_zscore_allch_path = neuralFR_save_folder+'/FR_zscore_allch.pkl'

                FR_already_processed = (os.path.exists(FR_timepoint_path) and
                                         os.path.exists(FR_allch_path) and
                                         os.path.exists(FR_zscore_allch_path))
                # 
                # FR_already_processed = 0 # force to redo

                if FR_already_processed:
                    print('FR already calculated for '+neural_record_condition+' -- loading saved results')
                    with open(FR_timepoint_path, 'rb') as f:
                        FR_timepoint_allch = pickle.load(f)
                    with open(FR_allch_path, 'rb') as f:
                        FR_allch = pickle.load(f)
                    with open(FR_zscore_allch_path, 'rb') as f:
                        FR_zscore_allch = pickle.load(f)

                else:
                    print('calculating FR for '+neural_record_condition)
                    _,FR_timepoint_allch,FR_allch,FR_zscore_allch = spike_analysis_FR_calculation(fps, FR_kernel, 
                                                     totalsess_time_forFR, spike_clusters_data, spike_time_data)

                    # save the neural fr data
                    print('save the firing rate data')

                    if not os.path.exists(neuralFR_save_folder):
                        os.makedirs(neuralFR_save_folder)
                    #
                    with open(FR_timepoint_path, 'wb') as f:
                        pickle.dump(FR_timepoint_allch, f)
                    with open(FR_allch_path, 'wb') as f:
                        pickle.dump(FR_allch, f)
                    with open(FR_zscore_allch_path, 'wb') as f:
                        pickle.dump(FR_zscore_allch, f)
                
            
        # load muae
        try:
            if 1:
                
                # totalsess_time_forFR = np.floor(np.shape(output_look_ornot['look_at_lever_or_not_merge']['dodson'])[0]/30)  # to match the total time of the video recording
                # totalsess_time_forFR = np.ceil(np.nanmax([np.nanmax(time_point_pull1), \
                #                                           np.nanmax(time_point_pull2)])+session_start_time)+10 # only the functioning time (pulling time)
                totalsess_time_forFR = np.ceil(np.nanmax([np.nanmax(time_point_pull1),                                                           np.nanmax(time_point_pull2)]))+10 # only the functioning time (pulling time)

                neuralFR_save_folder = neural_data_folder+neural_record_condition+'/FR_calculated/'
                muae_30hz_path = neuralFR_save_folder+'/muae_30hz_allch.pkl'
                muae_30hz_time_path = neuralFR_save_folder+'/muae_30hz_time_allch.pkl'
                muae_30hz_zscore_path = neuralFR_save_folder+'/muae_30hz_zscore_allch.pkl'

                already_processed = (os.path.exists(muae_30hz_path) and
                                      os.path.exists(muae_30hz_time_path) and
                                      os.path.exists(muae_30hz_zscore_path))
                #
                # already_processed = 0 # force to reanalyze

                if already_processed:
                    print('MUAe already processed for '+neural_record_condition+' -- loading saved results')
                    with open(muae_30hz_path, 'rb') as f:
                        muae_30hz = pickle.load(f)
                    with open(muae_30hz_time_path, 'rb') as f:
                        muae_30hz_time = pickle.load(f)
                    with open(muae_30hz_zscore_path, 'rb') as f:
                        muae_30hz_zscore = pickle.load(f)

                else:
                    print('load MUAe data for '+neural_record_condition)
                    muae_filename = neural_data_folder+neural_record_condition+'/MUAe.txt' # already downsample to 1000hz
                    muae_data_df = genfromtxt(muae_filename, delimiter=',')
                    #
                    # only consider time after session start and 10s after the last pull (to align with the FR time scale)
                    muae_time = np.arange(0,np.shape(muae_data_df)[1],1)/fs_lfp + neural_start_time_session_start_offset
                    muae_target_ind = (muae_time >= 0) & (muae_time <=totalsess_time_forFR)
                    muae_time = muae_time[muae_target_ind]
                    muae_data_df = muae_data_df[:,muae_target_ind]
                    # remove shared artifact transients (electrical noise, ground bounce, etc.)
                    muae_clean_df, artifact_mask, artifact_info = remove_common_artifact(
                        muae_data_df, pad_samples=30, mad_threshold=20, max_artifact_frac=0.05
                    )
                    if artifact_info['skipped_safety_cap']:
                        print(f"  ⚠ {neural_record_condition}: {artifact_info['frac_session_flagged']*100:.1f}% flagged "
                              f"(exceeds cap) — using UNCLEANED MUAe, needs manual review")
                    else:
                        print(f"  cleaned {artifact_info['n_artifact_events']} artifact event(s), "
                              f"{artifact_info['frac_session_flagged']*100:.2f}% of session")

                    # apply to your cleaned MUAe
                    muae_30hz, muae_30hz_time = bin_average_to_video_fps(muae_clean_df, target_fps=fps, source_fs=fs_lfp)
                    #
                    mean_muae = np.nanmean(muae_30hz, axis=1, keepdims=True)
                    std_muae = np.nanstd(muae_30hz, axis=1, keepdims=True)
                    std_muae[std_muae == 0] = 1  # guard against a flat/dead channel
                    muae_30hz_zscore = (muae_30hz - mean_muae) / std_muae

                    # save so future runs can skip reprocessing
                    if not os.path.exists(neuralFR_save_folder):
                        os.makedirs(neuralFR_save_folder)
                    with open(muae_30hz_time_path, 'wb') as f:
                        pickle.dump(muae_30hz_time, f)
                    with open(muae_30hz_path, 'wb') as f:
                        pickle.dump(muae_30hz, f)
                    with open(muae_30hz_zscore_path, 'wb') as f:
                        pickle.dump(muae_30hz_zscore, f)
                        
        except:
            continue

                    
        # load lfp
        if 0:
            print('load LFP data for '+neural_record_condition)
            lfp_filename = neural_data_folder+neural_record_condition+'/lfp_filt.txt' # already downsample to 1000hz
            lfp_data_df = genfromtxt(lfp_filename, delimiter=',')
            #
            # same time cropping as muae, so the two stay aligned to the same window
            lfp_time = np.arange(0,np.shape(lfp_data_df)[1],1)/fs_lfp + neural_start_time_session_start_offset
            lfp_target_ind = (lfp_time >= 0) & (lfp_time <=totalsess_time_forFR)
            lfp_time = lfp_time[lfp_target_ind]
            lfp_data_df = lfp_data_df[:,lfp_target_ind]

            # same bin-averaging to video fps, matching muae and FR time base
            lfp_30hz, lfp_30hz_time = bin_average_to_video_fps(lfp_data_df, target_fps=fps, source_fs=fs_lfp)
            #
            mean_lfp = np.nanmean(lfp_30hz, axis=1, keepdims=True)
            std_lfp = np.nanstd(lfp_30hz, axis=1, keepdims=True)
            std_lfp[std_lfp == 0] = 1  # guard against a flat/dead channel
            lfp_30hz_zscore = (lfp_30hz - mean_lfp) / std_lfp
            
            
        
        

    # save data
    if 0:
        
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
                
        # with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
        #     pickle.dump(DBN_input_data_alltypes, f)

        with open(data_saved_subfolder+'/owgaze1_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(owgaze1_num_all_dates, f)
        with open(data_saved_subfolder+'/owgaze2_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(owgaze2_num_all_dates, f)
        with open(data_saved_subfolder+'/mtgaze1_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(mtgaze1_num_all_dates, f)
        with open(data_saved_subfolder+'/mtgaze2_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(mtgaze2_num_all_dates, f)
        with open(data_saved_subfolder+'/pull1_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(pull1_num_all_dates, f)
        with open(data_saved_subfolder+'/pull2_num_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(pull2_num_all_dates, f)

        with open(data_saved_subfolder+'/tasktypes_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(tasktypes_all_dates, f)
        with open(data_saved_subfolder+'/coopthres_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(coopthres_all_dates, f)
        with open(data_saved_subfolder+'/succ_rate_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(succ_rate_all_dates, f)
        with open(data_saved_subfolder+'/interpullintv_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(interpullintv_all_dates, f)
        with open(data_saved_subfolder+'/trialnum_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(trialnum_all_dates, f)
            
        with open(data_saved_subfolder+'/totalsessiontime_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
            pickle.dump(totalsessiontime_all_dates, f)
        
    
    
    # only save a subset 
    if 0:
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
    
    
    


# In[18]:


# look at one session's data, for sanity check
if 0:
    plt.figure(figsize=(14, 5))
    for ch in range(muae_30hz_zscore.shape[0]):
        plt.plot(muae_30hz_time[4100:5100], muae_30hz_zscore[ch, 4100:5100], alpha=0.5, lw=0.7)
    4
    for pt in time_point_pull1:
        if muae_30hz_time[4100] <= pt <= muae_30hz_time[5100]:
            plt.axvline(pt, color='red', linestyle='--', alpha=0.7, lw=1.5)
    for pt in time_point_pull2:
        if muae_30hz_time[4100] <= pt <= muae_30hz_time[5100]:
            plt.axvline(pt, color='blue', linestyle='--', alpha=0.7, lw=1.5)

    plt.xlabel('time (s)'); plt.ylabel('MUAe (z-scored)')
    plt.title('All 64 channels, pull times overlaid (red=animal1, blue=animal2)')
    plt.show()


# In[19]:


# look at one session's data, for sanity check
if 0:
    plt.figure(figsize=(14, 5))
    for ch in range(lfp_30hz_zscore.shape[0]):
        plt.plot(lfp_30hz_time[4100:5100], lfp_30hz_zscore[ch, 4100:5100], alpha=0.5, lw=0.7)

    for pt in time_point_pull1:
        if lfp_30hz_time[4100] <= pt <= lfp_30hz_time[5100]:
            plt.axvline(pt, color='red', linestyle='--', alpha=0.7, lw=1.5)
    for pt in time_point_pull2:
        if lfp_30hz_time[4100] <= pt <= lfp_30hz_time[5100]:
            plt.axvline(pt, color='blue', linestyle='--', alpha=0.7, lw=1.5)

    plt.xlabel('time (s)'); plt.ylabel('LFP (z-scored)')
    plt.title('All 64 channels LFP, pull times overlaid (red=animal1, blue=animal2)')
    plt.show()


# ### attempt to use cebra to align MUAe population and bhv

# In[20]:


# prepare the data to adapt to cebra -- MUAe version

consider_othergaze = 0
if consider_othergaze:
    cebra_folder_suffix = '_withothergaze'
elif not consider_othergaze:
    cebra_folder_suffix = ''

consider_MCSR = 0
if not consider_MCSR:
    condition_tgt = ['MC']
    condition_prefix = ['MC_with']
    cebra_folder_suffix = cebra_folder_suffix + ''
elif consider_MCSR:
    condition_tgt = ['MC', 'SR']
    condition_prefix = ['MC_with', 'SR_with']
    cebra_folder_suffix = cebra_folder_suffix + '_MCSR'

try:

    data_saved_path = '/gpfs/marilyn/pi/nandy/VideoTracker_SocialInter/'+    '3d_recontruction_analysis_self_and_coop_task_neural_analysis_OFC_focus_saved/'+    'cebra_checkpoints_MUAe'+cebra_folder_suffix+'/'+savefile_sufix+'/'

    #
    with open(data_saved_path+recordedanimals[0]+'_multisessions_bhv_data_MUAe.pkl', 'rb') as f:
        multisessions_bhv_data = pickle.load(f)
    with open(data_saved_path+recordedanimals[0]+'_multisessions_neural_data_MUAe.pkl', 'rb') as f:
        multisessions_neural_data = pickle.load(f)
    with open(data_saved_path+recordedanimals[0]+'_multisessions_pullinfo_data_MUAe.pkl', 'rb') as f:
        multisessions_pullinfo_data = pickle.load(f)
    with open(data_saved_path+recordedanimals[0]+'_multisessions_dates_MUAe.pkl', 'rb') as f:
        multisessions_dates = pickle.load(f)
    with open(data_saved_path+recordedanimals[0]+'_multisessions_conditions_MUAe.pkl', 'rb') as f:
        multisessions_conditions = pickle.load(f)

    print('load cebra pre-processing dataset (MUAe)')


except:

    print('organize cebra pre-processing dataset (MUAe)')

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    ndates = np.shape(dates_list)[0]

    if not consider_othergaze:
        behavior_vars = ['socialgaze_prob','selfpull_prob', 'selfjuice_prob', 'otherpull_prob',]
    elif consider_othergaze:
        behavior_vars = ['socialgaze_prob','selfpull_prob', 'selfjuice_prob', 'otherpull_prob', 'othergaze_prob']

    min_channel_number = 10  # sanity-check floor; MUAe channel count is fixed (64) so this mainly
                              # catches a failed/corrupt load rather than a true low-yield session

    multisessions_neural_data = []
    multisessions_bhv_data = []
    multisessions_pullinfo_data = []

    multisessions_dates = []
    multisessions_conditions = []


    for idate in np.arange(0,ndates,1):

        date_tgt = dates_list[idate]
        neural_record_condition = neural_record_conditions[idate]
        task_condition = task_conditions[idate]
        session_start_time = session_start_times[idate]
        animal1_filename = animal1_filenames[idate]
        animal2_filename = animal2_filenames[idate]
        animal1_fixedorder = [animal1_fixedorders[idate]]
        animal2_fixedorder = [animal2_fixedorders[idate]]
        recordedanimal = recordedanimals[idate]

        self_animal = recordedanimal
        if animal1_fixedorder[0] == recordedanimal:
            partner_animal = animal2_fixedorder[0]
        elif animal2_fixedorder[0] == recordedanimal:
            partner_animal = animal1_fixedorder[0]

        if not ((task_condition in condition_tgt) | task_condition.startswith(tuple(condition_prefix))):
            continue

        # load the bhv data
        current_dir = data_saved_folder+'/bhv_events_singlecam_wholebody/'+animal1_fixedorder[0]+animal2_fixedorder[0]
        add_date_dir = os.path.join(current_dir,cameraID+'/'+date_tgt)

        with open(add_date_dir+'/data_summary_twoanimals.pkl', 'rb') as f:
            data_summary_twoanimals = pickle.load(f)
        with open(add_date_dir+'/data_summary_names.pkl', 'rb') as f:
            data_summary_names = pickle.load(f)

        behavior_data = data_summary_twoanimals[recordedanimal]
        behavior_time = np.arange(len(behavior_data[0])) / fps  - session_start_time

        # load the pull info behavioral data
        current_dir = data_saved_folder+'/bhv_events_singlecam_wholebody_with_glm_model'+savefile_sufix+'/'+animal1_fixedorder[0]+animal2_fixedorder[0]
        add_date_dir = os.path.join(current_dir,cameraID+'/'+date_tgt)

        with open(add_date_dir+'/pre_data_for_GLM.pkl', 'rb') as f:
            pre_data_for_GLM = pickle.load(f)

        pullinfo_data = pre_data_for_GLM[(recordedanimal,'X_all')]
        pullinfo_time = np.arange(np.shape(pullinfo_data)[0])/fps - session_start_time

        # load the neural MUAe data (already computed and cached at 30Hz, per-channel z-scored)
        neuralFR_save_folder = neural_data_folder+neural_record_condition+'/FR_calculated/'
        muae_30hz_time_path = neuralFR_save_folder+'/muae_30hz_time_allch.pkl'
        muae_30hz_zscore_path = neuralFR_save_folder+'/muae_30hz_zscore_allch.pkl'

        if not (os.path.exists(muae_30hz_time_path) and os.path.exists(muae_30hz_zscore_path)):
            print(f'  no MUAe cache found for {neural_record_condition}, skipping')
            continue

        with open(muae_30hz_time_path, 'rb') as f:
            muae_30hz_time = pickle.load(f)
        with open(muae_30hz_zscore_path, 'rb') as f:
            muae_30hz_zscore = pickle.load(f)

        # muae_30hz_zscore is (n_channels, n_timepoints) -- unlike FR_zscore_allch, which was
        # a dict keyed by neuron ID. Channels here play the role neurons did in the FR version.
        nchannels = muae_30hz_zscore.shape[0]
        if nchannels < min_channel_number:
            continue

        # find the common time of the behavior and the neural recording
        min_time = np.max([np.nanmin(behavior_time),np.nanmin(muae_30hz_time)])
        max_time = np.min([np.nanmax(behavior_time),np.nanmax(muae_30hz_time)])

        ind_bhv = (behavior_time>=min_time) & (behavior_time<=max_time)
        ind_MUAe = (muae_30hz_time>=min_time) & (muae_30hz_time<=max_time)
        ind_pullinfo = ((pullinfo_time>=min_time) & (pullinfo_time<=max_time))

        # 1. Create a rigid, absolute 10Hz master clock (1 bin = strictly 0.1 seconds / 100ms)
        common_time = np.arange(min_time, max_time, 0.1)
        singlesession_neural_data = []
        singlesession_bhv_data = []
        singlesession_pullinfo_data = []

        # Helper function to safely interpolate data
        def interpolate_trace(t_original, y_original, t_common):
            idx_sort = np.argsort(t_original)
            return np.interp(t_common, t_original[idx_sort], y_original[idx_sort])

        # 2. Extract and interpolate MUAe activity, per channel (rows of muae_30hz_zscore)
        t_MUAe = muae_30hz_time[ind_MUAe]
        for ich in range(nchannels):
            y_MUAe = muae_30hz_zscore[ich, :][ind_MUAe]
            if len(t_MUAe) > 0:
                singlesession_neural_data.append(interpolate_trace(t_MUAe, y_MUAe, common_time))

        singlesession_neural_data = np.vstack(singlesession_neural_data).T
        multisessions_neural_data.append(singlesession_neural_data)

        multisessions_dates.append(date_tgt)
        multisessions_conditions.append(task_condition)

        # 3. Extract and interpolate Continuous Behavior (unchanged from FR version)
        for var_name in behavior_vars:
            var_idx = data_summary_names.index(var_name)
            t_behav = behavior_time[ind_bhv]
            y_behav_current = behavior_data[var_idx][ind_bhv]
            y_behav_scaled = scaler.fit_transform(y_behav_current.reshape(-1, 1)).flatten()
            if len(t_behav) > 0:
                singlesession_bhv_data.append(interpolate_trace(t_behav, y_behav_scaled, common_time))

        singlesession_bhv_data = np.vstack(singlesession_bhv_data).T
        multisessions_bhv_data.append(singlesession_bhv_data)

        # get the pullinfo data (unchanged from FR version)
        if not (recordedanimal, 'raw_var_names') in pre_data_for_GLM.keys():
            print('no pullinfo data organized')
            break
        t_pullinfo = pullinfo_time[ind_pullinfo]
        y_consec_fail = pre_data_for_GLM[(recordedanimal,'X_all')][:,-1][ind_pullinfo]
        y_consec_fail_scaled = scaler.fit_transform(y_consec_fail.reshape(-1, 1)).flatten()
        if len(t_pullinfo) > 0:
            singlesession_pullinfo_data.append(interpolate_trace(t_pullinfo, y_consec_fail_scaled, common_time))
        y_time_since_succ = pre_data_for_GLM[(recordedanimal,'X_all')][:,-2][ind_pullinfo]
        y_time_since_succ_scaled = scaler.fit_transform(y_time_since_succ.reshape(-1, 1)).flatten()
        if len(t_pullinfo) > 0:
            singlesession_pullinfo_data.append(interpolate_trace(t_pullinfo, y_time_since_succ_scaled, common_time))
        #
        singlesession_pullinfo_data = np.vstack(singlesession_pullinfo_data).T
        multisessions_pullinfo_data.append(singlesession_pullinfo_data)


    # save the pre-processed data
    os.makedirs('/gpfs/marilyn/pi/nandy/VideoTracker_SocialInter/'+                '3d_recontruction_analysis_self_and_coop_task_neural_analysis_OFC_focus_saved/'+                'cebra_checkpoints_MUAe'+cebra_folder_suffix+'/'+savefile_sufix, exist_ok=True)
    data_saved_path = '/gpfs/marilyn/pi/nandy/VideoTracker_SocialInter/'+                '3d_recontruction_analysis_self_and_coop_task_neural_analysis_OFC_focus_saved/'+                'cebra_checkpoints_MUAe'+cebra_folder_suffix+'/'+savefile_sufix+'/'
    #
    savedata = 1
    if savedata:
        with open(data_saved_path+recordedanimals[0]+'_multisessions_bhv_data_MUAe.pkl', 'wb') as f:
            pickle.dump(multisessions_bhv_data, f)
        with open(data_saved_path+recordedanimals[0]+'_multisessions_pullinfo_data_MUAe.pkl', 'wb') as f:
            pickle.dump(multisessions_pullinfo_data, f)
        with open(data_saved_path+recordedanimals[0]+'_multisessions_neural_data_MUAe.pkl', 'wb') as f:
            pickle.dump(multisessions_neural_data, f)
        with open(data_saved_path+recordedanimals[0]+'_multisessions_dates_MUAe.pkl', 'wb') as f:
            pickle.dump(multisessions_dates, f)
        with open(data_saved_path+recordedanimals[0]+'_multisessions_conditions_MUAe.pkl', 'wb') as f:
            pickle.dump(multisessions_conditions, f)


# In[21]:


# check if i can use gpu
import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
# to be able to load the trained model, i have to use gpu


# ### cebra + hmm comparing two brain regions and two animals

# In[22]:


# organized the data contains OFC MUAe data across animals; optional check with only one seed of cebra
import cebra
if 1:
    brainregions = ['_OFCs']  # MUAe pipeline is OFC-only for now (dmPFC uses previously published spike-based data)
    tgtanimals = ['dodson','kanga']
    #
    consider_othergaze = 0 # this is where to set up the conditions
    if consider_othergaze:
        cebra_folder_suffix = '_withothergaze'
    elif not consider_othergaze:
        cebra_folder_suffix = ''

    #
    consider_MCSR = 0  # this is where to set up the conditions
    if not consider_MCSR:
        cebra_folder_suffix = cebra_folder_suffix + ''
    elif consider_MCSR:
        cebra_folder_suffix = cebra_folder_suffix + '_MCSR'
    #
    multisessions_bhv_all = {}
    multisessions_pullinfo_all = {}
    multisessions_neural_all = {}
    multisessions_dates_all = {}
    multisessions_conditions_all = {}
    cebra_model_all = {}
    import warnings
    warnings.filterwarnings('ignore', message='.*weights_only.*')
    gpfs_base = '/gpfs/marilyn/pi/nandy/VideoTracker_SocialInter/'                 '3d_recontruction_analysis_self_and_coop_task_neural_analysis_OFC_focus_saved/'                 'cebra_checkpoints_MUAe'+cebra_folder_suffix+'/'
    for ibrainregion in brainregions:
        for itgtanimal in tgtanimals:
            # load behavioral data
            data_saved_path = f'{gpfs_base}/{ibrainregion}/'
            #
            with open(data_saved_path+itgtanimal+'_multisessions_bhv_data_MUAe.pkl', 'rb') as f:
                multisessions_bhv_all[ibrainregion,itgtanimal] = pickle.load(f)
            with open(data_saved_path+itgtanimal+'_multisessions_pullinfo_data_MUAe.pkl', 'rb') as f:
                multisessions_pullinfo_all[ibrainregion,itgtanimal] = pickle.load(f)
            with open(data_saved_path+itgtanimal+'_multisessions_neural_data_MUAe.pkl', 'rb') as f:
                multisessions_neural_all[ibrainregion,itgtanimal] = pickle.load(f)
            with open(data_saved_path+itgtanimal+'_multisessions_dates_MUAe.pkl', 'rb') as f:
                multisessions_dates_all[ibrainregion,itgtanimal] = pickle.load(f)
            with open(data_saved_path+itgtanimal+'_multisessions_conditions_MUAe.pkl', 'rb') as f:
                multisessions_conditions_all[ibrainregion,itgtanimal] = pickle.load(f)


# In[23]:


cebra_folder_suffix


# #### test with multiple seed cebra sweep

# In[ ]:


# ============================================================
# MULTI-SEED CEBRA SWEEP (MUAe, OFC-only) — 20 seeds, same 5000 iterations,
# rerun projection + smoothing, then compute R^2 for behavioral variables per seed
# ============================================================
import cebra
import torch
import warnings
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score

brainregions = ['_OFCs']  # MUAe pipeline is OFC-only
tgtanimals = ['dodson','kanga']

#
if consider_othergaze:
    cebra_folder_suffix = '_withothergaze'
elif not consider_othergaze:
    cebra_folder_suffix = ''

#
if not consider_MCSR:
    cebra_folder_suffix = cebra_folder_suffix + ''
elif consider_MCSR:
    cebra_folder_suffix = cebra_folder_suffix + '_MCSR'


warnings.filterwarnings('ignore', message='.*weights_only.*')

SEEDS = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

MAX_ITERATIONS = 5000

TARGET_DWELL_SECONDS = 15
BIN_SIZE_SECONDS = 0.1
SMOOTHING_WINDOW_SECONDS = 3
SMOOTHING_WINDOW_BINS = int(SMOOTHING_WINDOW_SECONDS / BIN_SIZE_SECONDS)
target_dwell_bins = TARGET_DWELL_SECONDS / BIN_SIZE_SECONDS

def variance_explained_per_variable(all_days_3D, behavior_vars_dict):
    results = {}
    for var_name, var_values in behavior_vars_dict.items():
        valid = ~np.isnan(var_values)
        reg = LinearRegression()
        reg.fit(var_values[valid].reshape(-1, 1), all_days_3D[valid])
        pred = reg.predict(var_values[valid].reshape(-1, 1))
        results[var_name] = r2_score(all_days_3D[valid], pred)
    return results


# force to redo the cebra projection
force_to_redo = 0

gpfs_base = '/gpfs/marilyn/pi/nandy/VideoTracker_SocialInter/'             '3d_recontruction_analysis_self_and_coop_task_neural_analysis_OFC_focus_saved/'             'cebra_checkpoints_MUAe'+cebra_folder_suffix+'/'

multiseed_results = []

for ibrainregion in brainregions:
    for itgtanimal in tgtanimals:
        key = (ibrainregion, itgtanimal)
        neural_data = multisessions_neural_all[key]
        bhv_data = multisessions_bhv_all[key]
        all_days_bhv = np.vstack(bhv_data)
        gaze_color = all_days_bhv[:, 0]
        selfpull_color = all_days_bhv[:, 1]
        selfjuice_color = all_days_bhv[:, 2]
        otherpull_color = all_days_bhv[:, 3]
        if consider_othergaze:
            othergaze_color = all_days_bhv[:, 4]

        for seed in SEEDS:
            print(f"\n{'='*60}\n{key}, seed={seed} (MUAe)\n{'='*60}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_path = f'{gpfs_base}/{ibrainregion}/{itgtanimal}_cebra_model_seed{seed}.pt'
            try:
                if force_to_redo:
                    print('force to redo:')
                    dummy
                cebra_model_seed = cebra.CEBRA.load(model_path)
                print(f"Loaded existing checkpoint: {model_path}")
            except:
                print(f"Training fresh model for seed {seed}...")
                cebra_model_seed = cebra.CEBRA(
                    model_architecture='offset10-model',
                    batch_size=512,
                    learning_rate=3e-4,
                    temperature=1,
                    output_dimension=3,
                    max_iterations=MAX_ITERATIONS,
                    distance='cosine',
                    conditional='time_delta',
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    verbose=True,
                )
                cebra_model_seed.fit(neural_data, bhv_data)
                os.makedirs(f'{gpfs_base}/{ibrainregion}', exist_ok=True)
                cebra_model_seed.save(model_path)
                print(f"Saved: {model_path}")

            # --- Project and smooth --
            aligned_neural_spaces = [
                cebra_model_seed.transform(s, session_id=i) for i, s in enumerate(neural_data)
            ]
            smoothed_spaces = [uniform_filter1d(s, size=SMOOTHING_WINDOW_BINS, axis=0) for s in aligned_neural_spaces]
            all_days_3D_smoothed = np.vstack(smoothed_spaces)

            # --- Variance explained, all variables ---
            bhv_vars_to_test = {
                'gaze': gaze_color,
                'self_pull': selfpull_color,
                'self_juice': selfjuice_color,
                'other_pull': otherpull_color,
            }
            if consider_othergaze:
                bhv_vars_to_test['other_gaze'] = othergaze_color

            var_explained = variance_explained_per_variable(all_days_3D_smoothed, bhv_vars_to_test)

            result_row = {
                'region': ibrainregion, 'animal': itgtanimal, 'seed': seed,
                'data_type': 'MUAe',
                'gaze_R2': var_explained['gaze'],
                'self_pull_R2': var_explained['self_pull'],
                'self_juice_R2': var_explained['self_juice'],
                'other_pull_R2': var_explained['other_pull'],
            }
            if consider_othergaze:
                result_row['other_gaze_R2'] = var_explained['other_gaze']

            multiseed_results.append(result_row)

multiseed_df_MUAe = pd.DataFrame(multiseed_results)
print("\n\nAll seed results (MUAe):")
print(multiseed_df_MUAe.to_string(index=False))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





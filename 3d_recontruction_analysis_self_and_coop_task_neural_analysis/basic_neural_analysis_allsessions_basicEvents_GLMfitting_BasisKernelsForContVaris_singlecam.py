#!/usr/bin/env python
# coding: utf-8

# ### Basic neural activity analysis with single camera tracking
# #### use GLM model to analyze spike count trains, the GLM use continuous variables and use basis kernel to simplify the fitting
# #### also add the function to reduce the continuous variables into smaller dimensions to reduce the correlation and be more task related 

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


# ### function - plot inter-pull interval

# In[8]:


from ana_functions.plot_interpull_interval import plot_interpull_interval


# ### function - interval between all behavioral events

# In[9]:


from ana_functions.bhv_events_interval import bhv_events_interval


# ### function - GLM fitting for spike trains based on the discrete variables from single camera

# In[10]:


from ana_functions.singlecam_bhv_var_neuralGLM_fitting_BasisKernelsForContVaris import get_singlecam_bhv_var_for_neuralGLM_fitting_BasisKernelsForContVaris
from ana_functions.singlecam_bhv_var_neuralGLM_fitting_BasisKernelsForContVaris import neuralGLM_fitting_BasisKernelsForContVaris
from ana_functions.singlecam_bhv_var_neuralGLM_fitting_BasisKernelsForContVaris_PullGazeVectorProjection import neuralGLM_fitting_BasisKernelsForContVaris_PullGazeVectorProjection


# ### function - other useful functions

# In[11]:


# for defining the meaningful social gaze (the continuous gaze distribution that is closest to the pull) 
from ana_functions.keep_closest_cluster_single_trial import keep_closest_cluster_single_trial


# In[12]:


# get useful information about pulls
from ana_functions.get_pull_infos import get_pull_infos


# In[13]:


# use the gaze vector speed and face mass speed to find the pull action start time within IPI
from ana_functions.find_sharp_increases_withinIPI import find_sharp_increases_withinIPI
from ana_functions.find_sharp_increases_withinIPI import find_sharp_increases_withinIPI_dual_speed


# In[14]:


def cluster_based_correction_with_timing(real_mean, shuffled_coefs, alpha=0.05, time_axis=None):
    n_boot, n_vars, n_basis = shuffled_coefs.shape
    cluster_significance = np.zeros(n_vars, dtype=bool)
    cluster_timing = ['None'] * n_vars

    for var in range(n_vars):
        real_coef = real_mean[var, :]
        shuf_coef = shuffled_coefs[:, var, :]

        # Empirical p-values
        p_vals = (np.sum(np.abs(shuf_coef) >= np.abs(real_coef), axis=0) + 1) / (n_boot + 1)
        sig_mask = p_vals < alpha

        # Cluster label
        labeled_array, n_clusters = label(sig_mask)
        real_cluster_sizes = [
            np.sum(labeled_array == cluster_idx + 1)
            for cluster_idx in range(n_clusters)
        ]
        max_real_cluster = np.max(real_cluster_sizes) if real_cluster_sizes else 0

        # Compute max cluster size in each shuffled iteration
        shuf_max_clusters = []
        for b in range(n_boot):
            others = np.delete(shuf_coef, b, axis=0)
            p_vals_shuf = np.mean(np.abs(others) >= np.abs(shuf_coef[b, :]), axis=0)
            sig_shuf = p_vals_shuf < alpha
            lbl_shuf, n_lbl = label(sig_shuf)
            max_cluster = max([np.sum(lbl_shuf == i + 1) for i in range(n_lbl)], default=0)
            shuf_max_clusters.append(max_cluster)

        cluster_thresh = np.percentile(shuf_max_clusters, 100 * (1 - alpha))
        cluster_significance[var] = max_real_cluster > cluster_thresh

        # Dominant timing classification
        if cluster_significance[var] and time_axis is not None:
            sig_times = time_axis[sig_mask]
            n_before = np.sum(sig_times < 0)
            n_after = np.sum(sig_times > 0)

            if n_before > n_after:
                cluster_timing[var] = 'Reactive'
            elif n_after > n_before:
                cluster_timing[var] = 'Predictive'
            else:
                cluster_timing[var] = 'Both' if (n_before > 0) else 'None'

    return cluster_significance, cluster_timing


# In[15]:


# check the orthogonality among the three behavioral vectors
def check_orthogonality(*vectors, tol=1e-6):
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            dot = np.dot(vectors[i], vectors[j])
            print(f"Dot product between vector {i} and {j}: {dot:.6f}")
            if abs(dot) > tol:
                print("⚠️ Not orthogonal!")
            else:
                print("✅ Orthogonal")
                
#  Gram-Schmidt Orthogonalization
def gram_schmidt(vectors):
    """Orthogonalize a list of vectors using the Gram-Schmidt process."""
    orthogonal_vectors = []
    for v in vectors:
        # Subtract projection onto all previous orthogonal vectors
        for u in orthogonal_vectors:
            v = v - np.dot(v, u) * u
        # Normalize the vector
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            orthogonal_vectors.append(v / norm)
    return orthogonal_vectors


# ## Analyze each session

# ### prepare the basic behavioral data (especially the time stamps for each bhv events)

# In[16]:


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
redo_anystep = 0

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

# dodson ginger
if 1:
    if do_DLPFC:
        neural_record_conditions = [
                                    '20240531_Dodson_MC',
                                    '20240603_Dodson_MC_and_SR',
                                    '20240603_Dodson_MC_and_SR',
                                    '20240604_Dodson_MC',
                                    '20240605_Dodson_MC_and_SR',
                                    '20240605_Dodson_MC_and_SR',
                                    '20240606_Dodson_MC_and_SR',
                                    '20240606_Dodson_MC_and_SR',
                                    '20240607_Dodson_SR',
                                    '20240610_Dodson_MC',
                                    '20240611_Dodson_SR',
                                    '20240612_Dodson_MC',
                                    '20240613_Dodson_SR',
                                    '20240620_Dodson_SR',
                                    '20240719_Dodson_MC',
                                        
                                    '20250129_Dodson_MC',
                                    '20250130_Dodson_SR',
                                    '20250131_Dodson_MC',
                                
            
                                    '20250210_Dodson_SR_withKoala',
                                    '20250211_Dodson_MC_withKoala',
                                    '20250212_Dodson_SR_withKoala',
                                    '20250214_Dodson_MC_withKoala',
                                    '20250217_Dodson_SR_withKoala',
                                    '20250218_Dodson_MC_withKoala',
                                    '20250219_Dodson_SR_withKoala',
                                    '20250220_Dodson_MC_withKoala',
                                    '20250224_Dodson_KoalaAL_withKoala',
                                    '20250226_Dodson_MC_withKoala',
                                    '20250227_Dodson_KoalaAL_withKoala',
                                    '20250228_Dodson_DodsonAL_withKoala',
                                    '20250304_Dodson_DodsonAL_withKoala',
                                    '20250305_Dodson_MC_withKoala',
                                    '20250306_Dodson_KoalaAL_withKoala',
                                    '20250307_Dodson_DodsonAL_withKoala',
                                    '20250310_Dodson_MC_withKoala',
                                    '20250312_Dodson_NV_withKoala',
                                    '20250313_Dodson_NV_withKoala',
                                    '20250314_Dodson_NV_withKoala',
            
                                    '20250401_Dodson_MC_withKanga',
                                    '20250402_Dodson_MC_withKanga',
                                    '20250403_Dodson_MC_withKanga',
                                    '20250404_Dodson_SR_withKanga',
                                    '20250407_Dodson_SR_withKanga',
                                    '20250408_Dodson_SR_withKanga',
                                    '20250409_Dodson_MC_withKanga',
            
                                    '20250415_Dodson_MC_withKanga',
                                    # '20250416_Dodson_SR_withKanga', # has to remove from the later analysis, recording has problems
                                    '20250417_Dodson_MC_withKanga',
                                    '20250418_Dodson_SR_withKanga',
                                    '20250421_Dodson_SR_withKanga',
                                    '20250422_Dodson_MC_withKanga',
                                    '20250422_Dodson_SR_withKanga',
            
                                    '20250423_Dodson_MC_withKanga',
                                    '20250423_Dodson_SR_withKanga', 
                                    '20250424_Dodson_NV_withKanga',
                                    '20250424_Dodson_MC_withKanga',
                                    '20250424_Dodson_SR_withKanga',            
                                    '20250425_Dodson_NV_withKanga',
                                    '20250425_Dodson_SR_withKanga',
                                    '20250428_Dodson_NV_withKanga',
                                    '20250428_Dodson_MC_withKanga',
                                    '20250428_Dodson_SR_withKanga',  
                                    '20250429_Dodson_NV_withKanga',
                                    '20250429_Dodson_MC_withKanga',
                                    '20250429_Dodson_SR_withKanga',  
                                    '20250430_Dodson_NV_withKanga',
                                    '20250430_Dodson_MC_withKanga',
                                    '20250430_Dodson_SR_withKanga',  
            
                                   ]
        task_conditions = [
                            'MC',           
                            'MC',
                            'SR',
                            'MC',
                            'MC',
                            'SR',
                            'MC',
                            'SR',
                            'SR',
                            'MC',
                            'SR',
                            'MC',
                            'SR',
                            'SR',
                            'MC',
                            
                            'MC_withGingerNew',
                            'SR_withGingerNew',
                            'MC_withGingerNew',
            
                            'SR_withKoala',
                            'MC_withKoala',
                            'SR_withKoala',
                            'MC_withKoala',
                            'SR_withKoala',
                            'MC_withKoala',
                            'SR_withKoala',
                            'MC_withKoala',
                            'MC_KoalaAuto_withKoala',
                            'MC_withKoala',
                            'MC_KoalaAuto_withKoala',
                            'MC_DodsonAuto_withKoala',
                            'MC_DodsonAuto_withKoala',
                            'MC_withKoala',
                            'MC_KoalaAuto_withKoala',
                            'MC_DodsonAuto_withKoala',
                            'MC_withKoala',
                            'NV_withKoala',
                            'NV_withKoala',
                            'NV_withKoala',

                            'MC_withKanga',
                            'MC_withKanga',
                            'MC_withKanga',
                            'SR_withKanga',
                            'SR_withKanga',
                            'SR_withKanga',
                            'MC_withKanga',
            
                            'MC_withKanga',
                            # 'SR_withKanga',
                            'MC_withKanga',
                            'SR_withKanga',
                            'SR_withKanga',
                            'MC_withKanga',
                            'SR_withKanga',
            
                            'MC_withKanga',
                            'SR_withKanga', 
                            'NV_withKanga',
                            'MC_withKanga',
                            'SR_withKanga',            
                            'NV_withKanga',
                            'SR_withKanga',
                            'NV_withKanga',
                            'MC_withKanga',
                            'SR_withKanga',  
                            'NV_withKanga',
                            'MC_withKanga',
                            'SR_withKanga',  
                            'NV_withKanga',
                            'MC_withKanga',
                            'SR_withKanga',  
                          ]
        dates_list = [
                        '20240531',
                        '20240603_MC',
                        '20240603_SR',
                        '20240604',
                        '20240605_MC',
                        '20240605_SR',
                        '20240606_MC',
                        '20240606_SR',
                        '20240607',
                        '20240610_MC',
                        '20240611',
                        '20240612',
                        '20240613',
                        '20240620',
                        '20240719',
            
                        '20250129',
                        '20250130',
                        '20250131',
            
                        '20250210',
                        '20250211',
                        '20250212',
                        '20250214',
                        '20250217',
                        '20250218',
                        '20250219',
                        '20250220',
                        '20250224',
                        '20250226',
                        '20250227',
                        '20250228',
                        '20250304',
                        '20250305',
                        '20250306',
                        '20250307',
                        '20250310',
                        '20250312',
                        '20250313',
                        '20250314',
            
                        '20250401',
                        '20250402',
                        '20250403',
                        '20250404',
                        '20250407',
                        '20250408',
                        '20250409',
            
                        '20250415',
                        # '20250416',
                        '20250417',
                        '20250418',
                        '20250421',
                        '20250422',
                        '20250422_SR',
            
                        '20250423',
                        '20250423_SR', 
                        '20250424',
                        '20250424_MC',
                        '20250424_SR',            
                        '20250425',
                        '20250425_SR',
                        '20250428_NV',
                        '20250428_MC',
                        '20250428_SR',  
                        '20250429_NV',
                        '20250429_MC',
                        '20250429_SR',  
                        '20250430_NV',
                        '20250430_MC',
                        '20250430_SR',  
                     ]
        videodates_list = [
                            '20240531',
                            '20240603',
                            '20240603',
                            '20240604',
                            '20240605',
                            '20240605',
                            '20240606',
                            '20240606',
                            '20240607',
                            '20240610_MC',
                            '20240611',
                            '20240612',
                            '20240613',
                            '20240620',
                            '20240719',
            
                            '20250129',
                            '20250130',
                            '20250131',
                            
                            '20250210',
                            '20250211',
                            '20250212',
                            '20250214',
                            '20250217',
                            '20250218',          
                            '20250219',
                            '20250220',
                            '20250224',
                            '20250226',
                            '20250227',
                            '20250228',
                            '20250304',
                            '20250305',
                            '20250306',
                            '20250307',
                            '20250310',
                            '20250312',
                            '20250313',
                            '20250314',
            
                            '20250401',
                            '20250402',
                            '20250403',
                            '20250404',
                            '20250407',
                            '20250408',
                            '20250409',
            
                            '20250415',
                            # '20250416',
                            '20250417',
                            '20250418',
                            '20250421',
                            '20250422',
                            '20250422_SR',
            
                            '20250423',
                            '20250423_SR', 
                            '20250424',
                            '20250424_MC',
                            '20250424_SR',            
                            '20250425',
                            '20250425_SR',
                            '20250428_NV',
                            '20250428_MC',
                            '20250428_SR',  
                            '20250429_NV',
                            '20250429_MC',
                            '20250429_SR',  
                            '20250430_NV',
                            '20250430_MC',
                            '20250430_SR',  
            
                          ] # to deal with the sessions that MC and SR were in the same session
        session_start_times = [ 
                                0.00,
                                340,
                                340,
                                72.0,
                                60.1,
                                60.1,
                                82.2,
                                82.2,
                                35.8,
                                0.00,
                                29.2,
                                35.8,
                                62.5,
                                71.5,
                                54.4,
            
                                0.00,
                                0.00,
                                0.00,
            
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
            
                                0.00,
                                0.00,
                                73.5,
                                0.00,
                                76.1,
                                81.5,
                                0.00,
            
                                363,
                                # 0.00,
                                79.0,
                                162.6,
                                231.9,
                                109,
                                0.00,
            
                                0.00,
                                0.00, 
                                0.00,
                                0.00,
                                0.00,          
                                0.00,
                                93.0,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00, 
                                0.00,
                                274.4,
                                0.00,
            
                              ] # in second
        
        kilosortvers = list((np.ones(np.shape(dates_list))*4).astype(int))
        
        trig_channelnames = [ 'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0',# 'Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai9','Dev1/ai9','Dev1/ai9','Dev1/ai9',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                              'Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0','Dev1/ai0',
                             
                              ]
        animal1_fixedorders = ['dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson',# 'dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson','dodson',
                               'dodson','dodson','dodson','dodson','dodson',
                              ]
        recordedanimals = animal1_fixedorders 
        animal2_fixedorders = ['ginger','ginger','ginger','ginger','ginger','ginger','ginger','ginger','ginger',
                               'ginger','ginger','ginger','ginger','ginger','ginger','gingerNew','gingerNew','gingerNew',
                               'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala',
                               'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala', 'koala',
                               'koala', 'koala', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', # 'kanga', 
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                               'kanga', 'kanga', 'kanga', 'kanga', 'kanga',
                              ]

        animal1_filenames = ["Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             "Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson","Dodson",
                             'Dodson',# 'Dodson',
                             'Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson',
                             'Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson','Dodson',
                             'Dodson','Dodson','Dodson','Dodson','Dodson',
                            ]
        animal2_filenames = ["Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger",
                             "Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger","Ginger",
                             "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala",
                             "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala", "Koala",
                             "Koala", "Koala", "Kanga", "Kanga", "Kanga", "Kanga", "Kanga", "Kanga", "Kanga",
                             'Kanga', # 'Kanga', 
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                             'Kanga', 'Kanga', 'Kanga', 'Kanga', 'Kanga',
                            ]
        
    elif do_OFC:
        # pick only five sessions for each conditions
        neural_record_conditions = [
                                     '20231101_Dodson_withGinger_MC',
                                     '20231107_Dodson_withGinger_MC',
                                     '20231122_Dodson_withGinger_MC',
                                     '20231129_Dodson_withGinger_MC',
                                     '20231101_Dodson_withGinger_SR',
                                     '20231107_Dodson_withGinger_SR',
                                     '20231122_Dodson_withGinger_SR',
                                     '20231129_Dodson_withGinger_SR',
                                   ]
        task_conditions = [
                            'MC',
                            'MC',
                            'MC',
                            'MC',
                            'SR',
                            'SR',
                            'SR',
                            'SR',
                          ]
        dates_list = [
                      "20231101_MC",
                      "20231107_MC",
                      "20231122_MC",
                      "20231129_MC",
                      "20231101_SR",
                      "20231107_SR",
                      "20231122_SR",
                      "20231129_SR",      
                     ]
        videodates_list = dates_list
        session_start_times = [ 
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
                         2, 
                         2, 
                         4, 
                         4,
                         2, 
                         2, 
                         4, 
                         4,
                       ]
    
        trig_channelnames = ['Dev1/ai0']*np.shape(dates_list)[0]
        animal1_fixedorder = ['dodson']*np.shape(dates_list)[0]
        recordedanimals = animal1_fixedorders
        animal2_fixedorder = ['ginger']*np.shape(dates_list)[0]

        animal1_filename = ["Dodson"]*np.shape(dates_list)[0]
        animal2_filename = ["Ginger"]*np.shape(dates_list)[0]

    
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
            
                                     '20240808_Kanga_MC_withGinger',
                                     '20240809_Kanga_MC_withGinger',
                                     '20240812_Kanga_MC_withGinger',
                                     '20240813_Kanga_MC_withKoala',
                                     '20240814_Kanga_MC_withKoala',
                                     '20240815_Kanga_MC_withKoala',
                                     '20240819_Kanga_MC_withVermelho',
                                     '20240821_Kanga_MC_withVermelho',
                                     '20240822_Kanga_MC_withVermelho',
            
                                     '20250415_Kanga_MC_withDodson',
                                     '20250416_Kanga_SR_withDodson',
                                     '20250417_Kanga_MC_withDodson',
                                     '20250418_Kanga_SR_withDodson',
                                     '20250421_Kanga_SR_withDodson',
                                     '20250422_Kanga_MC_withDodson',
                                     '20250422_Kanga_SR_withDodson',
            
                                    '20250423_Kanga_MC_withDodson',
                                    '20250423_Kanga_SR_withDodson', 
                                    '20250424_Kanga_NV_withDodson',
                                    '20250424_Kanga_MC_withDodson',
                                    '20250424_Kanga_SR_withDodson',            
                                    '20250425_Kanga_NV_withDodson',
                                    '20250425_Kanga_SR_withDodson',
                                    '20250428_Kanga_NV_withDodson',
                                    '20250428_Kanga_MC_withDodson',
                                    '20250428_Kanga_SR_withDodson',  
                                    '20250429_Kanga_NV_withDodson',
                                    '20250429_Kanga_MC_withDodson',
                                    '20250429_Kanga_SR_withDodson',  
                                    '20250430_Kanga_NV_withDodson',
                                    '20250430_Kanga_MC_withDodson',
                                    '20250430_Kanga_SR_withDodson',  
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
            
                      "20240808",
                      "20240809",
                      "20240812",
                      "20240813",
                      "20240814",
                      "20240815",
                      "20240819",
                      "20240821",
                      "20240822",
            
                      "20250415",
                      "20250416",
                      "20250417",
                      "20250418",
                      "20250421",
                      "20250422",
                      "20250422_SR",
            
                        '20250423',
                        '20250423_SR', 
                        '20250424',
                        '20250424_MC',
                        '20250424_SR',            
                        '20250425',
                        '20250425_SR',
                        '20250428_NV',
                        '20250428_MC',
                        '20250428_SR',  
                        '20250429_NV',
                        '20250429_MC',
                        '20250429_SR',  
                        '20250430_NV',
                        '20250430_MC',
                        '20250430_SR',  
                     ]
        videodates_list = dates_list
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
                            
                             'MC_withGinger',
                             'MC_withGinger',
                             'MC_withGinger',
                             'MC_withKoala',
                             'MC_withKoala',
                             'MC_withKoala',
                             'MC_withVermelho',
                             'MC_withVermelho',
                             'MC_withVermelho',
            
                             'MC_withDodson',
                             'SR_withDodson',
                             'MC_withDodson',
                             'SR_withDodson',
                             'SR_withDodson',
                             'MC_withDodson',
                             'SR_withDodson',
            
                             'MC_withDodson',
                            'SR_withDodson', 
                            'NV_withDodson',
                            'MC_withDodson',
                            'SR_withDodson',            
                            'NV_withDodson',
                            'SR_withDodson',
                            'NV_withDodson',
                            'MC_withDodson',
                            'SR_withDodson',  
                            'NV_withDodson',
                            'MC_withDodson',
                            'SR_withDodson',  
                            'NV_withDodson',
                            'MC_withDodson',
                            'SR_withDodson',  
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
                                
                                 59.2,
                                 49.5,
                                 40.0,
                                 50.0,
                                 0.00,
                                 69.8,
                                 85.0,
                                 212.9,
                                 68.5,
            
                                 363,
                                 0.00,
                                 79.0,
                                 162.6,
                                 231.9,
                                 109,
                                 0.00,
            
                                0.00,
                                0.00, 
                                0.00,
                                0.00,
                                0.00,          
                                0.00,
                                93.0,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00,
                                0.00, 
                                0.00,
                                274.4,
                                0.00,
                              ] # in second
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
        # pick only five sessions for each conditions
        neural_record_conditions = [
                                     
                                   ]
        dates_list = [
                      
                     ]
        videodates_list = dates_list
        task_conditions = [
                           
                          ]
        session_start_times = [ 
                                
                              ] # in second
        kilosortvers = [ 

                       ]
    
        animal1_fixedorders = ['dannon']*np.shape(dates_list)[0]
        animal2_fixedorders = ['kanga']*np.shape(dates_list)[0]
        recordedanimals = animal2_fixedorders
        
        animal1_filenames = ["Dannon"]*np.shape(dates_list)[0]
        animal2_filenames = ["Kanga"]*np.shape(dates_list)[0]
    

    
# a test case
if 0: # kanga example
    neural_record_conditions = ['20250415_Kanga_MC_withDodson']
    dates_list = ["20250415"]
    videodates_list = dates_list
    task_conditions = ['MC_withDodson']
    session_start_times = [363] # in second
    kilosortvers = [4]
    trig_channelnames = ['Dev1/ai9']
    animal1_fixedorders = ['dodson']
    animal2_fixedorders = ['kanga']
    recordedanimals = animal2_fixedorders
    animal1_filenames = ["Dodson"]
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


# GLM related variables
Kernel_coefs_all_dates = dict.fromkeys(dates_list, [])
Kernel_spikehist_all_dates = dict.fromkeys(dates_list, [])
#
Kernel_coefs_all_shuffled_dates = dict.fromkeys(dates_list, [])
Kernel_spikehist_all_shuffled_dates = dict.fromkeys(dates_list, [])



# where to save the summarizing data
data_saved_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/3d_recontruction_analysis_self_and_coop_task_data_saved/'

# neural data folder
neural_data_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/Marmoset_neural_recording/'

    


# In[17]:


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


# In[ ]:


# basic behavior analysis (define time stamps for each bhv events, etc)

try:
    if redo_anystep:
        dummy
    
    # load saved data
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_neuralGLM_new'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'
    

    with open(data_saved_subfolder+'/Kernel_coefs_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_coefs_all_dates = pickle.load(f)         
    with open(data_saved_subfolder+'/Kernel_spikehist_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_spikehist_all_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/Kernel_coefs_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_coefs_all_shuffled_dates = pickle.load(f) 
    with open(data_saved_subfolder+'/Kernel_spikehist_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'rb') as f:
        Kernel_spikehist_all_shuffled_dates = pickle.load(f) 
        
    
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
        
        # load behavioral results
        try:
            bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_from_task_code/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
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
            bhv_data_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/marmoset_tracking_bhv_data_from_task_code/"+date_tgt+"_"+animal1_filename+"_"+animal2_filename+"/"
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

            # folder and file path
            camera12_analyzed_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera12/"
            camera23_analyzed_path = "/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+videodate_tgt+"_"+animal1_filename+"_"+animal2_filename+"_camera23/"
            
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
            with open(data_saved_folder+"bhv_events_singlecam_wholebody/"+animal1_fixedorder[0]+animal2_fixedorder[0]+"/"+cameraID+'/'+date_tgt+'/output_key_locations.pkl', 'wb') as f:
                pickle.dump(output_key_locations, f)
                

        look_at_other_or_not_merge = output_look_ornot['look_at_other_or_not_merge']
        look_at_tube_or_not_merge = output_look_ornot['look_at_tube_or_not_merge']
        look_at_lever_or_not_merge = output_look_ornot['look_at_lever_or_not_merge']
        look_at_otherlever_or_not_merge = output_look_ornot['look_at_otherlever_or_not_merge']
        look_at_otherface_or_not_merge = output_look_ornot['look_at_otherface_or_not_merge']
        
        # change the unit to second and align to the start of the session
        session_start_time = session_start_times[idate]
        look_at_other_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_other_or_not_merge['dodson'])[0],1)/fps - session_start_time
        look_at_lever_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_lever_or_not_merge['dodson'])[0],1)/fps - session_start_time
        look_at_tube_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_tube_or_not_merge['dodson'])[0],1)/fps - session_start_time 
        look_at_otherlever_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_otherlever_or_not_merge['dodson'])[0],1)/fps - session_start_time
        look_at_otherface_or_not_merge['time_in_second'] = np.arange(0,np.shape(look_at_otherface_or_not_merge['dodson'])[0],1)/fps - session_start_time

        
        # find time point of behavioral events
        output_time_points_socialgaze ,output_time_points_levertube = bhv_events_timepoint_singlecam(bhv_data,look_at_other_or_not_merge,look_at_lever_or_not_merge,look_at_tube_or_not_merge)
        time_point_pull1 = output_time_points_socialgaze['time_point_pull1']
        time_point_pull2 = output_time_points_socialgaze['time_point_pull2']
        oneway_gaze1 = output_time_points_socialgaze['oneway_gaze1']
        oneway_gaze2 = output_time_points_socialgaze['oneway_gaze2']
        mutual_gaze1 = output_time_points_socialgaze['mutual_gaze1']
        mutual_gaze2 = output_time_points_socialgaze['mutual_gaze2']
        lever_gaze1 = output_time_points_levertube['time_point_lookatlever1']
        lever_gaze2 = output_time_points_levertube['time_point_lookatlever2']
        # 
        # mostly just for the sessions in which MC and SR are in the same session 
        firstpulltime = np.nanmin([np.nanmin(time_point_pull1),np.nanmin(time_point_pull2)])
        oneway_gaze1 = oneway_gaze1[oneway_gaze1>(firstpulltime-15)] # 15s before the first pull (animal1 or 2) count as the active period
        oneway_gaze2 = oneway_gaze2[oneway_gaze2>(firstpulltime-15)]
        mutual_gaze1 = mutual_gaze1[mutual_gaze1>(firstpulltime-15)]
        mutual_gaze2 = mutual_gaze2[mutual_gaze2>(firstpulltime-15)]  
        lever_gaze1 = lever_gaze1[lever_gaze1>(firstpulltime-15)]
        lever_gaze2 = lever_gaze2[lever_gaze2>(firstpulltime-15)]
        #    
        # newly added condition: only consider gaze during the active pulling time (15s after the last pull)    
        lastpulltime = np.nanmax([np.nanmax(time_point_pull1),np.nanmax(time_point_pull2)])
        oneway_gaze1 = oneway_gaze1[oneway_gaze1<(lastpulltime+15)]    
        oneway_gaze2 = oneway_gaze2[oneway_gaze2<(lastpulltime+15)]
        mutual_gaze1 = mutual_gaze1[mutual_gaze1<(lastpulltime+15)]
        mutual_gaze2 = mutual_gaze2[mutual_gaze2<(lastpulltime+15)] 
        lever_gaze1 = lever_gaze1[lever_gaze1<(lastpulltime+15)] 
        lever_gaze2 = lever_gaze2[lever_gaze2<(lastpulltime+15)] 
            
        # define successful pulls and failed pulls
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
        
        # 
        # based on time point pull and juice, define some features for each pull action
        pull_infos = get_pull_infos(animal1, animal2, time_point_pull1, time_point_pull2, 
                                    time_point_juice1, time_point_juice2)
        
        
        # new total session time (instead of 600s) - total time of the video recording
        totalsess_time = np.ceil(np.shape(output_look_ornot['look_at_lever_or_not_merge']['dodson'])[0]/30) 
        #
        # remove task irrelavant period
        if totalsess_time > (lastpulltime+session_start_time+15):
            totalsess_time = np.ceil(lastpulltime+session_start_time+15)
            
        
        
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
        # get the dataset for GLM and run GLM
        if 1:
            # get the organized data for GLM
            print('get '+neural_record_condition+' data for single camera GLM fitting')
            #
            gausKernelsize = 4 # 4; 15
            
            data_summary, data_summary_names, spiketrain_summary = get_singlecam_bhv_var_for_neuralGLM_fitting_BasisKernelsForContVaris(gausKernelsize,fps, 
                                                                                        animal1, animal2, recordedanimal, animalnames_videotrack, 
                                                                                        session_start_time, time_point_pull1, time_point_pull2, time_point_juice1, time_point_juice2, 
                                                                                        oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, 
                                                                                        output_look_ornot, output_allvectors, output_allangles, output_key_locations, 
                                                                                        spike_clusters_data, spike_time_data, spike_channels_data)
                
            # MODIFICATION: Define kernel parameters here for easy adjustment
            KERNEL_DURATION_S = 8.0  # total span: -4s to +4s
            KERNEL_OFFSET_S = -4.0   # shift so it starts at -4s
            N_BASIS_FUNCS = 26     # The number of basis functions to represent the kernel
                
            var_toglm_names = ['gaze_other_angle', 'gaze_lever_angle', # 'gaze_tube_angle',
                               'animal_animal_dist', 'animal_lever_dist', # 'animal_tube_dist',
                               'mass_move_speed', 'gaze_angle_speed',
                               'otherani_otherlever_dist', # 'otherani_othertube_dist', # 'othergaze_self_angle',
                               'other_mass_move_speed',
                               'selfpull_prob',
                               'socialgaze_prob',
                               'otherpull_prob',
                              ]
            nvars_toglm = np.shape(var_toglm_names)[0]
            
            #
            # neuralGLM with all variables
            try:
                # dummy
                print('load the session wised data for neural GLM fitting')

                current_dir = data_saved_folder+'/bhv_events_singlecam_wholebody_with_neuralglm_model'+savefile_sufix+'/'+                              animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+recordedanimal+'Recorded'
                add_date_dir = os.path.join(current_dir,cameraID+'/'+date_tgt)

                with open(add_date_dir+'/neuralGLM_kernels_coef.pkl', 'rb') as f:
                    neuralGLM_kernels_coef = pickle.load(f)
                with open(add_date_dir+'/neuralGLM_kernels_tempFilter.pkl', 'rb') as f:
                    neuralGLM_kernels_tempFilter = pickle.load(f)
                with open(add_date_dir+'/neuralGLM_kernels_coef_shf.pkl', 'rb') as f:
                    neuralGLM_kernels_coef_shf = pickle.load(f)
                with open(add_date_dir+'/neuralGLM_kernels_tempFilter_shf.pkl', 'rb') as f:
                    neuralGLM_kernels_tempFilter_shf = pickle.load(f)
                
            except:
                
                print('do GLM fitting for spike trains with continuous variables')
                
                # dp the glm for n bootstraps, each bootstrap do 80/20 training/testing
                N_BOOTSTRAPS = 100
                test_size = 0.4
                #
                dospikehist = 0
                spikehist_twin = 2
                #
                neuralGLM_kernels_coef, neuralGLM_kernels_tempFilter,                 neuralGLM_kernels_coef_shf, neuralGLM_kernels_tempFilter_shf, _                         = neuralGLM_fitting_BasisKernelsForContVaris(KERNEL_DURATION_S, KERNEL_OFFSET_S,
                                                N_BASIS_FUNCS, fps, 
                                                animal1, animal2, recordedanimal,
                                                var_toglm_names, data_summary_names, data_summary, 
                                                spiketrain_summary, dospikehist, spikehist_twin, 
                                                N_BOOTSTRAPS,test_size )
                
                # save data
                if 1:
                    current_dir = data_saved_folder+'/bhv_events_singlecam_wholebody_with_neuralglm_model'+savefile_sufix+'/'+                                  animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+recordedanimal+'Recorded'
                    add_date_dir = os.path.join(current_dir,cameraID+'/'+date_tgt)
                    if not os.path.exists(add_date_dir):
                        os.makedirs(add_date_dir)
                    #
                    with open(add_date_dir+'/neuralGLM_kernels_coef.pkl', 'wb') as f:
                        pickle.dump(neuralGLM_kernels_coef, f)
                    with open(add_date_dir+'/neuralGLM_kernels_tempFilter.pkl', 'wb') as f:
                        pickle.dump(neuralGLM_kernels_tempFilter, f)
                    with open(add_date_dir+'/neuralGLM_kernels_coef_shf.pkl', 'wb') as f:
                        pickle.dump(neuralGLM_kernels_coef_shf, f)
                    with open(add_date_dir+'/neuralGLM_kernels_tempFilter_shf.pkl', 'wb') as f:
                        pickle.dump(neuralGLM_kernels_tempFilter_shf, f)
                        
                
            #
            # neuralGLM with all projected variables to the three axies
            try:
                # dummy
                print('load the session wised data for neural GLM fitting - pull gaze juice projected axes neural-glm')

                current_dir = data_saved_folder+'/bhv_events_singlecam_wholebody_with_neuralglm_model'+savefile_sufix+'/'+                              animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+recordedanimal+'Recorded'
                add_date_dir = os.path.join(current_dir,cameraID+'/'+date_tgt)

                with open(add_date_dir+'/neuralGLM_mainAxesProjected_kernels_coef.pkl', 'rb') as f:
                    neuralGLM_mainAxesProjected_kernels_coef = pickle.load(f)
                with open(add_date_dir+'/neuralGLM_mainAxesProjected_kernels_tempFilter.pkl', 'rb') as f:
                    neuralGLM_mainAxesProjected_kernels_tempFilter = pickle.load(f)
                with open(add_date_dir+'/neuralGLM_mainAxesProjected_kernels_coef_shf.pkl', 'rb') as f:
                    neuralGLM_mainAxesProjected_kernels_coef_shf = pickle.load(f)
                with open(add_date_dir+'/neuralGLM_mainAxesProjected_kernels_tempFilter_shf.pkl', 'rb') as f:
                    neuralGLM_mainAxesProjected_kernels_tempFilter_shf = pickle.load(f)
                
            except:
                
                print('do GLM fitting for spike trains with continuous variables - pull gaze juice projected axes neural-glm')
                
                # dp the glm for n bootstraps, each bootstrap do 80/20 training/testing
                N_BOOTSTRAPS = 100
                test_size = 0.4
                #
                dospikehist = 0
                spikehist_twin = 2
                #
                neuralGLM_mainAxesProjected_kernels_coef, neuralGLM_mainAxesProjected_kernels_tempFilter,                 neuralGLM_mainAxesProjected_kernels_coef_shf, neuralGLM_mainAxesProjected_kernels_tempFilter_shf,                 var_toglm_names_mainAxesProjected = neuralGLM_fitting_BasisKernelsForContVaris_PullGazeVectorProjection(
                                                KERNEL_DURATION_S, KERNEL_OFFSET_S,
                                                N_BASIS_FUNCS, fps, 
                                                animal1, animal2, recordedanimal,
                                                var_toglm_names, data_summary_names, data_summary, 
                                                spiketrain_summary, dospikehist, spikehist_twin, 
                                                N_BOOTSTRAPS,test_size )
                
                # save data
                if 1:
                    current_dir = data_saved_folder+'/bhv_events_singlecam_wholebody_with_neuralglm_model'+savefile_sufix+'/'+                                  animal1_fixedorder[0]+animal2_fixedorder[0]+'_'+recordedanimal+'Recorded'
                    add_date_dir = os.path.join(current_dir,cameraID+'/'+date_tgt)
                    if not os.path.exists(add_date_dir):
                        os.makedirs(add_date_dir)
                    #
                    with open(add_date_dir+'/neuralGLM_mainAxesProjected_kernels_coef.pkl', 'wb') as f:
                        pickle.dump(neuralGLM_mainAxesProjected_kernels_coef, f)
                    with open(add_date_dir+'/neuralGLM_mainAxesProjected_kernels_tempFilter.pkl', 'wb') as f:
                        pickle.dump(neuralGLM_mainAxesProjected_kernels_tempFilter, f)
                    with open(add_date_dir+'/neuralGLM_mainAxesProjected_kernels_coef_shf.pkl', 'wb') as f:
                        pickle.dump(neuralGLM_mainAxesProjected_kernels_coef_shf, f)
                    with open(add_date_dir+'/neuralGLM_mainAxesProjected_kernels_tempFilter_shf.pkl', 'wb') as f:
                        pickle.dump(neuralGLM_mainAxesProjected_kernels_tempFilter_shf, f)
                        

                        
            #
            # determine if each neuron is encoding each variables and are they predicting or reacting based on the time
            if 0:
                neuronIDs = np.array(list(neuralGLM_kernels_coef.keys()))
                nneurons = np.shape(neuronIDs)[0]

                for ineuron in np.arange(0,nneurons,1):
                    neuronID = neuronIDs[ineuron]

                    real_coefs = np.array(neuralGLM_kernels_coef[neuronID])       # shape: (N_BOOTSTRAPS, nvars_toglm, N_BASIS_FUNCS)
                    shuffled_coefs = np.array(neuralGLM_kernels_coef_shf[neuronID])  # shape: (N_BOOTSTRAPS, nvars_toglm, N_BASIS_FUNCS)

                    # Compute the mean beta over bootstraps for real data
                    real_mean = np.mean(real_coefs, axis=0)

                    time_axis = np.linspace(KERNEL_OFFSET_S, KERNEL_DURATION_S+KERNEL_OFFSET_S, N_BASIS_FUNCS)  # same time window used in kernel

                    sig_vars, sig_timing = cluster_based_correction_with_timing(real_mean, shuffled_coefs, 
                                                                alpha=0.01, time_axis=time_axis)

                    for i, (is_sig, timing) in enumerate(zip(sig_vars, sig_timing)):
                        if is_sig:
                            print(f"Variable {i}: Significant ({timing})")
                        else:
                            print(f"Variable {i}: Not significant")
                
            

    # save data
    if 0:
        
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_neuralGLM_new'+savefile_sufix+'/'+cameraID+'/'+animal1_fixedorders[0]+animal2_fixedorders[0]+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)
                
        # GLM to behavioral events (actions)
        if 0:
            with open(data_saved_subfolder+'/Kernel_coefs_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_coefs_all_dates, f)    
            with open(data_saved_subfolder+'/Kernel_spikehist_all_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_spikehist_all_dates, f)    
            with open(data_saved_subfolder+'/Kernel_coefs_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_coefs_all_shuffled_dates, f)    
            with open(data_saved_subfolder+'/Kernel_spikehist_all_shuffled_dates_'+animal1_fixedorders[0]+animal2_fixedorders[0]+'.pkl', 'wb') as f:
                pickle.dump(Kernel_spikehist_all_shuffled_dates, f)  
           

    
    
    


# In[ ]:


bb


# In[ ]:


np.floor(1)


# In[19]:


np.shape(data_summary[-3])


# In[20]:


data_summary_names[-3]


# In[21]:


np.shape(spiketrain_summary[95])


# In[54]:


spiketrain_summary


# In[ ]:





# In[39]:


print(neuralGLM_mainAxesProjected_kernels_coef.keys())
print(neuralGLM_kernels_coef_shf.keys())


# In[40]:


np.shape(neuralGLM_mainAxesProjected_kernels_coef[2])


# In[ ]:


# testing for now
# project the 8D continuous variables to smaller dimension that are more task relavant

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.linalg import orth

# 1. Load and extract behavioral data
var_toPCA_names = ['gaze_other_angle', 'gaze_lever_angle', 'gaze_tube_angle',
                   'animal_animal_dist', 'animal_lever_dist', 'animal_tube_dist',
                   'mass_move_speed', 'gaze_angle_speed']
pull_axis_name = ['selfpull_prob']
gaze_axis_name = ['socialgaze_prob']
juice_axis_name = ['selfjuice_prob']

# Indices
PCAindices_in_summary = [data_summary_names.index(var) for var in var_toPCA_names]
Pullindices_in_summary = [data_summary_names.index(var) for var in pull_axis_name]
Gazeindices_in_summary = [data_summary_names.index(var) for var in gaze_axis_name]
Juiceindices_in_summary = [data_summary_names.index(var) for var in juice_axis_name]

# Data extraction
data_summary = np.array(data_summary)
vars_toPCA = data_summary[PCAindices_in_summary]        # shape (8, T)
var_pull = data_summary[Pullindices_in_summary][0]      # shape (T,)
var_gaze = data_summary[Gazeindices_in_summary][0]
var_juice = data_summary[Juiceindices_in_summary][0]

# 2. Z-score behavioral variables
scaler = StandardScaler()
vars_z = scaler.fit_transform(vars_toPCA.T).T            # (8, T)

# 3. Get raw projection vectors (not yet orthogonal)
gaze_weights = vars_z @ var_gaze        # shape (8,)
pull_weights = vars_z @ var_pull
juice_weights = vars_z @ var_juice

# Normalize to get direction vectors
gaze_dir = gaze_weights / np.linalg.norm(gaze_weights)

# Orthogonalize pull_dir to gaze_dir
pull_dir = pull_weights - (gaze_dir @ pull_weights) * gaze_dir
pull_dir /= np.linalg.norm(pull_dir)

# Orthogonalize juice_dir to both gaze and pull
Q = orth(np.stack([gaze_dir, pull_dir], axis=1))  # (8,2)
proj_mat = Q @ Q.T
juice_resid = juice_weights - proj_mat @ juice_weights
juice_dir = juice_resid / np.linalg.norm(juice_resid)

# 4. Project onto mode directions
gaze_PC = gaze_dir @ vars_z       # (T,)
pull_PC = pull_dir @ vars_z
juice_PC = juice_dir @ vars_z


# In[ ]:


print("Correlation (pull vs gaze):", np.corrcoef(pull_PC, gaze_PC)[0, 1])
print("Correlation (pull vs juice):", np.corrcoef(pull_PC, juice_PC)[0, 1])
print("Correlation (gaze vs juice):", np.corrcoef(gaze_PC, juice_PC)[0, 1])


# In[ ]:


# Assume these are unit vectors of shape (n_dims,)
vectors = [pull_dir, gaze_dir, residual_dir]

# Enforce orthogonality
orthogonal_dirs = gram_schmidt(vectors)

# Unpack them
pull_orth, gaze_orth, residual_orth = orthogonal_dirs

# Check orthogonality again
check_orthogonality(pull_orth, gaze_orth, residual_orth)

# Project behavioral data into 3D mode traces:
# pull_PC = pull_orth @ vars_z  # shape: (T,)
# gaze_PC = gaze_orth @ vars_z
# residual_PC = residual_orth @ vars_z


# In[ ]:


ind_ = [10000,12000]
plt.plot(pull_PC[ind_[0]:ind_[1]],label='pull_PC')
plt.plot(gaze_PC[ind_[0]:ind_[1]],label='gaze_PC')
plt.plot(juice_PC[ind_[0]:ind_[1]],label='juice_PC')
# plt.plot(residual_PC[ind_[0]:ind_[1]],label='residual_PC')
plt.plot(var_pull[ind_[0]:ind_[1]]/4,label='var_pull')
# plt.plot(var_juice[ind_[0]:ind_[1]]/4,label='var_juice')
# plt.plot(var_gaze[ind_[0]:ind_[1]],label='var_gaze')
# plt.plot(vars_toPCA[6,ind_[0]:ind_[1]],label='mass_speed')
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


KERNEL_DURATION_S = 8.0  # total span: -4s to +4s
KERNEL_OFFSET_S = -4.0   # shift so it starts at -4s
N_BASIS_FUNCS = 20     # The number of basis functions to represent the kernel


# In[ ]:


N_BOOTSTRAPS =10
test_size = 0.6


# In[ ]:



import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.signal import convolve
import string
import warnings
import pickle    
import random as random
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

#
def make_raised_cosine_basis(duration_s, n_basis, dt, offset_s=0.0):
    t = np.arange(offset_s, offset_s + duration_s, dt)  # e.g., from -4 to +4 seconds
    centers = np.linspace(offset_s, offset_s + duration_s, n_basis)
    width = (centers[1] - centers[0]) * 1.5  # spread of each cosine

    basis = []
    for ci in centers:
        phi = (t - ci) * np.pi / width
        b = np.cos(np.clip(phi, -np.pi, np.pi))
        b = (b + 1) / 2
        b[(t < ci - width/2) | (t > ci + width/2)] = 0  # zero out beyond the support
        basis.append(b)

    basis = np.stack(basis, axis=1)  # shape: [time, n_basis]
    return basis, t


#
def make_gaussian_basis(duration_s, n_basis, dt, offset_s=0.0, sigma_scale=1.5):
    t = np.arange(offset_s, offset_s + duration_s, dt)  # e.g., -4 to 4s
    centers = np.linspace(offset_s, offset_s + duration_s, n_basis)
    sigma = (centers[1] - centers[0]) * sigma_scale

    basis = []
    for c in centers:
        b = np.exp(-0.5 * ((t - c) / sigma) ** 2)
        basis.append(b)

    basis = np.stack(basis, axis=1)  # shape: [time, n_basis]
    return basis, t

#
def make_square_basis(duration_s, n_basis, dt):
    """
    Create square (boxcar) basis functions evenly tiling [-duration_s/2, duration_s/2]
    """
    t = np.arange(-duration_s / 2, duration_s / 2, dt)
    n_timepoints = len(t)
    basis = np.zeros((n_timepoints, n_basis))

    # Get bin edges using np.array_split for even division
    indices = np.array_split(np.arange(n_timepoints), n_basis)
    
    for i, idx in enumerate(indices):
        basis[idx, i] = 1

    return basis, t

#
def convolve_with_basis(var, basis_funcs):
    return np.stack([
        convolve(var, basis, mode='full')[:len(var)]
        for basis in basis_funcs.T
    ], axis=1)


# In[ ]:


dt = 1 / fps

# basis_funcs, time_vector = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
basis_funcs, time_vector = make_gaussian_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
# basis_funcs, t_basis = make_square_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)

plt.figure(figsize=(8, 3))
for i in range(basis_funcs.shape[1]):
    plt.plot(basis_funcs[:, i], label=f'Basis {i+1}')
plt.title("Gaussian Temporal Basis Functions")
plt.xlabel("Time bins")
plt.ylabel("Amplitude")
plt.show()


# In[ ]:


data_summary = np.array(data_summary)
np.shape(data_summary)


# In[ ]:


np.shape(pull_PC)


# In[ ]:


np.shape(np.vstack([pull_PC, gaze_PC, juice_PC]))


# In[ ]:


# do the glm fitting with the projected vectors
# 
# def neuralGLM_fitting_BasisKernelsForContVaris(KERNEL_DURATION_S, N_BASIS_FUNCS, fps, animal1, animal2, 
#                                                recordedanimal, data_summary_names, data_summary, 
#                                                spiketrain_summary, nbootstraps, dospikehist, spikehist_twin, 
#                                                N_BOOTSTRAPS,test_size ):

dt = 1 / fps

# basis_funcs, time_vector = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
basis_funcs, time_vector = make_gaussian_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
# basis_funcs, t_basis = make_square_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)


####
# do the glm fitting
####

var_toglm_names = ['pull_PC', 'gaze_PC', 'juice_PC']

# projected pull, gaze, juice action vector
predictors = np.vstack([pull_PC, gaze_PC, juice_PC])

# Design matrix from continuous variables
X_continuous = np.hstack([convolve_with_basis(v, basis_funcs) for v in predictors])
#
# zscore again
scaler = StandardScaler()
X_continuous_z = scaler.fit_transform(X_continuous)


# do the glm for each neuron
neuron_clusters = list(spiketrain_summary.keys())
nclusters = np.shape(neuron_clusters)[0]


# Track kernel for each var × basis
n_vars = len(var_toglm_names)
n_basis = basis_funcs.shape[1]
T_kernel = basis_funcs.shape[0]  # length of time kernel

# storage
Kernel_coefs_allboots_allcells = {}
Kernel_coefs_spikehist_allboots_allcells = {}
Kernel_coefs_allboots_allcells_shf = {}
Kernel_coefs_spikehist_allboots_allcells_shf = {}
#
Temporal_filters_allcells = dict.fromkeys(neuron_clusters, None)
Temporal_filters_spikehist_allcells = dict.fromkeys(neuron_clusters, None)
Temporal_filters_allcells_shf = dict.fromkeys(neuron_clusters, None)
Temporal_filters_spikehist_allcells_shf = dict.fromkeys(neuron_clusters, None)

#    
# for icluster in np.arange(0,nclusters,1):
for icluster in np.arange(0,1,1):
    iclusterID = neuron_clusters[icluster]

    # Binary spike train
    # Y = (spiketrain_summary[iclusterID] > 0).astype(int)
    Y = spiketrain_summary[iclusterID]
    #
    Y_shuffled = np.random.permutation(Y)

    #
    Kernel_coefs_boots = []
    filters_boot = []
    #
    Kernel_coefs_boots_shf = []
    filters_boot_shf = []

    for i in range(N_BOOTSTRAPS):

        # Train/test split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_continuous, Y, test_size=0.2, random_state=random.randint(0, 10000)
            )

        # Fit Poisson GLM with L2 penalty
        clf_full = PoissonRegressor(alpha=10, max_iter=500)  # alpha controls regularization strength
        clf_full.fit(X_tr, y_tr)

        # Extract coefficients
        full_beta = clf_full.coef_.flatten()
        kernel_matrix = full_beta.reshape(n_vars, n_basis)
        Kernel_coefs_boots.append(kernel_matrix)

        # Reconstruct temporal filter
        temporal_filter = np.dot(kernel_matrix, basis_funcs.T)  # (n_vars, T_kernel)
        filters_boot.append(temporal_filter)

        # SHUFFLED CONTROL
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_continuous, Y_shuffled, test_size=0.2, random_state=random.randint(0, 10000)
        )

        clf_shuffled = PoissonRegressor(alpha=10, max_iter=500)
        clf_shuffled.fit(X_tr, y_tr)

        full_beta_shf = clf_shuffled.coef_.flatten()
        kernel_matrix_shf = full_beta_shf.reshape(n_vars, n_basis)
        Kernel_coefs_boots_shf.append(kernel_matrix_shf)

        temporal_filter_shf = np.dot(kernel_matrix_shf, basis_funcs.T)
        filters_boot_shf.append(temporal_filter_shf)


    # Save as array (n_boots, n_vars, T_kernel)
    Kernel_coefs_allboots_allcells[iclusterID] = np.array(Kernel_coefs_boots)  # shape: (n_boots, n_vars, n_basis)
    Temporal_filters_allcells[iclusterID] = np.array(filters_boot) # (n_boots, n_vars, T_kernel)

    Kernel_coefs_allboots_allcells_shf[iclusterID] = np.array(Kernel_coefs_boots_shf)  # shape: (n_boots, n_vars, n_basis)
    Temporal_filters_allcells_shf[iclusterID] = np.array(filters_boot_shf) # (n_boots, n_vars, T_kernel)



neuralGLM_kernels_coef = Kernel_coefs_allboots_allcells
neuralGLM_kernels_tempFilter = Temporal_filters_allcells
neuralGLM_kernels_coef_shf = Kernel_coefs_allboots_allcells_shf
neuralGLM_kernels_tempFilter_shf = Temporal_filters_allcells_shf

# return neuralGLM_kernels_coef, neuralGLM_kernels_tempFilter, neuralGLM_kernels_coef_shf, neuralGLM_kernels_tempFilter_shf




# In[ ]:


print(var_toglm_names[0])
plt.plot(np.nanmean(neuralGLM_kernels_tempFilter[2][:,0,:],axis=0))


# In[ ]:


plt.plot(neuralGLM_kernels_tempFilter_shf[2][:,1,:].T)


# In[ ]:


all_kernels = np.stack(neuralGLM_kernels_coef[2], axis=0)
print("Filter shape:", all_kernels.shape)  # (n_bootstraps, n_predictors * n_basis)

# Compute variance across bootstraps
filter_std = np.std(all_kernels, axis=0)
plt.plot(filter_std)
plt.title("Standard deviation across bootstraps")
plt.xlabel("Kernel index")
plt.ylabel("STD")
plt.show()


# In[ ]:


filters = np.stack(neuralGLM_kernels_tempFilter_shf[2], axis=0)  # shape: (n_bootstraps, n_vars, T)
mean_filter = filters.mean(axis=0)
std_filter = filters.std(axis=0)

# Plot mean ± std for each variable
for i, name in enumerate(var_toglm_names):
    plt.figure()
    plt.fill_between(np.arange(mean_filter.shape[1]), 
                     mean_filter[i] - std_filter[i], 
                     mean_filter[i] + std_filter[i], 
                     alpha=0.3, label='±1 STD')
    plt.plot(mean_filter[i], label='Mean filter')
    plt.title(name)
    plt.legend()
    plt.show()


# In[ ]:


np.unique(y_tr)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Example shape: 16 signals × 11315 timepoints
# data_summary = np.random.randn(16, 11315)  # <-- Replace with your actual data

print(var_toglm_names)

# Compute Pearson correlation across rows (pairwise)
# corr_matrix = np.corrcoef(data_summary)
corr_matrix = np.corrcoef(predictors)


# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='vlag', square=True, 
            xticklabels=np.arange(1, np.shape(var_toglm_names)[0]), 
            yticklabels=np.arange(1, np.shape(var_toglm_names)[0]))
plt.title("Cross-Correlation Heatmap (16x16)")
plt.xlabel("Signal Index")
plt.ylabel("Signal Index")
plt.tight_layout()
plt.show()


# In[ ]:


data_summary_names[3]


# In[ ]:


data_summary_names[11]


# In[ ]:


data_summary_names


# In[ ]:


animal2


# In[ ]:





# In[ ]:


np.shape(spiketrain_summary[2])


# In[ ]:


totalsess_time*30


# ## plot - for individual animal
# ### prepare the summarizing data set and run population level analysis such as PCA
# ### plot the kernel defined based on the stretagy (pair of action)

# In[ ]:


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





#!/usr/bin/env python
# coding: utf-8

# ### In this script, DBN has run and this script is used to make predictions
# ### In this script, DBN is run with 1s time bin, 3 time lag 
# ### In this script, the animal tracking is done with only one camera - camera 2 (middle) 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn
import scipy
import scipy.stats as st
import sklearn
from sklearn.neighbors import KernelDensity
import string
import warnings
import pickle

import os
import glob
import random
from time import time

from pgmpy.models import BayesianModel
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch,BicScore
from pgmpy.base import DAG
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


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


# ### function - make demo videos with skeleton and inportant vectors

# In[8]:


from ana_functions.tracking_video_singlecam_demo import tracking_video_singlecam_demo
from ana_functions.tracking_video_singlecam_wholebody_demo import tracking_video_singlecam_wholebody_demo


# ### function - interval between all behavioral events

# In[9]:


from ana_functions.bhv_events_interval import bhv_events_interval
from ana_functions.bhv_events_interval import bhv_events_interval_certainEdges


# ### function - train the dynamic bayesian network - multi time lag (3 lags)

# In[10]:


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

# In[11]:


# instead of using gaze angle threshold, use the target rectagon to deside gaze info
# ...need to update
sqr_thres_tubelever = 75 # draw the square around tube and lever
sqr_thres_face = 1.15 # a ratio for defining face boundary
sqr_thres_body = 4 # how many times to enlongate the face box boundry to the body


# get the fps of the analyzed video
fps = 30

# frame number of the demo video
# nframes = 0.5*30 # second*30fps
nframes = 2*30 # second*30fps

# re-analyze the video or not
reanalyze_video = 0
redo_anystep = 0

# session list options
do_bestsession = 1 # only analyze the best (five) sessions for each conditions during the training phase
do_trainedMCs = 1 # the list that only consider trained (1s) MC, together with SR and NV as controls
if do_bestsession:
    if not do_trainedMCs:
        savefile_sufix = '_bestsessions'
    elif do_trainedMCs:
        savefile_sufix = '_trainedMCsessions'
else:
    savefile_sufix = ''

# which camera to analyzed
cameraID = 'camera-2'
cameraID_short = 'cam2'

# where to save the summarizing data
data_saved_folder = '/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/3d_recontruction_analysis_self_and_coop_task_data_saved/'


    


# ### load the DBN related data for each dyad and run the prediction
# ### For each condition, only use the hypothetical dependencies
# ### For each dyad align animal1 as the subordinate animal and animal2 as the donimant animal
# ### for each dyad, for each iternation, train on 80% of data but test on 20% of other dyad

# In[43]:


# Suppress all warnings
warnings.simplefilter('ignore')

redoFitting = 0

do_succfull = 0

niters = 100

# PLOT multiple pairs in one plot, so need to load data seperately
moreSampSize = 0
mergetempRos = 0 # 1: merge different time bins

#
animal1_fixedorders = ['eddie','dodson','ginger','dannon','koala']
animal2_fixedorders = ['sparkle','scorch','kanga','kanga','vermelho']
# animal1_fixedorders = ['eddie',]
# animal2_fixedorders = ['sparkle',]
nanimalpairs = np.shape(animal1_fixedorders)[0]

# donimant animal name; since animal1 and 2 are already aligned, no need to check it; but keep it here for reference
dom_animal_names = ['sparkle','scorch','kanga','kanga','vermelho']

temp_resolu = 1

# ONLY FOR PLOT!! 
# define DBN related summarizing variables
# DBN_group_typenames = ['self','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
# DBN_group_typeIDs  =  [1,3,3,3,5]
# DBN_group_coopthres = [0,2,1.5,1,0]
DBN_group_typenames = ['coop(2s)']
DBN_group_typeIDs  =  [3,]
DBN_group_coopthres = [2,]
if do_trainedMCs:
    DBN_group_typenames = ['coop(1s)']
    DBN_group_typeIDs  =  [3]
    DBN_group_coopthres = [1]
nDBN_groups = np.shape(DBN_group_typenames)[0]

# DBN input data - to and from Nodes
toNodes = ['pull1_t3','pull2_t3','owgaze1_t3','owgaze2_t3']
fromNodes = ['pull1_t2','pull2_t2','owgaze1_t2','owgaze2_t2']
eventnames = ["M1pull","M2pull","M1gaze","M2gaze"]
nevents = np.shape(eventnames)[0]

timelagtype = '' # '' means 1secondlag, otherwise will be specificed
#
if timelagtype == '2secondlag':
    fromNodes = ['pull1_t1','pull2_t1','owgaze1_t1','owgaze2_t1']
if timelagtype == '3secondlag':
    fromNodes = ['pull1_t0','pull2_t0','owgaze1_t0','owgaze2_t0']

    
# hypothetical graph structure that reflect the strategies
# hypothetical graph structure that reflect the strategies
# strategynames = ['threeMains','sync_pulls','gaze_lead_pull','social_attention','other_dependencies','other_noself_dependcies']
strategynames = ['sync_pulls','gaze_lead_pull',]
# strategynames = ['threeMains',]
# strategynames = ['gaze_lead_pull'] # ['all_threes','sync_pulls','gaze_lead_pull','social_attention']
bina_graphs_specific_strategy = {
    'threeMains': np.array([[0,1,0,1],[1,0,1,0],[1,0,0,0],[0,1,0,0]]),
    'sync_pulls': np.array([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]),
    'gaze_lead_pull':np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]]),
    'social_attention':np.array([[0,0,0,1],[0,0,1,0],[0,0,0,0],[0,0,0,0]]),
    'other_dependencies': np.array([[1,0,1,0],[0,1,0,1],[0,1,1,1],[1,0,1,1]]),
    'other_noself_dependcies': np.array([[0,0,1,0],[0,0,0,1],[0,1,0,1],[1,0,1,0]]),
}
nstrategies_forplot = np.shape(strategynames)[0]


for istrg in np.arange(0,nstrategies_forplot,1):
    
    strategyname = strategynames[istrg]

    #
    bina_graph_mean_strg = bina_graphs_specific_strategy[strategyname]
    
    # translate the binary DAGs to edge
    nrows,ncols = np.shape(bina_graph_mean_strg)
    edgenames = []
    for irow in np.arange(0,nrows,1):
        for icol in np.arange(0,ncols,1):
            if bina_graph_mean_strg[irow,icol] > 0:
                edgenames.append((fromNodes[irow],toNodes[icol]))

    # define the DBN predicting model
    bn = BayesianNetwork()
    bn.add_nodes_from(fromNodes)
    bn.add_nodes_from(toNodes)
    bn.add_edges_from(edgenames)
    
    effect_slice = toNodes
    
    # load ROC_summary_all data
    try:
        if redoFitting:
            dumpy
        
        print('load all ROC data for hypothetical dependencies, and only plot the summary figure')
        
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions_cross_dyad_validation/'+                               savefile_sufix+'/'+cameraID+'/'

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_'+strategyname+timelagtype+'.pkl', 'rb') as f:
            ROC_summary_all = pickle.load(f)
   
    except:  
    
        # initialize a summary dataframe for plotting the summary figure across animals 
        ROC_summary_all = pd.DataFrame(columns=['train_animal','test_animal','action','testCondition','predROC'])

        #
        # session type to analyze
        for igroup in np.arange(0,nDBN_groups,1):
            DBN_group_typename = DBN_group_typenames[igroup]
               
            #    
            # load the dyad for training
            for ianimalpair_train in np.arange(0,nanimalpairs,1):

                #
                # load the DBN input data
                animal1_train = animal1_fixedorders[ianimalpair_train]
                animal2_train = animal2_fixedorders[ianimalpair_train]
                #
                # only for kanga
                if animal2_train == 'kanga':
                    if animal1_train == 'ginger':
                        animal2_train_nooverlap = 'kanga_withG'
                    elif animal1_train == 'dannon':
                        animal2_train_nooverlap = 'kanga_withD'
                    else:
                        animal2_train_nooverlap = animal2_train
                else:
                    animal2_train_nooverlap = animal2_train
                #
                if not do_succfull:
                    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_train+animal2_train+'/'
                    if not mergetempRos:
                        with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_train+animal2_train+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                            DBN_input_data_alltypes = pickle.load(f)
                    else:
                        with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_train+animal2_train+'_mergeTempsReSo.pkl', 'rb') as f:
                            DBN_input_data_alltypes = pickle.load(f)
                    #
                    DBN_input_data_train = DBN_input_data_alltypes[DBN_group_typename]
                #    
                elif do_succfull:
                    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_SuccAndFailedPull_newDefinition'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_train+animal2_train+'/'
                    if not mergetempRos:
                        with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_train+animal2_train+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                            DBN_input_data_alltypes = pickle.load(f)
                    else:
                        with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_train+animal2_train+'_mergeTempsReSo.pkl', 'rb') as f:
                            DBN_input_data_alltypes = pickle.load(f)
                    #
                    DBN_input_data_train = DBN_input_data_alltypes['succpull'][DBN_group_typename]
            
                #    
                # load the dyad for testing
                for ianimalpair_test in np.arange(0,nanimalpairs,1):

                    # load the DBN input data
                    animal1_test = animal1_fixedorders[ianimalpair_test]
                    animal2_test = animal2_fixedorders[ianimalpair_test]
                    #
                    # only for kanga
                    if animal2_test == 'kanga':
                        if animal1_test == 'ginger':
                            animal2_test_nooverlap = 'kanga_withG'
                        elif animal1_test == 'dannon':
                            animal2_test_nooverlap = 'kanga_withD'
                        else:
                            animal2_test_nooverlap = animal2_test
                    else:
                        animal2_test_nooverlap = animal2_test
                    #
                    if not do_succfull:
                        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_test+animal2_test+'/'
                        if not mergetempRos:
                            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_test+animal2_test+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                                DBN_input_data_alltypes = pickle.load(f)
                        else:
                            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_test+animal2_test+'_mergeTempsReSo.pkl', 'rb') as f:
                                DBN_input_data_alltypes = pickle.load(f)
                        #
                        DBN_input_data_test = DBN_input_data_alltypes[DBN_group_typename]
                    #    
                    elif do_succfull:
                        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody_SuccAndFailedPull_newDefinition'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_test+animal2_test+'/'
                        if not mergetempRos:
                            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_test+animal2_test+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                                DBN_input_data_alltypes = pickle.load(f)
                        else:
                            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_test+animal2_test+'_mergeTempsReSo.pkl', 'rb') as f:
                                DBN_input_data_alltypes = pickle.load(f)
                        #
                        DBN_input_data_test = DBN_input_data_alltypes['succpull'][DBN_group_typename]
           
        
                    #
                    # run niters iterations for each condition
                    for iiter in np.arange(0,niters,1):


                        # Split data into training and testing sets
                        train_data, _ = train_test_split(DBN_input_data_train, test_size=0.2)
                        _,  test_data = train_test_split(DBN_input_data_test, test_size=0.5)

                        # Perform parameter learning for each time slice
                        bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                        # Perform inference
                        infer = VariableElimination(bn)

                        # Prediction for each behavioral events
                        # With aligned animals across dyad - animal1:sub, animal2:dom
                        for ievent in np.arange(0,nevents,1):

                            var = effect_slice[ievent]
                            Pbehavior = [] # Initialize log-likelihood

                            for index, row in test_data.iterrows():
                                evidence = {fromNodes[0]: row[fromNodes[0]], 
                                            fromNodes[1]: row[fromNodes[1]], 
                                            fromNodes[2]: row[fromNodes[2]], 
                                            fromNodes[3]: row[fromNodes[3]], }

                                # Query the probability distribution for Pulls given evidence
                                aucPpredBehavior = infer.query(variables=[var], evidence=evidence) 

                                # Extract the probability of outcome = 1
                                prob = aucPpredBehavior.values[1]
                                Pbehavior = np.append(Pbehavior, prob)

                            # Calculate the AUC score
                            trueBeh = test_data[var].values
                            try:
                                auc = roc_auc_score(trueBeh, Pbehavior)
                            except:
                                auc = np.nan
                            print(f"AUC Score: {auc:.4f}")

                            # put data in the summarizing data frame
                            if (ievent == 0) | (ievent == 2): # for animal1
                                ROC_summary_all = ROC_summary_all.append({'train_animal':animal1_train,
                                                                          'test_animal':animal1_test,
                                                                          'train_dyadID':ianimalpair_train,
                                                                          'test_dyadID':ianimalpair_test,
                                                                          'action':eventnames[ievent][2:],
                                                                          'testCondition':DBN_group_typename,
                                                                          'predROC':auc,
                                                                          'iters': iiter,
                                                                         }, ignore_index=True)
                            else:
                                ROC_summary_all = ROC_summary_all.append({'train_animal':animal2_train_nooverlap,
                                                                          'test_animal':animal2_test_nooverlap,
                                                                          'train_dyadID':ianimalpair_train,
                                                                          'test_dyadID':ianimalpair_test,
                                                                          'action':eventnames[ievent][2:],
                                                                          'testCondition':DBN_group_typename,
                                                                          'predROC':auc,
                                                                          'iters': iiter,
                                                                     }, ignore_index=True)
                                
                        #         
                        # Prediction for each behavioral events with swapped animal1 and animal2
                        # With training set and testing set has swapped animal type:
                        # training set: animal1 - sub; animal2 - dom
                        # testing set: animal1 - dom; animal2 - sub
                        test_data_swap = test_data.copy()
                        # Create a column mapping
                        new_columns = {}
                        for col in test_data_swap.columns:
                            if 'pull1' in col:
                                new_columns[col] = col.replace('pull1', 'pull2')
                            elif 'pull2' in col:
                                new_columns[col] = col.replace('pull2', 'pull1')
                            elif 'owgaze1' in col:
                                new_columns[col] = col.replace('owgaze1', 'owgaze2')
                            elif 'owgaze2' in col:
                                new_columns[col] = col.replace('owgaze2', 'owgaze1')
                        # Rename the columns using the mapping
                        test_data_swap = test_data_swap.rename(columns=new_columns)
                        
                        for ievent in np.arange(0,nevents,1):

                            var = effect_slice[ievent]
                            Pbehavior = [] # Initialize log-likelihood

                            for index, row in test_data_swap.iterrows():
                                evidence = {fromNodes[0]: row[fromNodes[0]], 
                                            fromNodes[1]: row[fromNodes[1]], 
                                            fromNodes[2]: row[fromNodes[2]], 
                                            fromNodes[3]: row[fromNodes[3]], }

                                # Query the probability distribution for Pulls given evidence
                                aucPpredBehavior = infer.query(variables=[var], evidence=evidence) 

                                # Extract the probability of outcome = 1
                                prob = aucPpredBehavior.values[1]
                                Pbehavior = np.append(Pbehavior, prob)

                            # Calculate the AUC score
                            trueBeh = test_data_swap[var].values
                            try:
                                auc = roc_auc_score(trueBeh, Pbehavior)
                            except:
                                auc = np.nan
                            print(f"AUC Score: {auc:.4f}")

                            # put data in the summarizing data frame
                            if (ievent == 0) | (ievent == 2): # for animal1
                                ROC_summary_all = ROC_summary_all.append({'train_animal':animal1_train,
                                                                          'test_animal':animal2_test_nooverlap,
                                                                          'train_dyadID':ianimalpair_train,
                                                                          'test_dyadID':ianimalpair_test,
                                                                          'action':eventnames[ievent][2:],
                                                                          'testCondition':DBN_group_typename,
                                                                          'predROC':auc,
                                                                          'iters': iiter,
                                                                         }, ignore_index=True)
                            else:
                                ROC_summary_all = ROC_summary_all.append({'train_animal':animal2_train_nooverlap,
                                                                          'test_animal':animal1_test,
                                                                          'train_dyadID':ianimalpair_train,
                                                                          'test_dyadID':ianimalpair_test,
                                                                          'action':eventnames[ievent][2:],
                                                                          'testCondition':DBN_group_typename,
                                                                          'predROC':auc,
                                                                          'iters': iiter,
                                                                     }, ignore_index=True)

                            
                    
        
        
        # save the summarizing data ROC_summary_all
        savedata = 1
        if savedata:
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions_cross_dyad_validation/'+                                   savefile_sufix+'/'+cameraID+'/'
            if not os.path.exists(data_saved_subfolder):
                os.makedirs(data_saved_subfolder)

            with open(data_saved_subfolder+'/ROC_summary_all_dependencies_'+strategyname+timelagtype+'.pkl', 'wb') as f:
                pickle.dump(ROC_summary_all, f)

            
            
    


# In[47]:


if 0:
    ind = (ROC_summary_all['action']=='pull') & (ROC_summary_all['testCondition']=='coop(1s)')
    ROC_summary_all_tgt = ROC_summary_all[ind]

    print(ROC_summary_all_tgt.keys())

    import seaborn as sns

    # Pivot the DataFrame to have train_animal as rows and test_animal as columns
    heatmap_data = ROC_summary_all_tgt.pivot(index='train_animal', columns='test_animal', values='predROC')

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=0.5, vmax=1.0)
    plt.title('ROC AUC Heatmap')
    plt.xlabel('Test Animal')
    plt.ylabel('Train Animal')
    plt.tight_layout()
    plt.show()


# ### load the DBN related data for each dyad and run the prediction
# ### training set and testing set are from the same conditions
# ### use the DBN learned structure, only the 0 1 DAG, do not consider the weights
# ### do not consider the self dependencies (dependencies to variables themselves, no diagonal dependencies)

# In[48]:


redoFitting = 0

niters = 100

# PLOT multiple pairs in one plot, so need to load data seperately
moreSampSize = 0
mergetempRos = 0 # 1: merge different time bins

#
animal1_fixedorders = ['eddie','dodson','ginger','dannon','koala']
animal2_fixedorders = ['sparkle','scorch','kanga','kanga','vermelho']
# animal1_fixedorders = ['eddie',]
# animal2_fixedorders = ['sparkle',]
nanimalpairs = np.shape(animal1_fixedorders)[0]

# donimant animal name; since animal1 and 2 are already aligned, no need to check it; but keep it here for reference
dom_animal_names = ['sparkle','scorch','kanga','kanga','vermelho']

temp_resolu = 1

# ONLY FOR PLOT!! 
# define DBN related summarizing variables
DBN_group_typenames = ['self','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
DBN_group_typeIDs  =  [1,3,3,3,5]
DBN_group_coopthres = [0,2,1.5,1,0]
if do_trainedMCs:
    DBN_group_typenames = ['coop(1s)']
    DBN_group_typeIDs  =  [3]
    DBN_group_coopthres = [1]
nDBN_groups = np.shape(DBN_group_typenames)[0]

# DBN model
toNodes = ['pull1_t3','pull2_t3','owgaze1_t3','owgaze2_t3']
fromNodes = ['pull1_t2','pull2_t2','owgaze1_t2','owgaze2_t2']
eventnames = ["M1pull","M2pull","M1gaze","M2gaze"]
nevents = np.shape(eventnames)[0]

timelagtype = 'allthreelags'
time_lags = ['t_-3','t_-2','t_-1']
fromRowIDs =[[0,1,2,3], [4,5,6,7], [8,9,10,11]]
#
# timelagtype = '1and2secondlag'
# time_lags = ['t_-2','t_-1']
# fromRowIDs =[[4,5,6,7], [8,9,10,11]]
#
# timelagtype = '1secondlag'
# time_lags = ['t_-1']
# fromRowIDs =[[8,9,10,11]]
#
# timelagtype = '2secondlag'
# time_lags = ['t_-2']
# fromRowIDs =[[4,5,6,7]]
#
# timelagtype = '3secondlag'
# time_lags = ['t_-3']
# fromRowIDs =[[0,1,2,3]]
#
nlags = np.shape(fromRowIDs)[0]
#
if timelagtype == '2secondlag':
    fromNodes = ['pull1_t1','pull2_t1','owgaze1_t1','owgaze2_t1']
if timelagtype == '3secondlag':
    fromNodes = ['pull1_t0','pull2_t0','owgaze1_t0','owgaze2_t0']


# load ROC_summary_all data
try:
    if redoFitting:
        dumpy
    print('load all ROC data for within task condition (only binary dependencies without self dependencies), and only plot the summary figure')
        
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions_cross_dyad_validation/'+                               savefile_sufix+'/'+cameraID+'/'
    
    with open(data_saved_subfolder+'/ROC_summary_all_dependencies_DBNdependenciesAfterMI_binary_noself_'+timelagtype+'.pkl', 'rb') as f:
        ROC_summary_all = pickle.load(f)

except:

    # initialize a summary dataframe for plotting the summary figure across animals 
    ROC_summary_all = pd.DataFrame(columns=['train_animal','test_animal','action','testCondition','predROC'])
    
    for igroup in np.arange(0,nDBN_groups,1):
        DBN_group_typename = DBN_group_typenames[igroup]

        
        #    
        # load the dyad for training
        for ianimalpair_train in np.arange(0,nanimalpairs,1):

            #
            # load the DBN input data
            animal1_train = animal1_fixedorders[ianimalpair_train]
            animal2_train = animal2_fixedorders[ianimalpair_train]
            #
            # only for kanga
            if animal2_train == 'kanga':
                if animal1_train == 'ginger':
                    animal2_train_nooverlap = 'kanga_withG'
                elif animal1_train == 'dannon':
                    animal2_train_nooverlap = 'kanga_withD'
                else:
                    animal2_train_nooverlap = animal2_train
            else:
                animal2_train_nooverlap = animal2_train
            #
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_train+animal2_train+'/'
            if not mergetempRos:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_train+animal2_train+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                    DBN_input_data_alltypes = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_train+animal2_train+'_mergeTempsReSo.pkl', 'rb') as f:
                    DBN_input_data_alltypes = pickle.load(f)

            # load the DBN training outcome
            if moreSampSize:
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_train+animal2_train+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_train+animal2_train+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_train+animal2_train+'_moreSampSize.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_train+animal2_train+'.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_train+animal2_train+'.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_train+animal2_train+'.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)

            # make sure these variables are the same as in the previous steps
            # temp_resolus = [0.5,1,1.5,2] # temporal resolution in the DBN model, eg: 0.5 means 500ms
            temp_resolus = [1] # temporal resolution in the DBN model, eg: 0.5 means 500ms
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
                samplingsizes_name = ['min_row_number']   
            nsamplings = np.shape(samplingsizes_name)[0]

            #
            temp_resolu = temp_resolus[0]
            j_sampsize_name = samplingsizes_name[0]    

            #
            DBN_input_data_train = DBN_input_data_alltypes[DBN_group_typename]
            
            weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)
            
            # self reward as the baseline to compare with
            weighted_graphs_self = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)]['self']
            weighted_graphs_shuffled_self = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)]['self']
            sig_edges_self = get_significant_edges(weighted_graphs_self,weighted_graphs_shuffled_self)

            # calculate the modulation index
            MI_coop_self_all,sig_edges_coop_self = Modulation_Index(weighted_graphs_self, weighted_graphs_tgt,
                                          sig_edges_self, sig_edges_tgt, 150)
            # only consider the edges that has significant MI and enhanced
            nfromNodes = np.shape(MI_coop_self_all)[1]
            ntoNodes = np.shape(MI_coop_self_all)[2]
            sig_edges_MI = np.zeros((np.shape(sig_edges_coop_self)))
            #
            for ifromNode in np.arange(0,nfromNodes,1):
                for itoNode in np.arange(0,ntoNodes,1):
                    _,pp = st.ttest_1samp(MI_coop_self_all[:,ifromNode,itoNode],0)
                    
                    if (pp<0.01) & (np.nanmean(MI_coop_self_all[:,ifromNode,itoNode])>0):
                        sig_edges_MI[ifromNode,itoNode] = 1
                        
            bina_graphs_mean_tgt = sig_edges_MI*sig_edges_coop_self
            
            #
            # consider the time lags
            if nlags == 1:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:] 
            elif nlags == 2:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]
            elif nlags == 3:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]+bina_graphs_mean_tgt[fromRowIDs[2],:]

            #
            # translate the binary DAGs to edge
            nrows,ncols = np.shape(bina_graphs_mean_tgt)
            edgenames = []
            for irow in np.arange(0,nrows,1):
                for icol in np.arange(0,ncols,1):
                    
                    # remove the self dependencies
                    if irow == icol:
                        bina_graphs_mean_tgt[irow,icol] = 0
                    
                    if bina_graphs_mean_tgt[irow,icol] > 0:
                        edgenames.append((fromNodes[irow],toNodes[icol]))

            # define the DBN predicting model
            bn = BayesianNetwork()
            bn.add_nodes_from(fromNodes)
            bn.add_nodes_from(toNodes)
            bn.add_edges_from(edgenames)

            effect_slice = toNodes        
            
            #    
            # load the dyad for testing
            for ianimalpair_test in np.arange(0,nanimalpairs,1):

                # load the DBN input data
                animal1_test = animal1_fixedorders[ianimalpair_test]
                animal2_test = animal2_fixedorders[ianimalpair_test]
                #
                # only for kanga
                if animal2_test == 'kanga':
                    if animal1_test == 'ginger':
                        animal2_test_nooverlap = 'kanga_withG'
                    elif animal1_test == 'dannon':
                        animal2_test_nooverlap = 'kanga_withD'
                    else:
                        animal2_test_nooverlap = animal2_test
                else:
                    animal2_test_nooverlap = animal2_test
                #
                data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_test+animal2_test+'/'
                if not mergetempRos:
                    with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_test+animal2_test+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                        DBN_input_data_alltypes = pickle.load(f)
                else:
                    with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_test+animal2_test+'_mergeTempsReSo.pkl', 'rb') as f:
                        DBN_input_data_alltypes = pickle.load(f)
                #
                DBN_input_data_test = DBN_input_data_alltypes[DBN_group_typename]
               
            
                #
                # run niters iterations for each condition
                for iiter in np.arange(0,niters,1):


                    # Split data into training and testing sets
                    train_data, _ = train_test_split(DBN_input_data_train, test_size=0.2)
                    _,  test_data = train_test_split(DBN_input_data_test, test_size=0.5)

                    # Perform parameter learning for each time slice
                    bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                    # Perform inference
                    infer = VariableElimination(bn)

                    # Prediction for each behavioral events
                    # With aligned animals across dyad - animal1:sub, animal2:dom
                    for ievent in np.arange(0,nevents,1):

                        var = effect_slice[ievent]
                        Pbehavior = [] # Initialize log-likelihood

                        for index, row in test_data.iterrows():
                            evidence = {fromNodes[0]: row[fromNodes[0]], 
                                        fromNodes[1]: row[fromNodes[1]], 
                                        fromNodes[2]: row[fromNodes[2]], 
                                        fromNodes[3]: row[fromNodes[3]], }

                            # Query the probability distribution for Pulls given evidence
                            aucPpredBehavior = infer.query(variables=[var], evidence=evidence) 

                            # Extract the probability of outcome = 1
                            prob = aucPpredBehavior.values[1]
                            Pbehavior = np.append(Pbehavior, prob)

                        # Calculate the AUC score
                        trueBeh = test_data[var].values
                        try:
                            auc = roc_auc_score(trueBeh, Pbehavior)
                        except:
                            auc = np.nan
                        print(f"AUC Score: {auc:.4f}")

                        # put data in the summarizing data frame
                        if (ievent == 0) | (ievent == 2): # for animal1
                            ROC_summary_all = ROC_summary_all.append({'train_animal':animal1_train,
                                                                      'test_animal':animal1_test,
                                                                      'train_dyadID':ianimalpair_train,
                                                                      'test_dyadID':ianimalpair_test,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename,
                                                                      'predROC':auc,
                                                                      'iters': iiter,
                                                                     }, ignore_index=True)
                        else:
                            ROC_summary_all = ROC_summary_all.append({'train_animal':animal2_train_nooverlap,
                                                                      'test_animal':animal2_test_nooverlap,
                                                                      'train_dyadID':ianimalpair_train,
                                                                      'test_dyadID':ianimalpair_test,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename,
                                                                      'predROC':auc,
                                                                      'iters': iiter,
                                                                 }, ignore_index=True)

                    #         
                    # Prediction for each behavioral events with swapped animal1 and animal2
                    # With training set and testing set has swapped animal type:
                    # training set: animal1 - sub; animal2 - dom
                    # testing set: animal1 - dom; animal2 - sub
                    test_data_swap = test_data.copy()
                    # Create a column mapping
                    new_columns = {}
                    for col in test_data_swap.columns:
                        if 'pull1' in col:
                            new_columns[col] = col.replace('pull1', 'pull2')
                        elif 'pull2' in col:
                            new_columns[col] = col.replace('pull2', 'pull1')
                        elif 'owgaze1' in col:
                            new_columns[col] = col.replace('owgaze1', 'owgaze2')
                        elif 'owgaze2' in col:
                            new_columns[col] = col.replace('owgaze2', 'owgaze1')
                    # Rename the columns using the mapping
                    test_data_swap = test_data_swap.rename(columns=new_columns)

                    for ievent in np.arange(0,nevents,1):

                        var = effect_slice[ievent]
                        Pbehavior = [] # Initialize log-likelihood

                        for index, row in test_data_swap.iterrows():
                            evidence = {fromNodes[0]: row[fromNodes[0]], 
                                        fromNodes[1]: row[fromNodes[1]], 
                                        fromNodes[2]: row[fromNodes[2]], 
                                        fromNodes[3]: row[fromNodes[3]], }

                            # Query the probability distribution for Pulls given evidence
                            aucPpredBehavior = infer.query(variables=[var], evidence=evidence) 

                            # Extract the probability of outcome = 1
                            prob = aucPpredBehavior.values[1]
                            Pbehavior = np.append(Pbehavior, prob)

                        # Calculate the AUC score
                        trueBeh = test_data_swap[var].values
                        try:
                            auc = roc_auc_score(trueBeh, Pbehavior)
                        except:
                            auc = np.nan
                        print(f"AUC Score: {auc:.4f}")

                        # put data in the summarizing data frame
                        if (ievent == 0) | (ievent == 2): # for animal1
                            ROC_summary_all = ROC_summary_all.append({'train_animal':animal1_train,
                                                                      'test_animal':animal2_test_nooverlap,
                                                                      'train_dyadID':ianimalpair_train,
                                                                      'test_dyadID':ianimalpair_test,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename,
                                                                      'predROC':auc,
                                                                      'iters': iiter,
                                                                     }, ignore_index=True)
                        else:
                            ROC_summary_all = ROC_summary_all.append({'train_animal':animal2_train_nooverlap,
                                                                      'test_animal':animal1_test,
                                                                      'train_dyadID':ianimalpair_train,
                                                                      'test_dyadID':ianimalpair_test,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename,
                                                                      'predROC':auc,
                                                                      'iters': iiter,
                                                                 }, ignore_index=True)

                

   
    # save the summarizing data ROC_summary_all
    savedata = 1
    if savedata:
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions_cross_dyad_validation/'+                                       savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_DBNdependenciesAfterMI_binary_noself_'+timelagtype+'.pkl', 'wb') as f:
            pickle.dump(ROC_summary_all, f)
        


# In[49]:


if 0:
    ind = (ROC_summary_all['action']=='pull') & (ROC_summary_all['testCondition']=='coop(1s)')
    ROC_summary_all_tgt = ROC_summary_all[ind]

    print(ROC_summary_all_tgt.keys())

    import seaborn as sns

    # Pivot the DataFrame to have train_animal as rows and test_animal as columns
    heatmap_data = ROC_summary_all_tgt.pivot(index='train_animal', columns='test_animal', values='predROC')

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=0.5, vmax=1.0)
    plt.title('ROC AUC Heatmap')
    plt.xlabel('Test Animal')
    plt.ylabel('Train Animal')
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:





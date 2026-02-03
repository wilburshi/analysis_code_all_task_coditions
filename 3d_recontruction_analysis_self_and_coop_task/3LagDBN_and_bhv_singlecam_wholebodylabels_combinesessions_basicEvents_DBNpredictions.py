#!/usr/bin/env python
# coding: utf-8

# ### In this script, DBN has run and this script is used to make predictions
# ### In this script, DBN is run with 1s time bin, 3 time lag 
# ### In this script, the animal tracking is done with only one camera - camera 2 (middle) 

# In[ ]:


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


# ### function - make demo videos with skeleton and inportant vectors

# In[ ]:


from ana_functions.tracking_video_singlecam_demo import tracking_video_singlecam_demo
from ana_functions.tracking_video_singlecam_wholebody_demo import tracking_video_singlecam_wholebody_demo


# ### function - interval between all behavioral events

# In[ ]:


from ana_functions.bhv_events_interval import bhv_events_interval
from ana_functions.bhv_events_interval import bhv_events_interval_certainEdges


# ### function - train the dynamic bayesian network - multi time lag (3 lags)

# In[ ]:


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

# In[ ]:


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

# In[ ]:


# Suppress all warnings
warnings.simplefilter('ignore')

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
strategynames = ['threeMains','sync_pulls','gaze_lead_pull','social_attention',
                 'other_dependencies','other_noself_dependcies']
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
        
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_'+strategyname+timelagtype+'.pkl', 'rb') as f:
            ROC_summary_all = pickle.load(f)
   
    except:  
    
        # initialize a summary dataframe for plotting the summary figure across animals 
        ROC_summary_all = pd.DataFrame(columns=['animal','action','testCondition','predROC'])

        fig, axs = plt.subplots(nanimalpairs,nDBN_groups) # nDBN_groups(3) task conditions; 2 animal individual
        fig.set_figheight(8*nanimalpairs)
        fig.set_figwidth(8*nDBN_groups)

        for ianimalpair in np.arange(0,nanimalpairs,1):

            # initiate figure
            # fig, axs = plt.subplots(nDBN_groups,2) # nDBN_groups(3) task conditions; 2 animal individual
            # fig.set_figheight(8*nDBN_groups)
            # fig.set_figwidth(15*2)

            # load the DBN input data
            animal1_fixedorder = animal1_fixedorders[ianimalpair]
            animal2_fixedorder = animal2_fixedorders[ianimalpair]
            #
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder+animal2_fixedorder+'/'
            if not mergetempRos:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                    DBN_input_data_alltypes = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_mergeTempsReSo.pkl', 'rb') as f:
                    DBN_input_data_alltypes = pickle.load(f)

            # load the DBN training outcome
            if moreSampSize:
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
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



            for igroup in np.arange(0,nDBN_groups,1):
                DBN_group_typename = DBN_group_typenames[igroup]

                DBN_input_data_tgt = DBN_input_data_alltypes[DBN_group_typename]
                weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
                weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
                # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
                sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)

                # mean weight 
                weighted_graphs_mean_tgt = np.nanmean(weighted_graphs_tgt,axis=0)
                weighted_graphs_mean_tgt = weighted_graphs_mean_tgt * sig_edges_tgt

                bina_graphs_mean_tgt = sig_edges_tgt


                # run niters iterations for each condition
                for iiter in np.arange(0,niters,1):


                    # Split data into training and testing sets
                    train_data, test_data = train_test_split(DBN_input_data_tgt, test_size=0.2)
                    # test_data = DBN_input_data_tgt

                    # Perform parameter learning for each time slice
                    bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                    # Perform inference
                    infer = VariableElimination(bn)

                    # Prediction for each behavioral events
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

                        # Optionally, plot the ROC curve - only plot for the first iteration
                        if iiter == 0:
                            fpr, tpr, _ = roc_curve(trueBeh, Pbehavior)
                            try:
                                axs[ianimalpair,igroup].plot(fpr, tpr, 
                                                             label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                            except:
                                axs[ianimalpair].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                            
                        # put data in the summarizing data frame
                        if (ievent == 0) | (ievent == 2): # for animal1
                            ROC_summary_all = ROC_summary_all.append({'animal':animal1_fixedorder,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename,
                                                                      'predROC':auc
                                                                     }, ignore_index=True)
                        else:
                            ROC_summary_all = ROC_summary_all.append({'animal':animal2_fixedorder,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename,
                                                                      'predROC':auc
                                                                 }, ignore_index=True)

                    if iiter == 0:
                        try:
                            axs[ianimalpair,igroup].plot([0, 1], [0, 1], 'k--')
                            axs[ianimalpair,igroup].set_xlabel('False Positive Rate')
                            axs[ianimalpair,igroup].set_ylabel('True Positive Rate')
                            axs[ianimalpair,igroup].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                              +'\n'+DBN_group_typename+'; with dependency:'+strategyname+timelagtype)
                            axs[ianimalpair,igroup].legend(loc='lower right')
                        except:
                            axs[ianimalpair].plot([0, 1], [0, 1], 'k--')
                            axs[ianimalpair].set_xlabel('False Positive Rate')
                            axs[ianimalpair].set_ylabel('True Positive Rate')
                            axs[ianimalpair].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                              +'\n'+DBN_group_typename+'; with dependency:'+strategyname+timelagtype)
                            axs[ianimalpair].legend(loc='lower right')

        savefig = 1
        if savefig:
            figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
            if not os.path.exists(figsavefolder):
                os.makedirs(figsavefolder)
            fig.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_binaryGraph_onlydependency'+strategyname+timelagtype+'.pdf')

        
        
        # save the summarizing data ROC_summary_all
        savedata = 1
        if savedata:
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
            if not os.path.exists(data_saved_subfolder):
                os.makedirs(data_saved_subfolder)

            with open(data_saved_subfolder+'/ROC_summary_all_dependencies_'+strategyname+timelagtype+'.pkl', 'wb') as f:
                pickle.dump(ROC_summary_all, f)

            
            
    # average ROC across all iterations
    # separate Kanga into two groups based on her partner
    ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    #
    animals_ROC = np.unique(ROC_summary_all['animal'])
    nanis = np.shape(animals_ROC)[0]
    actions_ROC = np.unique(ROC_summary_all['action'])
    nacts = np.shape(actions_ROC)[0]
    testCons_ROC = np.unique(ROC_summary_all['testCondition'])
    ntcons = np.shape(testCons_ROC)[0]
    ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all.keys())
    #
    for iani in np.arange(0,nanis,1):
        animal_ROC = animals_ROC[iani]
        ind_ani = ROC_summary_all['animal']==animal_ROC
        ROC_summary_all_iani = ROC_summary_all[ind_ani]
        #
        for iact in np.arange(0,nacts,1):
            action_ROC = actions_ROC[iact]
            ind_act = ROC_summary_all_iani['action']==action_ROC
            ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
            #
            for itcon in np.arange(0,ntcons,1):
                testCon_ROC = testCons_ROC[itcon]
                ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
                ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
                #
                ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                    'action':action_ROC,
                                                                    'testCondition':testCon_ROC,
                                                                    'predROC':np.nanmean(ROC_summary_all_itcon['predROC'])
                                                                     }, ignore_index=True)        
    
    # plot the summarizing figure
    fig2, axs2 = plt.subplots(1,1)
    fig2.set_figheight(5)
    fig2.set_figwidth(5)
    seaborn.barplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                    errorbar='ci',alpha=.5)
    seaborn.swarmplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                      alpha=.9,size=6,dodge=True,legend=False)
    plt.plot([-0.5,1.5],[0.5,0.5],'k--')
    axs2.set_xlim([-0.5,1.5])
    axs2.set_title('all animal'+'; with dependency:'+strategyname+timelagtype)

    savefig = 1
    if savefig:
        figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        fig2.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_binaryGraph_onlydependency'+strategyname+timelagtype+'_summarizingplot.pdf')


# ### load the DBN related data for each dyad and run the prediction
# ### For each condition, only use the hypothetical dependencies together with the self dependencies

# In[ ]:


# Suppress all warnings
warnings.simplefilter('ignore')

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
strategynames = ['threeMains','sync_pulls','gaze_lead_pull','social_attention',
                 ]
# strategynames = ['threeMains',]
# strategynames = ['gaze_lead_pull'] # ['all_threes','sync_pulls','gaze_lead_pull','social_attention']
bina_graphs_specific_strategy = {
    'threeMains': np.array([[1,1,0,1],[1,1,1,0],[1,0,1,0],[0,1,0,1]]),
    'sync_pulls': np.array([[1,1,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]]),
    'gaze_lead_pull':np.array([[1,0,0,0],[0,1,0,0],[1,0,1,0],[0,1,0,1]]),
    'social_attention':np.array([[1,0,0,1],[0,1,1,0],[0,0,1,0],[0,0,0,1]]),
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
        
        print('load all ROC data for hypothetical dependencies (with self edges), and only plot the summary figure')
        
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_'+strategyname+timelagtype+'_withSelfEdge.pkl', 'rb') as f:
            ROC_summary_all = pickle.load(f)
   
    except:  
    
        # initialize a summary dataframe for plotting the summary figure across animals 
        ROC_summary_all = pd.DataFrame(columns=['animal','action','testCondition','predROC'])

        fig, axs = plt.subplots(nanimalpairs,nDBN_groups) # nDBN_groups(3) task conditions; 2 animal individual
        fig.set_figheight(8*nanimalpairs)
        fig.set_figwidth(8*nDBN_groups)

        for ianimalpair in np.arange(0,nanimalpairs,1):

            # initiate figure
            # fig, axs = plt.subplots(nDBN_groups,2) # nDBN_groups(3) task conditions; 2 animal individual
            # fig.set_figheight(8*nDBN_groups)
            # fig.set_figwidth(15*2)

            # load the DBN input data
            animal1_fixedorder = animal1_fixedorders[ianimalpair]
            animal2_fixedorder = animal2_fixedorders[ianimalpair]
            #
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder+animal2_fixedorder+'/'
            if not mergetempRos:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                    DBN_input_data_alltypes = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_mergeTempsReSo.pkl', 'rb') as f:
                    DBN_input_data_alltypes = pickle.load(f)

            # load the DBN training outcome
            if moreSampSize:
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
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



            for igroup in np.arange(0,nDBN_groups,1):
                DBN_group_typename = DBN_group_typenames[igroup]

                DBN_input_data_tgt = DBN_input_data_alltypes[DBN_group_typename]
                weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
                weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
                # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
                sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)

                # mean weight 
                weighted_graphs_mean_tgt = np.nanmean(weighted_graphs_tgt,axis=0)
                weighted_graphs_mean_tgt = weighted_graphs_mean_tgt * sig_edges_tgt

                bina_graphs_mean_tgt = sig_edges_tgt


                # run niters iterations for each condition
                for iiter in np.arange(0,niters,1):


                    # Split data into training and testing sets
                    train_data, test_data = train_test_split(DBN_input_data_tgt, test_size=0.2)
                    # test_data = DBN_input_data_tgt

                    # Perform parameter learning for each time slice
                    bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                    # Perform inference
                    infer = VariableElimination(bn)

                    # Prediction for each behavioral events
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

                        # Optionally, plot the ROC curve - only plot for the first iteration
                        if iiter == 0:
                            fpr, tpr, _ = roc_curve(trueBeh, Pbehavior)
                            try:
                                axs[ianimalpair,igroup].plot(fpr, tpr, 
                                                             label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                            except:
                                axs[ianimalpair].plot(fpr, tpr, 
                                                             label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                                
                        # put data in the summarizing data frame
                        if (ievent == 0) | (ievent == 2): # for animal1
                            ROC_summary_all = ROC_summary_all.append({'animal':animal1_fixedorder,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename,
                                                                      'predROC':auc
                                                                     }, ignore_index=True)
                        else:
                            ROC_summary_all = ROC_summary_all.append({'animal':animal2_fixedorder,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename,
                                                                      'predROC':auc
                                                                 }, ignore_index=True)

                    if iiter == 0:
                        try:
                            axs[ianimalpair,igroup].plot([0, 1], [0, 1], 'k--')
                            axs[ianimalpair,igroup].set_xlabel('False Positive Rate')
                            axs[ianimalpair,igroup].set_ylabel('True Positive Rate')
                            axs[ianimalpair,igroup].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                              +'\n'+DBN_group_typename+'; with dependency:'+strategyname+timelagtype)
                            axs[ianimalpair,igroup].legend(loc='lower right')
                        except:
                            axs[ianimalpair].plot([0, 1], [0, 1], 'k--')
                            axs[ianimalpair].set_xlabel('False Positive Rate')
                            axs[ianimalpair].set_ylabel('True Positive Rate')
                            axs[ianimalpair].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                              +'\n'+DBN_group_typename+'; with dependency:'+strategyname+timelagtype)
                            axs[ianimalpair].legend(loc='lower right')


        savefig = 1
        if savefig:
            figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
            if not os.path.exists(figsavefolder):
                os.makedirs(figsavefolder)
            fig.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_binaryGraph_onlydependency'+strategyname+timelagtype+'_withSelfEdge.pdf')

        
        
        # save the summarizing data ROC_summary_all
        savedata = 1
        if savedata:
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
            if not os.path.exists(data_saved_subfolder):
                os.makedirs(data_saved_subfolder)

            with open(data_saved_subfolder+'/ROC_summary_all_dependencies_'+strategyname+timelagtype+'_withSelfEdge.pkl', 'wb') as f:
                pickle.dump(ROC_summary_all, f)

            
            
    # average ROC across all iterations
    # separate Kanga into two groups based on her partner
    ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    #
    animals_ROC = np.unique(ROC_summary_all['animal'])
    nanis = np.shape(animals_ROC)[0]
    actions_ROC = np.unique(ROC_summary_all['action'])
    nacts = np.shape(actions_ROC)[0]
    testCons_ROC = np.unique(ROC_summary_all['testCondition'])
    ntcons = np.shape(testCons_ROC)[0]
    ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all.keys())
    #
    for iani in np.arange(0,nanis,1):
        animal_ROC = animals_ROC[iani]
        ind_ani = ROC_summary_all['animal']==animal_ROC
        ROC_summary_all_iani = ROC_summary_all[ind_ani]
        #
        for iact in np.arange(0,nacts,1):
            action_ROC = actions_ROC[iact]
            ind_act = ROC_summary_all_iani['action']==action_ROC
            ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
            #
            for itcon in np.arange(0,ntcons,1):
                testCon_ROC = testCons_ROC[itcon]
                ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
                ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
                #
                ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                    'action':action_ROC,
                                                                    'testCondition':testCon_ROC,
                                                                    'predROC':np.nanmean(ROC_summary_all_itcon['predROC'])
                                                                     }, ignore_index=True)        
    
    # plot the summarizing figure
    fig2, axs2 = plt.subplots(1,1)
    fig2.set_figheight(5)
    fig2.set_figwidth(5)
    seaborn.barplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                    errorbar='ci',alpha=.5)
    seaborn.swarmplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                      alpha=.9,size=6,dodge=True,legend=False)
    plt.plot([-0.5,1.5],[0.5,0.5],'k--')
    axs2.set_xlim([-0.5,1.5])
    axs2.set_title('all animal'+'; with dependency:'+strategyname+timelagtype)

    savefig = 1
    if savefig:
        figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        fig2.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_binaryGraph_onlydependency'+strategyname+timelagtype+'_withSelfEdge_summarizingplot.pdf')


# In[ ]:





# ### load the DBN related data for each dyad and run the prediction
# ### training set and testing set are from the same conditions
# ### use the DBN learned structure, only the 0 1 DAG, do not consider the weights 

# In[ ]:


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
    print('load all ROC data for within task condition (only binary dependencies), and only plot the summary figure')
        
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'

    with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_binary_'+timelagtype+'.pkl', 'rb') as f:
        ROC_summary_all = pickle.load(f)

except:
    # initialize a summary dataframe for plotting the summary figure across animals 
    ROC_summary_all = pd.DataFrame(columns=['animal','action','testCondition','predROC'])

    fig, axs = plt.subplots(nanimalpairs,nDBN_groups) # nDBN_groups(3) task conditions; 2 animal individual
    fig.set_figheight(7*nanimalpairs)
    fig.set_figwidth(7*nDBN_groups)

    for ianimalpair in np.arange(0,nanimalpairs,1):

        # load the DBN input data
        animal1_fixedorder = animal1_fixedorders[ianimalpair]
        animal2_fixedorder = animal2_fixedorders[ianimalpair]
        #
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder+animal2_fixedorder+'/'
        if not mergetempRos:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_mergeTempsReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)

        # load the DBN training outcome
        if moreSampSize:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                sig_edges_diffTempRo_diffSampSize = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
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



        for igroup in np.arange(0,nDBN_groups,1):
            DBN_group_typename = DBN_group_typenames[igroup]

            DBN_input_data_tgt = DBN_input_data_alltypes[DBN_group_typename]
            weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)

            # mean weight 
            weighted_graphs_mean_tgt = np.nanmean(weighted_graphs_tgt,axis=0)
            weighted_graphs_mean_tgt = weighted_graphs_mean_tgt * sig_edges_tgt

            bina_graphs_mean_tgt = sig_edges_tgt
            #
            # consider the time lags
            if nlags == 1:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:] 
            elif nlags == 2:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]
            elif nlags == 3:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]+bina_graphs_mean_tgt[fromRowIDs[2],:]

            

            # translate the binary DAGs to edge
            nrows,ncols = np.shape(bina_graphs_mean_tgt)
            edgenames = []
            for irow in np.arange(0,nrows,1):
                for icol in np.arange(0,ncols,1):
                    if bina_graphs_mean_tgt[irow,icol] > 0:
                        edgenames.append((fromNodes[irow],toNodes[icol]))

            # define the DBN predicting model
            bn = BayesianNetwork()
            bn.add_nodes_from(fromNodes)
            bn.add_nodes_from(toNodes)
            bn.add_edges_from(edgenames)

            effect_slice = toNodes        

            # run niters iterations for each condition
            for iiter in np.arange(0,niters,1):


                # Split data into training and testing sets
                train_data, test_data = train_test_split(DBN_input_data_tgt, test_size=0.2)
                # test_data = DBN_input_data_tgt

                # Perform parameter learning for each time slice
                bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                # Perform inference
                infer = VariableElimination(bn)

                # Prediction for each behavioral events
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

                    # Optionally, plot the ROC curve - only plot for the first iteration
                    if iiter == 0:
                        fpr, tpr, _ = roc_curve(trueBeh, Pbehavior)
                        try:
                            axs[ianimalpair,igroup].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        except:
                            axs[ianimalpair].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        
                    # put data in the summarizing data frame
                    if (ievent == 0) | (ievent == 2): # for animal1
                        ROC_summary_all = ROC_summary_all.append({'animal':animal1_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                                 }, ignore_index=True)
                    else:
                        ROC_summary_all = ROC_summary_all.append({'animal':animal2_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                             }, ignore_index=True)

                if iiter == 0:
                    try:
                        axs[ianimalpair,igroup].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair,igroup].set_xlabel('False Positive Rate')
                        axs[ianimalpair,igroup].set_ylabel('True Positive Rate')
                        axs[ianimalpair,igroup].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair,igroup].legend(loc='lower right')
                    except:
                        axs[ianimalpair].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair].set_xlabel('False Positive Rate')
                        axs[ianimalpair].set_ylabel('True Positive Rate')
                        axs[ianimalpair].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair].legend(loc='lower right')

    savefig = 1
    if savefig:
        figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        fig.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_binaryGraph_'+timelagtype+'.pdf')

        
    # save the summarizing data ROC_summary_all
    savedata = 1
    if savedata:
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_binary_'+timelagtype+'.pkl', 'wb') as f:
            pickle.dump(ROC_summary_all, f)
        

# average ROC across all iterations
# separate Kanga into two groups based on her partner
ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
#
animals_ROC = np.unique(ROC_summary_all['animal'])
nanis = np.shape(animals_ROC)[0]
actions_ROC = np.unique(ROC_summary_all['action'])
nacts = np.shape(actions_ROC)[0]
testCons_ROC = np.unique(ROC_summary_all['testCondition'])
ntcons = np.shape(testCons_ROC)[0]
ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all.keys())
#
for iani in np.arange(0,nanis,1):
    animal_ROC = animals_ROC[iani]
    ind_ani = ROC_summary_all['animal']==animal_ROC
    ROC_summary_all_iani = ROC_summary_all[ind_ani]
    #
    for iact in np.arange(0,nacts,1):
        action_ROC = actions_ROC[iact]
        ind_act = ROC_summary_all_iani['action']==action_ROC
        ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
        #
        for itcon in np.arange(0,ntcons,1):
            testCon_ROC = testCons_ROC[itcon]
            ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
            ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
            #
            ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                'action':action_ROC,
                                                                'testCondition':testCon_ROC,
                                                                'predROC':np.nanmean(ROC_summary_all_itcon['predROC'])
                                                                 }, ignore_index=True)

        
# plot the summarizing figure
fig2, axs2 = plt.subplots(1,1)
fig2.set_figheight(5)
fig2.set_figwidth(5)
seaborn.barplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                errorbar='ci',alpha=.5)
seaborn.swarmplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                  alpha=.9,size=6,dodge=True,legend=False)
plt.plot([-0.5,1.5],[0.5,0.5],'k--')
axs2.set_xlim([-0.5,1.5])
axs2.set_title('all animal')

savefig = 1
if savefig:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    fig2.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_binaryGraph_'+timelagtype+'_summarizingplot.pdf')




# ### load the DBN related data for each dyad and run the prediction
# ### training set and testing set are from the same conditions
# ### use the DBN learned structure, consider the weight as well

# In[ ]:


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
    print('load all ROC data for within task condition (weighted dependencies), and only plot the summary figure')
        
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'

    with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_'+timelagtype+'.pkl', 'rb') as f:
        ROC_summary_all = pickle.load(f)

except:
    # initialize a summary dataframe for plotting the summary figure across animals 
    ROC_summary_all = pd.DataFrame(columns=['animal','action','testCondition','predROC'])

    fig, axs = plt.subplots(nanimalpairs,nDBN_groups) # nDBN_groups(3) task conditions; 2 animal individual
    fig.set_figheight(7*nanimalpairs)
    fig.set_figwidth(7*nDBN_groups)

    for ianimalpair in np.arange(0,nanimalpairs,1):

        # load the DBN input data
        animal1_fixedorder = animal1_fixedorders[ianimalpair]
        animal2_fixedorder = animal2_fixedorders[ianimalpair]
        #
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder+animal2_fixedorder+'/'
        if not mergetempRos:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_mergeTempsReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)

        # load the DBN training outcome
        if moreSampSize:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                sig_edges_diffTempRo_diffSampSize = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
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



        for igroup in np.arange(0,nDBN_groups,1):
            DBN_group_typename = DBN_group_typenames[igroup]

            DBN_input_data_tgt = DBN_input_data_alltypes[DBN_group_typename]
            weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)

            # mean weight 
            weighted_graphs_mean_tgt = np.nanmean(weighted_graphs_tgt,axis=0)
            weighted_graphs_mean_tgt = weighted_graphs_mean_tgt * sig_edges_tgt

            bina_graphs_mean_tgt = (np.random.random(size=np.shape(weighted_graphs_mean_tgt))<weighted_graphs_mean_tgt).astype(int)
            
            #
            # consider the time lags
            if nlags == 1:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:] 
            elif nlags == 2:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]
            elif nlags == 3:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]+bina_graphs_mean_tgt[fromRowIDs[2],:]

            

            # translate the binary DAGs to edge
            nrows,ncols = np.shape(bina_graphs_mean_tgt)
            edgenames = []
            for irow in np.arange(0,nrows,1):
                for icol in np.arange(0,ncols,1):
                    if bina_graphs_mean_tgt[irow,icol] > 0:
                        edgenames.append((fromNodes[irow],toNodes[icol]))

            # define the DBN predicting model
            bn = BayesianNetwork()
            bn.add_nodes_from(fromNodes)
            bn.add_nodes_from(toNodes)
            bn.add_edges_from(edgenames)

            effect_slice = toNodes        

            # run niters iterations for each condition
            for iiter in np.arange(0,niters,1):


                # Split data into training and testing sets
                train_data, test_data = train_test_split(DBN_input_data_tgt, test_size=0.2)
                # test_data = DBN_input_data_tgt

                # Perform parameter learning for each time slice
                bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                # Perform inference
                infer = VariableElimination(bn)

                # Prediction for each behavioral events
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

                    # Optionally, plot the ROC curve - only plot for the first iteration
                    if iiter == 0:
                        fpr, tpr, _ = roc_curve(trueBeh, Pbehavior)
                        try:
                            axs[ianimalpair,igroup].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        except:
                            axs[ianimalpair].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        
                    # put data in the summarizing data frame
                    if (ievent == 0) | (ievent == 2): # for animal1
                        ROC_summary_all = ROC_summary_all.append({'animal':animal1_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                                 }, ignore_index=True)
                    else:
                        ROC_summary_all = ROC_summary_all.append({'animal':animal2_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                             }, ignore_index=True)

                if iiter == 0:
                    try:
                        axs[ianimalpair,igroup].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair,igroup].set_xlabel('False Positive Rate')
                        axs[ianimalpair,igroup].set_ylabel('True Positive Rate')
                        axs[ianimalpair,igroup].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair,igroup].legend(loc='lower right')
                    except:
                        axs[ianimalpair].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair].set_xlabel('False Positive Rate')
                        axs[ianimalpair].set_ylabel('True Positive Rate')
                        axs[ianimalpair].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair].legend(loc='lower right')


    savefig = 1
    if savefig:
        figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        fig.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_weightedGraph_'+timelagtype+'.pdf')

        
    # save the summarizing data ROC_summary_all
    savedata = 1
    if savedata:
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_'+timelagtype+'.pkl', 'wb') as f:
            pickle.dump(ROC_summary_all, f)
        

# average ROC across all iterations
# separate Kanga into two groups based on her partner
ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
#
animals_ROC = np.unique(ROC_summary_all['animal'])
nanis = np.shape(animals_ROC)[0]
actions_ROC = np.unique(ROC_summary_all['action'])
nacts = np.shape(actions_ROC)[0]
testCons_ROC = np.unique(ROC_summary_all['testCondition'])
ntcons = np.shape(testCons_ROC)[0]
ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all.keys())
#
for iani in np.arange(0,nanis,1):
    animal_ROC = animals_ROC[iani]
    ind_ani = ROC_summary_all['animal']==animal_ROC
    ROC_summary_all_iani = ROC_summary_all[ind_ani]
    #
    for iact in np.arange(0,nacts,1):
        action_ROC = actions_ROC[iact]
        ind_act = ROC_summary_all_iani['action']==action_ROC
        ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
        #
        for itcon in np.arange(0,ntcons,1):
            testCon_ROC = testCons_ROC[itcon]
            ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
            ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
            #
            ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                'action':action_ROC,
                                                                'testCondition':testCon_ROC,
                                                                'predROC':np.nanmean(ROC_summary_all_itcon['predROC'])
                                                                 }, ignore_index=True)

        
# plot the summarizing figure
fig2, axs2 = plt.subplots(1,1)
fig2.set_figheight(5)
fig2.set_figwidth(5)
seaborn.barplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                errorbar='ci',alpha=.5)
seaborn.swarmplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                  alpha=.9,size=6,dodge=True,legend=False)
plt.plot([-0.5,1.5],[0.5,0.5],'k--')
axs2.set_xlim([-0.5,1.5])
axs2.set_title('all animal')

savefig = 1
if savefig:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    fig2.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_weightedGraph_'+timelagtype+'_summarizingplot.pdf')




# ### load the DBN related data for each dyad and run the prediction
# ### training set and testing set are from the same conditions
# ### use the DBN learned structure, only the 0 1 DAG, do not consider the weights
# ### do not consider the self dependencies (dependencies to variables themselves, no diagonal dependencies)

# In[ ]:


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
        
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'

    with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_binary_noself_'+timelagtype+'.pkl', 'rb') as f:
        ROC_summary_all = pickle.load(f)

except:
    # initialize a summary dataframe for plotting the summary figure across animals 
    ROC_summary_all = pd.DataFrame(columns=['animal','action','testCondition','predROC'])

    fig, axs = plt.subplots(nanimalpairs,nDBN_groups) # nDBN_groups(3) task conditions; 2 animal individual
    fig.set_figheight(7*nanimalpairs)
    fig.set_figwidth(7*nDBN_groups)

    for ianimalpair in np.arange(0,nanimalpairs,1):

        # load the DBN input data
        animal1_fixedorder = animal1_fixedorders[ianimalpair]
        animal2_fixedorder = animal2_fixedorders[ianimalpair]
        #
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder+animal2_fixedorder+'/'
        if not mergetempRos:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_mergeTempsReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)

        # load the DBN training outcome
        if moreSampSize:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                sig_edges_diffTempRo_diffSampSize = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
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



        for igroup in np.arange(0,nDBN_groups,1):
            DBN_group_typename = DBN_group_typenames[igroup]

            DBN_input_data_tgt = DBN_input_data_alltypes[DBN_group_typename]
            weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)

            # mean weight 
            weighted_graphs_mean_tgt = np.nanmean(weighted_graphs_tgt,axis=0)
            weighted_graphs_mean_tgt = weighted_graphs_mean_tgt * sig_edges_tgt

            bina_graphs_mean_tgt = sig_edges_tgt
            #
            # consider the time lags
            if nlags == 1:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:] 
            elif nlags == 2:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]
            elif nlags == 3:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]+bina_graphs_mean_tgt[fromRowIDs[2],:]

            

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

            # run niters iterations for each condition
            for iiter in np.arange(0,niters,1):


                # Split data into training and testing sets
                train_data, test_data = train_test_split(DBN_input_data_tgt, test_size=0.2)
                # test_data = DBN_input_data_tgt

                # Perform parameter learning for each time slice
                bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                # Perform inference
                infer = VariableElimination(bn)

                # Prediction for each behavioral events
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

                    # Optionally, plot the ROC curve - only plot for the first iteration
                    if iiter == 0:
                        fpr, tpr, _ = roc_curve(trueBeh, Pbehavior)
                        try:
                            axs[ianimalpair,igroup].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        except:
                            axs[ianimalpair].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        
                    # put data in the summarizing data frame
                    if (ievent == 0) | (ievent == 2): # for animal1
                        ROC_summary_all = ROC_summary_all.append({'animal':animal1_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                                 }, ignore_index=True)
                    else:
                        ROC_summary_all = ROC_summary_all.append({'animal':animal2_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                             }, ignore_index=True)

                if iiter == 0:
                    try:
                        axs[ianimalpair,igroup].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair,igroup].set_xlabel('False Positive Rate')
                        axs[ianimalpair,igroup].set_ylabel('True Positive Rate')
                        axs[ianimalpair,igroup].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair,igroup].legend(loc='lower right')
                    except:
                        axs[ianimalpair].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair].set_xlabel('False Positive Rate')
                        axs[ianimalpair].set_ylabel('True Positive Rate')
                        axs[ianimalpair].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair].legend(loc='lower right')



    savefig = 1
    if savefig:
        figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        fig.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_binaryGraph_noself_'+timelagtype+'.pdf')

        
    # save the summarizing data ROC_summary_all
    savedata = 1
    if savedata:
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_binary_noself_'+timelagtype+'.pkl', 'wb') as f:
            pickle.dump(ROC_summary_all, f)
        

# average ROC across all iterations
# separate Kanga into two groups based on her partner
ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
#
animals_ROC = np.unique(ROC_summary_all['animal'])
nanis = np.shape(animals_ROC)[0]
actions_ROC = np.unique(ROC_summary_all['action'])
nacts = np.shape(actions_ROC)[0]
testCons_ROC = np.unique(ROC_summary_all['testCondition'])
ntcons = np.shape(testCons_ROC)[0]
ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all.keys())
#
for iani in np.arange(0,nanis,1):
    animal_ROC = animals_ROC[iani]
    ind_ani = ROC_summary_all['animal']==animal_ROC
    ROC_summary_all_iani = ROC_summary_all[ind_ani]
    #
    for iact in np.arange(0,nacts,1):
        action_ROC = actions_ROC[iact]
        ind_act = ROC_summary_all_iani['action']==action_ROC
        ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
        #
        for itcon in np.arange(0,ntcons,1):
            testCon_ROC = testCons_ROC[itcon]
            ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
            ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
            #
            ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                'action':action_ROC,
                                                                'testCondition':testCon_ROC,
                                                                'predROC':np.nanmean(ROC_summary_all_itcon['predROC'])
                                                                 }, ignore_index=True)

        
# plot the summarizing figure
fig2, axs2 = plt.subplots(1,1)
fig2.set_figheight(5)
fig2.set_figwidth(5)
seaborn.barplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                errorbar='ci',alpha=.5)
seaborn.swarmplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                  alpha=.9,size=6,dodge=True,legend=False)
plt.plot([-0.5,1.5],[0.5,0.5],'k--')
axs2.set_xlim([-0.5,1.5])
axs2.set_title('all animal')

savefig = 1
if savefig:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    fig2.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_binaryGraph_noself_'+timelagtype+'_summarizingplot.pdf')




# ### load the DBN related data for each dyad and run the prediction
# ### training set and testing set are from the same conditions
# ### use the DBN learned structure, only the 0 1 DAG, do not consider the weights
# ### do not consider the self dependencies (dependencies to variables themselves, no diagonal dependencies)
# ### consider the modulation index checking with self reward

# In[ ]:


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

temp_resolu = 1

# ONLY FOR PLOT!! 
# define DBN related summarizing variables
DBN_group_typenames = ['coop(2s)','coop(1.5s)','coop(1s)','no-vision']
DBN_group_typeIDs  =  [3,3,3,5]
DBN_group_coopthres = [2,1.5,1,0]
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
        
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'

    with open(data_saved_subfolder+'/ROC_summary_all_dependencies_DBNdependenciesAfterMI_binary_noself_'+timelagtype+'.pkl', 'rb') as f:
        ROC_summary_all = pickle.load(f)

except:
    # initialize a summary dataframe for plotting the summary figure across animals 
    ROC_summary_all = pd.DataFrame(columns=['animal','action','testCondition','predROC'])

    fig, axs = plt.subplots(nanimalpairs,nDBN_groups) # nDBN_groups(3) task conditions; 2 animal individual
    fig.set_figheight(7*nanimalpairs)
    fig.set_figwidth(7*nDBN_groups)

    for ianimalpair in np.arange(0,nanimalpairs,1):

        # load the DBN input data
        animal1_fixedorder = animal1_fixedorders[ianimalpair]
        animal2_fixedorder = animal2_fixedorders[ianimalpair]
        #
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder+animal2_fixedorder+'/'
        if not mergetempRos:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_mergeTempsReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)

        # load the DBN training outcome
        if moreSampSize:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                sig_edges_diffTempRo_diffSampSize = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
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


        # self reward as the baseline to compare with
        DBN_input_data_self = DBN_input_data_alltypes['self']
        weighted_graphs_self = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)]['self']
        weighted_graphs_shuffled_self = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)]['self']
        sig_edges_self = get_significant_edges(weighted_graphs_self,weighted_graphs_shuffled_self)
    
        

        for igroup in np.arange(0,nDBN_groups,1):
            DBN_group_typename = DBN_group_typenames[igroup]

            DBN_input_data_tgt = DBN_input_data_alltypes[DBN_group_typename]
            weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)
            
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

            # run niters iterations for each condition
            for iiter in np.arange(0,niters,1):


                # Split data into training and testing sets
                train_data, test_data = train_test_split(DBN_input_data_tgt, test_size=0.2)
                # test_data = DBN_input_data_tgt

                # Perform parameter learning for each time slice
                bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                # Perform inference
                infer = VariableElimination(bn)

                # Prediction for each behavioral events
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

                    # Optionally, plot the ROC curve - only plot for the first iteration
                    if iiter == 0:
                        fpr, tpr, _ = roc_curve(trueBeh, Pbehavior)
                        try:
                            axs[ianimalpair,igroup].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        except:
                            axs[ianimalpair].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        
                    # put data in the summarizing data frame
                    if (ievent == 0) | (ievent == 2): # for animal1
                        ROC_summary_all = ROC_summary_all.append({'animal':animal1_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                                 }, ignore_index=True)
                    else:
                        ROC_summary_all = ROC_summary_all.append({'animal':animal2_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                             }, ignore_index=True)

                if iiter == 0:
                    try:
                        axs[ianimalpair,igroup].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair,igroup].set_xlabel('False Positive Rate')
                        axs[ianimalpair,igroup].set_ylabel('True Positive Rate')
                        axs[ianimalpair,igroup].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair,igroup].legend(loc='lower right')
                    except:
                        axs[ianimalpair].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair].set_xlabel('False Positive Rate')
                        axs[ianimalpair].set_ylabel('True Positive Rate')
                        axs[ianimalpair].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair].legend(loc='lower right')

    savefig = 1
    if savefig:
        figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        fig.savefig(figsavefolder+'withinCondition_DBNpredicition_DBNdependenciesAfterMI_binary_noself_'+timelagtype+'.pdf')

        
    # save the summarizing data ROC_summary_all
    savedata = 1
    if savedata:
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_DBNdependenciesAfterMI_binary_noself_'+timelagtype+'.pkl', 'wb') as f:
            pickle.dump(ROC_summary_all, f)
        

# average ROC across all iterations
# separate Kanga into two groups based on her partner
ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
#
animals_ROC = np.unique(ROC_summary_all['animal'])
nanis = np.shape(animals_ROC)[0]
actions_ROC = np.unique(ROC_summary_all['action'])
nacts = np.shape(actions_ROC)[0]
testCons_ROC = np.unique(ROC_summary_all['testCondition'])
ntcons = np.shape(testCons_ROC)[0]
ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all.keys())
#
for iani in np.arange(0,nanis,1):
    animal_ROC = animals_ROC[iani]
    ind_ani = ROC_summary_all['animal']==animal_ROC
    ROC_summary_all_iani = ROC_summary_all[ind_ani]
    #
    for iact in np.arange(0,nacts,1):
        action_ROC = actions_ROC[iact]
        ind_act = ROC_summary_all_iani['action']==action_ROC
        ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
        #
        for itcon in np.arange(0,ntcons,1):
            testCon_ROC = testCons_ROC[itcon]
            ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
            ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
            #
            ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                'action':action_ROC,
                                                                'testCondition':testCon_ROC,
                                                                'predROC':np.nanmean(ROC_summary_all_itcon['predROC'])
                                                                 }, ignore_index=True)

        
# plot the summarizing figure
fig2, axs2 = plt.subplots(1,1)
fig2.set_figheight(5)
fig2.set_figwidth(5)
seaborn.barplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                errorbar='ci',alpha=.5)
seaborn.swarmplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                  alpha=.9,size=6,dodge=True,legend=False)
plt.plot([-0.5,1.5],[0.5,0.5],'k--')
axs2.set_xlim([-0.5,1.5])
axs2.set_title('all animal')

savefig = 1
if savefig:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    fig2.savefig(figsavefolder+'withinCondition_DBNpredicition_DBNdependenciesAfterMI_binary_noself_'+timelagtype+'_summarizingplot.pdf')




# ### load the DBN related data for each dyad and run the prediction
# ### training set and testing set are from the same conditions
# ### use the DBN learned structure, consider the weight as well
# ### do not consider the self dependencies (dependencies to variables themselves, no diagonal dependencies)

# In[ ]:


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
    print('load all ROC data for within task condition (weighted dependencies without self dependencies), and only plot the summary figure')
        
    data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'

    with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_noself_'+timelagtype+'.pkl', 'rb') as f:
        ROC_summary_all = pickle.load(f)

except:
    # initialize a summary dataframe for plotting the summary figure across animals 
    ROC_summary_all = pd.DataFrame(columns=['animal','action','testCondition','predROC'])

    fig, axs = plt.subplots(nanimalpairs,nDBN_groups) # nDBN_groups(3) task conditions; 2 animal individual
    fig.set_figheight(7*nanimalpairs)
    fig.set_figwidth(7*nDBN_groups)

    for ianimalpair in np.arange(0,nanimalpairs,1):

        # load the DBN input data
        animal1_fixedorder = animal1_fixedorders[ianimalpair]
        animal2_fixedorder = animal2_fixedorders[ianimalpair]
        #
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder+animal2_fixedorder+'/'
        if not mergetempRos:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_mergeTempsReSo.pkl', 'rb') as f:
                DBN_input_data_alltypes = pickle.load(f)

        # load the DBN training outcome
        if moreSampSize:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                sig_edges_diffTempRo_diffSampSize = pickle.load(f)
        else:
            with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
            with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
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



        for igroup in np.arange(0,nDBN_groups,1):
            DBN_group_typename = DBN_group_typenames[igroup]

            DBN_input_data_tgt = DBN_input_data_alltypes[DBN_group_typename]
            weighted_graphs_tgt = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            weighted_graphs_shuffled_tgt = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            # sig_edges_tgt = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename]
            sig_edges_tgt = get_significant_edges(weighted_graphs_tgt,weighted_graphs_shuffled_tgt)

            # mean weight 
            weighted_graphs_mean_tgt = np.nanmean(weighted_graphs_tgt,axis=0)
            weighted_graphs_mean_tgt = weighted_graphs_mean_tgt * sig_edges_tgt

            bina_graphs_mean_tgt = (np.random.random(size=np.shape(weighted_graphs_mean_tgt))<weighted_graphs_mean_tgt).astype(int)
            
            #
            # consider the time lags
            if nlags == 1:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:] 
            elif nlags == 2:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]
            elif nlags == 3:
                bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]+bina_graphs_mean_tgt[fromRowIDs[2],:]

            

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

            # run niters iterations for each condition
            for iiter in np.arange(0,niters,1):


                # Split data into training and testing sets
                train_data, test_data = train_test_split(DBN_input_data_tgt, test_size=0.2)
                # test_data = DBN_input_data_tgt

                # Perform parameter learning for each time slice
                bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                # Perform inference
                infer = VariableElimination(bn)

                # Prediction for each behavioral events
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

                    # Optionally, plot the ROC curve - only plot for the first iteration
                    if iiter == 0:
                        fpr, tpr, _ = roc_curve(trueBeh, Pbehavior)
                        try:
                            axs[ianimalpair,igroup].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        except:
                            axs[ianimalpair].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))
                        
                    # put data in the summarizing data frame
                    if (ievent == 0) | (ievent == 2): # for animal1
                        ROC_summary_all = ROC_summary_all.append({'animal':animal1_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                                 }, ignore_index=True)
                    else:
                        ROC_summary_all = ROC_summary_all.append({'animal':animal2_fixedorder,
                                                                  'action':eventnames[ievent][2:],
                                                                  'testCondition':DBN_group_typename,
                                                                  'predROC':auc
                                                             }, ignore_index=True)

                if iiter == 0:
                    try:
                        axs[ianimalpair,igroup].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair,igroup].set_xlabel('False Positive Rate')
                        axs[ianimalpair,igroup].set_ylabel('True Positive Rate')
                        axs[ianimalpair,igroup].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair,igroup].legend(loc='lower right')
                    except:
                        axs[ianimalpair].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair].set_xlabel('False Positive Rate')
                        axs[ianimalpair].set_ylabel('True Positive Rate')
                        axs[ianimalpair].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename+'; with dependency:'+strategyname)
                        axs[ianimalpair].legend(loc='lower right')


    savefig = 1
    if savefig:
        figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        fig.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_weightedGraph_noself_'+timelagtype+'.pdf')

        
    # save the summarizing data ROC_summary_all
    savedata = 1
    if savedata:
        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(data_saved_subfolder):
            os.makedirs(data_saved_subfolder)

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_noself_'+timelagtype+'.pkl', 'wb') as f:
            pickle.dump(ROC_summary_all, f)
        

# average ROC across all iterations
# separate Kanga into two groups based on her partner
ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
#
animals_ROC = np.unique(ROC_summary_all['animal'])
nanis = np.shape(animals_ROC)[0]
actions_ROC = np.unique(ROC_summary_all['action'])
nacts = np.shape(actions_ROC)[0]
testCons_ROC = np.unique(ROC_summary_all['testCondition'])
ntcons = np.shape(testCons_ROC)[0]
ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all.keys())
#
for iani in np.arange(0,nanis,1):
    animal_ROC = animals_ROC[iani]
    ind_ani = ROC_summary_all['animal']==animal_ROC
    ROC_summary_all_iani = ROC_summary_all[ind_ani]
    #
    for iact in np.arange(0,nacts,1):
        action_ROC = actions_ROC[iact]
        ind_act = ROC_summary_all_iani['action']==action_ROC
        ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
        #
        for itcon in np.arange(0,ntcons,1):
            testCon_ROC = testCons_ROC[itcon]
            ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
            ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
            #
            ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                'action':action_ROC,
                                                                'testCondition':testCon_ROC,
                                                                'predROC':np.nanmean(ROC_summary_all_itcon['predROC'])
                                                                 }, ignore_index=True)

        
# plot the summarizing figure
fig2, axs2 = plt.subplots(1,1)
fig2.set_figheight(5)
fig2.set_figwidth(5)
seaborn.barplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                errorbar='ci',alpha=.5)
seaborn.swarmplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',
                  alpha=.9,size=6,dodge=True,legend=False)
plt.plot([-0.5,1.5],[0.5,0.5],'k--')
axs2.set_xlim([-0.5,1.5])
axs2.set_title('all animal')

savefig = 1
if savefig:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    fig2.savefig(figsavefolder+'withinCondition_DBNpredicition_fullDBN_weightedGraph_noself_'+timelagtype+'_summarizingplot.pdf')




# In[ ]:





# ### load the DBN related data for each dyad and run the prediction
# ### from one task condition to test on another task condition
# ### use the DBN learned structure, only the 0 1 DAG, do not consider the weights 

# In[ ]:


if not do_trainedMCs:
    
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

    temp_resolu = 1

    # ONLY FOR PLOT!! 
    # define DBN related summarizing variables
    DBN_group_typenames_train = ['coop(1s)','coop(1s)','coop(1s)']
    DBN_group_typeIDs_train  =  [3,3,3]
    DBN_group_coopthres_train = [1,1,1]
    #
    DBN_group_typenames_test = ['self','coop(1s)','no-vision']
    DBN_group_typeIDs_test  =  [1,3,5]
    DBN_group_coopthres_test = [0,1,0]
    #
    nDBN_groups = np.shape(DBN_group_typenames_train)[0]


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
        print('load all ROC data for across task conditions (only binary dependencies), and only plot the summary figure')

        data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'

        with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_binary_'+timelagtype+'_trainedon'+DBN_group_typenames_train[0]+'.pkl', 'rb') as f:
            ROC_summary_all = pickle.load(f)

    except:
        # initialize a summary dataframe for plotting the summary figure across animals 
        ROC_summary_all = pd.DataFrame(columns=['animal','action','testCondition','predROC'])

        fig, axs = plt.subplots(nanimalpairs,nDBN_groups) # nDBN_groups(3) task conditions; 2 animal individual
        fig.set_figheight(7*nanimalpairs)
        fig.set_figwidth(7*nDBN_groups)

        for ianimalpair in np.arange(0,nanimalpairs,1):

            # load the DBN input data
            animal1_fixedorder = animal1_fixedorders[ianimalpair]
            animal2_fixedorder = animal2_fixedorders[ianimalpair]
            #
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebody'+savefile_sufix+'_3lags/'+cameraID+'/'+animal1_fixedorder+animal2_fixedorder+'/'
            if not mergetempRos:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_'+str(temp_resolu)+'sReSo.pkl', 'rb') as f:
                    DBN_input_data_alltypes = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/DBN_input_data_alltypes_'+animal1_fixedorder+animal2_fixedorder+'_mergeTempsReSo.pkl', 'rb') as f:
                    DBN_input_data_alltypes = pickle.load(f)

            # load the DBN training outcome
            if moreSampSize:
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'_moreSampSize.pkl', 'rb') as f:
                    sig_edges_diffTempRo_diffSampSize = pickle.load(f)
            else:
                with open(data_saved_subfolder+'/weighted_graphs_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                    weighted_graphs_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/weighted_graphs_shuffled_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
                    weighted_graphs_shuffled_diffTempRo_diffSampSize = pickle.load(f)
                with open(data_saved_subfolder+'/sig_edges_diffTempRo_diffSampSize_'+animal1_fixedorder+animal2_fixedorder+'.pkl', 'rb') as f:
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



            for igroup in np.arange(0,nDBN_groups,1):

                DBN_group_typename_train = DBN_group_typenames_train[igroup]
                DBN_group_typename_test  = DBN_group_typenames_test[igroup]

                #
                # training dataset
                DBN_input_data_tgt_train = DBN_input_data_alltypes[DBN_group_typename_train]
                weighted_graphs_tgt_train = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename_train]
                weighted_graphs_shuffled_tgt_train = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename_train]
                # sig_edges_tgt_train = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename_train]
                sig_edges_tgt_train = get_significant_edges(weighted_graphs_tgt_train,weighted_graphs_shuffled_tgt_train)

                # mean weight 
                weighted_graphs_mean_tgt_train = np.nanmean(weighted_graphs_tgt_train,axis=0)
                weighted_graphs_mean_tgt_train = weighted_graphs_mean_tgt_train * sig_edges_tgt_train

                bina_graphs_mean_tgt_train = sig_edges_tgt_train

                # train_data = DBN_input_data_tgt_train

                #
                # testing dataset
                DBN_input_data_tgt_test = DBN_input_data_alltypes[DBN_group_typename_test]
                weighted_graphs_tgt_test = weighted_graphs_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename_test]
                weighted_graphs_shuffled_tgt_test = weighted_graphs_shuffled_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename_test]
                # sig_edges_tgt_test = sig_edges_diffTempRo_diffSampSize[(str(temp_resolu),j_sampsize_name)][DBN_group_typename_test]
                sig_edges_tgt_test = get_significant_edges(weighted_graphs_tgt_test,weighted_graphs_shuffled_tgt_test)

                # mean weight 
                weighted_graphs_mean_tgt_test = np.nanmean(weighted_graphs_tgt_test,axis=0)
                weighted_graphs_mean_tgt_test = weighted_graphs_mean_tgt_test * sig_edges_tgt_test

                bina_graphs_mean_tgt_test = sig_edges_tgt_test

                # test_data = DBN_input_data_tgt_test


                bina_graphs_mean_tgt = bina_graphs_mean_tgt_train
                # consider the time lags
                #
                if nlags == 1:
                    bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:] 
                elif nlags == 2:
                    bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]
                elif nlags == 3:
                    bina_graphs_mean_tgt = bina_graphs_mean_tgt[fromRowIDs[0],:]+bina_graphs_mean_tgt[fromRowIDs[1],:]+bina_graphs_mean_tgt[fromRowIDs[2],:]



                # translate the binary DAGs to edge
                nrows,ncols = np.shape(bina_graphs_mean_tgt)
                edgenames = []
                for irow in np.arange(0,nrows,1):
                    for icol in np.arange(0,ncols,1):
                        if bina_graphs_mean_tgt[irow,icol] > 0:
                            edgenames.append((fromNodes[irow],toNodes[icol]))

                # define the DBN predicting model
                bn = BayesianNetwork()
                bn.add_nodes_from(fromNodes)
                bn.add_nodes_from(toNodes)
                bn.add_edges_from(edgenames)

                effect_slice = toNodes        

                # run niters iterations for each condition
                for iiter in np.arange(0,niters,1):

                    # To match other analysis case, split data randomly here well 
                    # Split data into training and testing sets
                    train_data, _ = train_test_split(DBN_input_data_tgt_train, test_size=0.2)
                    _, test_data  = train_test_split(DBN_input_data_tgt_test, test_size=0.2)

                    # Perform parameter learning for each time slice
                    bn.fit(train_data, estimator=MaximumLikelihoodEstimator)

                    # Perform inference
                    infer = VariableElimination(bn)

                    # Prediction for each behavioral events
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

                        # Optionally, plot the ROC curve - only plot for the first iteration
                        if iiter == 0:
                            fpr, tpr, _ = roc_curve(trueBeh, Pbehavior)

                            axs[ianimalpair,igroup].plot(fpr, tpr, 
                                                         label=f'AUC for '+eventnames[ievent]+' auc = '+"{:.2f}".format(auc))

                        # put data in the summarizing data frame
                        if (ievent == 0) | (ievent == 2): # for animal1
                            ROC_summary_all = ROC_summary_all.append({'animal':animal1_fixedorder,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename_test,
                                                                      'predROC':auc
                                                                     }, ignore_index=True)
                        else:
                            ROC_summary_all = ROC_summary_all.append({'animal':animal2_fixedorder,
                                                                      'action':eventnames[ievent][2:],
                                                                      'testCondition':DBN_group_typename_test,
                                                                      'predROC':auc
                                                                 }, ignore_index=True)

                    if iiter == 0:
                        axs[ianimalpair,igroup].plot([0, 1], [0, 1], 'k--')
                        axs[ianimalpair,igroup].set_xlabel('False Positive Rate')
                        axs[ianimalpair,igroup].set_ylabel('True Positive Rate')
                        axs[ianimalpair,igroup].set_title('ROC Curve '+animal1_fixedorder+'&'+animal2_fixedorder
                                                          +'\n'+DBN_group_typename_train+' to '+DBN_group_typename_test)
                        axs[ianimalpair,igroup].legend(loc='lower right')



        savefig = 1
        if savefig:
            figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
            if not os.path.exists(figsavefolder):
                os.makedirs(figsavefolder)
            fig.savefig(figsavefolder+'acrossCondition_DBNpredicition_fullDBN_binaryGraph_'+timelagtype+'_DBNtrainedon'+DBN_group_typename_train+'.pdf')


        # save the summarizing data ROC_summary_all
        savedata = 1
        if savedata:
            data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
            if not os.path.exists(data_saved_subfolder):
                os.makedirs(data_saved_subfolder)

            with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_binary_'+timelagtype+'_trainedon'+DBN_group_typenames_train[0]+'.pkl', 'wb') as f:
                pickle.dump(ROC_summary_all, f)


        # average ROC across all iterations
        # separate Kanga into two groups based on her partner
        ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all.iloc[np.where([ROC_summary_all['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
        #
        animals_ROC = np.unique(ROC_summary_all['animal'])
        nanis = np.shape(animals_ROC)[0]
        actions_ROC = np.unique(ROC_summary_all['action'])
        nacts = np.shape(actions_ROC)[0]
        testCons_ROC = np.unique(ROC_summary_all['testCondition'])
        ntcons = np.shape(testCons_ROC)[0]
        ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all.keys())
        #
        for iani in np.arange(0,nanis,1):
            animal_ROC = animals_ROC[iani]
            ind_ani = ROC_summary_all['animal']==animal_ROC
            ROC_summary_all_iani = ROC_summary_all[ind_ani]
            #
            for iact in np.arange(0,nacts,1):
                action_ROC = actions_ROC[iact]
                ind_act = ROC_summary_all_iani['action']==action_ROC
                ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
                #
                for itcon in np.arange(0,ntcons,1):
                    testCon_ROC = testCons_ROC[itcon]
                    ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
                    ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
                    #
                    ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                        'action':action_ROC,
                                                                        'testCondition':testCon_ROC,
                                                                        'predROC':np.nanmean(ROC_summary_all_itcon['predROC'])
                                                                     }, ignore_index=True)

    # plot the summarizing figure
    fig2, axs2 = plt.subplots(1,1)
    fig2.set_figheight(5)
    fig2.set_figwidth(5)
    seaborn.barplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',errorbar='ci',alpha=.5)
    seaborn.swarmplot(ax=axs2,data=ROC_summary_all_mean,x='action',y='predROC',hue='testCondition',alpha=.9,size=6,dodge=True,legend=False)
    plt.plot([-0.5,1.5],[0.5,0.5],'k--')
    axs2.set_xlim([-0.5,1.5])
    axs2.set_title('all animal')

    savefig = 1
    if savefig:
        figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
        if not os.path.exists(figsavefolder):
            os.makedirs(figsavefolder)
        fig2.savefig(figsavefolder+'acrossCondition_DBNpredicition_fullDBN_binaryGraph_'+timelagtype+'_DBNtrainedon'+DBN_group_typenames_train[0]+'_summarizingplot.pdf')




# In[ ]:





# ### plot the ROC across all analysis conditions
# ### only do this after all the previous analysis were done 

# In[ ]:


# DBN model information
timelagtype = 'allthreelags'

# load all the ROC_all data
#
print('load all ROC data for within task condition (only binary dependencies)')
#         
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_binary_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_binary = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_binary.iloc[np.where([ROC_summary_all_full_binary['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_binary.iloc[np.where([ROC_summary_all_full_binary['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for within task condition (only binary dependencies without self dependencies)')
#         
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_binary_noself_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_binary_noself = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_binary_noself.iloc[np.where([ROC_summary_all_full_binary_noself['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_binary_noself.iloc[np.where([ROC_summary_all_full_binary_noself['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for within task condition (weighted dependencies)')
#
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_weighted = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_weighted.iloc[np.where([ROC_summary_all_full_weighted['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_weighted.iloc[np.where([ROC_summary_all_full_weighted['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for within task condition (weighted dependencies without self dependencies)')
#
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_noself_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_weighted_noself = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_weighted_noself.iloc[np.where([ROC_summary_all_full_weighted_noself['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_weighted_noself.iloc[np.where([ROC_summary_all_full_weighted_noself['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')


#
print('load all ROC data for hypothetical dependencies - three main dependencies')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_threeMains.pkl', 'rb') as f:
    ROC_summary_all_threeMains = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_threeMains.iloc[np.where([ROC_summary_all_threeMains['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_threeMains.iloc[np.where([ROC_summary_all_threeMains['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for hypothetical dependencies - sync_pulls')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_sync_pulls.pkl', 'rb') as f:
    ROC_summary_all_sync_pulls = pickle.load(f)    
# separate Kanga into two groups based on her partner
ROC_summary_all_sync_pulls.iloc[np.where([ROC_summary_all_sync_pulls['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_sync_pulls.iloc[np.where([ROC_summary_all_sync_pulls['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for hypothetical dependencies - gaze_lead_pull')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_gaze_lead_pull.pkl', 'rb') as f:
    ROC_summary_all_gaze_lead_pull = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_gaze_lead_pull.iloc[np.where([ROC_summary_all_gaze_lead_pull['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_gaze_lead_pull.iloc[np.where([ROC_summary_all_gaze_lead_pull['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for hypothetical dependencies - social_attention')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_social_attention.pkl', 'rb') as f:
    ROC_summary_all_social_attention = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_social_attention.iloc[np.where([ROC_summary_all_social_attention['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_social_attention.iloc[np.where([ROC_summary_all_social_attention['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for hypothetical dependencies - three main dependencies with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_threeMains_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_threeMains_withSelfEdge = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_threeMains_withSelfEdge.iloc[np.where([ROC_summary_all_threeMains_withSelfEdge['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_threeMains_withSelfEdge.iloc[np.where([ROC_summary_all_threeMains_withSelfEdge['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for hypothetical dependencies - sync_pulls with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_sync_pulls_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_sync_pulls_withSelfEdge = pickle.load(f)    
# separate Kanga into two groups based on her partner
ROC_summary_all_sync_pulls_withSelfEdge.iloc[np.where([ROC_summary_all_sync_pulls_withSelfEdge['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_sync_pulls_withSelfEdge.iloc[np.where([ROC_summary_all_sync_pulls_withSelfEdge['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for hypothetical dependencies - gaze_lead_pull with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_gaze_lead_pull_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_gaze_lead_pull_withSelfEdge = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_gaze_lead_pull_withSelfEdge.iloc[np.where([ROC_summary_all_gaze_lead_pull_withSelfEdge['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_gaze_lead_pull_withSelfEdge.iloc[np.where([ROC_summary_all_gaze_lead_pull_withSelfEdge['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for hypothetical dependencies - social_attention with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_social_attention_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_social_attention_withSelfEdge = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_social_attention_withSelfEdge.iloc[np.where([ROC_summary_all_social_attention_withSelfEdge['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_social_attention_withSelfEdge.iloc[np.where([ROC_summary_all_social_attention_withSelfEdge['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
        
    
# put all data together
ROC_summary_all_full_binary['type']='binary'
ROC_summary_all_full_binary_noself['type']='binary_noself'
ROC_summary_all_full_weighted['type']='weighted'
ROC_summary_all_full_weighted_noself['type']='weighted_noself'
ROC_summary_all_threeMains['type']='threeMains'
ROC_summary_all_sync_pulls['type']='sync_pulls'
ROC_summary_all_gaze_lead_pull['type']='gaze_lead_pull'
ROC_summary_all_social_attention['type']='social_attention'
#
ROC_summary_all_threeMains_withSelfEdge['type']='threeMains_withSelfEdge'
ROC_summary_all_sync_pulls_withSelfEdge['type']='sync_pulls_withSelfEdge'
ROC_summary_all_gaze_lead_pull_withSelfEdge['type']='gaze_lead_pull_withSelfEdge'
ROC_summary_all_social_attention_withSelfEdge['type']='social_attention_withSelfEdge'

# 
ROC_summary_all_merged = pd.concat([ROC_summary_all_full_binary,
                                    ROC_summary_all_full_binary_noself,
                                    ROC_summary_all_full_weighted,
                                    ROC_summary_all_full_weighted_noself,
                                    ROC_summary_all_threeMains,
                                    ROC_summary_all_sync_pulls,
                                    ROC_summary_all_gaze_lead_pull,
                                    ROC_summary_all_social_attention,
                                    #
                                    ROC_summary_all_threeMains_withSelfEdge,
                                    ROC_summary_all_sync_pulls_withSelfEdge,
                                    ROC_summary_all_gaze_lead_pull_withSelfEdge,
                                    ROC_summary_all_social_attention_withSelfEdge,
                                   ])


# for each animal calculate the mean
#
animals_ROC = np.unique(ROC_summary_all_merged['animal'])
nanis = np.shape(animals_ROC)[0]
actions_ROC = np.unique(ROC_summary_all_merged['action'])
nacts = np.shape(actions_ROC)[0]
testCons_ROC = np.unique(ROC_summary_all_merged['testCondition'])
ntcons = np.shape(testCons_ROC)[0]
types_ROC = np.unique(ROC_summary_all_merged['type'])
ntypes = np.shape(types_ROC)[0]
ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all_merged.keys())
#
for iani in np.arange(0,nanis,1):
    animal_ROC = animals_ROC[iani]
    ind_ani = ROC_summary_all_merged['animal']==animal_ROC
    ROC_summary_all_iani = ROC_summary_all_merged[ind_ani]
    #
    for iact in np.arange(0,nacts,1):
        action_ROC = actions_ROC[iact]
        ind_act = ROC_summary_all_iani['action']==action_ROC
        ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
        #
        for itcon in np.arange(0,ntcons,1):
            testCon_ROC = testCons_ROC[itcon]
            ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
            ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
            #
            for itype in np.arange(0,ntypes,1):
                type_ROC = types_ROC[itype]
                ind_type = ROC_summary_all_itcon['type'] == type_ROC
                ROC_summary_all_itype = ROC_summary_all_itcon[ind_type]
                #
                ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                    'action':action_ROC,
                                                                    'testCondition':testCon_ROC,
                                                                    'type':type_ROC,
                                                                    'predROC':np.nanmean(ROC_summary_all_itype['predROC'])
                                                                 }, ignore_index=True)


# plot
testCond_forplot = 'coop(1s)' # 'no-vision','self','coop(1s)'
# ROC_summary_all_forplot = ROC_summary_all_merged[ROC_summary_all_merged['testCondition']==testCond_forplot]
ROC_summary_all_forplot = ROC_summary_all_mean[ROC_summary_all_mean['testCondition']==testCond_forplot]

# plot the summarizing figure - pool all animals together
#
fig2, axs2 = plt.subplots(1,1)
fig2.set_figheight(5)
fig2.set_figwidth(15)
# s=seaborn.barplot(ax=axs2,data=ROC_summary_all_forplot,x='action',y='predROC',hue='type',errorbar='ci',alpha=.5)
s=seaborn.boxplot(ax=axs2,data=ROC_summary_all_forplot,x='action',y='predROC',hue='type',whis=3)
# s=seaborn.swarmplot(ax=axs2,data=ROC_summary_all_forplot,x='action',y='predROC',hue='type',
#                   alpha=.9,size=6,dodge=True,legend=False)
plt.plot([-0.5,1.5],[0.5,0.5],'k--')
axs2.set_xlim([-0.5,1.5])
axs2.set_title('all animal')
s.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# plot the summarizing figure - separate each animal
#
fig3, axs3 = plt.subplots(2,1)
fig3.set_figheight(10)
fig3.set_figwidth(10)
#
events_forplot = ['pull','gaze'] 
nevents = np.shape(events_forplot)[0]

for ievent in np.arange(0,nevents,1):
    ievent_forplot = events_forplot[ievent]
    
    ROC_summary_all_ievent = ROC_summary_all_forplot[ROC_summary_all_forplot['action']==ievent_forplot]
    #
    s=seaborn.barplot(ax=axs3[ievent],data=ROC_summary_all_ievent,x='animal',y='predROC',hue='type',
                    errorbar='ci',alpha=.5)
    # s=seaborn.boxplot(ax=axs3[ievent],data=ROC_summary_all_ievent,x='animal',y='predROC',hue='type')
    # s=seaborn.swarmplot(ax=axs3[ievent],data=ROC_summary_all_ievent,x='animal',y='predROC',hue='type',
    #                   alpha=.9,size=6,dodge=True,legend=False)
    axs3[ievent].plot([-0.5,9.5],[0.5,0.5],'k--')
    axs3[ievent].set_xlim([-0.5,9.5])
    axs3[ievent].set_title('all animal; events of '+ievent_forplot)
    s.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
# plot the summarizing figure - separate male and female
maleanimals = np.array(['eddie','dannon','dodson','vermelho'])
allanimalnames = np.array(ROC_summary_all_forplot['animal'])
allanimalsexes = allanimalnames.copy()
#
allanimalsexes[np.isin(allanimalnames,maleanimals)]='male'
allanimalsexes[~np.isin(allanimalnames,maleanimals)]='female'
#
ROC_summary_all_forplot['sex'] = allanimalsexes

#
fig4, axs4 = plt.subplots(1,2)
fig4.set_figheight(5)
fig4.set_figwidth(20)
#
events_forplot = ['pull','gaze'] 
nevents = np.shape(events_forplot)[0]

for ievent in np.arange(0,nevents,1):
    ievent_forplot = events_forplot[ievent]
    
    ROC_summary_all_ievent = ROC_summary_all_forplot[ROC_summary_all_forplot['action']==ievent_forplot]
    #
    # s=seaborn.barplot(ax=axs4[ievent],data=ROC_summary_all_ievent,x='sex',y='predROC',hue='type',
    #                 errorbar='ci',alpha=.5)
    s=seaborn.boxplot(ax=axs4[ievent],data=ROC_summary_all_ievent,x='sex',y='predROC',hue='type',whis=3)
    # s=seaborn.swarmplot(ax=axs3[ievent],data=ROC_summary_all_ievent,x='animal',y='predROC',hue='type',
    #                   alpha=.9,size=6,dodge=True,legend=False)
    axs4[ievent].plot([-0.5,1.5],[0.5,0.5],'k--')
    axs4[ievent].set_xlim([-0.5,1.5])
    axs4[ievent].set_title('all animal; events of '+ievent_forplot)
    s.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plot the summarizing figure - only plot the speficified condition; plot all animals together
fig5, axs5 = plt.subplots(1,1)
fig5.set_figheight(5)
fig5.set_figwidth(15)
#
testCond_forplot = 'coop(1s)' # 'no-vision','self','coop(1s)'
# ROC_summary_all_forplot = ROC_summary_all_merged[ROC_summary_all_merged['testCondition']==testCond_forplot]
ROC_summary_all_forplot = ROC_summary_all_mean[ROC_summary_all_mean['testCondition']==testCond_forplot]
#
plottypes = ['threeMains','sync_pulls','gaze_lead_pull','social_attention',
    'threeMains_withSelfEdge','sync_pulls_withSelfEdge','gaze_lead_pull_withSelfEdge','social_attention_withSelfEdge']
ind_toplot = np.isin(ROC_summary_all_forplot['type'],plottypes)
ROC_summary_all_forplot2 = ROC_summary_all_forplot[ind_toplot]

# s=seaborn.barplot(ax=axs5,data=ROC_summary_all_forplot2,x='action',y='predROC',hue='type',errorbar='ci',alpha=.5)
s=seaborn.boxplot(ax=axs5,data=ROC_summary_all_forplot2,x='action',y='predROC',hue='type',whis=3)
# s=seaborn.swarmplot(ax=axs5,data=ROC_summary_all_forplot2,x='action',y='predROC',hue='type',
#                   alpha=.9,size=6,dodge=True,legend=False)
plt.plot([-0.5,1.5],[0.5,0.5],'k--')
axs5.set_xlim([-0.5,1.5])
axs5.set_title('all animal, compared with and without self edges')
s.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    
savefig = 1
if savefig:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    fig2.savefig(figsavefolder+'withinCondition_DBNpredicition_'+timelagtype+'DBN_andHypotheticalDependencies_'+testCond_forplot+'_summarizingplot_allanimals.pdf')
    fig3.savefig(figsavefolder+'withinCondition_DBNpredicition_'+timelagtype+'DBN_andHypotheticalDependencies_'+testCond_forplot+'_summarizingplot_eachanimals.pdf')
    fig4.savefig(figsavefolder+'withinCondition_DBNpredicition_'+timelagtype+'DBN_andHypotheticalDependencies_'+testCond_forplot+'_summarizingplot_malefemales.pdf')
    fig5.savefig(figsavefolder+'withinCondition_DBNpredicition_'+timelagtype+'DBN_andHypotheticalDependencies_'+testCond_forplot+'_summarizingplot_allanimals_withandwithoutselfedge.pdf')


# In[ ]:


iind=(ROC_summary_all_forplot['action']=='pull')&(ROC_summary_all_forplot['testCondition']=='coop(1s)')&(ROC_summary_all_forplot['type']=='social_attention')

ROCtgt = np.array(ROC_summary_all_forplot[iind]['predROC'])

st.ttest_1samp(ROCtgt,0.5)


# In[ ]:





# In[ ]:


ROC_summary_all_mean


# ### plot the ROC across all analysis conditions
# ### only do this after all the previous analysis were done 
# #### compare 1s 2s and 3s time lags (different timelagtype)

# In[ ]:


# load all the ROC_all data

# 1s time lags 
timelagtype = '1secondlag'

print('load all ROC data for 1secondlag within task condition (weighted dependencies with self dependencies)')
#
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_weighted_1secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_weighted_1secondlag.iloc[np.where([ROC_summary_all_full_weighted_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_weighted_1secondlag.iloc[np.where([ROC_summary_all_full_weighted_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 1secondlag within task condition (weighted dependencies without self dependencies)')
#
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_noself_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_weighted_noself_1secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_weighted_noself_1secondlag.iloc[np.where([ROC_summary_all_full_weighted_noself_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_weighted_noself_1secondlag.iloc[np.where([ROC_summary_all_full_weighted_noself_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 1secondlag hypothetical dependencies - three main dependencies')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_threeMains.pkl', 'rb') as f:
    ROC_summary_all_threeMains_1secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_threeMains_1secondlag.iloc[np.where([ROC_summary_all_threeMains_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_threeMains_1secondlag.iloc[np.where([ROC_summary_all_threeMains_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 1secondlag hypothetical dependencies - sync_pulls')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_sync_pulls.pkl', 'rb') as f:
    ROC_summary_all_sync_pulls_1secondlag = pickle.load(f)    
# separate Kanga into two groups based on her partner
ROC_summary_all_sync_pulls_1secondlag.iloc[np.where([ROC_summary_all_sync_pulls_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_sync_pulls_1secondlag.iloc[np.where([ROC_summary_all_sync_pulls_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 1secondlag hypothetical dependencies - gaze_lead_pull')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_gaze_lead_pull.pkl', 'rb') as f:
    ROC_summary_all_gaze_lead_pull_1secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_gaze_lead_pull_1secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_gaze_lead_pull_1secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 1secondlag hypothetical dependencies - social_attention')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_social_attention.pkl', 'rb') as f:
    ROC_summary_all_social_attention_1secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_social_attention_1secondlag.iloc[np.where([ROC_summary_all_social_attention_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_social_attention_1secondlag.iloc[np.where([ROC_summary_all_social_attention_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 1secondlag hypothetical dependencies - three main dependencies with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_threeMains_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_threeMains_withSelfEdge_1secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_threeMains_withSelfEdge_1secondlag.iloc[np.where([ROC_summary_all_threeMains_withSelfEdge_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_threeMains_withSelfEdge_1secondlag.iloc[np.where([ROC_summary_all_threeMains_withSelfEdge_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 1secondlag hypothetical dependencies - sync_pulls with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_sync_pulls_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_sync_pulls_withSelfEdge_1secondlag = pickle.load(f)    
# separate Kanga into two groups based on her partner
ROC_summary_all_sync_pulls_withSelfEdge_1secondlag.iloc[np.where([ROC_summary_all_sync_pulls_withSelfEdge_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_sync_pulls_withSelfEdge_1secondlag.iloc[np.where([ROC_summary_all_sync_pulls_withSelfEdge_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 1secondlag hypothetical dependencies - gaze_lead_pull with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_gaze_lead_pull_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_gaze_lead_pull_withSelfEdge_1secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_gaze_lead_pull_withSelfEdge_1secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_withSelfEdge_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_gaze_lead_pull_withSelfEdge_1secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_withSelfEdge_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 2secondlag hypothetical dependencies - social_attention with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_social_attention_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_social_attention_withSelfEdge_1secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_social_attention_withSelfEdge_1secondlag.iloc[np.where([ROC_summary_all_social_attention_withSelfEdge_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_social_attention_withSelfEdge_1secondlag.iloc[np.where([ROC_summary_all_social_attention_withSelfEdge_1secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
        

# 2s time lags 
timelagtype = '2secondlag'

print('load all ROC data for 2secondlag within task condition (weighted dependencies with self dependencies)')
#
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_weighted_2secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_weighted_2secondlag.iloc[np.where([ROC_summary_all_full_weighted_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_weighted_2secondlag.iloc[np.where([ROC_summary_all_full_weighted_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 2secondlag within task condition (weighted dependencies without self dependencies)')
#
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_noself_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_weighted_noself_2secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_weighted_noself_2secondlag.iloc[np.where([ROC_summary_all_full_weighted_noself_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_weighted_noself_2secondlag.iloc[np.where([ROC_summary_all_full_weighted_noself_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 2secondlag hypothetical dependencies - three main dependencies')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_threeMains'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_threeMains_2secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_threeMains_2secondlag.iloc[np.where([ROC_summary_all_threeMains_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_threeMains_2secondlag.iloc[np.where([ROC_summary_all_threeMains_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 2secondlag hypothetical dependencies - sync_pulls')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_sync_pulls'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_sync_pulls_2secondlag = pickle.load(f)    
# separate Kanga into two groups based on her partner
ROC_summary_all_sync_pulls_2secondlag.iloc[np.where([ROC_summary_all_sync_pulls_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_sync_pulls_2secondlag.iloc[np.where([ROC_summary_all_sync_pulls_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 2secondlag hypothetical dependencies - gaze_lead_pull')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_gaze_lead_pull'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_gaze_lead_pull_2secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_gaze_lead_pull_2secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_gaze_lead_pull_2secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 2secondlag hypothetical dependencies - social_attention')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_social_attention'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_social_attention_2secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_social_attention_2secondlag.iloc[np.where([ROC_summary_all_social_attention_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_social_attention_2secondlag.iloc[np.where([ROC_summary_all_social_attention_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 2secondlag hypothetical dependencies - three main dependencies with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_threeMains'+timelagtype+'_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_threeMains_withSelfEdge_2secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_threeMains_withSelfEdge_2secondlag.iloc[np.where([ROC_summary_all_threeMains_withSelfEdge_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_threeMains_withSelfEdge_2secondlag.iloc[np.where([ROC_summary_all_threeMains_withSelfEdge_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 2secondlag hypothetical dependencies - sync_pulls with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_sync_pulls'+timelagtype+'_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_sync_pulls_withSelfEdge_2secondlag = pickle.load(f)    
# separate Kanga into two groups based on her partner
ROC_summary_all_sync_pulls_withSelfEdge_2secondlag.iloc[np.where([ROC_summary_all_sync_pulls_withSelfEdge_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_sync_pulls_withSelfEdge_2secondlag.iloc[np.where([ROC_summary_all_sync_pulls_withSelfEdge_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 2secondlag hypothetical dependencies - gaze_lead_pull with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_gaze_lead_pull'+timelagtype+'_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_gaze_lead_pull_withSelfEdge_2secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_gaze_lead_pull_withSelfEdge_2secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_withSelfEdge_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_gaze_lead_pull_withSelfEdge_2secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_withSelfEdge_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 2secondlag hypothetical dependencies - social_attention with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_social_attention'+timelagtype+'_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_social_attention_withSelfEdge_2secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_social_attention_withSelfEdge_2secondlag.iloc[np.where([ROC_summary_all_social_attention_withSelfEdge_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_social_attention_withSelfEdge_2secondlag.iloc[np.where([ROC_summary_all_social_attention_withSelfEdge_2secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')


# 3s time lags 
timelagtype = '3secondlag'

print('load all ROC data for 3secondlag within task condition (weighted dependencies with self dependencies)')
#
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_weighted_3secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_weighted_3secondlag.iloc[np.where([ROC_summary_all_full_weighted_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_weighted_3secondlag.iloc[np.where([ROC_summary_all_full_weighted_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 3secondlag within task condition (weighted dependencies without self dependencies)')
#
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_fullDBNdependencies_weighted_noself_'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_full_weighted_noself_3secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_full_weighted_noself_3secondlag.iloc[np.where([ROC_summary_all_full_weighted_noself_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_full_weighted_noself_3secondlag.iloc[np.where([ROC_summary_all_full_weighted_noself_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 3secondlag hypothetical dependencies - three main dependencies')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_threeMains'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_threeMains_3secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_threeMains_3secondlag.iloc[np.where([ROC_summary_all_threeMains_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_threeMains_3secondlag.iloc[np.where([ROC_summary_all_threeMains_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 3secondlag hypothetical dependencies - sync_pulls')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_sync_pulls'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_sync_pulls_3secondlag = pickle.load(f)    
# separate Kanga into two groups based on her partner
ROC_summary_all_sync_pulls_3secondlag.iloc[np.where([ROC_summary_all_sync_pulls_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_sync_pulls_3secondlag.iloc[np.where([ROC_summary_all_sync_pulls_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 3secondlag hypothetical dependencies - gaze_lead_pull')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_gaze_lead_pull'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_gaze_lead_pull_3secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_gaze_lead_pull_3secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_gaze_lead_pull_3secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 3secondlag hypothetical dependencies - social_attention')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_social_attention'+timelagtype+'.pkl', 'rb') as f:
    ROC_summary_all_social_attention_3secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_social_attention_3secondlag.iloc[np.where([ROC_summary_all_social_attention_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_social_attention_3secondlag.iloc[np.where([ROC_summary_all_social_attention_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 3secondlag hypothetical dependencies - three main dependencies with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_threeMains'+timelagtype+'_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_threeMains_withSelfEdge_3secondlag = pickle.load(f)
# separate Kanga into two groups based on her partner
ROC_summary_all_threeMains_withSelfEdge_3secondlag.iloc[np.where([ROC_summary_all_threeMains_withSelfEdge_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_threeMains_withSelfEdge_3secondlag.iloc[np.where([ROC_summary_all_threeMains_withSelfEdge_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 3secondlag hypothetical dependencies - sync_pulls with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_sync_pulls'+timelagtype+'_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_sync_pulls_withSelfEdge_3secondlag = pickle.load(f)    
# separate Kanga into two groups based on her partner
ROC_summary_all_sync_pulls_withSelfEdge_3secondlag.iloc[np.where([ROC_summary_all_sync_pulls_withSelfEdge_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_sync_pulls_withSelfEdge_3secondlag.iloc[np.where([ROC_summary_all_sync_pulls_withSelfEdge_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')
    
#
print('load all ROC data for 3secondlag hypothetical dependencies - gaze_lead_pull with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_gaze_lead_pull'+timelagtype+'_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_gaze_lead_pull_withSelfEdge_3secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_gaze_lead_pull_withSelfEdge_3secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_withSelfEdge_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_gaze_lead_pull_withSelfEdge_3secondlag.iloc[np.where([ROC_summary_all_gaze_lead_pull_withSelfEdge_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')

#
print('load all ROC data for 3secondlag hypothetical dependencies - social_attention with Self Edge')
#        
data_saved_subfolder = data_saved_folder+'data_saved_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
with open(data_saved_subfolder+'/ROC_summary_all_dependencies_social_attention'+timelagtype+'_withSelfEdge.pkl', 'rb') as f:
    ROC_summary_all_social_attention_withSelfEdge_3secondlag = pickle.load(f) 
# separate Kanga into two groups based on her partner
ROC_summary_all_social_attention_withSelfEdge_3secondlag.iloc[np.where([ROC_summary_all_social_attention_withSelfEdge_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]] = ROC_summary_all_social_attention_withSelfEdge_3secondlag.iloc[np.where([ROC_summary_all_social_attention_withSelfEdge_3secondlag['animal']=='kanga'])[1][0:niters*nDBN_groups*2]].replace('kanga','kanga1')




# In[ ]:


# put all data together
ROC_summary_all_full_weighted_1secondlag['type']='weighted_1secondlag'
ROC_summary_all_full_weighted_noself_1secondlag['type']='weighted_noself_1secondlag'
ROC_summary_all_threeMains_1secondlag['type']='threeMains_1secondlag'
ROC_summary_all_sync_pulls_1secondlag['type']='sync_pulls_1secondlag'
ROC_summary_all_gaze_lead_pull_1secondlag['type']='gaze_lead_pull_1secondlag'
ROC_summary_all_social_attention_1secondlag['type']='social_attention_1secondlag'
ROC_summary_all_threeMains_withSelfEdge_1secondlag['type']='threeMains_withSelfEdge_1secondlag'
ROC_summary_all_sync_pulls_withSelfEdge_1secondlag['type']='sync_pulls_withSelfEdge_1secondlag'
ROC_summary_all_gaze_lead_pull_withSelfEdge_1secondlag['type']='gaze_lead_pull_withSelfEdge_1secondlag'
ROC_summary_all_social_attention_withSelfEdge_1secondlag['type']='social_attention_withSelfEdge_1secondlag'
#
ROC_summary_all_full_weighted_2secondlag['type']='weighted_2secondlag'
ROC_summary_all_full_weighted_noself_2secondlag['type']='weighted_noself_2secondlag'
ROC_summary_all_threeMains_2secondlag['type']='threeMains_2secondlag'
ROC_summary_all_sync_pulls_2secondlag['type']='sync_pulls_2secondlag'
ROC_summary_all_gaze_lead_pull_2secondlag['type']='gaze_lead_pull_2secondlag'
ROC_summary_all_social_attention_2secondlag['type']='social_attention_2secondlag'
ROC_summary_all_threeMains_withSelfEdge_2secondlag['type']='threeMains_withSelfEdge_2secondlag'
ROC_summary_all_sync_pulls_withSelfEdge_2secondlag['type']='sync_pulls_withSelfEdge_2secondlag'
ROC_summary_all_gaze_lead_pull_withSelfEdge_2secondlag['type']='gaze_lead_pull_withSelfEdge_2secondlag'
ROC_summary_all_social_attention_withSelfEdge_2secondlag['type']='social_attention_withSelfEdge_2secondlag'
#
ROC_summary_all_full_weighted_3secondlag['type']='weighted_3secondlag'
ROC_summary_all_full_weighted_noself_3secondlag['type']='weighted_noself_3secondlag'
ROC_summary_all_threeMains_3secondlag['type']='threeMains_3secondlag'
ROC_summary_all_sync_pulls_3secondlag['type']='sync_pulls_3secondlag'
ROC_summary_all_gaze_lead_pull_3secondlag['type']='gaze_lead_pull_3secondlag'
ROC_summary_all_social_attention_3secondlag['type']='social_attention_3secondlag'
ROC_summary_all_threeMains_withSelfEdge_3secondlag['type']='threeMains_withSelfEdge_3secondlag'
ROC_summary_all_sync_pulls_withSelfEdge_3secondlag['type']='sync_pulls_withSelfEdge_3secondlag'
ROC_summary_all_gaze_lead_pull_withSelfEdge_3secondlag['type']='gaze_lead_pull_withSelfEdge_3secondlag'
ROC_summary_all_social_attention_withSelfEdge_3secondlag['type']='social_attention_withSelfEdge_3secondlag'


#
ROC_summary_all_merged = pd.concat([
                                    ROC_summary_all_full_weighted_1secondlag,
                                    ROC_summary_all_full_weighted_noself_1secondlag,
                                    ROC_summary_all_threeMains_1secondlag,
                                    ROC_summary_all_sync_pulls_1secondlag,
                                    ROC_summary_all_gaze_lead_pull_1secondlag,
                                    ROC_summary_all_social_attention_1secondlag,
                                    ROC_summary_all_threeMains_withSelfEdge_1secondlag,
                                    ROC_summary_all_sync_pulls_withSelfEdge_1secondlag,
                                    ROC_summary_all_gaze_lead_pull_withSelfEdge_1secondlag,
                                    ROC_summary_all_social_attention_withSelfEdge_1secondlag,
                                    #
                                    ROC_summary_all_full_weighted_2secondlag,
                                    ROC_summary_all_full_weighted_noself_2secondlag,
                                    ROC_summary_all_threeMains_2secondlag,
                                    ROC_summary_all_sync_pulls_2secondlag,
                                    ROC_summary_all_gaze_lead_pull_2secondlag,
                                    ROC_summary_all_social_attention_2secondlag,
                                    ROC_summary_all_threeMains_withSelfEdge_2secondlag,
                                    ROC_summary_all_sync_pulls_withSelfEdge_2secondlag,
                                    ROC_summary_all_gaze_lead_pull_withSelfEdge_2secondlag,
                                    ROC_summary_all_social_attention_withSelfEdge_2secondlag,
                                    #
                                    ROC_summary_all_full_weighted_3secondlag,
                                    ROC_summary_all_full_weighted_noself_3secondlag,
                                    ROC_summary_all_threeMains_3secondlag,
                                    ROC_summary_all_sync_pulls_3secondlag,
                                    ROC_summary_all_gaze_lead_pull_3secondlag,
                                    ROC_summary_all_social_attention_3secondlag,
                                    ROC_summary_all_threeMains_withSelfEdge_3secondlag,
                                    ROC_summary_all_sync_pulls_withSelfEdge_3secondlag,
                                    ROC_summary_all_gaze_lead_pull_withSelfEdge_3secondlag,
                                    ROC_summary_all_social_attention_withSelfEdge_3secondlag,
                                   ])


# In[ ]:




# for each animal calculate the mean
#
animals_ROC = np.unique(ROC_summary_all_merged['animal'])
nanis = np.shape(animals_ROC)[0]
actions_ROC = np.unique(ROC_summary_all_merged['action'])
nacts = np.shape(actions_ROC)[0]
testCons_ROC = np.unique(ROC_summary_all_merged['testCondition'])
ntcons = np.shape(testCons_ROC)[0]
types_ROC = np.unique(ROC_summary_all_merged['type'])
ntypes = np.shape(types_ROC)[0]
ROC_summary_all_mean = pd.DataFrame(columns=ROC_summary_all_merged.keys())
#
for iani in np.arange(0,nanis,1):
    animal_ROC = animals_ROC[iani]
    ind_ani = ROC_summary_all_merged['animal']==animal_ROC
    ROC_summary_all_iani = ROC_summary_all_merged[ind_ani]
    #
    for iact in np.arange(0,nacts,1):
        action_ROC = actions_ROC[iact]
        ind_act = ROC_summary_all_iani['action']==action_ROC
        ROC_summary_all_iact = ROC_summary_all_iani[ind_act]
        #
        for itcon in np.arange(0,ntcons,1):
            testCon_ROC = testCons_ROC[itcon]
            ind_tcon = ROC_summary_all_iact['testCondition']==testCon_ROC
            ROC_summary_all_itcon = ROC_summary_all_iact[ind_tcon]
            #
            for itype in np.arange(0,ntypes,1):
                type_ROC = types_ROC[itype]
                ind_type = ROC_summary_all_itcon['type'] == type_ROC
                ROC_summary_all_itype = ROC_summary_all_itcon[ind_type]
                #
                ROC_summary_all_mean = ROC_summary_all_mean.append({'animal':animal_ROC,
                                                                    'action':action_ROC,
                                                                    'testCondition':testCon_ROC,
                                                                    'type':type_ROC,
                                                                    'predROC':np.nanmean(ROC_summary_all_itype['predROC'])
                                                                 }, ignore_index=True)
                


# #### plot for one specific task condition

# In[ ]:



# plot
testCond_forplot = 'coop(1.5s)' # 'no-vision','self','coop(1s)'
# testCond_forplot = 'self' # 'no-vision','self','coop(1s)'
# ROC_summary_all_forplot = ROC_summary_all_merged[ROC_summary_all_merged['testCondition']==testCond_forplot]
ROC_summary_all_forplot = ROC_summary_all_mean[ROC_summary_all_mean['testCondition']==testCond_forplot]

# plot the summarizing figure - pool all animals together
#
fig2, axs2 = plt.subplots(1,1)
fig2.set_figheight(5)
fig2.set_figwidth(15)
# s=seaborn.barplot(ax=axs2,data=ROC_summary_all_forplot,x='action',y='predROC',hue='type',errorbar='ci',alpha=.5)
s=seaborn.boxplot(ax=axs2,data=ROC_summary_all_forplot,x='action',y='predROC',hue='type',whis=3)
# s=seaborn.swarmplot(ax=axs2,data=ROC_summary_all_forplot,x='action',y='predROC',hue='type',
#                   alpha=.9,size=6,dodge=True,legend=False)
plt.plot([-0.5,1.5],[0.5,0.5],'k--')
axs2.set_xlim([-0.5,1.5])
axs2.set_title('all animal')
s.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# plot the summarizing figure - separate each animal
#
fig3, axs3 = plt.subplots(2,1)
fig3.set_figheight(10)
fig3.set_figwidth(10)
#
events_forplot = ['pull','gaze'] 
nevents = np.shape(events_forplot)[0]

for ievent in np.arange(0,nevents,1):
    ievent_forplot = events_forplot[ievent]
    
    ROC_summary_all_ievent = ROC_summary_all_forplot[ROC_summary_all_forplot['action']==ievent_forplot]
    #
    s=seaborn.barplot(ax=axs3[ievent],data=ROC_summary_all_ievent,x='animal',y='predROC',hue='type',
                    errorbar='ci',alpha=.5)
    # s=seaborn.boxplot(ax=axs3[ievent],data=ROC_summary_all_ievent,x='animal',y='predROC',hue='type')
    # s=seaborn.swarmplot(ax=axs3[ievent],data=ROC_summary_all_ievent,x='animal',y='predROC',hue='type',
    #                   alpha=.9,size=6,dodge=True,legend=False)
    axs3[ievent].plot([-0.5,9.5],[0.5,0.5],'k--')
    axs3[ievent].set_xlim([-0.5,9.5])
    axs3[ievent].set_title('all animal; events of '+ievent_forplot)
    s.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
# plot the summarizing figure - separate male and female
maleanimals = np.array(['eddie','dannon','dodson','vermelho'])
allanimalnames = np.array(ROC_summary_all_forplot['animal'])
allanimalsexes = allanimalnames.copy()
#
allanimalsexes[np.isin(allanimalnames,maleanimals)]='male'
allanimalsexes[~np.isin(allanimalnames,maleanimals)]='female'
#
ROC_summary_all_forplot['sex'] = allanimalsexes

#
fig4, axs4 = plt.subplots(1,2)
fig4.set_figheight(5)
fig4.set_figwidth(20)
#
events_forplot = ['pull','gaze'] 
nevents = np.shape(events_forplot)[0]

for ievent in np.arange(0,nevents,1):
    ievent_forplot = events_forplot[ievent]
    
    ROC_summary_all_ievent = ROC_summary_all_forplot[ROC_summary_all_forplot['action']==ievent_forplot]
    #
    # s=seaborn.barplot(ax=axs4[ievent],data=ROC_summary_all_ievent,x='sex',y='predROC',hue='type',
    #                 errorbar='ci',alpha=.5)
    s=seaborn.boxplot(ax=axs4[ievent],data=ROC_summary_all_ievent,x='sex',y='predROC',hue='type',whis=3)
    # s=seaborn.swarmplot(ax=axs3[ievent],data=ROC_summary_all_ievent,x='animal',y='predROC',hue='type',
    #                   alpha=.9,size=6,dodge=True,legend=False)
    axs4[ievent].plot([-0.5,1.5],[0.5,0.5],'k--')
    axs4[ievent].set_xlim([-0.5,1.5])
    axs4[ievent].set_title('all animal; events of '+ievent_forplot)
    if ievent == nevents-1:
        s.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        axs4[ievent].legend([])

if 0:
    # plot the summarizing figure - only plot the speficified condition; plot all animals together
    fig5, axs5 = plt.subplots(1,1)
    fig5.set_figheight(5)
    fig5.set_figwidth(15)
    #
    testCond_forplot = 'coop(1s)' # 'no-vision','self','coop(1s)'
    # ROC_summary_all_forplot = ROC_summary_all_merged[ROC_summary_all_merged['testCondition']==testCond_forplot]
    ROC_summary_all_forplot = ROC_summary_all_mean[ROC_summary_all_mean['testCondition']==testCond_forplot]
    #
    plottypes = ['threeMains','sync_pulls','gaze_lead_pull','social_attention',
        'threeMains_withSelfEdge','sync_pulls_withSelfEdge','gaze_lead_pull_withSelfEdge','social_attention_withSelfEdge']
    ind_toplot = np.isin(ROC_summary_all_forplot['type'],plottypes)
    ROC_summary_all_forplot2 = ROC_summary_all_forplot[ind_toplot]

    # s=seaborn.barplot(ax=axs5,data=ROC_summary_all_forplot2,x='action',y='predROC',hue='type',errorbar='ci',alpha=.5)
    s=seaborn.boxplot(ax=axs5,data=ROC_summary_all_forplot2,x='action',y='predROC',hue='type',whis=3)
    # s=seaborn.swarmplot(ax=axs5,data=ROC_summary_all_forplot2,x='action',y='predROC',hue='type',
    #                   alpha=.9,size=6,dodge=True,legend=False)
    plt.plot([-0.5,1.5],[0.5,0.5],'k--')
    axs5.set_xlim([-0.5,1.5])
    axs5.set_title('all animal, compared with and without self edges')
    s.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    
savefig = 1
if savefig:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    fig2.savefig(figsavefolder+'withinCondition_DBNpredicition_andHypotheticalDependencies_differenttimelag_'+testCond_forplot+'_summarizingplot_allanimals.pdf')
    fig3.savefig(figsavefolder+'withinCondition_DBNpredicition_andHypotheticalDependencies_differenttimelag_'+testCond_forplot+'_summarizingplot_eachanimals.pdf')
    fig4.savefig(figsavefolder+'withinCondition_DBNpredicition_andHypotheticalDependencies_differenttimelag_'+testCond_forplot+'_summarizingplot_malefemales.pdf')
    if 0:
        fig5.savefig(figsavefolder+'withinCondition_DBNpredicition_andHypotheticalDependencies_differenttimelag_'+testCond_forplot+'_summarizingplot_allanimals_withandwithoutselfedge.pdf')


# #### plot, for multiple task condition, across multiple time lags

# In[ ]:


ROC_summary_all_forplot = ROC_summary_all_mean.copy()

taskconds_order = ['self','coop(2s)','coop(1.5s)','coop(1s)','no-vision']
ROC_summary_all_forplot['testCondition'] = pd.Categorical(ROC_summary_all_forplot['testCondition'], 
                                                          categories=taskconds_order, ordered=True)
ROC_summary_all_forplot = ROC_summary_all_forplot.sort_values(by='testCondition')

#
events_forplot = ['pull','gaze'] 
nevents = np.shape(events_forplot)[0]

# 
# 
examinewithselfedge = 0
if examinewithselfedge:
    dependencies_forplot = ['weighted','threeMains_withSelfEdge','sync_pulls_withSelfEdge',
                            'gaze_lead_pull_withSelfEdge','social_attention_withSelfEdge']
else:
    dependencies_forplot = ['weighted_noself','threeMains','sync_pulls','gaze_lead_pull','social_attention']
    
ndepends = np.shape(dependencies_forplot)[0]


# separate the action and stretagy structure; box plot
fig4, axs4 = plt.subplots(ndepends,nevents)
fig4.set_figheight(5*ndepends)
fig4.set_figwidth(10*nevents)

for idepend in np.arange(0,ndepends,1):
    idepend_forplot = dependencies_forplot[idepend]
    
    idepend_alllags_forplot = [idepend_forplot+'_1secondlag',
                               idepend_forplot+'_2secondlag',
                               idepend_forplot+'_3secondlag',]
    
    
    ROC_summary_all_idepend = ROC_summary_all_forplot[np.isin(ROC_summary_all_forplot['type'],idepend_alllags_forplot)]
    ROC_summary_all_idepend['type'] = pd.Categorical(ROC_summary_all_idepend['type'],
                                                     categories = idepend_alllags_forplot, ordered=True)
    ROC_summary_all_idepend = ROC_summary_all_idepend.sort_values(by='type')
    
    
    for ievent in np.arange(0,nevents,1):
        ievent_forplot = events_forplot[ievent]

        ROC_summary_all_ievent = ROC_summary_all_idepend[ROC_summary_all_idepend['action']==ievent_forplot]
        #
        # s=seaborn.barplot(ax=axs4[idepend,ievent],data=ROC_summary_all_ievent,x='sex',y='predROC',hue='type',
        #                 errorbar='ci',alpha=.5)
        s=seaborn.boxplot(ax=axs4[idepend,ievent],data=ROC_summary_all_ievent,
                          x='testCondition',y='predROC',hue='type',whis=3)
        # s=seaborn.swarmplot(ax=axs3[idepend,ievent],data=ROC_summary_all_ievent,x='animal',y='predROC',hue='type',
        #                   alpha=.9,size=6,dodge=True,legend=False)
        # s=seaborn.lineplot(ax=axs4[idepend,ievent],data=ROC_summary_all_ievent,
        #                   x='testCondition',y='predROC',hue='type')
        axs4[idepend,ievent].plot([-0.5,4.5],[0.5,0.5],'k--')
        axs4[idepend,ievent].set_xlim([-0.5,4.5])
        axs4[idepend,ievent].set_ylim([0.45,0.8])
        axs4[idepend,ievent].set_title('all animal; events of '+ievent_forplot)
        if ievent == nevents-1:
            s.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            axs4[idepend,ievent].legend([])
            
        if 0:
            # Connect the same individuals with lines across categories
            x_ticks_offset = [-0.33,0,0.33]
            linecolors = ['b','r','g']
            for animal in ROC_summary_all_idepend['animal'].unique():
                individual_data = ROC_summary_all_ievent[ROC_summary_all_idepend['animal'] == animal]
                
                types_toplot = np.unique(individual_data['type'])
                ntypes = np.shape(types_toplot)[0]
                
                for itype in np.arange(0,ntypes,1):
                    indidivual_data_itype = individual_data[individual_data['type']==types_toplot[itype]]
                
                    indidivual_data_itype['testCondition'] = pd.Categorical(indidivual_data_itype['testCondition'], 
                                                          categories=taskconds_order, ordered=True)
                    indidivual_data_itype =indidivual_data_itype.sort_values(by='testCondition')
                
                    xxx = axs4[idepend,ievent].get_xticks()
                    xticklabels = axs4[idepend,ievent].get_xticklabels()
                    xtick_labels_text = [label.get_text() for label in xticklabels]
                    nticks = np.shape(xtick_labels_text)[0]
                    #
                    xxx_toplots = np.zeros(np.shape(indidivual_data_itype['testCondition']))
                    #
                    for itick in np.arange(0,nticks,1):
                        indd = np.array(indidivual_data_itype['testCondition']) == xtick_labels_text[itick]
                        xxx_toplots[indd] = xxx[itick]
                                       
                    
                    axs4[idepend,ievent].plot(xxx_toplots+x_ticks_offset[itype], indidivual_data_itype['predROC'],
                                              '-', color=linecolors[itype])



# separate the dependency structures and time lags, similar to the DBN fitting summarizing results; line plot
timelags_forplot = ['1secondlag','2secondlag','3secondlag']
nlags = np.shape(timelags_forplot)[0]
fig5, axs5 = plt.subplots(nlags,ndepends)
fig5.set_figheight(6*nlags)
fig5.set_figwidth(6*ndepends)
     
for idepend in np.arange(0,ndepends,1):
    idepend_forplot = dependencies_forplot[idepend]
    
    for ilag in np.arange(0,nlags,1):
        ilag_forplot = timelags_forplot[ilag]
        
        idepend_ilag_forplot = idepend_forplot+'_'+ilag_forplot

        ROC_summary_all_idepend_ilag = ROC_summary_all_forplot[np.isin(ROC_summary_all_forplot['type'],idepend_ilag_forplot)]
        ROC_summary_all_idepend_ilag['action'] = pd.Categorical(ROC_summary_all_idepend_ilag['action'],
                                                     categories = events_forplot, ordered=True)
        ROC_summary_all_idepend_ilag = ROC_summary_all_idepend_ilag.sort_values(by='action')
        
        
        s=seaborn.lineplot(ax=axs5[ilag,idepend],data=ROC_summary_all_idepend_ilag,
                          x='testCondition',y='predROC',hue='action')
        axs5[ilag,idepend].plot([-0.5,4.5],[0.5,0.5],'k--')
        axs5[ilag,idepend].set_xlim([-0.5,4.5])
        axs5[ilag,idepend].set_ylim([0.49,0.7])
        axs5[ilag,idepend].set_title('all animal; events of '+idepend_ilag_forplot)
        if idepend == ndepends-1:
            s.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            axs5[ilag,idepend].legend([])
        
savefig = 1
if savefig:
    figsavefolder = data_saved_folder+'figs_for_3LagDBN_and_bhv_singlecam_wholebodylabels_combinesessions_basicEvents_DBNpredictions/'+savefile_sufix+'/'+cameraID+'/'
    if not os.path.exists(figsavefolder):
        os.makedirs(figsavefolder)
    if examinewithselfedge:
        fig4.savefig(figsavefolder+'withinCondition_DBNpredicition_andHypotheticalDependencies_withselfedge_differenttimelag_multiconditions_summarizingplot.pdf')
        fig5.savefig(figsavefolder+'withinCondition_DBNpredicition_andHypotheticalDependencies_withselfedge_differenttimelag_multiconditions_summarizingplot_lineplot.pdf')       
    else:
        fig4.savefig(figsavefolder+'withinCondition_DBNpredicition_andHypotheticalDependencies_noselfedge_differenttimelag_multiconditions_summarizingplot.pdf')
        fig5.savefig(figsavefolder+'withinCondition_DBNpredicition_andHypotheticalDependencies_noselfedge_differenttimelag_multiconditions_summarizingplot_lineplot.pdf')


# In[ ]:





# In[ ]:





# In[ ]:





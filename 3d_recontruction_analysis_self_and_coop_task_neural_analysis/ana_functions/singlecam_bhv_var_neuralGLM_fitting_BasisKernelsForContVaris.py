#  function - get singlecam variables and neurons and then do the GLM fitting

# helper functions for the glm fitting

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
def make_gaussian_basis(duration_s, n_basis, dt, offset_s=0.0, sigma_scale=1.0):
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


def get_singlecam_bhv_var_for_neuralGLM_fitting_BasisKernelsForContVaris(gausKernelsize, fps, animal1, animal2, recordedanimal, animalnames_videotrack, session_start_time, time_point_pull1, time_point_pull2, time_point_juice1, time_point_juice2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, output_look_ornot, output_allvectors, output_allangles, output_key_locations, spike_clusters_data, spike_time_data, spike_channels_data):
    

    

    gaze_thresold = 0.2 # min length threshold to define if a gaze is real gaze or noise, in the unit of second

   
    ###### 
    # # prepare the continuous data
    ######
    
    # prepare some time stamp data
    # merge oneway gaze and mutual gaze # note!! the time stamps are already aligned to the start of session (instead of the start of video recording)
    oneway_gaze1 = np.sort(np.hstack((oneway_gaze1,mutual_gaze1)))
    oneway_gaze2 = np.sort(np.hstack((oneway_gaze2,mutual_gaze2)))

    # get the gaze start and stop
    #animal1_gaze = np.concatenate([oneway_gaze1, mutual_gaze1])
    animal1_gaze = oneway_gaze1
    animal1_gaze = np.sort(np.unique(animal1_gaze))
    animal1_gaze_stop = animal1_gaze[np.concatenate(((animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal1_gaze_start = np.concatenate(([animal1_gaze[0]],animal1_gaze[np.where(animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)[0]+1]))
    animal1_gaze_flash = np.intersect1d(animal1_gaze_start, animal1_gaze_stop)
    animal1_gaze_start = animal1_gaze_start[~np.isin(animal1_gaze_start,animal1_gaze_flash)]
    animal1_gaze_stop = animal1_gaze_stop[~np.isin(animal1_gaze_stop,animal1_gaze_flash)]
    #
    #animal2_gaze = np.concatenate([oneway_gaze2, mutual_gaze2])
    animal2_gaze = oneway_gaze2
    animal2_gaze = np.sort(np.unique(animal2_gaze))
    animal2_gaze_stop = animal2_gaze[np.concatenate(((animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal2_gaze_start = np.concatenate(([animal2_gaze[0]],animal2_gaze[np.where(animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)[0]+1]))
    animal2_gaze_flash = np.intersect1d(animal2_gaze_start, animal2_gaze_stop)
    animal2_gaze_start = animal2_gaze_start[~np.isin(animal2_gaze_start,animal2_gaze_flash)]
    animal2_gaze_stop = animal2_gaze_stop[~np.isin(animal2_gaze_stop,animal2_gaze_flash)] 


    con_vars_plot = ['gaze_other_angle', 'gaze_tube_angle', 'gaze_lever_angle', 'animal_animal_dist', 'animal_tube_dist', 'animal_lever_dist', 'othergaze_self_angle', 'mass_move_speed', 'other_mass_move_speed', 'gaze_angle_speed', 'otherani_otherlever_dist', 'otherani_othertube_dist', 'socialgaze_prob', 'othergaze_prob', 'otherpull_prob', 'selfpull_prob', 'selfjuice_prob']
    
    data_summary_names = con_vars_plot
    
    nconvarplots = np.shape(con_vars_plot)[0]    
    
    nanimals = np.shape(animalnames_videotrack)[0]
    
    data_summary = dict.fromkeys([animal1,animal2],[])
    
    #
    # define and organize the continuous variables corresponding to the both animals
    for ianimal in np.arange(0,nanimals,1):

        #
        # load the video tracking animal names
        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]
            
        if ianimal == 0:
            animal_name_forsaving = animal1
        elif ianimal == 1:
            animal_name_forsaving = animal2

        #
        # get the variables
        gaze_other_angle = output_allangles['other_eye_angle_all_merge'][animal_name]
        gaze_other_angle = np.array(gaze_other_angle)
        # fill NaNs
        nans = np.isnan(gaze_other_angle)
        if np.any(~nans):
            gaze_other_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), gaze_other_angle[~nans])
        #
        gaze_other_angle = scipy.ndimage.gaussian_filter1d(gaze_other_angle,gausKernelsize)  # smooth the curve, use 30 before, change to 3 

        gaze_tube_angle = output_allangles['tube_eye_angle_all_merge'][animal_name]
        gaze_tube_angle = np.array(gaze_tube_angle)
        # fill NaNs
        nans = np.isnan(gaze_tube_angle)
        if np.any(~nans):
            gaze_tube_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), gaze_tube_angle[~nans])
        #
        gaze_tube_angle = scipy.ndimage.gaussian_filter1d(gaze_tube_angle,gausKernelsize)  

        gaze_lever_angle = output_allangles['lever_eye_angle_all_merge'][animal_name]
        gaze_lever_angle = np.array(gaze_lever_angle)
        # fill NaNs
        nans = np.isnan(gaze_lever_angle)
        if np.any(~nans):
            gaze_lever_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), gaze_lever_angle[~nans])
        #
        gaze_lever_angle = scipy.ndimage.gaussian_filter1d(gaze_lever_angle,gausKernelsize)  

        othergaze_self_angle = output_allangles['other_eye_angle_all_merge'][animal_name_other]
        othergaze_self_angle = np.array(othergaze_self_angle)
        # fill NaNs
        nans = np.isnan(othergaze_self_angle)
        if np.any(~nans):
            othergaze_self_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), othergaze_self_angle[~nans])
        #
        othergaze_self_angle = scipy.ndimage.gaussian_filter1d(othergaze_self_angle,gausKernelsize)  

        a = output_key_locations['facemass_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['facemass_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_animal_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(animal_animal_dist)
        if np.any(~nans):
            animal_animal_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), animal_animal_dist[~nans])
        #
        animal_animal_dist = scipy.ndimage.gaussian_filter1d(animal_animal_dist,gausKernelsize)  

        a = output_key_locations['tube_loc_all_merge'][animal_name].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_tube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(animal_tube_dist)
        if np.any(~nans):
            animal_tube_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), animal_tube_dist[~nans])
        #
        animal_tube_dist = scipy.ndimage.gaussian_filter1d(animal_tube_dist,gausKernelsize)  

        a = output_key_locations['lever_loc_all_merge'][animal_name].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_lever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(animal_lever_dist)
        if np.any(~nans):
            animal_lever_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), animal_lever_dist[~nans])
        #
        animal_lever_dist = scipy.ndimage.gaussian_filter1d(animal_lever_dist,gausKernelsize)  

        a = output_key_locations['facemass_loc_all_merge'][animal_name].transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        # fill NaNs
        nans = np.isnan(mass_move_speed)
        if np.any(~nans):
            mass_move_speed[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), mass_move_speed[~nans])
        #
        mass_move_speed = scipy.ndimage.gaussian_filter1d(mass_move_speed,gausKernelsize)  

        a = output_key_locations['facemass_loc_all_merge'][animal_name_other].transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        other_mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        # fill NaNs
        nans = np.isnan(other_mass_move_speed)
        if np.any(~nans):
            other_mass_move_speed[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), other_mass_move_speed[~nans])
        #
        other_mass_move_speed = scipy.ndimage.gaussian_filter1d(other_mass_move_speed,gausKernelsize)

        a = np.array(output_allvectors['head_vect_all_merge'][animal_name]).transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1 = a[:,1:]
        at0 = a[:,:-1] 
        nframes = np.shape(at1)[1]
        gaze_angle_speed = np.empty((1,nframes,))
        gaze_angle_speed[:] = np.nan
        gaze_angle_speed = gaze_angle_speed[0]
        #
        for iframe in np.arange(0,nframes,1):
            gaze_angle_speed[iframe] = np.arccos(np.clip(np.dot(at1[:,iframe]/np.linalg.norm(at1[:,iframe]), at0[:,iframe]/np.linalg.norm(at0[:,iframe])), -1.0, 1.0))    
        # fill NaNs
        nans = np.isnan(gaze_angle_speed)
        if np.any(~nans):
            gaze_angle_speed[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), gaze_angle_speed[~nans])
        #
        gaze_angle_speed = scipy.ndimage.gaussian_filter1d(gaze_angle_speed,gausKernelsize)  

        a = output_key_locations['lever_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name_other].transpose()
        a_min_b = a - b
        otherani_otherlever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(otherani_otherlever_dist)
        if np.any(~nans):
            otherani_otherlever_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), otherani_otherlever_dist[~nans])
        #
        otherani_otherlever_dist = scipy.ndimage.gaussian_filter1d(otherani_otherlever_dist,gausKernelsize)

        a = output_key_locations['tube_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name_other].transpose()
        a_min_b = a - b
        otherani_othertube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(otherani_othertube_dist)
        if np.any(~nans):
            otherani_othertube_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), otherani_othertube_dist[~nans])
        #
        otherani_othertube_dist = scipy.ndimage.gaussian_filter1d(otherani_othertube_dist,gausKernelsize)

        #
        # get the self social gaze time series
        # align to the start of the video recording
        # self social gaze
        if ianimal == 0:
            timepoint_gaze = animal1_gaze+session_start_time
        elif ianimal == 1:
            timepoint_gaze = animal2_gaze+session_start_time
        #
        try:
            timeseries_gaze = np.zeros(np.shape(gaze_angle_speed))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        except: # some videos are shorter than the task 
            timeseries_gaze = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_gaze*fps))))+1,))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        socialgaze_prob = timeseries_gaze
        # socialgaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,gausKernelsize)
        socialgaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,1)

        #
        # get the other social gaze time series
        # align to the start of the video recording
        # other social gaze
        if ianimal == 0:
            timepoint_gaze = animal2_gaze+session_start_time
        elif ianimal == 1:
            timepoint_gaze = animal1_gaze+session_start_time
        #
        try:
            timeseries_gaze = np.zeros(np.shape(gaze_angle_speed))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        except: # some videos are shorter than the task 
            timeseries_gaze = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_gaze*fps))))+1,))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        othergaze_prob = timeseries_gaze
        # othergaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,gausKernelsize)
        othergaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,1)

        #
        # get the other pull time series
        # align to the start of the video recording
        # other pull
        if ianimal == 0:
            timepoint_otherpull = time_point_pull2 + session_start_time
        elif ianimal == 1:
            timepoint_otherpull = time_point_pull1 + session_start_time
        #
        try:
            timeseries_otherpull = np.zeros(np.shape(gaze_angle_speed))
            timeseries_otherpull[list(map(int,list(np.round(timepoint_otherpull*fps))))]=1
        except: # some videos are shorter than the task 
            timeseries_otherpull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_otherpull*fps))))+1,))
            timeseries_otherpull[list(map(int,list(np.round(timepoint_otherpull*fps))))]=1
        otherpull_prob = scipy.ndimage.gaussian_filter1d(timeseries_otherpull,1)

        #
        # get the self pull time series for comparison
        # align to the start of the video recording
        # self pull
        if ianimal == 0:
            timepoint_selfpull = time_point_pull1 + session_start_time
        elif ianimal == 1:
            timepoint_selfpull = time_point_pull2 + session_start_time
        #
        try:
            timeseries_selfpull = np.zeros(np.shape(gaze_angle_speed))
            timeseries_selfpull[list(map(int,list(np.round(timepoint_selfpull*fps))))]=1
        except: # some videos are shorter than the task 
            timeseries_selfpull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_selfpull*fps))))+1,))
            timeseries_selfpull[list(map(int,list(np.round(timepoint_selfpull*fps))))]=1
        selfpull_prob = scipy.ndimage.gaussian_filter1d(timeseries_selfpull,1)

        #
        # get the self juice time series 
        # align to the start of the video recording
        # self juice
        if ianimal == 0:
            timepoint_selfjuice = time_point_juice1 + session_start_time
        elif ianimal == 1:
            timepoint_selfjuice = time_point_juice2 + session_start_time
        #
        try:
            timeseries_selfjuice = np.zeros(np.shape(gaze_angle_speed))
            timeseries_selfjuice[list(map(int,list(np.round(timepoint_selfjuice*fps))))]=1
        except: # some videos are shorter than the task 
            timeseries_selfjuice = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_selfjuice*fps))))+1,))
            timeseries_selfjuice[list(map(int,list(np.round(timepoint_selfjuice*fps))))]=1
        selfjuice_prob = scipy.ndimage.gaussian_filter1d(timeseries_selfjuice,1)


        # put all the data together in the same order as the con_vars_plot
        data_summary[animal_name_forsaving] = [gaze_other_angle, gaze_tube_angle, gaze_lever_angle, animal_animal_dist, animal_tube_dist, animal_lever_dist, othergaze_self_angle, mass_move_speed, other_mass_move_speed, gaze_angle_speed, otherani_otherlever_dist, otherani_othertube_dist, socialgaze_prob, othergaze_prob, otherpull_prob, selfpull_prob, selfjuice_prob]

        #
        # only plot the active meaning period
        for ivar in np.arange(0,nconvarplots,1):

            if ianimal == 0:
                timepoint_pull = time_point_pull1
            elif ianimal == 1:
                timepoint_pull = time_point_pull2
            #
            xxx_time = np.arange(0,np.shape(data_summary[animal_name_forsaving][ivar])[0],1)/fps
            # only plot the active meaning period
            xxx_time_range = [np.max([xxx_time[0],np.array(timepoint_pull)[0]+session_start_time-5]),
                              np.min([xxx_time[-1],np.array(timepoint_pull)[-1]+session_start_time+5])]
            # 
            if xxx_time_range[1]<np.floor(np.shape(gaze_angle_speed)[0]/fps):
                ind_time_range = (xxx_time >= xxx_time_range[0]) & (xxx_time <=xxx_time_range[1])   
            else:
                xxx_time_range[1] = np.floor(np.shape(gaze_angle_speed)[0]/fps)
                ind_time_range = (xxx_time >= xxx_time_range[0]) & (xxx_time <=xxx_time_range[1])   

            #
            data_summary[animal_name_forsaving][ivar] = data_summary[animal_name_forsaving][ivar][ind_time_range]

            data_summary[animal_name_forsaving][ivar] = (data_summary[animal_name_forsaving][ivar] - np.nanmean(data_summary[animal_name_forsaving][ivar])) / np.nanstd(data_summary[animal_name_forsaving][ivar])
        
        
        
    ###### 
    # # get the spike related data
    ######

    # align the spike time to the start of the video recording
    # only get the spike_time that is within the range of starttime(frame) and endtime(frame)
    spike_time_data_new = spike_time_data + session_start_time*fps

    spike_clusters_unique = np.unique(spike_clusters_data)
    nclusters = np.shape(spike_clusters_unique)[0]

    spiketrain_summary = dict.fromkeys(spike_clusters_unique,[])

    for icluster in np.arange(0,nclusters,1):
        iclusterID = spike_clusters_unique[icluster]

        ind_clusterID = np.isin(spike_clusters_data,iclusterID)

        spike_time_icluster = spike_time_data_new[ind_clusterID]
        spike_time_icluster = np.array(list(map(int,spike_time_icluster)))
        spiketimes, counts = np.unique(spike_time_icluster, return_counts=True)

        nspikes = np.shape(spiketimes)[0]
        spiketrain = np.zeros((1,int(np.max(spike_time_data_new)+1)))[0]

        for ispike in np.arange(0,nspikes,1):
            spiketrain[spiketimes[ispike]] = counts[ispike]

        if ianimal == 0:
            timepoint_pull = time_point_pull1
        elif ianimal == 1:
            timepoint_pull = time_point_pull2
        #
        xxx_time = np.arange(0,np.shape(spiketrain)[0],1)/fps
        # only plot the active meaning period
        xxx_time_range = [np.max([xxx_time[0],np.array(timepoint_pull)[0]+session_start_time-5]),
                          np.min([xxx_time[-1],np.array(timepoint_pull)[-1]+session_start_time+5])]
        # 
        if xxx_time_range[1]<np.floor(np.shape(gaze_angle_speed)[0]/fps):
            ind_time_range = (xxx_time >= xxx_time_range[0]) & (xxx_time <=xxx_time_range[1])   
        else:
            xxx_time_range[1] = np.floor(np.shape(gaze_angle_speed)[0]/fps)
            ind_time_range = (xxx_time >= xxx_time_range[0]) & (xxx_time <=xxx_time_range[1])    
        #
        spiketrain_summary[iclusterID] = spiketrain[ind_time_range]




    return data_summary, data_summary_names, spiketrain_summary








########################

def neuralGLM_fitting_BasisKernelsForContVaris(KERNEL_DURATION_S, KERNEL_OFFSET_S, N_BASIS_FUNCS, fps, animal1, animal2, recordedanimal, var_toglm_names,  data_summary_names, data_summary, spiketrain_summary, dospikehist, spikehist_twin, N_BOOTSTRAPS,test_size):

    
    dt = 1 / fps

    # basis_funcs, time_vector = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    basis_funcs, time_vector = make_gaussian_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    # basis_funcs, t_basis = make_square_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)


    ####
    # do the glm fitting
    ####

    #
    # This gives you indices for the variables in var_toglm_names (in that same order)
    indices_in_summary = [data_summary_names.index(var) for var in var_toglm_names if var in data_summary_names]

    data_summary = np.array(data_summary)  # if it’s still a list
    predictors = data_summary[indices_in_summary]

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
    for icluster in np.arange(0,nclusters,1):
    # for icluster in np.arange(0,1,1):
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

    return neuralGLM_kernels_coef, neuralGLM_kernels_tempFilter, neuralGLM_kernels_coef_shf, neuralGLM_kernels_tempFilter_shf, var_toglm_names





 














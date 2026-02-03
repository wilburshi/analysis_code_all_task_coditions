# helper functions for the glm fitting

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.signal import convolve

#
def make_raised_cosine_basis(duration_s, n_basis, dt):
    t = np.arange(0, duration_s, dt)
    c = np.linspace(0, duration_s, n_basis)
    width = (c[1] - c[0]) * 1.5

    basis = []
    for ci in c:
        phi = (t - ci) * np.pi / width
        b = np.cos(np.clip(phi, -np.pi, np.pi))
        b = (b + 1) / 2
        b[(t < ci - width/2) | (t > ci + width/2)] = 0  # apply cutoff mask
        basis.append(b)

    basis = np.stack(basis, axis=1)  # shape: [time, n_basis]
    return basis

#
def convolve_with_basis(var, basis_funcs):
    return np.stack([
        convolve(var, basis, mode='full')[:len(var)]
        for basis in basis_funcs.T
    ], axis=1)


# calculate the glm for the continuous variables and save the key output

def continuous_variable_create_data_forGLM(KERNEL_DURATION_S, N_BASIS_FUNCS, fps, animal1, animal2, session_start_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, animalnames_videotrack, output_look_ornot, output_allvectors, output_allangles, output_key_locations):



    gausKernelsize = 4 # 15

    gaze_thresold = 0.2 # min length threshold to define if a gaze is real gaze or noise, in the unit of second

    dt = 1 / fps
    basis_funcs = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)

    nanimals = np.shape(animalnames_videotrack)[0]

    ###### 
    # # prepare the glm input data
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


    con_vars_plot = ['gaze_other_angle','gaze_tube_angle','gaze_lever_angle','animal_animal_dist','animal_tube_dist','animal_lever_dist','othergaze_self_angle','mass_move_speed', 'other_mass_move_speed', 'gaze_angle_speed','otherani_otherlever_dist','otherani_othertube_dist','socialgaze_prob','othergaze_prob','otherpull_prob', 'selfpull_prob']
    nconvarplots = np.shape(con_vars_plot)[0]

    # initialize the summarizing data
    glm_fitting_summary = {}

    # 
    # run for each animal
    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]

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
        socialgaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,gausKernelsize)

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
        othergaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,gausKernelsize)

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


        # put all the data together in the same order as the con_vars_plot
        data_summary = [gaze_other_angle, gaze_tube_angle, gaze_lever_angle, animal_animal_dist, animal_tube_dist, animal_lever_dist, othergaze_self_angle, mass_move_speed, other_mass_move_speed, gaze_angle_speed, otherani_otherlever_dist, otherani_othertube_dist, socialgaze_prob, othergaze_prob, otherpull_prob, selfpull_prob]

        
        #
        # only plot the active meaning period
        if ianimal == 0:
            timepoint_pull = time_point_pull1
        elif ianimal == 1:
            timepoint_pull = time_point_pull2
        #
        xxx_time = np.arange(0,np.shape(data_summary[0])[0],1)/fps
        # only plot the active meaning period
        xxx_time_range = [np.max([xxx_time[0],np.array(timepoint_pull)[0]+session_start_time-5]),
                          np.min([xxx_time[-1],np.array(timepoint_pull)[-1]+session_start_time+5])]
        # xxx_time_range = [15+session_start_time, 45+session_start_time]
        ind_time_range = (xxx_time >= xxx_time_range[0]) & (xxx_time <=xxx_time_range[1])
        

        ####
        # do the glm fitting
        ####

        # Variables
        predictors = [
            gaze_other_angle[ind_time_range],
            gaze_tube_angle[ind_time_range],
            gaze_lever_angle[ind_time_range],
            animal_animal_dist[ind_time_range],
            animal_tube_dist[ind_time_range],
            animal_lever_dist[ind_time_range],
            mass_move_speed[ind_time_range],
            gaze_angle_speed[ind_time_range],
        ]
        var_names = ['gaze_other_angle', 'gaze_tube_angle', 'gaze_lever_angle',
                     'animal_animal_dist', 'animal_tube_dist', 'animal_lever_dist',
                     'mass_move_speed', 'gaze_angle_speed']

        # Design matrix from continuous variables
        X_continuous = np.hstack([convolve_with_basis(v, basis_funcs) for v in predictors])

        # prepare the Y
        Y = np.zeros(len(gaze_angle_speed))
        pull_idx = np.round(timepoint_selfpull * fps).astype(int)
        pull_idx = pull_idx[pull_idx < len(Y)]  # exclude out-of-bounds
        Y[pull_idx] = 1
        Y = Y[ind_time_range]


        # save data into the summary data set
        if ianimal == 0:
            glm_fitting_summary[(animal1,'var_names')] = var_names
            glm_fitting_summary[(animal1,'X_all')] = X_continuous
            glm_fitting_summary[(animal1,'Y')] = Y
        elif ianimal == 1:
            glm_fitting_summary[(animal2,'var_names')] = var_names
            glm_fitting_summary[(animal2,'X_all')] = X_continuous
            glm_fitting_summary[(animal2,'Y')] = Y

    return glm_fitting_summary


    

    

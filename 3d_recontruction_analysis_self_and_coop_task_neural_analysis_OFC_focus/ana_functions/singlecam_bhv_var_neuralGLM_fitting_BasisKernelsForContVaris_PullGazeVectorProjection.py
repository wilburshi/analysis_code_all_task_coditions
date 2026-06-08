#  function - get singlecam variables and neurons and then do the GLM fitting
# # project the 8D continuous variables to smaller dimension that are more task relavant

# helper functions for the glm fitting

import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.signal import convolve
import string
import warnings
import pickle    
import random as random
import statsmodels.api as sm

from sklearn.decomposition import PCA


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from scipy.linalg import orth

from scipy.special import gammaln


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




########################

# project the 8D continuous variables to smaller dimension that are more task relavant

########################

def neuralGLM_fitting_BasisKernelsForContVaris_PullGazeVectorProjection(KERNEL_DURATION_S, KERNEL_OFFSET_S, N_BASIS_FUNCS, fps, animal1, animal2, recordedanimal, var_toglm_names,  data_summary_names, data_summary, spiketrain_summary, dospikehist, spikehist_twin, N_BOOTSTRAPS,test_size):

    ####
    # prepare the projected behavioral data
    ####
    
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
    
    
    ####
    # do the glm fitting
    ####
    
    dt = 1 / fps

    # basis_funcs, time_vector = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    basis_funcs, time_vector = make_gaussian_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    # basis_funcs, t_basis = make_square_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)
    
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
    for icluster in np.arange(0,nclusters,1):
    # for icluster in np.arange(0,1,1):
        iclusterID = neuron_clusters[icluster]

        # Binary spike train
        # Y = (spiketrain_summary[iclusterID] > 0).astype(int)
        Y = spiketrain_summary[iclusterID]
        
        
        # make sure X_continous_z and Y has the same shape
        # Get the number of time points (samples) from the first dimension
        n_time_points_X = X_continuous_z.shape[0]
        n_time_points_Y = Y.shape[0]

        # Find the minimum of the two lengths
        min_len = min(n_time_points_X, n_time_points_Y)

        # Truncate both arrays to the minimum length
        Y = Y[:min_len]
        X_continuous_z = X_continuous_z[:min_len, :]
        
        
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
                X_continuous_z, Y, test_size=0.2, random_state=random.randint(0, 10000)
                )

            #
            finite_mask = np.isfinite(X_tr).all(axis=1)
            X_tr = X_tr[finite_mask]
            y_tr = y_tr[finite_mask]
            #
            
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
                X_continuous_z, Y_shuffled, test_size=0.2, random_state=random.randint(0, 10000)
            )

            #
            finite_mask = np.isfinite(X_tr).all(axis=1)
            X_tr = X_tr[finite_mask]
            y_tr = y_tr[finite_mask]
            #
            
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





########################

# project the 8D continuous variables to smaller dimension that are more task relavant; also consider partner's action's pc1

########################

def neuralGLM_fitting_BasisKernelsForContVaris_PullGazeVectorProjection_partnerPC1(KERNEL_DURATION_S, KERNEL_OFFSET_S, N_BASIS_FUNCS, fps, animal1, animal2, recordedanimal, var_toglm_names,  data_summary_names, data_summary_twoanimals, spiketrain_summary, dospikehist, spikehist_twin, N_BOOTSTRAPS,test_size):

    ####
    # prepare the projected behavioral data
    ####
    
    
    # 1. Load and extract behavioral data
    var_toPCA_names = ['gaze_other_angle', 'gaze_lever_angle', 'gaze_tube_angle',
                       'animal_animal_dist', 'animal_lever_dist', 'animal_tube_dist',
                       'mass_move_speed', 'gaze_angle_speed']
    pull_axis_name = ['selfpull_prob']
    gaze_axis_name = ['socialgaze_prob']
    juice_axis_name = ['selfjuice_prob']

    # for self animal projection
    data_summary = data_summary_twoanimals[recordedanimal]
    
    # for partner animal pca
    if animal1 == recordedanimal:
        data_summary_partner = data_summary_twoanimals[animal2]
    elif animal2 == recordedanimal:
        data_summary_partner = data_summary_twoanimals[animal1]
        
    # for self animal projection
    
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
    
    # partner animal PCA
    
    # Data extraction
    data_summary_partner = np.array(data_summary_partner)
    vars_toPCA = data_summary_partner[PCAindices_in_summary]        # shape (8, T)
    
    data_for_pca = vars_toPCA.T
    # Normalize the data (Z-score scaling)
    # Each of the 8 variables will now have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_pca)
    # Initialize and run PCA
    # We want to reduce the 8 variables to 3 principal components
    pca = PCA(n_components=3)
    # Fit the model and transform the data
    principal_components = pca.fit_transform(data_scaled)
    principal_components_transposed = principal_components.T
    # Get the explained variance for each component
    explained_variance = pca.explained_variance_ratio_
    #
    partner_PC1 = principal_components_transposed[0,:]
    
    
    ####
    # do the glm fitting
    ####
    
    dt = 1 / fps

    # basis_funcs, time_vector = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    basis_funcs, time_vector = make_gaussian_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    # basis_funcs, t_basis = make_square_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)
    
    var_toglm_names = ['pull_PC', 'gaze_PC', 'juice_PC', 'partner_PC1']

    # projected pull, gaze, juice action vector
    # List of arrays to be stacked
    arrays_to_stack = [pull_PC, gaze_PC, juice_PC, partner_PC1]
    # Find the minimum length among all arrays
    min_len = min(len(arr) for arr in arrays_to_stack)
    # Truncate all arrays to the minimum length to ensure they match
    truncated_arrays = [arr[:min_len] for arr in arrays_to_stack]
    # Now, stack the perfectly aligned arrays
    predictors = np.vstack(truncated_arrays)

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
        
        # make sure X_continous_z and Y has the same shape
        # Get the number of time points (samples) from the first dimension
        n_time_points_X = X_continuous_z.shape[0]
        n_time_points_Y = Y.shape[0]

        # Find the minimum of the two lengths
        min_len = min(n_time_points_X, n_time_points_Y)

        # Truncate both arrays to the minimum length
        Y = Y[:min_len]
        X_continuous_z = X_continuous_z[:min_len, :]
        
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
                X_continuous_z, Y, test_size=0.2, random_state=random.randint(0, 10000)
                )

            #
            finite_mask = np.isfinite(X_tr).all(axis=1)
            X_tr = X_tr[finite_mask]
            y_tr = y_tr[finite_mask]
            #
            
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
                X_continuous_z, Y_shuffled, test_size=0.2, random_state=random.randint(0, 10000)
            )
            
            #
            finite_mask = np.isfinite(X_tr).all(axis=1)
            X_tr = X_tr[finite_mask]
            y_tr = y_tr[finite_mask]
            #

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




########################

# directly use the pull, gaze and juice as the regressor; also consider partner's action's pc1

########################

def neuralGLM_fitting_BasisKernelsForContVaris_PullGazeAxis_partnerPC1(KERNEL_DURATION_S, KERNEL_OFFSET_S, N_BASIS_FUNCS, fps, animal1, animal2, recordedanimal, var_toglm_names,  data_summary_names, data_summary_twoanimals, spiketrain_summary, dospikehist, spikehist_twin, N_BOOTSTRAPS,test_size):

    ####
    # prepare the projected behavioral data
    ####
    
    
    # 1. Load and extract behavioral data
    var_toPCA_names = ['gaze_other_angle', 'gaze_lever_angle', 'gaze_tube_angle',
                       'animal_animal_dist', 'animal_lever_dist', 'animal_tube_dist',
                       'mass_move_speed', 'gaze_angle_speed']
    pull_axis_name = ['selfpull_prob']
    gaze_axis_name = ['socialgaze_prob']
    juice_axis_name = ['selfjuice_prob']

    # for self animal projection
    data_summary = data_summary_twoanimals[recordedanimal]
    
    # for partner animal pca
    if animal1 == recordedanimal:
        data_summary_partner = data_summary_twoanimals[animal2]
    elif animal2 == recordedanimal:
        data_summary_partner = data_summary_twoanimals[animal1]
        
    # for self animal projection
    
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

    # 2. smooth the gaze, pull, jucie axis more
    var_pull = scipy.ndimage.gaussian_filter1d(var_pull,4)
    var_gaze = scipy.ndimage.gaussian_filter1d(var_gaze,4)
    var_juice = scipy.ndimage.gaussian_filter1d(var_juice,4)
    
    
    # partner animal PCA
    
    # Data extraction
    data_summary_partner = np.array(data_summary_partner)
    vars_toPCA = data_summary_partner[PCAindices_in_summary]        # shape (8, T)
    
    data_for_pca = vars_toPCA.T
    # Normalize the data (Z-score scaling)
    # Each of the 8 variables will now have a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_pca)
    # Initialize and run PCA
    # We want to reduce the 8 variables to 3 principal components
    pca = PCA(n_components=3)
    # Fit the model and transform the data
    principal_components = pca.fit_transform(data_scaled)
    principal_components_transposed = principal_components.T
    # Get the explained variance for each component
    explained_variance = pca.explained_variance_ratio_
    #
    partner_PC1 = principal_components_transposed[0,:]
    
    
    ####
    # do the glm fitting
    ####
    
    dt = 1 / fps

    # basis_funcs, time_vector = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    basis_funcs, time_vector = make_gaussian_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    # basis_funcs, t_basis = make_square_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)
    
    var_toglm_names = ['var_pull', 'var_gaze', 'var_juice', 'partner_PC1']

    # projected pull, gaze, juice action vector
    # List of arrays to be stacked
    arrays_to_stack = [var_pull, var_gaze, var_juice, partner_PC1]
    # Find the minimum length among all arrays
    min_len = min(len(arr) for arr in arrays_to_stack)
    # Truncate all arrays to the minimum length to ensure they match
    truncated_arrays = [arr[:min_len] for arr in arrays_to_stack]
    # Now, stack the perfectly aligned arrays
    predictors = np.vstack(truncated_arrays)


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
        
        # make sure X_continous_z and Y has the same shape
        # Get the number of time points (samples) from the first dimension
        n_time_points_X = X_continuous_z.shape[0]
        n_time_points_Y = Y.shape[0]

        # Find the minimum of the two lengths
        min_len = min(n_time_points_X, n_time_points_Y)

        # Truncate both arrays to the minimum length
        Y = Y[:min_len]
        X_continuous_z = X_continuous_z[:min_len, :]
        
        
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
                X_continuous_z, Y, test_size=0.2, random_state=random.randint(0, 10000)
                )

            #
            finite_mask = np.isfinite(X_tr).all(axis=1)
            X_tr = X_tr[finite_mask]
            y_tr = y_tr[finite_mask]
            #
            
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
                X_continuous_z, Y_shuffled, test_size=0.2, random_state=random.randint(0, 10000)
            )
            
            #
            finite_mask = np.isfinite(X_tr).all(axis=1)
            X_tr = X_tr[finite_mask]
            y_tr = y_tr[finite_mask]
            #

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





########################

# directly use the pull, gaze and juice as the regressor; also consider partner's actions: pull gaze juice(?)

########################

def neuralGLM_fitting_BasisKernelsForContVaris_PullGazeAxis_partnerPullGazeAxis(KERNEL_DURATION_S, KERNEL_OFFSET_S, N_BASIS_FUNCS, fps, animal1, animal2, recordedanimal, var_toglm_names,  data_summary_names, data_summary_twoanimals, spiketrain_summary, dospikehist, spikehist_twin, N_BOOTSTRAPS,test_size):

    ####
    # prepare the projected behavioral data
    ####
    
    
    # 1. Load and extract behavioral data
    var_toPCA_names = ['gaze_other_angle', 'gaze_lever_angle', 'gaze_tube_angle',
                       'animal_animal_dist', 'animal_lever_dist', 'animal_tube_dist',
                       'mass_move_speed', 'gaze_angle_speed']
    pull_axis_name = ['selfpull_prob']
    gaze_axis_name = ['socialgaze_prob']
    juice_axis_name = ['selfjuice_prob']

    # for self animal projection
    data_summary = data_summary_twoanimals[recordedanimal]
    
    # for partner animal
    if animal1 == recordedanimal:
        data_summary_partner = data_summary_twoanimals[animal2]
    elif animal2 == recordedanimal:
        data_summary_partner = data_summary_twoanimals[animal1]
        
    # for self animal projection
    
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

    # 2. smooth the gaze, pull, jucie axis more
    var_pull = scipy.ndimage.gaussian_filter1d(var_pull,4)
    var_gaze = scipy.ndimage.gaussian_filter1d(var_gaze,4)
    var_juice = scipy.ndimage.gaussian_filter1d(var_juice,4)
    
    
    # partner animal PCA
    
    # Data extraction
    data_summary_partner = np.array(data_summary_partner)
    vars_toPCA_partner = data_summary_partner[PCAindices_in_summary]        # shape (8, T)
    var_pull_partner = data_summary_partner[Pullindices_in_summary][0]      # shape (T,)
    var_gaze_partner = data_summary_partner[Gazeindices_in_summary][0]
    var_juice_partner = data_summary_partner[Juiceindices_in_summary][0]

    # 2. smooth the gaze, pull, jucie axis more
    var_pull_partner = scipy.ndimage.gaussian_filter1d(var_pull_partner,4)
    var_gaze_partner = scipy.ndimage.gaussian_filter1d(var_gaze_partner,4)
    var_juice_partner = scipy.ndimage.gaussian_filter1d(var_juice_partner,4)
    
    
    
    ####
    # do the glm fitting
    ####
    
    dt = 1 / fps

    # basis_funcs, time_vector = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    basis_funcs, time_vector = make_gaussian_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)
    # basis_funcs, t_basis = make_square_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)
    
    var_toglm_names = ['var_pull', 'var_gaze', 'var_juice', 'var_pull_partner', 'var_gaze_partner', 'var_juice_partner',]

    # projected pull, gaze, juice action vector
    # List of arrays to be stacked
    arrays_to_stack = [var_pull, var_gaze, var_juice, var_pull_partner, var_gaze_partner, var_juice_partner]
    # Find the minimum length among all arrays
    min_len = min(len(arr) for arr in arrays_to_stack)
    # Truncate all arrays to the minimum length to ensure they match
    truncated_arrays = [arr[:min_len] for arr in arrays_to_stack]
    # Now, stack the perfectly aligned arrays
    predictors = np.vstack(truncated_arrays)


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
        
        # make sure X_continous_z and Y has the same shape
        # Get the number of time points (samples) from the first dimension
        n_time_points_X = X_continuous_z.shape[0]
        n_time_points_Y = Y.shape[0]

        # Find the minimum of the two lengths
        min_len = min(n_time_points_X, n_time_points_Y)

        # Truncate both arrays to the minimum length
        Y = Y[:min_len]
        X_continuous_z = X_continuous_z[:min_len, :]
        
        
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
                X_continuous_z, Y, test_size=0.2, random_state=random.randint(0, 10000)
                )

            #
            finite_mask = np.isfinite(X_tr).all(axis=1)
            X_tr = X_tr[finite_mask]
            y_tr = y_tr[finite_mask]
            #
            
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
                X_continuous_z, Y_shuffled, test_size=0.2, random_state=random.randint(0, 10000)
            )
            
            #
            finite_mask = np.isfinite(X_tr).all(axis=1)
            X_tr = X_tr[finite_mask]
            y_tr = y_tr[finite_mask]
            #

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





#########################

# identify the tuning property with the leave one out method

# directly use the pull, gaze and juice as the regressor; also consider partner's action's pc1

########################

# helper function
#
def likelihood_ratio_test(ll_full, ll_reduced, df_full, df_reduced):
    """
    Performs a likelihood ratio test to compare two nested models.

    Args:
        ll_full (float): Log-likelihood of the full model.
        ll_reduced (float): Log-likelihood of the reduced model.
        df_full (int): Degrees of freedom (number of parameters) of the full model.
        df_reduced (int): Degrees of freedom of the reduced model.

    Returns:
        tuple: A tuple containing the test statistic (D) and the p-value.
    """
    # Calculate the test statistic, D
    D = 2 * (ll_full - ll_reduced)
    # Calculate the difference in degrees of freedom
    df_diff = df_full - df_reduced
    # Calculate the p-value from the chi-squared distribution
    p_value = chi2.sf(D, df_diff) if df_diff > 0 else np.nan
    return D, p_value


def loglike_poisson(y_true, y_pred):
    """
    Calculates the log-likelihood of a Poisson model.

    Args:
        y_true (array): The observed data (e.g., spike counts).
        y_pred (array): The predicted rate (lambda) from the model.

    Returns:
        float: The total log-likelihood.
    """
    # Add a small epsilon to y_pred to avoid log(0) errors
    epsilon = 1e-9
    
    # Poisson log-likelihood formula: sum(y*log(lambda) - lambda - log(y!))
    # We use gammaln(y + 1) as a numerically stable way to compute log(y!)
    ll = np.sum(y_true * np.log(y_pred + epsilon) - y_pred - gammaln(y_true + 1))
    
    return ll


def neuralGLM_fitting_BasisKernelsForContVaris_PullGazeAxis_partnerPC1_LOOmethods(KERNEL_DURATION_S, KERNEL_OFFSET_S, N_BASIS_FUNCS, fps, animal1, animal2, recordedanimal, var_toglm_names,  data_summary_names, data_summary_twoanimals, spiketrain_summary, dospikehist, spikehist_twin, N_BOOTSTRAPS,test_size):

    ####
    # prepare the projected behavioral data
    ####
    
    
    ####
    # 1. Prepare Behavioral Data (largely unchanged)
    ####
    var_to_PCA_names = ['gaze_other_angle', 'gaze_lever_angle', 'gaze_tube_angle',
                      'animal_animal_dist', 'animal_lever_dist', 'animal_tube_dist',
                      'mass_move_speed', 'gaze_angle_speed']
    pull_axis_name = ['selfpull_prob']
    gaze_axis_name = ['socialgaze_prob']
    juice_axis_name = ['selfjuice_prob']

    # Data for the recorded animal
    data_summary = data_summary_twoanimals[recordedanimal]
    PCA_indices = [data_summary_names.index(var) for var in var_to_PCA_names]
    var_pull = np.array(data_summary)[data_summary_names.index(pull_axis_name[0])]
    var_gaze = np.array(data_summary)[data_summary_names.index(gaze_axis_name[0])]
    var_juice = np.array(data_summary)[data_summary_names.index(juice_axis_name[0])]
    var_pull = scipy.ndimage.gaussian_filter1d(var_pull, 4)
    var_gaze = scipy.ndimage.gaussian_filter1d(var_gaze, 4)
    var_juice = scipy.ndimage.gaussian_filter1d(var_juice, 4)

    # Data for the partner animal
    partner = animal2 if recordedanimal == animal1 else animal1
    data_summary_partner = data_summary_twoanimals[partner]
    vars_to_PCA_partner = np.array(data_summary_partner)[PCA_indices]
    scaler_partner = StandardScaler()
    data_scaled_partner = scaler_partner.fit_transform(vars_to_PCA_partner.T)
    pca = PCA(n_components=1)
    partner_PC1 = pca.fit_transform(data_scaled_partner).flatten()

    # Align and stack all behavioral predictor time-series
    original_var_names = ['var_pull', 'var_gaze', 'var_juice', 'partner_PC1']
    arrays_to_stack = [var_pull, var_gaze, var_juice, partner_PC1]
    min_len_behav = min(len(arr) for arr in arrays_to_stack)
    truncated_arrays = [arr[:min_len_behav] for arr in arrays_to_stack]
    predictors = np.vstack(truncated_arrays)

    ####
    # 2. Build the Split (Past/Future) Design Matrix
    ####
    dt = 1 / fps
    basis_funcs, time_vector = make_gaussian_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt, offset_s=KERNEL_OFFSET_S)

    kernel_centers_idx = np.argmax(basis_funcs, axis=0)
    kernel_centers_time = time_vector[kernel_centers_idx]
    
    past_kernels_mask = kernel_centers_time < 0
    future_kernels_mask = kernel_centers_time >= 0
    
    basis_past = basis_funcs[:, past_kernels_mask]
    basis_future = basis_funcs[:, future_kernels_mask]
    
    var_toglm_names_split = []
    X_blocks = []
    block_sizes = [] # To keep track of the number of columns for each split-variable

    for i, name in enumerate(original_var_names):
        # Add "past" component
        var_toglm_names_split.append(f"{name}_past")
        X_past_block = convolve_with_basis(predictors[i, :], basis_past)
        X_blocks.append(X_past_block)
        block_sizes.append(X_past_block.shape[1])
        
        # Add "future" component
        var_toglm_names_split.append(f"{name}_future")
        X_future_block = convolve_with_basis(predictors[i, :], basis_future)
        X_blocks.append(X_future_block)
        block_sizes.append(X_future_block.shape[1])

    X_continuous = np.hstack(X_blocks)
    scaler = StandardScaler()
    X_continuous_z = scaler.fit_transform(X_continuous)

    # Pre-calculate the column indices for each of the 8 blocks
    block_indices = np.cumsum([0] + block_sizes)

    ###
    # 3. Run Leave-One-Out Model Comparison for Each Neuron
    ####
    neuron_clusters = list(spiketrain_summary.keys())
    model_p_values = pd.DataFrame(index=neuron_clusters, columns=var_toglm_names_split, dtype=float)
    full_model_filters = {}

    for iclusterID in neuron_clusters:
        Y = spiketrain_summary[iclusterID]

        # Align neural and behavioral data
        min_len = min(X_continuous_z.shape[0], Y.shape[0])
        Y = Y[:min_len]
        X = X_continuous_z[:min_len, :]

        # Fit the FULL model (with all 8 predictor blocks)
        clf_full = PoissonRegressor(alpha=0, max_iter=1000, tol=1e-6)
        clf_full.fit(X, Y)
        y_pred_full = clf_full.predict(X)
        ll_full = loglike_poisson(Y, y_pred_full)
        df_full = X.shape[1] + 1

        # --- START: Finalized Kernel Reconstruction ---

        # Store the kernels from the full model
        full_beta = clf_full.coef_.flatten()

        # Create a dictionary to hold all 8 reconstructed filters for the current neuron
        reconstructed_filters = {}

        # Loop through all 8 of the split variable names (e.g., 'var_pull_past', 'var_pull_future', etc.)
        for i, var_name in enumerate(var_toglm_names_split):

            # Determine which basis set to use (past or future) based on the variable name
            if "_past" in var_name:
                basis_matrix_to_use = basis_past
            else: # The name contains "_future"
                basis_matrix_to_use = basis_future

            # Get the start and end column indices for this specific variable's coefficients
            start_col, end_col = block_indices[i], block_indices[i+1]

            # Slice full_beta to get the block of coefficients for this variable
            beta_block = full_beta[start_col:end_col]

            # Reconstruct the temporal filter by dotting the coefficients with the basis functions
            temporal_filter = np.dot(beta_block, basis_matrix_to_use.T)

            # Store the reconstructed filter in the dictionary with its full name as the key
            reconstructed_filters[var_name] = temporal_filter

        # Assign the complete dictionary of 8 filters to the main storage dictionary for this neuron
        full_model_filters[iclusterID] = reconstructed_filters

        # --- END: Finalized Kernel Reconstruction ---

        # Fit REDUCED models, leaving one block out at a time
        for i, var_name in enumerate(var_toglm_names_split):
            start_col, end_col = block_indices[i], block_indices[i+1]
            X_reduced = np.delete(X, np.s_[start_col:end_col], axis=1)

            clf_reduced = PoissonRegressor(alpha=0, max_iter=1000, tol=1e-6)
            clf_reduced.fit(X_reduced, Y)
            y_pred_reduced = clf_reduced.predict(X_reduced)
            ll_reduced = loglike_poisson(Y, y_pred_reduced)
            df_reduced = X_reduced.shape[1] + 1

            # Perform the Likelihood Ratio Test
            _, p_value = likelihood_ratio_test(ll_full, ll_reduced, df_full, df_reduced)
            model_p_values.loc[iclusterID, var_name] = p_value
            
    return model_p_values, full_model_filters, var_toglm_names_split




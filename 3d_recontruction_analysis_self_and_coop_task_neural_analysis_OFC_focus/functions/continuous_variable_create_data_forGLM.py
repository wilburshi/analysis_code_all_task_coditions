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

def continuous_variable_create_data_forGLM(KERNEL_DURATION_S, N_BASIS_FUNCS, fps, animal1, animal2, session_start_time,time_point_pull1, time_point_pull2, time_point_pulls_succfail, data_summary_twoanimals, data_summary_names, glm_tgt_variables, addpullinfo):

    nanimals = 2
    dt = 1 / fps
    basis_funcs = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)

    glm_fitting_summary = {}

    for ianimal in np.arange(0, nanimals, 1):

        if ianimal == 0:
            timepoint_selfpull = time_point_pull1 + session_start_time
            timepoint_pull = time_point_pull1
            data_summary = data_summary_twoanimals[animal1]
            succ_pulls = time_point_pulls_succfail["pull1_succ"] + session_start_time
            fail_pulls = time_point_pulls_succfail["pull1_fail"] + session_start_time
        elif ianimal == 1:
            timepoint_selfpull = time_point_pull2 + session_start_time
            timepoint_pull = time_point_pull2
            data_summary = data_summary_twoanimals[animal2]
            succ_pulls = time_point_pulls_succfail["pull2_succ"] + session_start_time
            fail_pulls = time_point_pulls_succfail["pull2_fail"] + session_start_time
            
        xxx_time = np.arange(0, np.shape(data_summary[0])[0], 1) / fps
        
        xxx_time_range = [np.max([xxx_time[0], np.array(timepoint_pull)[0] + session_start_time - 5]),
                          np.min([xxx_time[-1], np.array(timepoint_pull)[-1] + session_start_time + 5])]
        
        # only consider the active time and aligned at the session start time
        # ind_time_range = (xxx_time >= xxx_time_range[0]) & (xxx_time <= xxx_time_range[1])
        # only consider the active time but keep the session start time, and aligned at the session start time
        ind_time_range = (xxx_time >= 0) & (xxx_time <= xxx_time_range[1])

        # 1. Prepare Convolved Variables
        indices = [data_summary_names.index(var) for var in glm_tgt_variables]
        predictors = [data_summary[i] for i in indices]
        X_continuous = np.hstack([convolve_with_basis(v, basis_funcs) for v in predictors])

        # 2. Prepare Y
        Y = np.zeros(np.shape(data_summary[0])[0])
        pull_idx = np.round(timepoint_selfpull * fps).astype(int)
        pull_idx = pull_idx[pull_idx < len(Y)]  
        Y[pull_idx] = 1
        
        # Apply the time range mask first
        X_continuous = X_continuous[ind_time_range]
        Y = Y[ind_time_range]
        time_vector = xxx_time[ind_time_range] # Time vector for the raw variables

        # ==============================================================
        # NEW: Calculate and Append Raw Pull History Variables
        # ==============================================================
        raw_var_names = []
        if addpullinfo == 1:
            
            succ_pulls = np.array(succ_pulls)
            fail_pulls = np.array(fail_pulls)
            timepoint_pull = np.array(timepoint_pull)
            timepoint_selfpull = np.array(timepoint_selfpull)
            
            
            all_pulls = np.sort(np.concatenate((succ_pulls, fail_pulls)))

            # Variable 1: Absolute time
            abs_time = time_vector.copy()

            # Variable 2: Time since previous pull
            time_since_pull = np.zeros_like(time_vector)
            idx_all = np.searchsorted(all_pulls, time_vector, side='right') - 1
            valid_all = idx_all >= 0
            time_since_pull[valid_all] = time_vector[valid_all] - all_pulls[idx_all[valid_all]]

            # Variable 3: Time since previous successful pull
            time_since_succ = np.zeros_like(time_vector)
            idx_succ = np.searchsorted(succ_pulls, time_vector, side='right') - 1
            valid_succ = idx_succ >= 0
            time_since_succ[valid_succ] = time_vector[valid_succ] - succ_pulls[idx_succ[valid_succ]]

            # Variable 4: Number of consecutive failed pulls since last success
            consec_fails = np.zeros_like(time_vector)
            fails_before_current = np.searchsorted(fail_pulls, time_vector, side='right')
            
            last_succ_time = np.zeros_like(time_vector)
            last_succ_time[valid_succ] = succ_pulls[idx_succ[valid_succ]]
            fails_before_last_succ = np.searchsorted(fail_pulls, last_succ_time, side='right')
            
            consec_fails[valid_succ] = fails_before_current[valid_succ] - fails_before_last_succ[valid_succ]
            consec_fails[~valid_succ] = fails_before_current[~valid_succ]

            # Stack the raw variables horizontally
            X_raw = np.column_stack((abs_time, time_since_pull, time_since_succ, consec_fails))
            
            # Combine convolved and raw design matrices
            X_continuous = np.hstack((X_continuous, X_raw))
            raw_var_names = ['abs_time', 'time_since_pull', 'time_since_succ', 'consec_fails']
        # ==============================================================

        # Filter out any rows that contain NaNs from both X and Y
        valid_mask = ~np.isnan(X_continuous).any(axis=1)
        X_continuous = X_continuous[valid_mask]
        Y = Y[valid_mask]

        # save data into the summary data set
        animal_key = animal1 if ianimal == 0 else animal2
        glm_fitting_summary[(animal_key, 'convolved_var_names')] = glm_tgt_variables
        glm_fitting_summary[(animal_key, 'raw_var_names')] = raw_var_names # Save separately
        glm_fitting_summary[(animal_key, 'X_all')] = X_continuous
        glm_fitting_summary[(animal_key, 'Y')] = Y

    return glm_fitting_summary


# run the glm

import statsmodels.api as sm
from scipy.special import expit # This is the sigmoid/logistic function

def fit_glm_and_predict(glm_fitting_summary, animal_name):
    """
    Fits a Binomial GLM and returns the continuous predicted probability.
    """
    # 1. Extract the prepared data
    X = glm_fitting_summary[(animal_name, 'X_all')]
    Y = glm_fitting_summary[(animal_name, 'Y')]
    
    # 2. Add an intercept (constant) to the design matrix
    X_with_intercept = sm.add_constant(X)
    
    # 3. Fit the Logistic Regression (Binomial family, Logit link)
    # Using 'fit_regularized' or setting 'disp=0' can help if the model struggles to converge
    glm_model = sm.GLM(Y, X_with_intercept, family=sm.families.Binomial())
    results = glm_model.fit()
    
    # 4. Generate the continuous predicted likelihood (probability)
    # The .predict() function automatically applies the dot product and inverse link function
    predicted_likelihood = results.predict(X_with_intercept)
    
    # Alternatively, you can do it manually like this:
    # linear_predictor = np.dot(X_with_intercept, results.params)
    # predicted_likelihood = expit(linear_predictor) 
    
    # Save the results back to your summary dictionary
    glm_fitting_summary[(animal_name, 'predicted_likelihood')] = predicted_likelihood
    glm_fitting_summary[(animal_name, 'model_results')] = results
    
    return glm_fitting_summary
    
    
# check the glm
import matplotlib.pyplot as plt

def plot_pull_likelihood(glm_fitting_summary, animal_name, fps):
    
    # Extract the data
    likelihood = glm_fitting_summary[(animal_name, 'predicted_likelihood')]
    Y_actual = glm_fitting_summary[(animal_name, 'Y')]
    
    # Create a time vector based on the frames per second
    time_vector = np.arange(len(likelihood)) / fps
    
    plt.figure(figsize=(15, 4))
    
    # Plot the predicted probability
    plt.plot(time_vector, likelihood, label='Predicted Likelihood', color='blue', linewidth=1.5)
    
    # Plot the actual pulls as vertical dashed lines (or dots)
    pull_times = time_vector[Y_actual == 1]
    for pt in pull_times:
        plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
        
    # Just to add a legend label for the pulls without creating 100 legend entries
    plt.plot([], [], color='red', linestyle='--', label='Actual Pull Event')
    
    plt.title(f"Continuous Pull Likelihood for {animal_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("P(Pull)")
    plt.ylim([0, 0.125])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    
def plot_glm_temporal_filters(glm_fitting_summary, animal_name, fps, KERNEL_DURATION_S, N_BASIS_FUNCS):
    
    dt = 1 / fps
    basis_funcs = make_raised_cosine_basis(KERNEL_DURATION_S, N_BASIS_FUNCS, dt)
    
    results = glm_fitting_summary[(animal_name, 'model_results')]
    
    # Only grab the names of the variables that were convolved
    var_names = glm_fitting_summary[(animal_name, 'convolved_var_names')]
    
    # 1. Calculate how many weights belong to the convolved variables
    n_basis = basis_funcs.shape[1]
    num_convolved_weights = len(var_names) * n_basis
    
    # 2. Extract ONLY those specific weights (skipping the intercept at [0])
    # The raw variables are at the end, so we ignore them by slicing up to num_convolved_weights + 1
    weights = np.array(results.params[1 : num_convolved_weights + 1])
    
    # Reshape will now work perfectly
    weights_reshaped = weights.reshape(len(var_names), n_basis)
    
    time_axis = np.linspace(-KERNEL_DURATION_S, 0, basis_funcs.shape[0])
    fig, axes = plt.subplots(1, len(var_names), figsize=(4 * len(var_names), 4), sharey=True)
    if len(var_names) == 1: axes = [axes]
        
    for i, var in enumerate(var_names):
        temporal_filter = np.dot(basis_funcs, weights_reshaped[i])
        
        axes[i].plot(time_axis, temporal_filter, color='purple', linewidth=2.5)
        axes[i].fill_between(time_axis, temporal_filter, 0, color='purple', alpha=0.2)
        axes[i].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[i].axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
        axes[i].set_title(f"Impact of {var}")
        axes[i].set_xlabel("Time relative to pull (s)")
        if i == 0: axes[i].set_ylabel("GLM Weight")

    plt.suptitle(f"Temporal Filters for {animal_name.capitalize()}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
    
    # Optional: Print the single weights for the raw variables
    raw_vars = glm_fitting_summary[(animal_name, 'raw_var_names')]
    if len(raw_vars) > 0:
        raw_weights = results.params[num_convolved_weights + 1 :]
        print("\n--- Raw Trial-History Weights ---")
        for name, weight in zip(raw_vars, raw_weights):
            print(f"{name}: {weight:.4f}")


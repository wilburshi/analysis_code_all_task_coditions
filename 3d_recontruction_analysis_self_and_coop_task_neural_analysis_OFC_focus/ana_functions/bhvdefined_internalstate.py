# # define the function to use neural PC1,2,3 to decode states and relate them to the behavioral measures
import seaborn as sns
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from statsmodels.stats.outliers_influence import variance_inflation_factor
    

def bhvdefined_internalstate(glm_fitting_summary, data_summary_twoanimals, data_summary_names, 
                            animal1, animal2, recordedanimal, session_start_time,  N_BASIS_FUNCS, fps, addpullinfo, 
                            FRPCs_zscore_allch, FR_timepoint_allch, bhv_smooth_size, minimal_state_size, 
                            force_two_states, force_one_state,
                            time_point_pull1, time_point_pull2, time_point_pulls_succfail, do_shuffle):

    likelihood = glm_fitting_summary[(recordedanimal, 'predicted_likelihood')]
    X_all = glm_fitting_summary[(recordedanimal, 'X_all')]
    convolved_vars = glm_fitting_summary[(recordedanimal, 'convolved_var_names')]
    raw_vars = glm_fitting_summary[(recordedanimal, 'raw_var_names')]

    # Align the likelihood/GLM time vector
    abs_time_idx = len(convolved_vars) * N_BASIS_FUNCS + raw_vars.index('abs_time')
    likelihood_time = X_all[:, abs_time_idx] - session_start_time

    # Define behavioral variables for independent panels
    behavior_vars = ['mass_move_speed', 'gaze_angle_speed', 'gaze_other_angle', 'gaze_tube_angle',
                     'gaze_lever_angle', 'animal_animal_dist', 'animal_tube_dist', 'animal_lever_dist',         
                     'socialgaze_prob', 'selfpull_prob' ,'social_evidence']
    behavior_data = data_summary_twoanimals[recordedanimal]
    behavior_time = np.arange(len(behavior_data[0])) / fps - session_start_time
    # ---------------------------------------------------------

    time_point_pull1 = np.array(time_point_pull1)
    time_point_pull2 = np.array(time_point_pull2)

    # plot_min_time = 100
    # plot_max_time = 450
    plot_min_time = np.floor(np.nanmin([np.nanmin(time_point_pull1),np.nanmin(time_point_pull2)]))-10
    plot_max_time = np.ceil(np.nanmax([np.nanmax(time_point_pull1),np.nanmax(time_point_pull2)]))+10

    time_point_pull1_plot = time_point_pull1[(time_point_pull1 < plot_max_time) & (time_point_pull1 > plot_min_time)]
    time_point_pull2_plot = time_point_pull2[(time_point_pull2 < plot_max_time) & (time_point_pull2 > plot_min_time)]

    ind_FR = (FR_timepoint_allch < plot_max_time) & (FR_timepoint_allch > plot_min_time)
    ind_like = (likelihood_time < plot_max_time) & (likelihood_time > plot_min_time)
    ind_behav = (behavior_time < plot_max_time) & (behavior_time > plot_min_time)

    # --- 2. DYNAMIC FIGURE SETUP ---
    pcs = ['pc1', 'pc2', 'pc3']
    raw_vars_to_plot = ['time_since_pull', 'time_since_succ', 'consec_fails']

    # <--- DEFINE SMOOTHING SIGMA HERE
    sigma_smooth = bhv_smooth_size

    # <--- PRE-CALCULATE SMOOTHED PCs SO PLOTS AND CORRELATIONS MATCH EXACTLY
    smoothed_pcs = {pc: gaussian_filter1d(FRPCs_zscore_allch[pc], sigma=sigma_smooth) for pc in pcs}

    # =========================================================
    # --- 7. CORRELATION HEATMAP ---
    # =========================================================

    # 1. Create a rigid, absolute 10Hz master clock (1 bin = strictly 0.1 seconds / 100ms)
    common_time = np.arange(plot_min_time, plot_max_time, 0.1)
    corr_dict = {}

    # Helper function to safely interpolate data
    def interpolate_trace(t_original, y_original, t_common):
        # np.interp requires the x-coordinates to be strictly increasing
        idx_sort = np.argsort(t_original)
        return np.interp(t_common, t_original[idx_sort], y_original[idx_sort])

    # 2. Extract and interpolate Neural PCs
    for pc in pcs:
        t_FR = FR_timepoint_allch[ind_FR]

        # <--- USE THE SMOOTHED TRACE FOR CORRELATION TOO
        y_FR = smoothed_pcs[pc][ind_FR]

        if len(t_FR) > 0:
            corr_dict[f"Neural {pc.upper()}"] = interpolate_trace(t_FR, y_FR, common_time)

    # 3. Extract and interpolate Continuous Behavior
    for var_name in behavior_vars:
        var_idx = data_summary_names.index(var_name)
        t_behav = behavior_time[ind_behav]
        y_behav = behavior_data[var_idx][ind_behav]
        if len(t_behav) > 0:
            corr_dict[var_name] = interpolate_trace(t_behav, y_behav, common_time)

    # 4. Extract and interpolate GLM Raw History Variables
    if addpullinfo == 1:
        for r_var in raw_vars_to_plot:
            idx = len(convolved_vars) * N_BASIS_FUNCS + raw_vars.index(r_var)
            t_like = likelihood_time[ind_like]
            y_raw = X_all[:, idx][ind_like]
            if len(t_like) > 0:
                corr_dict[r_var] = interpolate_trace(t_like, y_raw, common_time)

    # 5. Extract and interpolate GLM Likelihood
    t_like = likelihood_time[ind_like]
    y_like = likelihood[ind_like]
    if len(t_like) > 0:
        corr_dict['P(Pull)'] = interpolate_trace(t_like, y_like, common_time)


    # =========================================================
    # 1. DEFINE CANDIDATE POOL & PLOTTING VARIABLES
    # =========================================================
    # Your 8 candidate kinematic/spatial variables for VIF screening
    candidate_vars = [
        'mass_move_speed', 'gaze_angle_speed', 'gaze_other_angle', 
        'gaze_tube_angle', 'gaze_lever_angle', 'animal_animal_dist', 
        'animal_tube_dist', 'animal_lever_dist',
    ]

    # Variables we strictly want to plot or force into the HMM
    # fixed_hmm_vars = ['socialgaze_prob', 'consec_fails']
    # fixed_hmm_vars = ['socialgaze_prob','time_since_pull']
    fixed_hmm_vars = ['socialgaze_prob_smoothed', 'selfpull_prob_smoothed']
    # vars_to_plot   = ['mass_move_speed', 'socialgaze_prob', 'consec_fails','time_since_pull', 'Neural PC1']
    vars_to_plot   = ['socialgaze_prob', 'selfpull_prob',
                      'Neural PC1', 'Neural PC2', 'Neural PC3']

    # Create a master list of unique variables to extract from corr_dict
    all_required_vars = list(set(candidate_vars + fixed_hmm_vars + vars_to_plot))

    corr_dict['socialgaze_prob_smoothed'] = corr_dict['socialgaze_prob'].copy()
    corr_dict['selfpull_prob_smoothed'] = corr_dict['selfpull_prob'].copy()
    
    # =========================================================
    # 2. EXTRACT, SMOOTH, AND SANITIZE EVERYTHING FIRST
    # =========================================================
    trace_dict = {}
    np.random.seed(42) 

    for var_name in all_required_vars:
        trace = corr_dict[var_name].copy()

        # Cognitive Smoothing for twitchy behavioral features
        if var_name in ['socialgaze_prob_smoothed']:
            trace = gaussian_filter1d(trace, sigma=sigma_smooth)
        
        if var_name in candidate_vars + ['selfpull_prob_smoothed']:
            trace = gaussian_filter1d(trace, sigma=5)

        # Heavy smoothing & micro-noise injection for rigid task variables
        if var_name in ['time_since_pull', 'time_since_succ', 'consec_fails']:
            trace = gaussian_filter1d(trace, sigma=1)
            white_noise = np.random.normal(0, 0.05, size=len(trace))
            trace = trace + white_noise

        trace_dict[var_name] = trace

    # Build Master DataFrame and permanently sanitize NaNs/Infs
    df_raw = pd.DataFrame(trace_dict)
    df_clean = df_raw.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0.0)

    # =========================================================
    # 3. RUN VIF SCREENING ON SANITIZED DATA
    # =========================================================
    # print("Running VIF screening to identify orthogonal continuous features...")
    X_vif = df_clean[candidate_vars].values

    vif_data = pd.DataFrame()
    vif_data["feature"] = candidate_vars
    vif_data["VIF"] = [variance_inflation_factor(X_vif, i) for i in range(len(candidate_vars))]

    # Sort by lowest VIF and pick the top two
    top_two_vars = vif_data.sort_values("VIF").head(2)["feature"].tolist()
    # print(f"[+] Top 2 lowest-VIF features selected: {top_two_vars}\n")

    # Dynamically assemble the final feature matrix for the HMM
    # hmm_vars = top_two_vars + fixed_hmm_vars
    hmm_vars = fixed_hmm_vars
    X_raw_hmm = df_clean[hmm_vars].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw_hmm)
    N_samples, N_features = X_scaled.shape

    # =========================================================
    # 4. AUTOMATED BIC MODEL SELECTION SWEEP
    # =========================================================
    candidate_states = range(1, 8)  # Test 2 through 6 hidden states
    bic_scores = []
    models = {}

    # print("Sweeping candidate HMM architectures...")

    for k in candidate_states:
        test_model = hmm.GaussianHMM(
            n_components=k, 
            covariance_type="diag",  
            min_covar=1e-3,          
            n_iter=1000, 
            random_state=42
        )
        test_model.fit(X_scaled)

        if test_model.monitor_.converged:
            log_likelihood = test_model.score(X_scaled) * N_samples

            # Free parameters calculation
            n_params = (k - 1) + (k * (k - 1)) + (k * N_features) + (k * N_features)
            bic = (-2 * log_likelihood) + (n_params * np.log(N_samples))

            bic_scores.append(bic)
            models[k] = test_model
            # print(f"  Fit k={k} States | Log-Likelihood: {log_likelihood:.1f} | BIC: {bic:.1f}")
        else:
            bic_scores.append(np.inf)
            # print(f"  Fit k={k} States | Failed to converge.")
    

    # Select the number of states
    if force_one_state:
        best_n_states = 1
    elif force_two_states:
        best_n_states = 2
    else:
        best_n_states = candidate_states[np.argmin(bic_scores)]
   


    model = models[best_n_states]
    latent_states = model.predict(X_scaled)

    
    # 5 seconds at 10Hz = 50 bins. medfilt requires an odd integer window size.
    window_size = int(minimal_state_size)
    if window_size % 2 == 0:  # If it is an even number
        window_size += 1      # Make it odd
        
    # Smooth the discrete states directly
    latent_states_smoothed = medfilt(latent_states, kernel_size=window_size)
    # Use the smoothed states for your plotting and ethological math
    latent_states = latent_states_smoothed.astype(int)
    

    # =========================================================
    # 5. AUTOMATED ETHOLOGICAL & NEURAL QUANTIFICATION
    # =========================================================
    state_quantification = {}
    unique_states = np.unique(latent_states)
    
    # Identify the exact start and end indices of every contiguous epoch
    state_changes = np.where(np.diff(latent_states) != 0)[0] + 1
    epoch_starts = np.insert(state_changes, 0, 0)
    epoch_ends = np.append(state_changes, len(latent_states))
    epoch_states = latent_states[epoch_starts]
    
    # --- Integrate User Pull Tracking Logic ---
    time_point_pull1_succ = time_point_pulls_succfail['pull1_succ']
    time_point_pull2_succ = time_point_pulls_succfail['pull2_succ']
    time_point_pull1_fail = time_point_pulls_succfail['pull1_fail']
    time_point_pull2_fail = time_point_pulls_succfail['pull2_fail']

    if animal1 == recordedanimal:
        time_point_pull1_succ = np.array(time_point_pull1_succ)
        succpulls_in_window = time_point_pull1_succ[(time_point_pull1_succ >= plot_min_time) & (time_point_pull1_succ <= plot_max_time)]
        
        time_point_pull1_fail = np.array(time_point_pull1_fail)
        failpulls_in_window = time_point_pull1_fail[(time_point_pull1_fail >= plot_min_time) & (time_point_pull1_fail <= plot_max_time)]   
    elif animal2 == recordedanimal:
        time_point_pull2_succ = np.array(time_point_pull2_succ)
        succpulls_in_window = time_point_pull2_succ[(time_point_pull2_succ >= plot_min_time) & (time_point_pull2_succ <= plot_max_time)]
        
        time_point_pull2_fail = np.array(time_point_pull2_fail)
        failpulls_in_window = time_point_pull2_fail[(time_point_pull2_fail >= plot_min_time) & (time_point_pull2_fail <= plot_max_time)]
    # ------------------------------------------

    for state_id in unique_states:
        state_mask = (latent_states == state_id)
        state_indices = np.where(state_mask)[0]
        
        # 1. Macro State Dwell Time
        total_bins = len(state_indices)
        total_duration_sec = total_bins * 0.1
        
        # 2. Assign Pulls to States
        state_succ_count = 0
        state_fail_count = 0

        for t_succ in succpulls_in_window:
            idx = np.argmin(np.abs(common_time - t_succ))
            if latent_states[idx] == state_id:
                state_succ_count += 1

        for t_fail in failpulls_in_window:
            idx = np.argmin(np.abs(common_time - t_fail))
            if latent_states[idx] == state_id:
                state_fail_count += 1
                
        state_pull_count = state_succ_count + state_fail_count
        succ_ratio = (state_succ_count / state_pull_count * 100) if state_pull_count > 0 else 0.0

        # 3. Continuous Variable Means & Neural Variances (Entire State)
        state_means = {}
        state_stds = {}
        for var_name in all_required_vars:
            state_means[var_name] = np.mean(df_clean[var_name].values[state_mask])
            if 'Neural' in var_name:
                state_stds[var_name] = np.std(df_clean[var_name].values[state_mask])

        # 4. Epoch-by-Epoch Metrics
        state_epoch_starts = epoch_starts[epoch_states == state_id]
        state_epoch_ends = epoch_ends[epoch_states == state_id]
        
        epoch_durations = (state_epoch_ends - state_epoch_starts) * 0.1
        
        epoch_level_data = []
        for start, end in zip(state_epoch_starts, state_epoch_ends):
            epoch_dict = {'duration_sec': (end - start) * 0.1}
            for var_name in all_required_vars:
                epoch_dict[f'mean_{var_name}'] = np.mean(df_clean[var_name].values[start:end])
            epoch_level_data.append(epoch_dict)

        # 5. Compile State Dictionary
        state_quantification[state_id] = {
            'num_epochs': len(state_epoch_starts),
            'total_duration_sec': total_duration_sec,
            'mean_epoch_duration_sec': np.mean(epoch_durations) if len(epoch_durations) > 0 else 0,
            'total_pulls': state_pull_count,
            'succ_pulls': state_succ_count,
            'fail_pulls': state_fail_count,
            'pull_success_rate': succ_ratio,
            'state_means': state_means,
            'state_neural_stds': state_stds,
            'epoch_details': epoch_level_data
        }

    return common_time, latent_states, state_quantification
    

   
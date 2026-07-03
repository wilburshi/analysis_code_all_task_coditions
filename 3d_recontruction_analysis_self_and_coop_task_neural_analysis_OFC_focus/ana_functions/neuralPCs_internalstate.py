# # define the function to use neural PC1,2,3 to decode states and relate them to the behavioral measures
import seaborn as sns
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
    

def neuralPCs_internalstate(glm_fitting_summary, data_summary_twoanimals, data_summary_names, 
                            animal1, animal2, recordedanimal, session_start_time,  N_BASIS_FUNCS, fps, addpullinfo, 
                            FRPCs_zscore_allch, FR_timepoint_allch, PC_smooth_size, force_two_states, force_one_state,
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
                     'socialgaze_prob', 'social_evidence']
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
    sigma_smooth = PC_smooth_size

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
    # BRAIN -> BEHAVIOR HMM DECODING (NEURAL PC1, PC2, PC3)
    # =========================================================

    # 1. Define the input space strictly as the Neural Manifold
    neural_vars = ['Neural PC1', 'Neural PC2', 'Neural PC3']

    # 2. Define the external behavioral variables we want to check against the neural states
    behavior_to_plot = ['mass_move_speed', 'socialgaze_prob', 'consec_fails']
    all_required = list(set(neural_vars+behavior_to_plot))

    # =========================================================
    # 2. EXTRACT, SMOOTH NEURAL PCs, AND SANITIZE
    # =========================================================
    trace_dict = {}
    np.random.seed(42)

    for var_name in all_required:
        trace = corr_dict[var_name].copy()

        # CRITICAL: Smooth Neural PCs (1.0s window at 10Hz) 
        # This prevents the brain state from flickering 40 times a minute
        if var_name in neural_vars:
            # trace = gaussian_filter1d(trace, sigma=10)
            trace = trace


        trace_dict[var_name] = trace

    # Build Master DataFrame and permanently sanitize NaNs/Infs
    df_raw = pd.DataFrame(trace_dict)
    df_clean = df_raw.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0.0)

    # Extract strictly the Neural PCs to train the HMM
    X_neural = df_clean[neural_vars].values
    
    # =======================================================
    # NULL CONTROL: Unique random shift for EACH session
    # =======================================================
    if do_shuffle:
        shift_amount = np.random.randint(600, 1200) 
        # print(f"Null Control: Circularly shifting this session's manifold by {shift_amount} bins ({shift_amount/10} sec)")
        X_neural = np.roll(X_neural, shift_amount, axis=0)
    # =======================================================

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_neural)
    N_samples, N_features = X_scaled.shape

    # =========================================================
    # 3. AUTOMATED BIC MODEL SELECTION SWEEP ON NEURAL MANIFOLD
    # =========================================================
    candidate_states = range(1, 7)
    bic_scores = []
    models = {}

    # print("Sweeping HMM architectures strictly on Prefrontal Neural PCs...")

    for k in candidate_states:
        test_model = hmm.GaussianHMM(
            n_components=k, 
            covariance_type="diag", 
            min_covar=1e-3, 
            n_iter=1500, 
            random_state=42
        )
        test_model.fit(X_scaled)

        if test_model.monitor_.converged:
            log_likelihood = test_model.score(X_scaled) * N_samples
            n_params = (k - 1) + (k * (k - 1)) + (k * N_features) + (k * N_features)
            bic = (-2 * log_likelihood) + (n_params * np.log(N_samples))

            bic_scores.append(bic)
            models[k] = test_model
            # print(f"  Fit k={k} Neural States | Log-Likelihood: {log_likelihood:.1f} | BIC: {bic:.1f}")
        else:
            bic_scores.append(np.inf)
            # print(f"  Fit k={k} Neural States | Failed to converge.")

    # Identify winning model directly from the BIC sweep
    # best_n_states = candidate_states[np.argmin(bic_scores)]
    # print(f"\n[+] Optimal Neural Manifold Model: {best_n_states} Latent States (Min BIC: {min(bic_scores):.1f})")

    # # FORCE to choose two states
    if force_two_states:
        best_n_states = 2
    else:
        best_n_states = candidate_states[np.argmin(bic_scores)]
        
    # # Force to choose one state as a control
    if force_one_state:
        best_n_states = 1
    else:
        best_n_states = candidate_states[np.argmin(bic_scores)]
   


    model = models[best_n_states]
    latent_states = model.predict(X_scaled)


    # =========================================================
    # 2. HMM STATE ETHOLOGICAL QUANTIFIER
    # =========================================================

    state_bhv_summary = {}

    time_point_pull1_succ = time_point_pulls_succfail['pull1_succ']
    time_point_pull2_succ = time_point_pulls_succfail['pull2_succ']
    time_point_pull1_fail = time_point_pulls_succfail['pull1_fail']
    time_point_pull2_fail = time_point_pulls_succfail['pull2_fail']

    #
    if animal1 == recordedanimal:
        pulls_in_window = time_point_pull1[(time_point_pull1 >= plot_min_time) & (time_point_pull1 <= plot_max_time)]
        time_point_pull1_succ = np.array(time_point_pull1_succ)
        succpulls_in_window = time_point_pull1_succ[(time_point_pull1_succ >= plot_min_time) \
                                                  & (time_point_pull1_succ <= plot_max_time)]
        time_point_pull1_fail = np.array(time_point_pull1_fail)
        failpulls_in_window = time_point_pull1_fail[(time_point_pull1_fail >= plot_min_time) \
                                                  & (time_point_pull1_fail <= plot_max_time)]   
    elif animal2 == recordedanimal:
        pulls_in_window = time_point_pull2[(time_point_pull2 >= plot_min_time) & (time_point_pull2 <= plot_max_time)]
        time_point_pull2_succ = np.array(time_point_pull2_succ)
        succpulls_in_window = time_point_pull2_succ[(time_point_pull2_succ >= plot_min_time) \
                                                  & (time_point_pull2_succ <= plot_max_time)]
        time_point_pull2_fail = np.array(time_point_pull2_fail)
        failpulls_in_window = time_point_pull2_fail[(time_point_pull2_fail >= plot_min_time) \
                                                  & (time_point_pull2_fail <= plot_max_time)]


    # Re-verify active traces from our sanitized dataframe
    gaze_trace = df_clean['socialgaze_prob'].values
    fails_trace = df_clean['consec_fails'].values  # <--- NEW: Extract consecutive fails trace
    speed_trace = df_clean['mass_move_speed'].values # <--- 1. NEW: Extract speed trace
   
    unique_states = np.sort(np.unique(latent_states))

    for state_id in unique_states:

        state_bhv_summary['state'+str(int(state_id))]={}

        state_mask = (latent_states == state_id)

        # 1. Calculate Dwell Time (Total seconds spent in this state)
        state_time_sec = np.sum(state_mask) * 0.1  # 10Hz resolution = 0.1s per bin
        total_time_sec = len(latent_states) * 0.1
        time_pct = (state_time_sec / total_time_sec) * 100

        # 2. Extract continuous Social Gaze & Frustration strictly during this state
        mean_gaze = np.mean(gaze_trace[state_mask])
        mean_fails = np.mean(fails_trace[state_mask])  # <--- NEW: Calculate state-specific mean
        mean_speed = np.mean(speed_trace[state_mask]) # <--- 2. NEW: Calculate mean speed

        # 3. Quantify discrete motor events falling inside these specific time blocks
        state_pull_count = 0
        state_succ_count = 0

        for t_succ in succpulls_in_window:
            idx = np.argmin(np.abs(common_time - t_succ))
            if latent_states[idx] == state_id:
                state_succ_count += 1
                state_pull_count += 1

        for t_fail in failpulls_in_window:
            idx = np.argmin(np.abs(common_time - t_fail))
            if latent_states[idx] == state_id:
                state_pull_count += 1

        # Calculate Success Ratio
        succ_ratio = (state_succ_count / state_pull_count * 100) if state_pull_count > 0 else 0.0


        state_bhv_summary['state'+str(int(state_id))]['state_time_sec'] = state_time_sec
        state_bhv_summary['state'+str(int(state_id))]['mean_gaze'] = mean_gaze
        state_bhv_summary['state'+str(int(state_id))]['mean_consec_fails'] = mean_fails
        state_bhv_summary['state'+str(int(state_id))]['mean_speed'] = mean_speed # <--- 3. NEW: Save to dict
        state_bhv_summary['state'+str(int(state_id))]['pull_count'] = state_pull_count
        state_bhv_summary['state'+str(int(state_id))]['succ_ratio'] = succ_ratio
    
    return common_time, latent_states, state_bhv_summary

   
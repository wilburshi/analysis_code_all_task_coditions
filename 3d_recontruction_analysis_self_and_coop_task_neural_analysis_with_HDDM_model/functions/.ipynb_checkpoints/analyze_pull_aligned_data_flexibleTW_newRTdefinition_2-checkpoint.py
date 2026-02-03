import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Added import for Pandas

from scipy.ndimage import label  # import once

from functions.get_aligned_segment import get_aligned_segment

# helper function
# there are two function using different way but define pull 'onset' similarly, choose one of them to use
# old function: In the IPI, find the lowest movement point, and from there look for when the movement start
def find_sharp_increases_withinIPI_not_use(pull_data,speed_data,session_start_time,fps):

    import numpy as np
    
    sharp_increase_frames = np.zeros(np.shape(speed_data))

    session_start_frame = np.round(session_start_time * fps).astype(int)
    
    pull_frames = np.where(pull_data)[0]
    npulls = np.shape(pull_frames)[0]
    
    for ipull in np.arange(0,npulls,1):
    
        pull_frame = pull_frames[ipull]
        #
        if ipull == 0:
            pullstart_frame = session_start_frame
        else:
            pullstart_frame = pull_frames[ipull-1] 
    
        try:
            speed_IPI_tgt = speed_data[pullstart_frame:pull_frame-int(1*fps)] # allows a 1000ms error time window
        except:
            speed_IPI_tgt = speed_data[pullstart_frame:pull_frame-int(0.5*fps)] # allows a 500ms error time window
    
        # Find index of minimum point
        try:
            min_idx = np.argmin(speed_IPI_tgt) 
            # Compute first derivative
            diff = np.diff(speed_IPI_tgt) 
            # Look for first positive change after the minimum
            post_min_diff = diff[min_idx+1:]
            above_zero = np.where(post_min_diff > 0)[0]
            # Compute actual index in the original array
            if len(above_zero) > 0:
                first_increase_idx = min_idx + 1 + above_zero[0]
            else:
                first_increase_idx = 0
        except:
            first_increase_idx = 0
            
        sharp_increase_frames[first_increase_idx + pullstart_frame] = 1
    
    return sharp_increase_frames

# new function: in each IPI, find the movement peak that is closest to the current pull, then retrospectively find when the movement start
def find_sharp_increases_withinIPI(pull_data, speed_data, session_start_time, fps):
    import numpy as np
    from scipy.signal import argrelextrema

    sharp_increase_frames = np.zeros_like(speed_data)
    session_start_frame = int(np.round(session_start_time * fps))
    pull_frames = np.where(pull_data)[0]
    npulls = len(pull_frames)

    for ipull in range(npulls):
        pull_frame = pull_frames[ipull]
        pullstart_frame = session_start_frame if ipull == 0 else pull_frames[ipull - 1]

        try:
            speed_IPI_tgt = speed_data[pullstart_frame:pull_frame-int(1*fps)] # allows a 1000ms error time window
        except:
            speed_IPI_tgt = speed_data[pullstart_frame:pull_frame-int(0.5*fps)] # allows a 500ms error time window

        onset_global_idx = pullstart_frame  # default fallback

        if len(speed_IPI_tgt) >= 3:
            # Find local maxima
            local_max_idx = argrelextrema(speed_IPI_tgt, np.greater_equal)[0]

            if len(local_max_idx) > 0:
                # Get the one closest to the current pull (end of interval)
                peak_idx = local_max_idx[np.argmax(local_max_idx)]

                # Go backward from the peak to find where it started rising
                for j in range(peak_idx - 1, 0, -1):
                    if speed_IPI_tgt[j] < speed_IPI_tgt[j - 1]:
                        onset_local = j
                        onset_global_idx = pullstart_frame + onset_local
                        break

        sharp_increase_frames[onset_global_idx] = 1

    return sharp_increase_frames




def analyze_pull_aligned_data_flexibleTW_newRTdefinition_2(pull1_ts, pull2_ts, juice1_ts, juice2_ts, gaze1_ts, gaze2_ts, levergaze2_ts, levergaze1_ts, speed1_ts, speed2_ts, anglespeed1_ts, anglespeed2_ts, resolution_s, session_start_time, fps):
    
   
    """
    Performs analysis on pull-aligned data with two key modifications:
    1. Successful pulls are defined by finding the closest preceding pull for each juice event.
    2. RT is behaviorally defined by gaze, but falls back to the IPI if no gaze is found.
    """
    num_time_points = len(pull1_ts)
    
    # --- Define successful pulls by mapping from juice to the preceding pull ---
    def get_successful_pull_indices(juice_ts, pull_ts):
        juice_indices = np.where(juice_ts == 1)[0]
        pull_indices = np.where(pull_ts == 1)[0]
        if len(juice_indices) == 0 or len(pull_indices) == 0:
            return set()
        
        insertion_points = np.searchsorted(pull_indices, juice_indices)
        valid_mask = insertion_points > 0
        successful_indices = pull_indices[insertion_points[valid_mask] - 1]
        
        return set(successful_indices)
    
    successful_pull_indices_a1 = get_successful_pull_indices(juice1_ts, pull1_ts)
    successful_pull_indices_a2 = get_successful_pull_indices(juice2_ts, pull2_ts)

    # use one of the method to define the start of one trial
    if 1:
        speed1_increase = find_sharp_increases_withinIPI(pull1_ts,speed1_ts,session_start_time,fps)
        anglespeed1_increase = find_sharp_increases_withinIPI(pull1_ts,anglespeed1_ts,session_start_time,fps)
        start_gaze_indices_a1 = np.where((speed1_increase + anglespeed1_increase) > 0)[0]
        #
        speed2_increase = find_sharp_increases_withinIPI(pull2_ts,speed2_ts,session_start_time,fps)
        anglespeed2_increase = find_sharp_increases_withinIPI(pull2_ts,anglespeed2_ts,session_start_time,fps)
        start_gaze_indices_a2 = np.where((speed2_increase + anglespeed2_increase) > 0)[0]
    if 0:
        # use the gaze events as the start of a trial
        # Pre-calculate gaze event indices for efficient lookup
        start_gaze_indices_a1 = np.where((gaze1_ts + levergaze1_ts) > 0)[0]
        start_gaze_indices_a2 = np.where((gaze2_ts + levergaze2_ts) > 0)[0]
    
    # --- Data structures for HDDM ---
    hddm_data_a1 = []
    hddm_data_a2 = []
    
    
    # --- Metrics for Animal 1's Pulls ---
    a1_pull_indices = np.where(pull1_ts == 1)[0]
    last_reward_time_a1_context = session_start_time
    failed_count_a1_context = 0
    prev_pull_idx_a1 = int(session_start_time / resolution_s)
    
    for i, p_idx in enumerate(a1_pull_indices):
        current_pull_time_s = p_idx * resolution_s
        prev_pull_time_s = prev_pull_idx_a1 * resolution_s
        
        start_trial_idx = np.nan
        current_rt = np.nan
        time_since_last_reward = np.nan
        window_pre_steps = 0
        
        if i > 0:
            if was_successful:
                # possible_start_indices = start_gaze_indices_a1[start_gaze_indices_a1 > (prev_pull_idx_a1+fps)] # 1s for juice delievery for successful pull
                possible_start_indices = start_gaze_indices_a1[start_gaze_indices_a1 > (prev_pull_idx_a1)]
            else:
                # possible_start_indices = start_gaze_indices_a1[start_gaze_indices_a1 > (prev_pull_idx_a1+0.5*fps)] # 0.5s for examiniation for failed pull
                possible_start_indices = start_gaze_indices_a1[start_gaze_indices_a1 > (prev_pull_idx_a1)]
            if len(possible_start_indices) > 0:
                temp_start_idx = possible_start_indices[0]
                if temp_start_idx < p_idx:
                    start_trial_idx = temp_start_idx
                    current_rt = current_pull_time_s - (start_trial_idx * resolution_s)
                    window_pre_steps = p_idx - start_trial_idx
                
            
        if np.isnan(start_trial_idx):
            current_rt = current_pull_time_s - prev_pull_time_s
            window_pre_steps = p_idx - prev_pull_idx_a1
        
        mean_gaze_in_window = np.nan
        partner_speed_mean = np.nan
        self_speed_mean = np.nan
        partner_speed_std = np.nan
        self_speed_std = np.nan

        if window_pre_steps > 0:
            window_total_steps = window_pre_steps + 1
            self_gaze_segment = get_aligned_segment(gaze1_ts, p_idx, window_pre_steps, 0, window_total_steps, num_time_points)
            gaze_auc = np.nansum(self_gaze_segment) * resolution_s
            # mean_gaze_in_window = gaze_auc / current_rt if current_rt > 0 else 0
            mean_gaze_in_window = gaze_auc

            # only care about the partner speed when self animal is looking
            gaze_mask = self_gaze_segment > 0
    
            partner_speed_segment = get_aligned_segment(speed2_ts, p_idx, window_pre_steps, 0, window_total_steps, num_time_points)
            #
            partner_speed_segment = partner_speed_segment * gaze_mask
            #
            partner_speed_mean = np.nanmean(partner_speed_segment)
            partner_speed_std = np.nanstd(partner_speed_segment) # / window_total_steps
            #
            if np.sum(gaze_mask)==0:
                partner_speed_std = np.nan
                partner_speed_mean = np.nan
    
            self_speed_segment = get_aligned_segment(speed1_ts, p_idx, window_pre_steps, 0, window_total_steps, num_time_points)
            self_speed_mean = np.nanmean(self_speed_segment)
            self_speed_std = np.nanstd(self_speed_segment) # / window_total_steps
    
        was_successful = 1 if p_idx in successful_pull_indices_a1 else 0
        time_since_last_reward = current_pull_time_s - last_reward_time_a1_context
    
        hddm_data_a1.append({
            'subj_idx': 'animal1', 
            'rt': current_rt, 
            'response': was_successful,
            'self_gaze_auc': mean_gaze_in_window, 
            'partner_mean_speed': partner_speed_mean,
            'self_mean_speed': self_speed_mean, 
            'partner_speed_std': partner_speed_std,
            'self_speed_std': self_speed_std, 
            'failed_pulls_before_reward': failed_count_a1_context,
            'time_since_last_reward': time_since_last_reward,
            'current_pull_time': current_pull_time_s - session_start_time,
            'previous_pull_time': prev_pull_time_s - session_start_time,
            'current_trial_start_time': current_pull_time_s - session_start_time - current_rt,
        })
    
        if was_successful:
            failed_count_a1_context = 0
            last_reward_time_a1_context = current_pull_time_s
        else:
            failed_count_a1_context += 1
        prev_pull_idx_a1 = p_idx
    
    
    # --- Repeat for Animal 2 (Symmetrical Logic) ---
    a2_pull_indices = np.where(pull2_ts == 1)[0]
    last_reward_time_a2_context = session_start_time 
    failed_count_a2_context = 0
    prev_pull_idx_a2 = int(session_start_time / resolution_s)
    
    for i, p_idx in enumerate(a2_pull_indices):
        current_pull_time_s = p_idx * resolution_s
        prev_pull_time_s = prev_pull_idx_a2 * resolution_s
        
        start_trial_idx = np.nan
        current_rt = np.nan
        time_since_last_reward = np.nan
        window_pre_steps = 0
    
        
        if i > 0:
            if was_successful:
                possible_start_indices = start_gaze_indices_a2[start_gaze_indices_a2 > (prev_pull_idx_a2)] # 1s for juice delievery for successful pull
            else:
                possible_start_indices = start_gaze_indices_a2[start_gaze_indices_a2 > (prev_pull_idx_a2)] # 0.5s for examiniation for failed pull
            if len(possible_start_indices) > 0:
                temp_start_idx = possible_start_indices[0]
                if temp_start_idx < p_idx:
                    start_trial_idx = temp_start_idx
                    current_rt = current_pull_time_s - (start_trial_idx * resolution_s)
                    window_pre_steps = p_idx - start_trial_idx
            
        if np.isnan(start_trial_idx):
            current_rt = current_pull_time_s - prev_pull_time_s
            window_pre_steps = p_idx - prev_pull_idx_a2
        
        mean_gaze_in_window = np.nan
        partner_speed_mean = np.nan
        self_speed_mean = np.nan
        partner_speed_std = np.nan
        self_speed_std = np.nan
        
        if window_pre_steps > 0:
            window_total_steps = window_pre_steps + 1
            self_gaze_segment = get_aligned_segment(gaze2_ts, p_idx, window_pre_steps, 0, window_total_steps, num_time_points)
            gaze_auc = np.nansum(self_gaze_segment) * resolution_s
            # mean_gaze_in_window = gaze_auc / current_rt if current_rt > 0 else 0
            mean_gaze_in_window = gaze_auc

            # only care about the partner speed when self animal is looking
            gaze_mask = self_gaze_segment > 0
            
            partner_speed_segment = get_aligned_segment(speed1_ts, p_idx, window_pre_steps, 0, window_total_steps, num_time_points)
            #
            partner_speed_segment = partner_speed_segment * gaze_mask
            #
            partner_speed_mean = np.nanmean(partner_speed_segment)
            partner_speed_std = np.nanstd(partner_speed_segment) # / window_total_steps
            #
            if np.sum(gaze_mask)==0:
                partner_speed_std = np.nan
                partner_speed_mean = np.nan
    
            self_speed_segment = get_aligned_segment(speed2_ts, p_idx, window_pre_steps, 0, window_total_steps, num_time_points)
            self_speed_mean = np.nanmean(self_speed_segment)
            self_speed_std = np.nanstd(self_speed_segment) # / window_total_steps
        
        was_successful = 1 if p_idx in successful_pull_indices_a2 else 0
        time_since_last_reward = current_pull_time_s - last_reward_time_a2_context
        
        hddm_data_a2.append({
            'subj_idx': 'animal2', 
            'rt': current_rt, 
            'response': was_successful,
            'self_gaze_auc': mean_gaze_in_window, 
            'partner_mean_speed': partner_speed_mean,
            'self_mean_speed': self_speed_mean, 
            'partner_speed_std': partner_speed_std,
            'self_speed_std': self_speed_std, 
            'failed_pulls_before_reward': failed_count_a2_context,
            'time_since_last_reward': time_since_last_reward,
            'current_pull_time': current_pull_time_s - session_start_time,
            'previous_pull_time': prev_pull_time_s - session_start_time,
            'current_trial_start_time': current_pull_time_s - session_start_time - current_rt,
        })
    
        if was_successful:
            failed_count_a2_context = 0
            last_reward_time_a2_context = current_pull_time_s
        else:
            failed_count_a2_context += 1
        prev_pull_idx_a2 = p_idx
        
    
    # --- Final Data Preparation ---
    df_a1 = pd.DataFrame(hddm_data_a1)
    df_a2 = pd.DataFrame(hddm_data_a2)
    
    # Add 'prev_trial_outcome' column
    if not df_a1.empty:
        df_a1['prev_trial_outcome'] = df_a1['response'].shift(1).fillna(0).astype(int)
    if not df_a2.empty:
        df_a2['prev_trial_outcome'] = df_a2['response'].shift(1).fillna(0).astype(int)
    
    df_a1['response'] = 1
    df_a2['response'] = 1
    
    # MODIFICATION: Keep the first pull, which will have NaN values for rt and calculated covariates.
    # The .dropna() calls have been removed.
    df_a1 = df_a1.reset_index(drop=True)
    df_a2 = df_a2.reset_index(drop=True)
    
    # Outlier removal (optional but good practice)
    # This will now correctly ignore the first row if its rt is NaN.
    if 1:
        if not df_a1.empty:
            Q1 = df_a1['rt'].quantile(0.25)
            Q3 = df_a1['rt'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Only apply outlier logic to valid rt values
            df_a1 = df_a1[(df_a1['rt'] >= lower_bound) & (df_a1['rt'] <= upper_bound) | (df_a1['rt'].isna())]
        
        if not df_a2.empty:
            Q1 = df_a2['rt'].quantile(0.25)
            Q3 = df_a2['rt'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Only apply outlier logic to valid rt values
            df_a2 = df_a2[(df_a2['rt'] >= lower_bound) & (df_a2['rt'] <= upper_bound) | (df_a2['rt'].isna())]
    
    results = {'hddm_data_animal1': df_a1, 'hddm_data_animal2': df_a2}
    
    
    # 
    hddm_data_a1 = df_a1.to_dict(orient='records')
    hddm_data_a2 = df_a2.to_dict(orient='records')


    # --- Calculate Correlations (using the collected lists for plotting) ---
    # Need to filter out the first pull's metrics if RT is NaN
    # For correlation, it's safer to align lists after filtering RTs
    a1_gaze_aucs_filtered = [item['self_gaze_auc'] for item in hddm_data_a1 if not np.isnan(item['rt'])]
    a2_speed_means_for_a1_pulls_filtered = [item['partner_mean_speed'] for item in hddm_data_a1 if not np.isnan(item['rt'])]
    a1_failed_pulls_before_reward_filtered = [item['failed_pulls_before_reward'] for item in hddm_data_a1 if not np.isnan(item['rt'])]
    a1_time_since_last_reward_filtered = [item['time_since_last_reward'] for item in hddm_data_a1 if not np.isnan(item['rt'])]
    a1_prev_trial_outcome_filtered = [item['prev_trial_outcome'] for item in hddm_data_a1 if not np.isnan(item['rt'])]

    a2_gaze_aucs_filtered = [item['self_gaze_auc'] for item in hddm_data_a2 if not np.isnan(item['rt'])]
    a1_speed_means_for_a2_pulls_filtered = [item['partner_mean_speed'] for item in hddm_data_a2 if not np.isnan(item['rt'])]
    a2_failed_pulls_before_reward_filtered = [item['failed_pulls_before_reward'] for item in hddm_data_a2 if not np.isnan(item['rt'])]
    a2_time_since_last_reward_filtered = [item['time_since_last_reward'] for item in hddm_data_a2 if not np.isnan(item['rt'])]
    a2_prev_trial_outcome_filtered = [item['prev_trial_outcome'] for item in hddm_data_a2 if not np.isnan(item['rt'])]


    if len(a1_gaze_aucs_filtered) > 1 and len(a2_speed_means_for_a1_pulls_filtered) > 1:
        corr_matrix_a1_gaze_speed = np.corrcoef(a1_gaze_aucs_filtered, a2_speed_means_for_a1_pulls_filtered)
        results['animal1_corr_gaze_auc_vs_partner_speed_mean'] = corr_matrix_a1_gaze_speed[0, 1]
    else:
        results['animal1_corr_gaze_auc_vs_partner_speed_mean'] = np.nan
        print("Warning: Not enough pull events for Animal 1 to calculate correlation (Gaze vs Speed).")
    
    if len(a2_gaze_aucs_filtered) > 1 and len(a1_speed_means_for_a2_pulls_filtered) > 1:
        corr_matrix_a2_gaze_speed = np.corrcoef(a2_gaze_aucs_filtered, a1_speed_means_for_a2_pulls_filtered)
        results['animal2_corr_gaze_auc_vs_partner_speed_mean'] = corr_matrix_a2_gaze_speed[0, 1]
    else:
        results['animal2_corr_gaze_auc_vs_partner_speed_mean'] = np.nan
        print("Warning: Not enough pull events for Animal 2 to calculate correlation (Gaze vs Speed).")


    if len(a1_gaze_aucs_filtered) > 1 and len(a1_failed_pulls_before_reward_filtered) > 1:
        corr_matrix_a1_gaze_failed = np.corrcoef(a1_gaze_aucs_filtered, a1_failed_pulls_before_reward_filtered)
        results['animal1_corr_gaze_auc_vs_failed_pulls'] = corr_matrix_a1_gaze_failed[0, 1]
    else:
        results['animal1_corr_gaze_auc_vs_failed_pulls'] = np.nan
        print("Warning: Not enough pull events for Animal 1 to calculate correlation (Gaze vs Failed Pulls).")

    if len(a2_gaze_aucs_filtered) > 1 and len(a2_failed_pulls_before_reward_filtered) > 1:
        corr_matrix_a2_gaze_failed = np.corrcoef(a2_gaze_aucs_filtered, a2_failed_pulls_before_reward_filtered)
        results['animal2_corr_gaze_auc_vs_failed_pulls'] = corr_matrix_a2_gaze_failed[0, 1]
    else:
        results['animal2_corr_gaze_auc_vs_failed_pulls'] = np.nan
        print("Warning: Not enough pull events for Animal 2 to calculate correlation (Gaze vs Failed Pulls).")


    if len(a1_gaze_aucs_filtered) > 1 and len(a1_time_since_last_reward_filtered) > 1:
        corr_matrix_a1_gaze_time_since_reward = np.corrcoef(a1_gaze_aucs_filtered, a1_time_since_last_reward_filtered)
        results['animal1_corr_gaze_auc_vs_time_since_reward'] = corr_matrix_a1_gaze_time_since_reward[0, 1]
    else:
        results['animal1_corr_gaze_auc_vs_time_since_reward'] = np.nan
        print("Warning: Not enough pull events for Animal 1 to calculate correlation (Gaze vs Time Since Reward).")

    if len(a2_gaze_aucs_filtered) > 1 and len(a2_time_since_last_reward_filtered) > 1:
        corr_matrix_a2_gaze_time_since_reward = np.corrcoef(a2_gaze_aucs_filtered, a2_time_since_last_reward_filtered)
        results['animal2_corr_gaze_auc_vs_time_since_reward'] = corr_matrix_a2_gaze_time_since_reward[0, 1]
    else:
        results['animal2_corr_gaze_auc_vs_time_since_reward'] = np.nan
        print("Warning: Not enough pull events for Animal 2 to calculate correlation (Gaze vs Time Since Reward).")

    # --- Plotting all scatter plots in one figure (2x3 layout) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Correlations with Self Gaze AUC (-4s to 0s)', fontsize=16)

    # Row 1, Column 1: Animal 1 Gaze AUC vs. Partner Mean Speed
    ax = axes[0, 0]
    if not np.isnan(results.get('animal1_corr_gaze_auc_vs_partner_speed_mean', np.nan)):
        ax.scatter(a1_gaze_aucs_filtered, a2_speed_means_for_a1_pulls_filtered, alpha=0.6)
        
        # Add regression line
        m, b = np.polyfit(a1_gaze_aucs_filtered, a2_speed_means_for_a1_pulls_filtered, 1)
        ax.plot(np.array(a1_gaze_aucs_filtered), m*np.array(a1_gaze_aucs_filtered) + b, color='red', linestyle='--', label='Regression Line')

        ax.set_title(f'A1 Gaze AUC vs. A2 Mean Speed (Corr: {results["animal1_corr_gaze_auc_vs_partner_speed_mean"]:.3f})')
        ax.set_xlabel('Self Gaze AUC (-4s to 0s)')
        ax.set_ylabel('Partner (Animal 2) Mean Speed (-4s to 0s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not enough data for A1 Gaze AUC vs. A2 Speed", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('A1 Gaze AUC vs. A2 Mean Speed')

    # Row 1, Column 2: Animal 2 Gaze AUC vs. Partner Mean Speed
    ax = axes[0, 1]
    if not np.isnan(results.get('animal2_corr_gaze_auc_vs_partner_speed_mean', np.nan)):
        ax.scatter(a2_gaze_aucs_filtered, a1_speed_means_for_a2_pulls_filtered, alpha=0.6)

        # Add regression line
        m, b = np.polyfit(a2_gaze_aucs_filtered, a1_speed_means_for_a2_pulls_filtered, 1)
        ax.plot(np.array(a2_gaze_aucs_filtered), m*np.array(a2_gaze_aucs_filtered) + b, color='red', linestyle='--', label='Regression Line')

        ax.set_title(f'A2 Gaze AUC vs. A1 Mean Speed (Corr: {results["animal2_corr_gaze_auc_vs_partner_speed_mean"]:.3f})')
        ax.set_xlabel('Self Gaze AUC (-4s to 0s)')
        ax.set_ylabel('Partner (Animal 1) Mean Speed (-4s to 0s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not enough data for A2 Gaze AUC vs. A1 Speed", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('A2 Gaze AUC vs. A1 Mean Speed')

    # Row 1, Column 3: Animal 1 Gaze AUC vs. Failed Pulls
    ax = axes[0, 2]
    if not np.isnan(results.get('animal1_corr_gaze_auc_vs_failed_pulls', np.nan)):
        ax.scatter(a1_gaze_aucs_filtered, a1_failed_pulls_before_reward_filtered, alpha=0.6)
        m, b = np.polyfit(a1_gaze_aucs_filtered, a1_failed_pulls_before_reward_filtered, 1)
        ax.plot(np.array(a1_gaze_aucs_filtered), m*np.array(a1_gaze_aucs_filtered) + b, color='red', linestyle='--', label='Regression Line')
        ax.set_title(f'A1 Gaze AUC vs. A1 Failed Pulls (Corr: {results["animal1_corr_gaze_auc_vs_failed_pulls"]:.3f})')
        ax.set_xlabel('Self Gaze AUC (-4s to 0s)')
        ax.set_ylabel('Number of Failed Pulls Before Reward')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not enough data for A1 Gaze AUC vs. Failed Pulls", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('A1 Gaze AUC vs. A1 Failed Pulls')

    # Row 2, Column 1: Animal 2 Gaze AUC vs. Failed Pulls
    ax = axes[1, 0]
    if not np.isnan(results.get('animal2_corr_gaze_auc_vs_failed_pulls', np.nan)):
        ax.scatter(a2_gaze_aucs_filtered, a2_failed_pulls_before_reward_filtered, alpha=0.6)
        m, b = np.polyfit(a2_gaze_aucs_filtered, a2_failed_pulls_before_reward_filtered, 1)
        ax.plot(np.array(a2_gaze_aucs_filtered), m*np.array(a2_gaze_aucs_filtered) + b, color='red', linestyle='--', label='Regression Line')
        ax.set_title(f'A2 Gaze AUC vs. A2 Failed Pulls (Corr: {results["animal2_corr_gaze_auc_vs_failed_pulls"]:.3f})')
        ax.set_xlabel('Self Gaze AUC (-4s to 0s)')
        ax.set_ylabel('Number of Failed Pulls Before Reward')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not enough data for A2 Gaze AUC vs. Failed Pulls", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('A2 Gaze AUC vs. A2 Failed Pulls')

    # Row 2, Column 2: Animal 1 Gaze AUC vs. Time Since Last Reward
    ax = axes[1, 1]
    if not np.isnan(results.get('animal1_corr_gaze_auc_vs_time_since_reward', np.nan)):
        ax.scatter(a1_gaze_aucs_filtered, a1_time_since_last_reward_filtered, alpha=0.6)
        m, b = np.polyfit(a1_gaze_aucs_filtered, a1_time_since_last_reward_filtered, 1)
        ax.plot(np.array(a1_gaze_aucs_filtered), m*np.array(a1_gaze_aucs_filtered) + b, color='red', linestyle='--', label='Regression Line')
        ax.set_title(f'A1 Gaze AUC vs. A1 Time Since Reward (Corr: {results["animal1_corr_gaze_auc_vs_time_since_reward"]:.3f})')
        ax.set_xlabel('Self Gaze AUC (-4s to 0s)')
        ax.set_ylabel('Time Since Last Reward (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not enough data for A1 Gaze AUC vs. Time Since Reward", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('A1 Gaze AUC vs. A1 Time Since Reward')

    # Row 2, Column 3: Animal 2 Gaze AUC vs. Time Since Last Reward
    ax = axes[1, 2]
    if not np.isnan(results.get('animal2_corr_gaze_auc_vs_time_since_reward', np.nan)):
        ax.scatter(a2_gaze_aucs_filtered, a2_time_since_last_reward_filtered, alpha=0.6)
        m, b = np.polyfit(a2_gaze_aucs_filtered, a2_time_since_last_reward_filtered, 1)
        ax.plot(np.array(a2_gaze_aucs_filtered), m*np.array(a2_gaze_aucs_filtered) + b, color='red', linestyle='--', label='Regression Line')
        ax.set_title(f'A2 Gaze AUC vs. A2 Time Since Reward (Corr: {results["animal2_corr_gaze_auc_vs_time_since_reward"]:.3f})')
        ax.set_xlabel('Self Gaze AUC (-4s to 0s)')
        ax.set_ylabel('Time Since Last Reward (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not enough data for A2 Gaze AUC vs. Time Since Reward", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('A2 Gaze AUC vs. A2 Time Since Reward')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    plt.close(fig)

    return results

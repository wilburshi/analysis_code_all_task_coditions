import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Added import for Pandas

from functions.get_aligned_segment import get_aligned_segment

def analyze_pull_aligned_data_flexibleTW(pull1_ts, pull2_ts, gaze1_ts, gaze2_ts, speed1_ts, speed2_ts, resolution_s,
                                         time_window_start_s, time_window_end_s, coop_window_s):
    """
    Performs analysis on pull-aligned data:
    1. Calculates AUC of self-gaze before self-pull.
    2. Calculates mean of partner speed before self-pull.
    3. Calculates mean of self speed before self-pull.
    4. Calculates number of failed pulls before reset.
    5. Calculates time since last reward.
    6. Computes and plots correlations.
    7. Prepares dataframes for HDDM.
    """

    num_time_points = len(pull1_ts)
    coop_window_steps = int(coop_window_s / resolution_s) # Define coop_window_steps here
    
    window_start_steps_relative_to_pull = int(time_window_start_s / resolution_s)
    window_end_steps_relative_to_pull = int(time_window_end_s / resolution_s)
    
    window_steps_pre_for_helper = -window_start_steps_relative_to_pull
    window_steps_post_for_helper = window_end_steps_relative_to_pull

    total_analysis_window_steps = window_steps_pre_for_helper + window_steps_post_for_helper + 1

    results = {}

    # --- Identify cooperative events (rewards) and store them for later use ---
    all_pull_events = []
    for idx in np.where(pull1_ts == 1)[0]:
        all_pull_events.append({'time_idx': idx, 'animal': 1})
    for idx in np.where(pull2_ts == 1)[0]:
        all_pull_events.append({'time_idx': idx, 'animal': 2})
    all_pull_events.sort(key=lambda x: x['time_idx'])

    reward_times_indices = []
    processed_indices_for_reward = set()

    for i, p_event in enumerate(all_pull_events):
        if p_event['time_idx'] not in processed_indices_for_reward:
            partner_found = False
            for j in range(i + 1, len(all_pull_events)):
                p_partner_event = all_pull_events[j]
                if p_event['animal'] != p_partner_event['animal'] and abs(p_event['time_idx'] - p_partner_event['time_idx']) <= coop_window_steps:
                    reward_time_idx = max(p_event['time_idx'], p_partner_event['time_idx'])
                    reward_times_indices.append(reward_time_idx)
                    processed_indices_for_reward.add(p_event['time_idx'])
                    processed_indices_for_reward.add(p_partner_event['time_idx'])
                    partner_found = True
                    break


    reward_times_indices.sort()


    # --- Data structures for HDDM ---
    hddm_data_a1 = []
    hddm_data_a2 = []

    # --- Metrics for Animal 1's Pulls ---
    a1_pull_indices = np.where(pull1_ts == 1)[0]
    a1_pull_times_s = a1_pull_indices * resolution_s
    
    a1_gaze_aucs = []
    a1_speed_means_for_a1_pulls = []
    a2_speed_means_for_a1_pulls = []
    a1_failed_pulls_before_reward = []
    a1_time_since_last_reward = []

    last_reward_time_a1_context = 0.0
    failed_count_a1_context = 0
    prev_pull_time_a1 = 0.0
    is_rewarded_current_pull_a1 = 0 # Default to 0 (failure) for the first pull

    for i, p_idx in enumerate(a1_pull_indices):
        current_pull_time_s = p_idx * resolution_s

        # Flexible window: from previous self-pull to current self-pull (RT duration)
        current_rt = current_pull_time_s - prev_pull_time_a1
        if i == 0:
            current_rt = np.nan # No previous pull for the very first one. Will be dropped.
            
        # Define flexible window steps for get_aligned_segment
        if not np.isnan(current_rt) and current_rt > 0:
            flexible_window_pre_steps = int(current_rt / resolution_s)
            flexible_window_total_steps = flexible_window_pre_steps + 1
        else: # Handle first pull or zero/negative RTs for segment slicing
            flexible_window_pre_steps = 0
            flexible_window_total_steps = 1 # Just the current point
            
        # Gaze AUC and Partner Speed Mean (calculated over flexible window)
        self_gaze_segment = get_aligned_segment(gaze1_ts, p_idx, flexible_window_pre_steps, 0, flexible_window_total_steps, num_time_points)
        gaze_auc = np.nansum(self_gaze_segment) * resolution_s # AUC over the flexible window
        
        # Normalize gaze AUC by the flexible window length (RT) to get mean gaze intensity
        mean_gaze_in_window = gaze_auc / current_rt if not np.isnan(current_rt) and current_rt > 0 else np.nan
        a1_gaze_aucs.append(mean_gaze_in_window)


        partner_speed_segment = get_aligned_segment(speed2_ts, p_idx, flexible_window_pre_steps, 0, flexible_window_total_steps, num_time_points)
        partner_speed_mean = np.nanmean(partner_speed_segment)
        a2_speed_means_for_a1_pulls.append(partner_speed_mean)

        self_speed_segment = get_aligned_segment(speed1_ts, p_idx, flexible_window_pre_steps, 0, flexible_window_total_steps, num_time_points)
        self_speed_mean = np.nanmean(self_speed_segment)
        a1_speed_means_for_a1_pulls.append(self_speed_mean)

        # Determine if current pull is part of a cooperative (rewarded) event
        current_pull_is_rewarded = False
        for rw_idx in reward_times_indices:
            if abs(p_idx - rw_idx) <= coop_window_steps:
                current_pull_is_rewarded = True
                break
        
        a1_failed_pulls_before_reward.append(failed_count_a1_context)
        a1_time_since_last_reward.append(current_pull_time_s - last_reward_time_a1_context)
        
        prev_pull_time_a1 = current_pull_time_s
        if current_pull_is_rewarded:
            failed_count_a1_context = 0
            last_reward_time_a1_context = current_pull_time_s
            is_rewarded_current_pull_a1 = 1 # Outcome of THIS pull for NEXT trial
        else:
            failed_count_a1_context += 1
            is_rewarded_current_pull_a1 = 0

        if not np.isnan(current_rt): # Use current_rt for validity check
            hddm_data_a1.append({
                'subj_idx': 'animal1',
                'rt': current_rt, # Use current_rt for the DDM RT
                'response': 1,
                'self_gaze_auc': mean_gaze_in_window, # Use normalized gaze
                'partner_mean_speed': partner_speed_mean,
                'self_mean_speed': self_speed_mean,
                'failed_pulls_before_reward': a1_failed_pulls_before_reward[-1],
                'time_since_last_reward': a1_time_since_last_reward[-1],
                'prev_trial_outcome': is_rewarded_current_pull_a1
            })

    # Correct 'prev_trial_outcome' after all data collected for Animal 1
    if len(hddm_data_a1) > 0:
        # Shift outcomes: outcome of pull i becomes prev_trial_outcome for pull i+1
        # The first entry's prev_trial_outcome should be 0 (no previous reward)
        # The stored 'is_rewarded_current_pull_a1' is the outcome of the *current* pull.
        # So, we shift this list to become the 'previous outcome' for the next entry.
        shifted_outcomes = [0] + [d['prev_trial_outcome'] for d in hddm_data_a1[:-1]]
        for i, d in enumerate(hddm_data_a1):
            d['prev_trial_outcome'] = shifted_outcomes[i]


    # --- Metrics for Animal 2's Pulls ---
    a2_pull_indices = np.where(pull2_ts == 1)[0]

    a2_gaze_aucs = []
    a1_speed_means_for_a2_pulls = []
    a2_speed_means_for_a2_pulls = []
    a2_failed_pulls_before_reward = []
    a2_time_since_last_reward = []

    last_reward_time_a2_context = 0.0
    failed_count_a2_context = 0
    prev_pull_time_a2 = 0.0
    is_rewarded_current_pull_a2 = 0 # Default to 0 (failure) for the first pull

    for i, p_idx in enumerate(a2_pull_indices):
        current_pull_time_s = p_idx * resolution_s

        current_rt = current_pull_time_s - prev_pull_time_a2
        if i == 0:
            current_rt = np.nan
        
        if not np.isnan(current_rt) and current_rt > 0:
            flexible_window_pre_steps = int(current_rt / resolution_s)
            flexible_window_total_steps = flexible_window_pre_steps + 1
        else:
            flexible_window_pre_steps = 0
            flexible_window_total_steps = 1
            
        self_gaze_segment = get_aligned_segment(gaze2_ts, p_idx, flexible_window_pre_steps, 0, flexible_window_total_steps, num_time_points)
        gaze_auc = np.nansum(self_gaze_segment) * resolution_s
        mean_gaze_in_window = gaze_auc / current_rt if not np.isnan(current_rt) and current_rt > 0 else np.nan
        a2_gaze_aucs.append(mean_gaze_in_window)

        partner_speed_segment = get_aligned_segment(speed1_ts, p_idx, flexible_window_pre_steps, 0, flexible_window_total_steps, num_time_points)
        partner_speed_mean = np.nanmean(partner_speed_segment)
        a1_speed_means_for_a2_pulls.append(partner_speed_mean)

        self_speed_segment = get_aligned_segment(speed2_ts, p_idx, flexible_window_pre_steps, 0, flexible_window_total_steps, num_time_points)
        self_speed_mean = np.nanmean(self_speed_segment)
        a2_speed_means_for_a2_pulls.append(self_speed_mean)


        current_pull_is_rewarded = False
        for rw_idx in reward_times_indices:
            if abs(p_idx - rw_idx) <= coop_window_steps:
                current_pull_is_rewarded = True
                break
        
        a2_failed_pulls_before_reward.append(failed_count_a2_context)
        a2_time_since_last_reward.append(current_pull_time_s - last_reward_time_a2_context)

        prev_pull_time_a2 = current_pull_time_s
        if current_pull_is_rewarded:
            failed_count_a2_context = 0
            last_reward_time_a2_context = current_pull_time_s
            is_rewarded_current_pull_a2 = 1
        else:
            failed_count_a2_context += 1
            is_rewarded_current_pull_a2 = 0

        if not np.isnan(current_rt):
            hddm_data_a2.append({
                'subj_idx': 'animal2',
                'rt': current_rt,
                'response': 1,
                'self_gaze_auc': mean_gaze_in_window,
                'partner_mean_speed': partner_speed_mean,
                'self_mean_speed': self_speed_mean,
                'failed_pulls_before_reward': a2_failed_pulls_before_reward[-1],
                'time_since_last_reward': a2_time_since_last_reward[-1],
                'prev_trial_outcome': is_rewarded_current_pull_a2
            })

    if len(hddm_data_a2) > 0:
        shifted_outcomes = [0] + [d['prev_trial_outcome'] for d in hddm_data_a2[:-1]]
        for i, item in enumerate(hddm_data_a2):
            item['prev_trial_outcome'] = shifted_outcomes[i]


    # Convert to DataFrame, drop rows with NaN RT (first pull of each animal)
    df_a1 = pd.DataFrame(hddm_data_a1).dropna(subset=['rt'])
    df_a2 = pd.DataFrame(hddm_data_a2).dropna(subset=['rt'])

    # Remove outliers based on 'rt' for df_a1
    Q1_a1 = df_a1['rt'].quantile(0.25)
    Q3_a1 = df_a1['rt'].quantile(0.75)
    IQR_a1 = Q3_a1 - Q1_a1
    lower_bound_a1 = Q1_a1 - 1.5 * IQR_a1
    upper_bound_a1 = Q3_a1 + 1.5 * IQR_a1
    df_a1 = df_a1[(df_a1['rt'] >= lower_bound_a1) & (df_a1['rt'] <= upper_bound_a1)]

    # Remove outliers based on 'rt' for df_a2
    Q1_a2 = df_a2['rt'].quantile(0.25)
    Q3_a2 = df_a2['rt'].quantile(0.75)
    IQR_a2 = Q3_a2 - Q1_a2
    lower_bound_a2 = Q1_a2 - 1.5 * IQR_a2
    upper_bound_a2 = Q3_a2 + 1.5 * IQR_a2
    df_a2 = df_a2[(df_a2['rt'] >= lower_bound_a2) & (df_a2['rt'] <= upper_bound_a2)]

    # Save results
    results['hddm_data_animal1'] = df_a1
    results['hddm_data_animal2'] = df_a2

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

def align_and_plot_data(pull1_ts, pull2_ts, gaze1_ts, gaze2_ts, speed1_ts, speed2_ts, resolution_s, window_s_pre, window_s_post):
    """
    Aligns and plots continuous data and binary pull events around self-pulls,
    including error areas.
    """
        
    import numpy as np
    import matplotlib.pyplot as plt
    
    from functions.get_aligned_segment import get_aligned_segment
    from functions.compute_stats import compute_stats
    
    num_time_points = len(pull1_ts)
    window_steps_pre = int(window_s_pre / resolution_s)
    window_steps_post = int(window_s_post / resolution_s)
    total_window_steps = window_steps_pre + window_steps_post + 1

    aligned_time_axis = np.linspace(-window_s_pre, window_s_post, total_window_steps)

    aligned_gaze1_raw = []
    aligned_speed1_raw = []
    aligned_other_pull_dist_for_a1_raw = []
    aligned_other_gaze_for_a1_raw = []
    aligned_other_speed_for_a1_raw = []

    aligned_gaze2_raw = []
    aligned_speed2_raw = []
    aligned_other_pull_dist_for_a2_raw = []
    aligned_other_gaze_for_a2_raw = []
    aligned_other_speed_for_a2_raw = []

    pull1_indices = np.where(pull1_ts == 1)[0]
    for p_idx in pull1_indices:
        aligned_gaze1_raw.append(get_aligned_segment(gaze1_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))
        aligned_speed1_raw.append(get_aligned_segment(speed1_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))
        aligned_other_pull_dist_for_a1_raw.append(get_aligned_segment(pull2_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))
        aligned_other_gaze_for_a1_raw.append(get_aligned_segment(gaze2_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))
        aligned_other_speed_for_a1_raw.append(get_aligned_segment(speed2_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))

    pull2_indices = np.where(pull2_ts == 1)[0]
    for p_idx in pull2_indices:
        aligned_gaze2_raw.append(get_aligned_segment(gaze2_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))
        aligned_speed2_raw.append(get_aligned_segment(speed2_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))
        aligned_other_pull_dist_for_a2_raw.append(get_aligned_segment(pull1_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))
        aligned_other_gaze_for_a2_raw.append(get_aligned_segment(gaze1_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))
        aligned_other_speed_for_a2_raw.append(get_aligned_segment(speed1_ts, p_idx, window_steps_pre, window_steps_post, total_window_steps, num_time_points))


    np_aligned_gaze1 = np.array(aligned_gaze1_raw)
    np_aligned_speed1 = np.array(aligned_speed1_raw)
    np_aligned_other_pull_dist_for_a1 = np.array(aligned_other_pull_dist_for_a1_raw)
    np_aligned_other_gaze_for_a1 = np.array(aligned_other_gaze_for_a1_raw)
    np_aligned_other_speed_for_a1 = np.array(aligned_other_speed_for_a1_raw)

    np_aligned_gaze2 = np.array(aligned_gaze2_raw)
    np_aligned_speed2 = np.array(aligned_speed2_raw)
    np_aligned_other_pull_dist_for_a2 = np.array(aligned_other_pull_dist_for_a2_raw)
    np_aligned_other_gaze_for_a2 = np.array(aligned_other_gaze_for_a2_raw)
    np_aligned_other_speed_for_a2 = np.array(aligned_other_speed_for_a2_raw)


    mean_gaze1, sem_gaze1 = compute_stats(np_aligned_gaze1)
    mean_speed1, sem_speed1 = compute_stats(np_aligned_speed1)
    
    mean_other_pull_dist_for_a1 = np.nanmean(np_aligned_other_pull_dist_for_a1, axis=0)
    sem_other_pull_dist_for_a1 = np.sqrt(mean_other_pull_dist_for_a1 * (1 - mean_other_pull_dist_for_a1) / np_aligned_other_pull_dist_for_a1.shape[0])
    
    mean_other_gaze_for_a1, sem_other_gaze_for_a1 = compute_stats(np_aligned_other_gaze_for_a1)
    mean_other_speed_for_a1, sem_other_speed_for_a1 = compute_stats(np_aligned_other_speed_for_a1)

    mean_gaze2, sem_gaze2 = compute_stats(np_aligned_gaze2)
    mean_speed2, sem_speed2 = compute_stats(np_aligned_speed2)

    mean_other_pull_dist_for_a2 = np.nanmean(np_aligned_other_pull_dist_for_a2, axis=0)
    sem_other_pull_dist_for_a2 = np.sqrt(mean_other_pull_dist_for_a2 * (1 - mean_other_pull_dist_for_a2) / np_aligned_other_pull_dist_for_a2.shape[0])
    
    mean_other_gaze_for_a2, sem_other_gaze_for_a2 = compute_stats(np_aligned_other_gaze_for_a2)
    mean_other_speed_for_a2, sem_other_speed_for_a2 = compute_stats(np_aligned_other_speed_for_a2)


    # --- Plotting ---
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), sharex=True)
    fig.suptitle('Pull-Aligned Data for Marmosets (-4 to +4 seconds)', fontsize=16)

    # Plot for Animal 1's Self-Pulls
    axes[0, 0].plot(aligned_time_axis, mean_gaze1, color='blue', label='Self Gaze')
    axes[0, 0].fill_between(aligned_time_axis, mean_gaze1 - sem_gaze1, mean_gaze1 + sem_gaze1, color='blue', alpha=0.2)
    axes[0, 0].plot(aligned_time_axis, mean_other_gaze_for_a1, color='orange', linestyle='--', label='Other Gaze')
    axes[0, 0].fill_between(aligned_time_axis, mean_other_gaze_for_a1 - sem_other_gaze_for_a1, mean_other_gaze_for_a1 + sem_other_gaze_for_a1, color='orange', alpha=0.2)
    axes[0, 0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0, 0].set_title('Animal 1: Gaze (Self & Other)')
    axes[0, 0].set_ylabel('Gaze (Arb. Units)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    axes[0, 1].plot(aligned_time_axis, mean_speed1, color='green', label='Self Speed')
    axes[0, 1].fill_between(aligned_time_axis, mean_speed1 - sem_speed1, mean_speed1 + sem_speed1, color='green', alpha=0.2)
    axes[0, 1].plot(aligned_time_axis, mean_other_speed_for_a1, color='purple', linestyle='--', label='Other Speed')
    axes[0, 1].fill_between(aligned_time_axis, mean_other_speed_for_a1 - sem_other_speed_for_a1, mean_other_speed_for_a1 + sem_other_speed_for_a1, color='purple', alpha=0.2)
    axes[0, 1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0, 1].set_title('Animal 1: Speed (Self & Other)')
    axes[0, 1].set_ylabel('Speed (Arb. Units)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    axes[0, 2].plot(aligned_time_axis, mean_other_pull_dist_for_a1, color='red', label='Other Pull')
    axes[0, 2].fill_between(aligned_time_axis, mean_other_pull_dist_for_a1 - sem_other_pull_dist_for_a1, mean_other_pull_dist_for_a1 + sem_other_pull_dist_for_a1, color='red', alpha=0.2)
    axes[0, 2].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0, 2].set_title('Animal 1 Pull Aligned: Other Pull Distribution (Animal 2)')
    axes[0, 2].set_ylabel('Pull Probability')
    axes[0, 2].legend()
    axes[0, 2].grid(True, linestyle=':', alpha=0.6)
    axes[0, 2].set_ylim([-0.05, 1.05]) # For binary distribution

    # Plot for Animal 2's Self-Pulls
    axes[1, 0].plot(aligned_time_axis, mean_gaze2, color='blue', label='Self Gaze')
    axes[1, 0].fill_between(aligned_time_axis, mean_gaze2 - sem_gaze2, mean_gaze2 + sem_gaze2, color='blue', alpha=0.2)
    axes[1, 0].plot(aligned_time_axis, mean_other_gaze_for_a2, color='orange', linestyle='--', label='Other Gaze')
    axes[1, 0].fill_between(aligned_time_axis, mean_other_gaze_for_a2 - sem_other_gaze_for_a2, mean_other_gaze_for_a2 + sem_other_gaze_for_a2, color='orange', alpha=0.2)
    axes[1, 0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1, 0].set_title('Animal 2: Gaze (Self & Other)')
    axes[1, 0].set_xlabel('Time relative to Self Pull (s)')
    axes[1, 0].set_ylabel('Gaze (Arb. Units)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    axes[1, 1].plot(aligned_time_axis, mean_speed2, color='green', label='Self Speed')
    axes[1, 1].fill_between(aligned_time_axis, mean_speed2 - sem_speed2, mean_speed2 + sem_speed2, color='green', alpha=0.2)
    axes[1, 1].plot(aligned_time_axis, mean_other_speed_for_a2, color='purple', linestyle='--', label='Other Speed')
    axes[1, 1].fill_between(aligned_time_axis, mean_other_speed_for_a2 - sem_other_speed_for_a2, mean_other_speed_for_a2 + sem_other_speed_for_a2, color='purple', alpha=0.2)
    axes[1, 1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1, 1].set_title('Animal 2: Speed (Self & Other)')
    axes[1, 1].set_xlabel('Time relative to Self Pull (s)')
    axes[1, 1].set_ylabel('Speed (Arb. Units)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    axes[1, 2].plot(aligned_time_axis, mean_other_pull_dist_for_a2, color='red', label='Other Pull')
    axes[1, 2].fill_between(aligned_time_axis, mean_other_pull_dist_for_a2 - sem_other_pull_dist_for_a2, mean_other_pull_dist_for_a2 + sem_other_pull_dist_for_a2, color='red', alpha=0.2)
    axes[1, 2].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1, 2].set_title('Animal 2 Pull Aligned: Other Pull Distribution (Animal 1)')
    axes[1, 2].set_xlabel('Time relative to Self Pull (s)')
    axes[1, 2].set_ylabel('Pull Probability')
    axes[1, 2].legend()
    axes[1, 2].grid(True, linestyle=':', alpha=0.6)
    axes[1, 2].set_ylim([-0.05, 1.05])


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
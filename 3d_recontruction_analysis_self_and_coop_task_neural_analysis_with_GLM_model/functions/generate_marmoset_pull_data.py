def generate_marmoset_pull_data(duration_s, resolution_s,  num_pulls_animal1, num_pulls_animal2, num_coop_pulls, coop_window_s, correlation_strength, prob_self_gaze_pre_self_pull,  prob_self_gaze_post_partner_pull): # New parameter
    """
    Generates simulated pull data for two marmosets, including pull times,
    social gaze distribution, and movement speed, with a tunable correlation
    between self-gaze accumulation and partner speed.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from functions._apply_gaussian_burst import _apply_gaussian_burst
    from functions.get_aligned_segment import get_aligned_segment
    
    num_time_points = int(duration_s / resolution_s)
    coop_window_steps = int(coop_window_s / resolution_s)

    # Initialize time series arrays
    pull1_ts = np.zeros(num_time_points, dtype=int)
    pull2_ts = np.zeros(num_time_points, dtype=int)

    # --- Generate Animal 1 pulls ---
    animal1_pull_indices = np.random.choice(num_time_points, num_pulls_animal1, replace=False)
    animal1_pull_indices.sort()
    pull1_ts[animal1_pull_indices] = 1

    # --- Generate Animal 2 pulls ---
    pull2_indices = set()
    coop_target_pull1_indices = np.random.choice(animal1_pull_indices, num_coop_pulls, replace=False)

    for a1_idx in coop_target_pull1_indices:
        while True:
            offset = np.random.randint(-coop_window_steps, coop_window_steps + 1)
            a2_coop_idx = a1_idx + offset
            if 0 <= a2_coop_idx < num_time_points and a2_coop_idx not in pull2_indices:
                pull2_indices.add(a2_coop_idx)
                break

    num_independent_pulls = num_pulls_animal2 - num_coop_pulls
    forbidden_indices = set()
    for a1_idx in animal1_pull_indices:
        for i in range(max(0, a1_idx - coop_window_steps), min(num_time_points, a1_idx + coop_window_steps + 1)):
            forbidden_indices.add(i)

    available_independent_indices = np.array(list(set(range(num_time_points)) - forbidden_indices - pull2_indices))

    if len(available_independent_indices) < num_independent_pulls:
        print(f"Warning: Not enough space for {num_independent_pulls} independent pulls for Animal 2. Adjust parameters.")
        num_independent_pulls = len(available_independent_indices)

    independent_pull_indices = np.random.choice(available_independent_indices, num_independent_pulls, replace=False)
    for idx in independent_pull_indices:
        pull2_indices.add(idx)

    final_pull2_indices = sorted(list(pull2_indices))
    pull2_ts[final_pull2_indices] = 1

    # --- Parameters for Continuous Variables ---
    # Parameters for Gaze
    gaze_baseline_mean = 0.1
    gaze_baseline_std = 0.1
    gaze_peak_amplitude_mean = 1.5
    gaze_peak_amplitude_std = 1.0
    gaze_pre_pull_offset_s = -0.55
    gaze_burst_std_s = 0.9 # Increased from 0.2

    # New parameters for Gaze AFTER partner pull
    gaze_post_partner_pull_offset_s = 0.3
    gaze_post_partner_pull_amplitude_mean = 0.8
    gaze_post_partner_pull_amplitude_std = 0.3
    gaze_post_partner_burst_std_s = 0.6 # Increased from 0.1


    # Parameters for Speed
    speed_baseline_mean = 0.05
    speed_baseline_std = 0.05
    speed_peak_amplitude_mean = 0.8
    speed_peak_amplitude_std = 0.4
    speed_at_pull_offset_s = 0.0
    speed_burst_std_s = 0.9 # Increased from 0.7

    # --- Initialize all continuous time series to baseline ---
    gaze1_ts = np.random.normal(loc=gaze_baseline_mean, scale=gaze_baseline_std, size=num_time_points)
    gaze2_ts = np.random.normal(loc=gaze_baseline_mean, scale=gaze_baseline_std, size=num_time_points)
    speed1_ts = np.random.normal(loc=speed_baseline_mean, scale=speed_baseline_std, size=num_time_points)
    speed2_ts = np.random.normal(loc=speed_baseline_mean, scale=speed_baseline_std, size=num_time_points)

    gaze1_ts[gaze1_ts < 0] = 0
    gaze2_ts[gaze2_ts < 0] = 0
    speed1_ts[speed1_ts < 0] = 0
    speed2_ts[speed2_ts < 0] = 0

    # --- First pass: Apply independent speed bursts for both animals ---
    for p1_idx in animal1_pull_indices:
        s1_amp = np.random.normal(loc=speed_peak_amplitude_mean, scale=speed_peak_amplitude_std)
        _apply_gaussian_burst(speed1_ts, p1_idx, speed_at_pull_offset_s, speed_burst_std_s, max(0, s1_amp), resolution_s, num_time_points)

    for p2_idx in final_pull2_indices:
        s2_amp = np.random.normal(loc=speed_peak_amplitude_mean, scale=speed_peak_amplitude_std)
        _apply_gaussian_burst(speed2_ts, p2_idx, speed_at_pull_offset_s, speed_burst_std_s, max(0, s2_amp), resolution_s, num_time_points)

    # --- Second pass: Apply self-gaze bursts, influenced by partner speed ---
    # Window for partner speed influence on self-gaze
    partner_speed_influence_window_start_s = -4.0
    partner_speed_influence_window_end_s = 0.0

    partner_speed_influence_window_pre_steps = int(-partner_speed_influence_window_start_s / resolution_s)
    partner_speed_influence_window_post_steps = int(partner_speed_influence_window_end_s / resolution_s)
    partner_speed_influence_window_total_steps = partner_speed_influence_window_pre_steps + partner_speed_influence_window_post_steps + 1

    # Apply Animal 1's self-gaze, influenced by Animal 2's speed (and new probability constraint)
    for p1_idx in animal1_pull_indices:
        if np.random.rand() < prob_self_gaze_pre_self_pull: # Apply probability constraint
            # Get Animal 2's mean speed in the window before Animal 1's pull
            a2_speed_segment_for_g1 = get_aligned_segment(speed2_ts, p1_idx,
                                                        partner_speed_influence_window_pre_steps,
                                                        partner_speed_influence_window_post_steps,
                                                        partner_speed_influence_window_total_steps,
                                                        num_time_points, fill_value=speed_baseline_mean)

            mean_a2_speed_in_window = np.nanmean(a2_speed_segment_for_g1)
            
            # Scale mean partner speed to influence gaze amplitude
            speed_range = speed_peak_amplitude_mean + 2*speed_peak_amplitude_std - speed_baseline_mean
            # Ensure speed_range is not zero to prevent division by zero
            if speed_range == 0: speed_range = 1e-6 

            speed_deviation_from_baseline_scaled = (mean_a2_speed_in_window - speed_baseline_mean) / speed_range

            # Modulate gaze amplitude based on this scaled speed deviation
            g1_amp_mod = gaze_peak_amplitude_mean + correlation_strength * speed_deviation_from_baseline_scaled * gaze_peak_amplitude_std * 2 # Factor of 2 to enhance effect
            g1_amp = np.random.normal(loc=g1_amp_mod, scale=gaze_peak_amplitude_std * (1-abs(correlation_strength))) # Reduce variance if strong correlation
            g1_amp = max(0, g1_amp)
            _apply_gaussian_burst(gaze1_ts, p1_idx, gaze_pre_pull_offset_s, gaze_burst_std_s, g1_amp, resolution_s, num_time_points)

    # Apply Animal 2's self-gaze, influenced by Animal 1's speed (and new probability constraint)
    for p2_idx in final_pull2_indices:
        if np.random.rand() < prob_self_gaze_pre_self_pull: # Apply probability constraint
            # Get Animal 1's mean speed in the window before Animal 2's pull
            a1_speed_segment_for_g2 = get_aligned_segment(speed1_ts, p2_idx,
                                                        partner_speed_influence_window_pre_steps,
                                                        partner_speed_influence_window_post_steps,
                                                        partner_speed_influence_window_total_steps,
                                                        num_time_points, fill_value=speed_baseline_mean)

            mean_a1_speed_in_window = np.nanmean(a1_speed_segment_for_g2)

            speed_range = speed_peak_amplitude_mean + 2*speed_peak_amplitude_std - speed_baseline_mean
            # Ensure speed_range is not zero to prevent division by zero
            if speed_range == 0: speed_range = 1e-6 

            speed_deviation_from_baseline_scaled = (mean_a1_speed_in_window - speed_baseline_mean) / speed_range

            g2_amp_mod = gaze_peak_amplitude_mean + correlation_strength * speed_deviation_from_baseline_scaled * gaze_peak_amplitude_std * 2
            g2_amp = np.random.normal(loc=g2_amp_mod, scale=gaze_peak_amplitude_std * (1-abs(correlation_strength)))
            g2_amp = max(0, g2_amp)
            _apply_gaussian_burst(gaze2_ts, p2_idx, gaze_pre_pull_offset_s, gaze_burst_std_s, g2_amp, resolution_s, num_time_points)


    # --- Apply gaze after partner pull (with new probability constraint) ---
    # Animal 2's gaze after Animal 1's pull
    for p1_idx in animal1_pull_indices:
        if np.random.rand() < prob_self_gaze_post_partner_pull: # Apply probability constraint
            g2_post_amp = np.random.normal(loc=gaze_post_partner_pull_amplitude_mean, scale=gaze_post_partner_pull_amplitude_std)
            _apply_gaussian_burst(gaze2_ts, p1_idx, gaze_post_partner_pull_offset_s, gaze_post_partner_burst_std_s, max(0, g2_post_amp), resolution_s, num_time_points)

    # Animal 1's gaze after Animal 2's pull
    for p2_idx in final_pull2_indices:
        if np.random.rand() < prob_self_gaze_post_partner_pull: # Apply probability constraint
            g1_post_amp = np.random.normal(loc=gaze_post_partner_pull_amplitude_mean, scale=gaze_post_partner_pull_amplitude_std)
            _apply_gaussian_burst(gaze1_ts, p2_idx, gaze_post_partner_pull_offset_s, gaze_post_partner_burst_std_s, max(0, g1_post_amp), resolution_s, num_time_points)


    # Final clipping to ensure values remain within a reasonable range
    gaze1_ts = np.clip(gaze1_ts, 0, gaze_baseline_mean + gaze_peak_amplitude_mean * 2.5 + gaze_post_partner_pull_amplitude_mean * 2.5)
    gaze2_ts = np.clip(gaze2_ts, 0, gaze_baseline_mean + gaze_peak_amplitude_mean * 2.5 + gaze_post_partner_pull_amplitude_mean * 2.5)
    speed1_ts = np.clip(speed1_ts, 0, speed_baseline_mean + speed_peak_amplitude_mean * 2.5)
    speed2_ts = np.clip(speed2_ts, 0, speed_baseline_mean + speed_peak_amplitude_mean * 2.5)


    summary = {
        "duration_s": duration_s,
        "resolution_s": resolution_s,
        "num_time_points": num_time_points,
        "coop_window_s": coop_window_s,
        "total_pulls_animal1_expected": num_pulls_animal1,
        "total_pulls_animal1_actual": np.sum(pull1_ts),
        "total_pulls_animal2_expected": num_pulls_animal2,
        "total_pulls_animal2_actual": np.sum(pull2_ts),
        "animal2_cooperative_pulls_generated": num_coop_pulls,
        "animal2_independent_pulls_generated": num_independent_pulls,
        "actual_cooperative_pulls_in_window": sum(1 for a2_idx in np.where(pull2_ts == 1)[0] for a1_idx in animal1_pull_indices if abs(a2_idx - a1_idx) <= coop_window_steps),
        "time_points": np.arange(0, duration_s, resolution_s)
    }

    return pull1_ts, pull2_ts, gaze1_ts, gaze2_ts, speed1_ts, speed2_ts, summary
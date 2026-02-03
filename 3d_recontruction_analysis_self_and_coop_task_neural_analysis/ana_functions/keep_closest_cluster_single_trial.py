import numpy as np

# for defining the meaningful social gaze (the continuous gaze distribution that is closest to the pull) 
def keep_closest_cluster_single_trial(trace, time_trace):
    """
    Keep only the contiguous region of non-zero gaze activity that's closest to time = 0.

    Args:
        trace (np.ndarray): 1D array of gaze values over time for one trial.
        time_trace (np.ndarray): 1D array of time values corresponding to trace.

    Returns:
        np.ndarray: Same shape as trace, with only the closest cluster kept.
    """
    # Step 1: Binary mask of non-zero gaze activity
    binary = trace > 0

    # Step 2: Label contiguous clusters of gaze
    labeled, num_features = label(binary)

    if num_features == 0:
        return np.zeros_like(trace)

    # Step 3: Identify cluster whose center is closest to time 0
    closest_id = None
    min_dist = np.inf

    for region_id in range(1, num_features + 1):
        inds = np.where(labeled == region_id)[0]
        region_center_time = np.mean(time_trace[inds])
        dist_to_zero = abs(region_center_time)
        if dist_to_zero < min_dist:
            min_dist = dist_to_zero
            closest_id = region_id

    # Step 4: Keep only the closest cluster
    keep_mask = (labeled == closest_id)
    return trace * keep_mask

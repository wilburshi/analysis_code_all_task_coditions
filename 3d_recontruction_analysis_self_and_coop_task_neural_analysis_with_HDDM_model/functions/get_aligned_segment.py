import numpy as np
import matplotlib.pyplot as plt

def get_aligned_segment(data_ts, center_idx, window_pre_steps, window_post_steps, total_len, num_time_points, fill_value=np.nan):
    """Helper to extract and pad segments. (Duplicate for self-containment)"""
    
    start_idx = center_idx - window_pre_steps
    end_idx = center_idx + window_post_steps + 1
    
    segment = np.full(total_len, fill_value=fill_value)
    
    slice_start = max(0, start_idx)
    slice_end = min(num_time_points, end_idx)
    
    target_start = max(0, -start_idx)
    target_end = target_start + (slice_end - slice_start)
    
    segment[target_start:target_end] = data_ts[slice_start:slice_end]
    return segment
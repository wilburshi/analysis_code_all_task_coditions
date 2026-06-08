# Helper function to apply a single Gaussian burst (defined globally for reusability)
def _apply_gaussian_burst(signal_array, center_idx, offset_s, burst_std_s, amplitude, resolution_s, num_total_time_points):

    import numpy as np
    import matplotlib.pyplot as plt
    
    event_offset_steps = int(offset_s / resolution_s)
    burst_std_steps = burst_std_s / resolution_s
    
    gaussian_center_idx = center_idx + event_offset_steps

    start_burst_idx = int(max(0, gaussian_center_idx - 3 * burst_std_steps))
    end_burst_idx = int(min(num_total_time_points, gaussian_center_idx + 3 * burst_std_steps + 1))
    
    time_points_in_window = np.arange(start_burst_idx, end_burst_idx)
    
    if len(time_points_in_window) > 0:
        kernel = amplitude * np.exp(-0.5 * ((time_points_in_window - gaussian_center_idx) / burst_std_steps)**2)
        # Add the kernel to the signal_array, ensuring we don't go out of bounds on the signal_array
        # Fixed: Changed end_idx to end_burst_idx
        signal_array[start_burst_idx:end_burst_idx] += kernel[:len(signal_array[start_burst_idx:end_burst_idx])]
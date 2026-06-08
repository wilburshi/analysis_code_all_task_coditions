def compute_stats(data_array):
    """Compute mean and standard error of the mean (SEM)."""
    
    import numpy as np
    import matplotlib.pyplot as plt

    if data_array.shape[0] == 0:
        return np.full(data_array.shape[1] if data_array.ndim > 1 else 1, np.nan), np.full(data_array.shape[1] if data_array.ndim > 1 else 1, np.nan)
    mean = np.nanmean(data_array, axis=0)
    sem = np.nanstd(data_array, axis=0) / np.sqrt(data_array.shape[0])
    return mean, sem
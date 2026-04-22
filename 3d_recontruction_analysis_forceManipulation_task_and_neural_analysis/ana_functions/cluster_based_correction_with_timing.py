import pandas as pd
import numpy as np

from scipy.ndimage import label


def cluster_based_correction_with_timing(real_mean, shuffled_coefs, alpha_form, alpha_sig, time_axis=None):
    n_boot, n_vars, n_basis = shuffled_coefs.shape
    cluster_significance = np.zeros(n_vars, dtype=bool)
    cluster_timing = ['None'] * n_vars

    for var in range(n_vars):
        real_coef = real_mean[var, :]
        shuf_coef = shuffled_coefs[:, var, :]

        # Empirical p-values
        p_vals = (np.sum(np.abs(shuf_coef) >= np.abs(real_coef), axis=0) + 1) / (n_boot + 1)
        sig_mask = p_vals < alpha_form

        # Cluster label
        labeled_array, n_clusters = label(sig_mask)
        real_cluster_sizes = [
            np.sum(labeled_array == cluster_idx + 1)
            for cluster_idx in range(n_clusters)
        ]
        max_real_cluster = np.max(real_cluster_sizes) if real_cluster_sizes else 0

        # Compute max cluster size in each shuffled iteration
        shuf_max_clusters = []
        for b in range(n_boot):
            others = np.delete(shuf_coef, b, axis=0)
            p_vals_shuf = np.mean(np.abs(others) >= np.abs(shuf_coef[b, :]), axis=0)
            sig_shuf = p_vals_shuf < alpha_form
            lbl_shuf, n_lbl = label(sig_shuf)
            max_cluster = max([np.sum(lbl_shuf == i + 1) for i in range(n_lbl)], default=0)
            shuf_max_clusters.append(max_cluster)

        cluster_thresh = np.percentile(shuf_max_clusters, 100 * (1 - alpha_sig))
        cluster_significance[var] = max_real_cluster > cluster_thresh

        # Dominant timing classification
        if cluster_significance[var] and time_axis is not None:
            sig_times = time_axis[sig_mask]
            n_before = np.sum(sig_times < 0)
            n_after = np.sum(sig_times > 0)

            if n_before > n_after:
                cluster_timing[var] = 'Reactive'
            elif n_after > n_before:
                cluster_timing[var] = 'Predictive'
            else:
                cluster_timing[var] = 'Both' if (n_before > 0) else 'None'

    return cluster_significance, cluster_timing

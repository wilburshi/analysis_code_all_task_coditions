# spike analysis function

def spike_analysis_FR_calculation(fs_spikes, FR_kernel, totalsess_time, spike_clusters_data, spike_time_data):
    
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KernelDensity
    import scipy
    import scipy.stats as st
 
    clusters_unique = np.unique(spike_clusters_data)
    nclusters = np.shape(clusters_unique)[0]

    spike_time_allclusters = dict.fromkeys(np.char.mod('%d', clusters_unique),[])
    FR_allclusters = dict.fromkeys(np.char.mod('%d', clusters_unique),[])
    FR_zscore_allclusters = dict.fromkeys(np.char.mod('%d', clusters_unique),[])

    for icluster in np.arange(0,nclusters,1):

        ind = spike_clusters_data == clusters_unique[icluster]

        # spike time stamps
        spike_time_icluster = spike_time_data[ind]
        spike_time_allclusters[str(clusters_unique[icluster])] = spike_time_icluster/fs_spikes # change the unit to second

        # firing rate
        xxx = spike_time_icluster
        xxx_plot = np.linspace(0, totalsess_time*fs_spikes, int(totalsess_time/FR_kernel))
        # xxx_plot = np.linspace(0, totalsess_time*fs_spikes, int(totalsess_time*fs_spikes/200))
        kde = KernelDensity(kernel="gaussian", bandwidth=FR_kernel*fs_spikes).fit(xxx.reshape(-1, 1)) # 100ms bandwith gaussian kernel
        log_dens = kde.score_samples(xxx_plot.reshape(-1, 1))
        FR_icluster = np.exp(log_dens)*fs_spikes
        
        #
        FR_allclusters[str(clusters_unique[icluster])] = FR_icluster
        FR_zscore_allclusters[str(clusters_unique[icluster])] = st.zscore(FR_icluster)
        
    FR_timepoint_allclusters = xxx_plot/fs_spikes
    # FR_timepoint_allclusters = xxx_plot
    
    return spike_time_allclusters,FR_timepoint_allclusters,FR_allclusters,FR_zscore_allclusters
        

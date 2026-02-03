def cross_correlation_function_for_bhv_variables(time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, block_starttime, block_endtime, lag_tgt):  

    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import ccf

    # Sampling rate
    sample_rate = 0.1  # in seconds
    total_duration = block_endtime-block_starttime  # total duration in seconds

    # Determine the number of samples
    num_samples = int(total_duration / sample_rate) + 1

    # change time for pull and gaze
    pull_times1 = np.array(time_point_pull1)-block_starttime  
    pull_times2 = np.array(time_point_pull2)-block_starttime 
    #
    gaze_times1 = np.unique(np.sort(np.hstack((oneway_gaze1,mutual_gaze1))))
    gaze_times2 = np.unique(np.sort(np.hstack((oneway_gaze2,mutual_gaze2))))
    gaze_times1 = np.array(gaze_times1)-block_starttime  
    gaze_times2 = np.array(gaze_times2)-block_starttime  
    #
    # 
    pull_times1 = pull_times1[(pull_times1<total_duration)&(pull_times1>=0)]
    pull_times2 = pull_times2[(pull_times2<total_duration)&(pull_times2>=0)]
    #
    gaze_times1 = gaze_times1[(gaze_times1<total_duration)&(gaze_times1>=0)]
    gaze_times2 = gaze_times2[(gaze_times2<total_duration)&(gaze_times2>=0)]
    # 
    timeseries_all = {'pull1':pull_times1,
                      'pull2':pull_times2,
                      'gaze1':gaze_times1,
                      'gaze2':gaze_times2,
                      }

    #
    center_bhvs = ['pull1','pull2','gaze1','gaze2']
    ncenterbhv = np.shape(center_bhvs)[0]
    #
    ccf_bhvs = [['pull1','pull2','gaze1','gaze2'],
                ['pull1','pull2','gaze1','gaze2'],
                ['pull1','pull2','gaze1','gaze2'],
                ['pull1','pull2','gaze1','gaze2'],]
    #
    ccf_summary = dict.fromkeys(center_bhvs)

    for icenterbhv in np.arange(0,ncenterbhv,1):
        center_bhv = center_bhvs[icenterbhv]

        ccf_bhvs_icenter = ccf_bhvs[icenterbhv]
        nccfbhvs = np.shape(ccf_bhvs_icenter)[0]

        ccf_summary[center_bhv] = dict.fromkeys(ccf_bhvs_icenter)

        for iccfbhv in np.arange(0,nccfbhvs,1):

            ccf_bhv = ccf_bhvs_icenter[iccfbhv]

            # get the time series data
            timeseries1 = timeseries_all[center_bhv]
            timeseries2 = timeseries_all[ccf_bhv]

            # Create binary (0,1) time series
            binary_series1 = np.zeros(num_samples)
            binary_series2 = np.zeros(num_samples)

            # Mark the pull events in the binary series
            indices1 = (timeseries1 / sample_rate).astype(int)
            indices2 = (timeseries2 / sample_rate).astype(int)
            binary_series1[indices1] = 1
            binary_series2[indices2] = 1

            # Calculate the cross-correlation function for binary time series
            # Since ccf only returns positive lags, calculate both positive and negative lags manually
            cross_corr = np.correlate(binary_series1 - np.mean(binary_series1),
                                      binary_series2 - np.mean(binary_series2), mode='full')
            # Normalize the cross-correlation
            norm_cross_corr = cross_corr / (np.std(binary_series1) * np.std(binary_series2) * num_samples)


            lags = np.arange(-len(binary_series1) + 1, len(binary_series1))

            # Convert lags from samples to seconds
            lags_seconds = lags * sample_rate

            # Mask for -4 to 4 seconds
            lag_mask = (lags_seconds >= -lag_tgt) & (lags_seconds <= lag_tgt)
            filtered_lags_seconds = lags_seconds[lag_mask]
            filtered_cross_corr = norm_cross_corr[lag_mask]

            ccf_summary[center_bhv][ccf_bhv] = filtered_cross_corr


    return ccf_summary, filtered_lags_seconds

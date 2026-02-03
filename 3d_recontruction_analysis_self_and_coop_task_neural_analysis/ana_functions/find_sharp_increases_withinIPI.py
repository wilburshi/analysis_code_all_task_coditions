# method 1: find the closest speed peak before pull and trace back to find when it starts
def find_sharp_increases_withinIPI(pull_data, speed_data, session_start_time, fps):

    import numpy as np
    from scipy.signal import argrelextrema

    sharp_increase_frames = np.zeros_like(speed_data)
    session_start_frame = int(np.round(session_start_time * fps))
    pull_frames = np.where(pull_data)[0]
    npulls = len(pull_frames)

    for ipull in range(npulls):
        pull_frame = pull_frames[ipull]
        pullstart_frame = session_start_frame if ipull == 0 else pull_frames[ipull - 1]

        try:
            speed_IPI_tgt = speed_data[pullstart_frame:pull_frame-int(1*fps)] # allows a 1000ms error time window
        except:
            speed_IPI_tgt = speed_data[pullstart_frame:pull_frame-int(0.5*fps)] # allows a 500ms error time window

        onset_global_idx = pullstart_frame  # default fallback

        if len(speed_IPI_tgt) >= 3:
            # Find local maxima
            local_max_idx = argrelextrema(speed_IPI_tgt, np.greater_equal)[0]

            if len(local_max_idx) > 0:
                # Get the one closest to the current pull (end of interval)
                peak_idx = local_max_idx[np.argmax(local_max_idx)]

                # Go backward from the peak to find where it started rising
                for j in range(peak_idx - 1, 0, -1):
                    if speed_IPI_tgt[j] < speed_IPI_tgt[j - 1]:
                        onset_local = j
                        onset_global_idx = pullstart_frame + onset_local
                        break

        sharp_increase_frames[onset_global_idx] = 1

    return sharp_increase_frames

# take both speed and angle speed 
def find_sharp_increases_withinIPI_dual_speed(pull_data, speed_data, anglespeed_data, session_start_time, fps):
    import numpy as np
    from scipy.signal import argrelextrema

    sharp_increase_frames = np.zeros_like(speed_data)
    session_start_frame = int(np.round(session_start_time * fps))
    pull_frames = np.where(pull_data)[0]
    npulls = len(pull_frames)

    for ipull in range(npulls):
        pull_frame = pull_frames[ipull]
        pullstart_frame = session_start_frame if ipull == 0 else pull_frames[ipull - 1]

        try:
            speed_segment = speed_data[pullstart_frame:pull_frame-int(1*fps)] # allows a 1000ms error time window
            anglespeed_segment = anglespeed_data[pullstart_frame:pull_frame-int(1*fps)] # allows a 1000ms error time window

        except:
            speed_segment = speed_data[pullstart_frame:pull_frame-int(0.5*fps)] # allows a 500ms error time window
            anglespeed_segment = anglespeed_data[pullstart_frame:pull_frame-int(0.5*fps)] # allows a 500ms error time window

        def find_onset(segment):
            onset = pullstart_frame  # default
            if len(segment) >= 3:
                local_max_idx = argrelextrema(segment, np.greater_equal)[0]
                if len(local_max_idx) > 0:
                    peak_idx = local_max_idx[np.argmax(local_max_idx)]
                    for j in range(peak_idx - 1, 0, -1):
                        if segment[j] < segment[j - 1]:
                            return pullstart_frame + j
            return onset

        onset_speed = find_onset(speed_segment)
        onset_anglespeed = find_onset(anglespeed_segment)

        # Take the earlier one
        final_onset = min(onset_speed, onset_anglespeed)
        sharp_increase_frames[final_onset] = 1

    return sharp_increase_frames

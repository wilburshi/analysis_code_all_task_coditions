# method 2: find the lowest timepoint then the increase point as the pull onset
def find_rising_onset_after_min_withinIPI(pull_data, speed_data, session_start_time, fps):
    
    import numpy as np

    sharp_onset_frames = np.zeros_like(speed_data)
    session_start_frame = int(np.round(session_start_time * fps))
    pull_frames = np.where(pull_data)[0]
    npulls = len(pull_frames)

    for ipull in range(npulls):
        pull_frame = pull_frames[ipull]
        pullstart_frame = session_start_frame if ipull == 0 else pull_frames[ipull - 1]

        try:
            speed_IPI_tgt = speed_data[pullstart_frame:pull_frame - int(1 * fps)]  # 1000ms window
        except:
            speed_IPI_tgt = speed_data[pullstart_frame:pull_frame - int(0.5 * fps)]  # fallback

        onset_global_idx = pullstart_frame  # fallback default

        if len(speed_IPI_tgt) >= 3:
            # Find the minimum point in the IPI
            min_idx = np.argmin(speed_IPI_tgt)

            # Go forward from min to find first clear rise
            for j in range(min_idx + 1, len(speed_IPI_tgt) - 1):
                if speed_IPI_tgt[j] > speed_IPI_tgt[j - 1]:
                    onset_local = j
                    onset_global_idx = pullstart_frame + onset_local
                    break

        sharp_onset_frames[onset_global_idx] = 1

    return sharp_onset_frames

#
def find_rising_onset_after_min_dual_speed(pull_data, speed_data, anglespeed_data, session_start_time, fps):
    
    import numpy as np

    sharp_onset_frames = np.zeros_like(speed_data)
    session_start_frame = int(np.round(session_start_time * fps))
    pull_frames = np.where(pull_data)[0]
    npulls = len(pull_frames)

    for ipull in range(npulls):
        pull_frame = pull_frames[ipull]
        pullstart_frame = session_start_frame if ipull == 0 else pull_frames[ipull - 1]

        try:
            speed_segment = speed_data[pullstart_frame:pull_frame - int(1 * fps)]
            anglespeed_segment = anglespeed_data[pullstart_frame:pull_frame - int(1 * fps)]
        except:
            speed_segment = speed_data[pullstart_frame:pull_frame - int(0.5 * fps)]
            anglespeed_segment = anglespeed_data[pullstart_frame:pull_frame - int(0.5 * fps)]

        def find_rising_onset(segment):
            if len(segment) >= 3:
                min_idx = np.argmin(segment)
                for j in range(min_idx + 1, len(segment) - 1):
                    if segment[j] > segment[j - 1]:
                        return pullstart_frame + j
            return pullstart_frame

        onset_speed = find_rising_onset(speed_segment)
        onset_anglespeed = find_rising_onset(anglespeed_segment)

        final_onset = min(onset_speed, onset_anglespeed)
        sharp_onset_frames[final_onset] = 1

    return sharp_onset_frames

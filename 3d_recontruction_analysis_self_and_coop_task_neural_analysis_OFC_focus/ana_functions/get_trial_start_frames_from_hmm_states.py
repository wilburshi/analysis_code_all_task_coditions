# methods 3: based on the HMM fitting
from hmmlearn import hmm
from scipy.stats import zscore
import numpy as np

def get_trial_start_frames_from_hmm_states(hmm_states, pull1_data, fps, session_start_time):
    """
    Determine trial start frames based on HMM state changes and pull events.

    Parameters:
    - hmm_states: 1D numpy array of inferred HMM states at each time frame
    - pull1_data: 1D boolean numpy array, True where pull occurs (frame-based)
    - fps: frames per second
    - session_start_time: float, session start time in seconds

    Returns:
    - trial_start_frames: list of ints (frame indices)
    """

    pull_frames = np.where(pull1_data)[0]
    trial_start_frames = []

    session_start_idx = int(session_start_time * fps)
    initial_state = hmm_states[session_start_idx]

    # First trial: use first HMM state change after session start
    for t in range(session_start_idx + 1, len(hmm_states)):
        if hmm_states[t] != initial_state:
            trial_start_frames.append(t)
            break
    else:
        # Fallback: if no change found, use session start
        trial_start_frames.append(session_start_idx)

    # Subsequent trials: first HMM state change after each pull
    for i in range(len(pull_frames) - 1):
        current_pull_idx = pull_frames[i]
        current_state = hmm_states[current_pull_idx]

        # Look for next state change after the pull
        for t in range(current_pull_idx + 1, len(hmm_states)):
            if hmm_states[t] != current_state:
                trial_start_frames.append(t)
                break
        else:
            # Fallback: use next pull frame if no state change found
            trial_start_frames.append(pull_frames[i + 1])

    return trial_start_frames



def get_trial_start_frames_from_HMM(speed_data, anglespeed_data, pull_data, fps, session_start_time, n_states):
    """
    Use Gaussian HMM on speed and angle speed to infer state transitions and define trial starts.

    Parameters:
    - speed_data: array-like, 1D speed trace
    - anglespeed_data: array-like, 1D angle speed trace
    - pull_data: array-like, 1D binary pull events (frame-based)
    - fps: float, sampling rate in Hz
    - session_start_time: float, start of the session in seconds
    - n_states: int, number of HMM states to infer (default=3)

    Returns:
    - trial_start_frames: list of int, frame indices for inferred trial starts
    - hmm_states: array of HMM-inferred hidden state per frame (same length as cropped data)
    """

    # Convert inputs to numpy arrays
    speed_data = np.array(speed_data)
    anglespeed_data = np.array(anglespeed_data)
    pull_data = np.array(pull_data)

    # Define cropping window: 5s before first pull to 5s after last pull (if available)
    pull_frames = np.where(pull_data)[0]

    if len(pull_frames) == 0:
        start_idx = 0
        end_idx = len(pull_data)
    else:
        first_pull = pull_frames[0]
        last_pull = pull_frames[-1]
        start_idx = 0  # or max(0, first_pull - 5*fps)
        end_idx = min(len(pull_data), last_pull + int(5 * fps))

    # Crop data
    speed_crop = speed_data[start_idx:end_idx]
    angle_crop = anglespeed_data[start_idx:end_idx]
    pull_crop = pull_data[start_idx:end_idx]

    # replace nan with 0
    speed_crop = np.nan_to_num(speed_crop, nan=0.0)
    angle_crop = np.nan_to_num(angle_crop, nan=0.0)

    # Z-score normalization
    speed_z = zscore(speed_crop)
    angle_z = zscore(angle_crop)

    # Stack for HMM
    obs = np.column_stack([speed_z, angle_z, pull_crop])

    # Fit HMM
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(obs)
    hmm_states = model.predict(obs)

    # Infer trial starts using the HMM state changes
    trial_start_frames = get_trial_start_frames_from_hmm_states(hmm_states, pull_crop, fps, session_start_time)

    # Adjust back to full-session frame indices
    trial_start_frames = [f + start_idx for f in trial_start_frames]

    return trial_start_frames, hmm_states

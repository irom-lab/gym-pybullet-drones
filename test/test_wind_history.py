import numpy as np
from collections import deque


# == frame stacking and skipping ==
def get_frames(prev_obs, traj_size, frame_skip):
    """
    Assume at least one obs in prev_obs - reasonable since we can always get the initial obs, and then get frames.
    """
    traj_cover = (traj_size-1) * frame_skip + traj_size
    default_frame_seq = np.arange(0, traj_cover, (frame_skip + 1))

    if len(prev_obs) == 1:
        seq = np.zeros((traj_size), dtype='int')
    elif len(prev_obs) < traj_cover:  # always pick zero (most recent one)
        seq_random = np.random.choice(
            np.arange(1, len(prev_obs)), traj_size - 1, replace=True
        )
        seq_random = np.sort(seq_random)  # ascending
        seq = np.insert(seq_random, 0, 0)  # add to front, then flipped
    else:
        seq = default_frame_seq
    seq = np.flip(seq)  #! since prev_obs appends left
    obs_stack = np.concatenate([prev_obs[obs_ind] for obs_ind in seq])
    return obs_stack


# Cfg
filter_step = 10    # rolling max step
wind_num_frame = 5
wind_frame_skip = 2
max_wind = 1000
wind_frame_cover = (wind_num_frame - 1) * wind_frame_skip + wind_num_frame  # maxlen in deque; if fewer, randomly sample

# Initialize wind observation history - this is filtered
wind_obs_frames = deque(maxlen=wind_frame_cover)

# Save all true wind
wind_all = []

# Run - assume control and wind observation run at the same frequency
for i in range(1000):

    # Monotonically increasing wind
    wind_current = i

    # Rolling max filter - this simplifies to taking current wind in this case... since wind monotonically increasing
    wind_all += [wind_current]
    wind_obs_filtered = max(wind_all[-filter_step:])

    # Add current wind to history
    wind_obs_frames.appendleft([wind_obs_filtered, 0, 0])

    # Get wind raw observation
    wind_obs_current = get_frames(wind_obs_frames, 
                                  wind_num_frame, 
                                  wind_frame_skip)

    # Normalize
    wind_obs_current = np.clip(wind_obs_current/max_wind, -1, 1)

    # Debug
    if i % 80 == 0 or i < 10:
        print('Step {}: wind observation is {}'.format(i, wind_obs_current))

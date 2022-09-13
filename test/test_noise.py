import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12


rng = np.random.default_rng()

#
t_bottom_low = 1
t_bottom_high = 7
slope_low = 5
slope_high = 10
vel_bottom_low = 0
vel_bottom_high = 1
vel_top_low = 4.5
vel_top_high = 5.5
#
vel_std_bottom = 0.1
vel_std_slope = 0.2
vel_std_dip = 0.2
vel_std_top = 0.4
# sensor_filter_ratio = 0.5
# sensor_std = 0.05
# add one dip for each trial
dip_period_low = 1
dip_period_high = 2
vel_dip_low = 0.
vel_dip_high = 2.5

t_total = 10
freq = 40
rolling_period = 0.1
rolling_step = int(rolling_period*freq)

fig, axarr = plt.subplots(2, 4, figsize=(16, 8))
for trial in range(8):

    t_bottom = rng.random() * (t_bottom_high - t_bottom_low) + t_bottom_low
    slope = rng.random() * (slope_high - slope_low) + slope_low
    vel_bottom = rng.random() * (vel_bottom_high - vel_bottom_low) + vel_bottom_low
    vel_top = rng.random() * (vel_top_high - vel_top_low) + vel_top_low
    rise_time = (vel_top - vel_bottom) / slope
    t_top = t_bottom + rise_time   

    dip_period = rng.random() * (dip_period_high - dip_period_low) + dip_period_low
    t_dip_start = rng.uniform(t_top, t_total)
    t_dip_end = rng.uniform(t_dip_start, t_dip_start+dip_period)

    vel_dip = rng.random() * (vel_dip_high - vel_dip_low) + vel_dip_low

    t_all = np.linspace(0, t_total, t_total*freq)
    vel_all = []
    sensor_all = []
    for t in t_all:
        if t < t_bottom:
            true_noise = -rng.gumbel(0, vel_std_bottom)
            vel = vel_bottom + true_noise
            # sensor = vel_bottom + true_noise * sensor_filter_ratio + rng.normal(0, sensor_std)
        elif t < t_top:
            true_noise = -rng.gumbel(0, vel_std_slope)
            vel = (t-t_bottom)*slope + vel_bottom + true_noise
            # sensor = (t-t_bottom)*slope + vel_bottom  + true_noise * sensor_filter_ratio + rng.normal(0, sensor_std)
        elif t < t_dip_end and t > t_dip_start:
            true_noise = -rng.gumbel(0, vel_std_dip)
            vel = vel_dip + true_noise
        else:
            true_noise = -rng.gumbel(0, vel_std_top)
            vel = vel_top + true_noise
            # sensor = vel_top + true_noise * sensor_filter_ratio + rng.normal(0, sensor_std)
        vel_all += [max(0, vel)]
        # sensor_all += [max(0, sensor)]
        sensor_all += [max(vel_all[-rolling_step:])]

    trial_x = int(trial / 4)
    trial_y = int(trial % 4)
    print(trial_x, trial_y)
    axarr[trial_x, trial_y].plot(t_all, vel_all, label='simulated wind')
    axarr[trial_x, trial_y].plot(t_all, sensor_all, label='simulated sensor')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
# plt.show()
plt.savefig('test.png')

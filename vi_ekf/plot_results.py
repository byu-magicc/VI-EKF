#!/usr/bin/env python

import os
import numpy as np
from data import History
from plot_helper import plot_side_by_side, init_plots
from tqdm import tqdm
import scipy.signal
from pyquat import Quaternion

log_dir = os.path.dirname(os.path.realpath(__file__)) + "/../logs/"
log_folders =  [int(name) for name in os.listdir(log_dir) if os.path.isdir(log_dir + name)]

latest_folder = max(log_folders)

prop_file = open(log_dir + str(latest_folder) + "/prop.txt")
perf_file = open(log_dir + str(latest_folder) + "/perf.txt")
meas_file = open(log_dir + str(latest_folder) + "/meas.txt")
conf_file = open(log_dir + str(latest_folder) + "/conf.txt")

conf = dict()
for line in conf_file:
    key = line.split(":")[0]
    item = line.split(":")[1]
    if key == 'Test Num' or key == 'Using Drag Term':
        conf[key] = line.split(":")[1]
    else:
        conf[key] = np.array([[float(it) for it in item.split()]])

h = History()
len_prop_file = 0
for line in prop_file:
    line_arr = np.array([float(item) for item in line.split()])
    if len_prop_file == 0: len_prop_file = len(line_arr)
    if len(line_arr) < len_prop_file: continue
    num_features = (len(line_arr) - 34) / 8
    X = 1
    COV = 1 + 17 + 5*num_features
    t = line_arr[0]
    h.store(t, xhat=line_arr[1:18], cov=np.diag(line_arr[COV:]), num_features=num_features)

for line in perf_file:
    line_arr = np.array([float(item) for item in line.split()])
    t = line_arr[0]
    if len(line_arr) == 12:
        h.store(t, prop_time=line_arr[1], acc_time=line_arr[2], pos_time=line_arr[5], feat_time=line_arr[8], depth_time=line_arr[10])

ids = []
for line in meas_file:
    meas_type = line.split()[0]
    line_arr = np.array([float(item) for item in line.split()[2:]])
    t = float(line.split()[1])

    if meas_type == 'ACC':
        if len(line_arr) < 5: continue
        h.store(t, acc=line_arr[0:2], acc_hat=line_arr[2:4])
    elif meas_type == 'ATT':
        if len(line_arr) < 8: continue
        h.store(t, att=line_arr[0:4], att_hat=line_arr[4:8])
    elif meas_type == 'POS':
        if len(line_arr) < 7: continue
        h.store(t, pos=line_arr[0:3], pos_hat=line_arr[3:6])
    elif meas_type == 'FEAT':
        if len(line_arr) < 5: continue
        id = line_arr[4]
        h.store(t, line_arr[4], feat=line_arr[0:2], feat_hat=line_arr[2:4])
        ids.append(id) if id not in ids else None
    elif meas_type == 'DEPTH':
        if len(line_arr) < 4: continue
        # Invert the covariance measurement
        p = 1.0/line_arr[1]
        s = line_arr[2]
        cov = 1./(p+s) - 1./p
        h.store(t, line_arr[3], depth=line_arr[0], depth_hat=line_arr[1], depth_cov=[[cov   ]])
    elif meas_type == 'ALT':
        if len(line_arr) < 3: continue
        h.store(t, line_arr[2], alt=line_arr[0], alt_hat=line_arr[1])
    else:
        print("unsupported measurement type ", meas_type)


h.tonumpy()

# Calculate body-fixed velocity by differentiating position and rotating
# into the body frame
b, a = scipy.signal.butter(8, 0.03)  # Create a Butterworth Filter
# differentiate Position
delta_t = np.diff(h.t.pos)
good_ids = delta_t != 0
delta_t = delta_t[good_ids]
v_t = h.t.pos[np.hstack((good_ids, False))]
delta_x = np.diff(h.pos, axis=0)
delta_x = delta_x[good_ids]
unfiltered_inertial_velocity = np.vstack((np.zeros((1, 3)), delta_x / delta_t[:, None]))
# Filter
v_inertial = scipy.signal.filtfilt(b, a, unfiltered_inertial_velocity, axis=0)
# Rotate into Body Frame
vel_data = []
try:
    att = h.att[np.hstack((good_ids))]
except:
    att = h.att[np.hstack((good_ids, False))]
for i in range(len(v_t)):
    q_I_b = Quaternion(att[i, :, None])
    vel_data.append(q_I_b.rot(v_inertial[i, None].T).T)

vel_data = np.array(vel_data).squeeze()


start = h.t.xhat[0]
end = h.t.xhat[-1]
fig_dir = os.path.dirname(os.path.realpath(__file__)) + "/../ROSplots/"

init_plots(start, end, fig_dir)

plot_cov = True
pose_cov = True

plot_side_by_side('x_pos', 0, 3, h.t.xhat, h.xhat, cov=h.cov if pose_cov else None, truth_t=h.t.pos, truth=h.pos, labels=['x', 'y', 'z'], start_t=start, end_t=end)
plot_side_by_side('x_vel', 3, 6, h.t.xhat, h.xhat, cov=h.cov if plot_cov else None, truth_t=v_t, truth=vel_data, labels=['x', 'y', 'z'], start_t=start, end_t=end)
plot_side_by_side('x_att', 6, 10, h.t.xhat, h.xhat, cov=None, truth_t=h.t.att, truth=h.att, labels=['w','x', 'y', 'z'], start_t=start, end_t=end)
plot_side_by_side('z_acc', 0, 2, h.t.acc, h.acc, labels=['x', 'y'], start_t=start, end_t=end)
plot_side_by_side('bias', 10, 17, h.t.xhat, h.xhat, labels=['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mu'], start_t=start, end_t=end, cov=h.cov if plot_cov else None, cov_bounds=(9,16))


for i in tqdm(ids):
    if i not in h.depth_hat: continue
    plot_side_by_side('lambda/x_{}'.format(i), 0, 2, h.t.feat_hat[i], h.feat_hat[i], truth_t=h.t.feat[i],
                      truth=h.feat[i], labels=['u', 'v'], start_t=start, end_t=end)
    plot_side_by_side('rho/x_{}'.format(i), 0, 1, h.t.depth_hat[i], h.depth_hat[i][:, None], truth_t=h.t.depth[i],
                      truth=h.depth[i][:, None], labels=[r'$\frac{1}{\rho}$'], start_t=start, end_t=end,
                      cov=h.depth_cov[i] if plot_cov else None)







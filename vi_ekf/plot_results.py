#!/usr/bin/env python

import os
import numpy as np
from data import History
from plot_helper import plot_side_by_side, init_plots
from tqdm import tqdm
import scipy.signal
from pyquat import Quaternion

# Shift truth timestamp
# offset = -0.35
offset = 0.0

plot_cov = True
pose_cov = False

log_dir = os.path.dirname(os.path.realpath(__file__)) + "/../logs/"
log_folders =  [int(name) for name in os.listdir(log_dir) if os.path.isdir(log_dir + name)]

latest_folder = max(log_folders)

prop_file = open(log_dir + str(latest_folder) + "/prop.txt")
perf_file = open(log_dir + str(latest_folder) + "/perf.txt")
meas_file = open(log_dir + str(latest_folder) + "/meas.txt")
conf_file = open(log_dir + str(latest_folder) + "/conf.txt")
fig_dir = log_dir + str(latest_folder) + "/plots"
os.system("mkdir " + fig_dir)

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
    try:
        line_arr = np.array([float(item) for item in line.split()[2:]])
        t = float(line.split()[1])
    except:
        pass


    if meas_type == 'ACC':
        if len(line_arr) < 5: continue
        h.store(t, acc=line_arr[0:2], acc_hat=line_arr[2:4])
    elif meas_type == 'ATT':
        if len(line_arr) < 8: continue
        h.store(t, att=line_arr[0:4], att_hat=line_arr[4:8])
    elif meas_type == 'GLOBAL_ATT':
        if len(line_arr) < 8: continue
        h.store(t, global_att=line_arr[0:4], global_att_hat=line_arr[4:8])
    elif meas_type == 'POS':
        if len(line_arr) < 7: continue
        h.store(t, pos=line_arr[0:3], pos_hat=line_arr[3:6])
    elif meas_type == 'GLOBAL_POS':
        if len(line_arr) < 6: continue
        h.store(t, global_pos=line_arr[0:3], global_pos_hat=line_arr[3:6])
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
        h.store(t, alt=line_arr[0], alt_hat=line_arr[1])
    else:
        print("unsupported measurement type ", meas_type)


h.tonumpy()

# Calculate body-fixed velocity by differentiating position and rotating
# into the body frame
delta_t = np.diff(h.t.global_pos)
good_ids = delta_t > 0.001 # only take truth measurements with a reasonable time difference
delta_t = delta_t[good_ids]
v_t = h.t.global_pos[np.hstack((good_ids, False))]
delta_x = np.diff(h.global_pos, axis=0)
delta_x = delta_x[good_ids]
unfiltered_inertial_velocity = np.vstack(delta_x / delta_t[:, None]) # Differentiate Truth
b_vel, a_vel = scipy.signal.butter(3, 0.50) # Smooth Differentiated Truth
v_inertial = scipy.signal.filtfilt(b_vel, a_vel, unfiltered_inertial_velocity, axis=0)

# Rotate into Body Frame
vel_data = []
try:
    att = h.global_att[np.hstack((good_ids))]
except:
    att = h.global_att[np.hstack((good_ids, False))]
for i in range(len(v_t)):
    q_I_b = Quaternion(att[i, :, None])
    vel_data.append(q_I_b.invrot(v_inertial[i, None].T).T)

vel_data = np.array(vel_data).squeeze()


# Create Plots
start = h.t.xhat[0]
end = h.t.xhat[-1]
init_plots(start, end, fig_dir)

plot_side_by_side(r'$p_{b/n}^n$', 0, 3, h.t.xhat, h.xhat, cov=h.cov if pose_cov else None, truth_t=h.t.pos, truth=h.pos, labels=['p_x', 'p_y', 'p_z'], start_t=start, end_t=end, truth_offset=offset)
plot_side_by_side(r'$p_{b/I}^I$', 0, 3, h.t.global_pos_hat, h.global_pos_hat, cov=h.cov if pose_cov else None, truth_t=h.t.global_pos, truth=h.global_pos, labels=['p_x', 'p_y', 'p_z'], start_t=start, end_t=end, truth_offset=offset)
plot_side_by_side(r'$v_{b/I}^b$', 3, 6, h.t.xhat, h.xhat, cov=h.cov if plot_cov else None, truth_t=v_t, truth=vel_data, labels=['v_x', 'v_y', 'v_z'], start_t=start, end_t=end, truth_offset=offset)
plot_side_by_side(r'$q_n^b$', 6, 10, h.t.xhat, h.xhat, cov=None, truth_t=h.t.att, truth=h.att, labels=['q_w','q_x', 'q_y', 'q_z'], start_t=start, end_t=end, truth_offset=offset)
plot_side_by_side(r'$q_I^b$', 0, 4, h.t.global_att_hat, h.global_att_hat, cov=None, truth_t=h.t.global_att, truth=h.global_att, labels=['q_w','q_x', 'q_y', 'q_z'], start_t=start, end_t=end, truth_offset=offset)
# Convert relative attitude to euler angles
true_euler, est_euler = np.zeros((len(h.att),3)), np.zeros((len(h.xhat),3))
for i, true_quat in enumerate(h.att): true_euler[i,:,None] = Quaternion(true_quat[:,None]).euler
for i, est_quat in enumerate(h.xhat[:,6:10]): est_euler[i,:,None] = (Quaternion(est_quat[:,None]).euler)
plot_side_by_side('relative_euler', 0, 3, h.t.xhat, est_euler, truth_t=h.t.att, truth=true_euler, start_t=start, end_t=end, labels=[r'\phi', r'\theta', r'\psi'], truth_offset=offset)
# Convert global attitude to euler angles
true_euler, est_euler = np.zeros((len(h.global_att),3)), np.zeros((len(h.global_att_hat),3))
for i, true_quat in enumerate(h.global_att): true_euler[i,:,None] = Quaternion(true_quat[:,None]).euler
for i, est_quat in enumerate(h.global_att_hat): est_euler[i,:,None] = (Quaternion(est_quat[:,None]).euler)
plot_side_by_side('global_euler', 0, 3, h.t.global_att_hat, est_euler, truth_t=h.t.global_att, truth=true_euler, start_t=start, end_t=end, labels=[r'\phi', r'\theta', r'\psi'], truth_offset=offset)
# Filter accelerometer
b_acc, a_acc = scipy.signal.butter(6, 0.05)
acc_smooth = scipy.signal.filtfilt(b_acc, a_acc, h.acc, axis=0)
plot_side_by_side('$y_{a}$', 0, 2, h.t.acc, acc_smooth, truth=h.acc, truth_t=h.t.acc, labels=[r'y_{a,x}', r'y_{a,y}'], start_t=start, end_t=end, truth_offset=offset)
plot_side_by_side('$y_{alt}$', 0, 1, h.t.alt, h.alt_hat[:,None], truth=h.alt[:,None], truth_t=h.t.alt, labels=[r'-p_z'], start_t=start, end_t=end)
plot_side_by_side('Bias Terms', 10, 17, h.t.xhat, h.xhat, labels=[r'\beta_{a,x}', r'\beta_{a,y}', r'\beta_{a,z}', r'\beta_{\omega,x}', r'\beta_{\omega,y}', r'\beta_{\omega,z}', 'b'], start_t=start, end_t=end, cov=h.cov if plot_cov else None, cov_bounds=(9,16), truth_offset=offset)


# print Final States for baiases for tuning
print "\nFinal bias States"
print "Accel", h.xhat[-1, 10:13]
print "Gyro", h.xhat[-1, 13:16]
print "Drag", h.xhat[-1, 16]


for i in tqdm(ids):
    if i not in h.depth_hat: continue
    plot_side_by_side('x_{}'.format(i), 0, 2, h.t.feat_hat[i], h.feat_hat[i], truth_t=h.t.feat[i],
                      truth=h.feat[i], labels=['u', 'v'], start_t=start, end_t=end, truth_offset=None, subdir='lambda')
    if hasattr(h, 'depth') and hasattr(h, 'depth_hat'): 
        plot_side_by_side('x_{}'.format(i), 0, 1, h.t.depth_hat[i], h.depth_hat[i][:, None], truth_t=h.t.depth[i],
                      truth=h.depth[i][:, None], labels=[r'\frac{1}{\rho}'], start_t=start, end_t=end,
                      cov=h.depth_cov[i] if plot_cov else None, truth_offset=None, subdir='rho')









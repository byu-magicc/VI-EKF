import os
import numpy as np
from data import History
from plot_helper import plot_side_by_side, init_plots
from tqdm import tqdm

log_dir = os.path.dirname(os.path.realpath(__file__)) + "/../logs/"
log_folders =  [int(name) for name in os.listdir(log_dir) if os.path.isdir(log_dir + name)]

latest_folder = max(log_folders)

prop_file = open(log_dir + str(latest_folder) + "/prop.txt")
perf_file = open(log_dir + str(latest_folder) + "/perf.txt")
meas_file = open(log_dir + str(latest_folder) + "/meas.txt")

h = History()
for line in prop_file:
    line_arr = np.array([float(item) for item in line.split()])
    num_features = len(line_arr) - 1 - 17 - 16
    X = 1
    COV = 1 + 17 + 5*num_features
    t = line_arr[0]
    h.store(t, xhat=line_arr[1:18], cov=np.diag(line_arr[COV:COV+16]), num_features=num_features)

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
        h.store(t, acc=line_arr[0:2], acc_hat=line_arr[2:4])
    elif meas_type == 'ATT':
        h.store(t, att=line_arr[0:4], att_hat=line_arr[4:8])
    elif meas_type == 'POS':
        h.store(t, pos=line_arr[0:3], pos_hat=line_arr[3:6])
    elif meas_type == 'FEAT':
        id = line_arr[4]
        h.store(t, line_arr[4], feat=line_arr[0:2], feat_hat=line_arr[2:4])
        ids.append(id) if id not in ids else None
    elif meas_type == 'DEPTH':
        h.store(t, line_arr[2], depth=line_arr[0], depth_hat=line_arr[1])
    else:
        print("unsupported measurement type ", meas_type)

h.tonumpy()
start = h.t.xhat[0]
end = h.t.xhat[-1]
fig_dir = os.path.dirname(os.path.realpath(__file__)) + "/../plots/"

init_plots(start, end, fig_dir)

plot_side_by_side('x_pos', 0, 3, h.t.xhat, h.xhat, cov=None, truth_t=h.t.pos, truth=h.pos, labels=['x', 'y', 'z'], start_t=start, end_t=end)
plot_side_by_side('x_vel', 3, 6, h.t.xhat, h.xhat, cov=None, labels=['x', 'y', 'z'], start_t=start, end_t=end)
plot_side_by_side('x_att', 6, 10, h.t.xhat, h.xhat, cov=None, truth_t=h.t.att, truth=h.att, labels=['w','x', 'y', 'z'], start_t=start, end_t=end)


for i in tqdm(ids):
    plot_side_by_side('lambda/x_{}'.format(i), 0, 2, h.t.feat_hat[i], h.feat_hat[i], truth_t=h.t.feat[i],
                      truth=h.feat[i], labels=['u', 'v'], start_t=start, end_t=end)
    plot_side_by_side('rho/x_{}'.format(i), 0, 1, h.t.depth_hat[i], h.depth_hat[i][:, None], truth_t=h.t.depth[i],
                      truth=h.depth[i][:, None], labels=[r'$\frac{1}{\rho}$'], start_t=start, end_t=end)






import vi_ekf as viekf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from plot_helper import plot_cube, plot_side_by_side
from data import FakeData, ETHData
from pyquat import Quaternion
import time

data = ETHData(start=1, end=4)
data.__test__()

ekf = viekf.VI_EKF(data.x0)
last, history = time.time(), []

for i, (t, dt, pos, vel, att, gyro, acc, zetas, depths, ids) in enumerate(tqdm(data)):
    x_hat, P = ekf.propagate(acc, gyro, dt) if (acc is not None and gyro is not None) else (None, None)

    # truth updates
    alt_hat = ekf.update(pos[2], 'alt', data.R['alt'], passive=True) if pos is not None else None
    acc_hat = ekf.update(acc[:2], 'acc', data.R['acc'], passive=True) if acc is not None else None
    att_hat = ekf.update(att, 'att', data.R['att'], passive=True) if att is not None else None
    pos_hat = ekf.update(pos, 'pos', data.R['pos'], passive=True) if pos is not None else None

    # feature updates
    zeta_hats, depth_hats = [], []
    for zeta, depth, id in zip(zetas, depths, ids):
        zeta_hats.append(ekf.update(zeta, 'feat', data.R['zeta'], passive=True, i=id))
        depth_hats.append(ekf.update(depth, 'depth', data.R['depth'], passive=True, i=id))

    history.append([t, x_hat, P, alt_hat, acc_hat, att_hat, pos_hat, zeta_hats,
                      depth_hats, pos, vel, att, gyro, acc, zetas, depths])

    # every 1/30th of a second, update zeta cube
    if time.time() - last > 1/30. and x_hat is not None and len(zeta_hats) > 0 and len(zetas) > 0:
        plot_cube(Quaternion(x_hat[6:10]), zeta_hats, zetas)
        last = time.time()

# convert the list of tuples of data into indvidual lists of data
history = zip(*[[d for d in instance] for instance in history])

# replace Nones with nan arrays, create nan arrays for each data type, and convert to numpy arrays
prototype = [next(np.array(item) for item in dt if item is not None) * np.nan for dt in history]
history = [[inst if inst is not None else prototype[i] for inst in dt] for i, dt in enumerate(history)]
(time, all_x_hat, all_P, all_alt_hat, all_acc_hat, all_att_hat, all_pos_hat, all_zeta_hats,
 all_depth_hats, all_pos, all_vel, all_att, all_gyro, all_acc, all_zetas, all_depths) = list(map(np.array, history))

# plot
if True:
    plot_side_by_side('pos', viekf.xPOS, viekf.xPOS+3, time, all_x_hat, cov=all_P, truth_t=time, truth=all_pos, labels=['x', 'y', 'z'])
    plot_side_by_side('vel', viekf.xVEL, viekf.xVEL+3, time, all_x_hat, cov=all_P, truth_t=time, truth=all_vel, labels=['x', 'y', 'z'])
    plot_side_by_side('att', viekf.xATT, viekf.xATT+4, time, all_x_hat, cov=None, truth_t=time, truth=all_att, labels=['w', 'x', 'y', 'z'])
    plot_side_by_side('mu', viekf.xMU, viekf.xMU+1, time, all_x_hat, cov=None, truth_t=None, truth=None, labels=['mu'])
    plot_side_by_side('z_alt', 0, 1, time, all_alt_hat, truth_t=time, truth=-all_pos[:,2], labels=['z_alt_'])
    plot_side_by_side('z_att', 0, 4, time, all_att_hat, truth_t=time, truth=all_att, labels='z_att')
    plot_side_by_side('z_acc', 0, 2, time, all_acc_hat, truth_t=time, truth=all_acc[:, :2], labels=['x', 'y'])
    plot_side_by_side('z_pos', 0, 2, time, all_pos_hat, truth_t=time, truth=all_pos, labels=['x', 'y', 'z'])
    plot_side_by_side('gyro', 0, 3, time, all_gyro, labels=['x', 'y', 'z'])
    plot_side_by_side('acc', 0, 3, time, all_acc, labels=['x', 'y', 'z'])

    for i in range(ekf.len_features):
        truth = np.hstack((all_zetas[:, i, :], all_depths[:, i]))
        est = np.hstack((all_zeta_hats[:, i, :], all_depth_hats[:, i]))
        plot_side_by_side('z_feat_{}'.format(i), 0, 3, time, all_zeta_hats[:, i], truth_t=time, truth=all_zetas[:, i, :], labels=['x', 'y', 'z'])
        plot_side_by_side('z_rho_{}'.format(i), 0, 1, time, all_depth_hats[:, i], truth_t=time, truth=all_depths[:, i], labels=['1/rho'])
        plot_side_by_side('zeta_{}'.format(i), 0, 4, time, est, cov=None, truth_t=time, truth=truth, labels=['zx', 'zy', 'zz', 'rho'])

    plt.show()

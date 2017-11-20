import vi_ekf as viekf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from plot_helper import plot_cube, plot_side_by_side
from data import FakeData, ETHData
from pyquat import Quaternion, quat_arr_to_euler
import time
from math_helper import  norm

data = ETHData(filename='data/V1_01_easy/mav0', start=2.0, end=30.0, sim_features=True, load_new=True)
data.__test__()

ekf = viekf.VI_EKF(data.x0)
ekf.x[viekf.xMU] = 1.0
last, history = time.time(), []

for i, (t, dt, pos, vel, att, gyro, acc, b_w, b_a, zetas, depths, ids) in enumerate(tqdm(data)):
    x_hat, P = ekf.propagate(acc, gyro, dt) if (acc is not None and gyro is not None) else (None, None)

    ekf.set_imu_bias(b_w, b_a)

    # truth updates
    alt_hat = ekf.update(pos[2], 'alt', data.R['alt'], passive=True) if pos is not None else None
    acc_hat = ekf.update(acc[:2], 'acc', data.R['acc'], passive=True) if acc is not None else None
    att_hat = ekf.update(att, 'att', data.R['att'], passive=True) if att is not None else None
    pos_hat = ekf.update(pos, 'pos', data.R['pos'], passive=False) if pos is not None else None

    # feature updates
    if len(ids) > 0:
        ekf.keep_only_features(ids)

    zeta_hats, depth_hats = [], []
    for zeta, id in zip(zetas, ids):
        zeta_hats.append(ekf.update(zeta, 'feat', data.R['zeta'], passive=True, i=id))

    for depth, id in zip(depths, ids):
        depth_hats.append(ekf.update(depth, 'depth', data.R['depth'], passive=True, i=id))

    # store data for plotting
    history.append([t,
                    x_hat[:viekf.xZ], P[:viekf.dxZ, :viekf.dxZ],
                    alt_hat, acc_hat, att_hat, pos_hat, zeta_hats, depth_hats, pos, vel, att, b_w, b_a, gyro, acc, zetas, depths, ids])

    # every 1/60th of a second, update zeta cube
    if time.time() - last > 1/10. and x_hat is not None and len(zeta_hats) > 0 and len(zetas) > 0:
        # plot_cube(Quaternion(x_hat[6:10]), zeta_hats, zetas)
        last = time.time()

# convert the list of tuples of data into indvidual lists of data
history = zip(*[[d for d in instance] for instance in history])


# replace Nones with nan arrays, create nan arrays for each data type, and convert to numpy arrays
prototype = [next(np.array(item) for item in dt if item is not None) * np.nan for dt in history]
history = [[inst if inst is not None else prototype[i] for inst in dt] for i, dt in enumerate(history)]
(tm, all_x_hat, all_P, all_alt_hat, all_acc_hat, all_att_hat, all_pos_hat, all_zeta_hats,
 all_depth_hats, all_pos, all_vel, all_att, all_b_w, all_b_a, all_gyro, all_acc, all_zetas, all_depths, all_ids) = list(map(np.array, history))



# Massage Feature Data
feature_ids = list(np.unique(np.concatenate(all_ids)).astype('uint32'))

zeta_hat_array = np.nan*np.ones((len(all_zetas), 3*len(feature_ids)))
depth_hat_array = np.nan*np.ones((len(all_zetas), len(feature_ids)))
zeta_array = np.nan*np.ones((len(all_zetas), 3*len(feature_ids)))
depth_array = np.nan*np.ones((len(all_zetas), len(feature_ids)))
for i in range(len(all_zetas)):
    for l in all_ids[i]:
        zeta_hat_array[i,3*l:3*l+3,None] = all_zeta_hats[i][l]
        zeta_array[i,3*l:3*l+3,None] = all_zetas[i][l]
        depth_hat_array[i,l] = all_depth_hats[i][l]
        depth_array[i, l] = all_depths[i][l]

euler = quat_arr_to_euler(all_att.squeeze().T).T
euler_hat = quat_arr_to_euler(all_x_hat[:, viekf.xATT:viekf.xATT+4].squeeze().T).T

# plot
if True:
    plot_side_by_side('pos', viekf.xPOS, viekf.xPOS+3, tm, all_x_hat, cov=all_P, truth_t=tm, truth=all_pos, labels=['x', 'y', 'z'])
    plot_side_by_side('vel', viekf.xVEL, viekf.xVEL+3, tm, all_x_hat, cov=all_P, truth_t=tm, truth=all_vel, labels=['x', 'y', 'z'])
    plot_side_by_side('att', viekf.xATT, viekf.xATT+4, tm, all_x_hat, cov=None, truth_t=tm, truth=all_att, labels=['w', 'x', 'y', 'z'])
    plot_side_by_side('euler', 0, 3, tm, euler_hat, cov=None, truth_t=tm, truth=euler, labels=[r'$\phi$', r'$\rho$', r'$\psi$'])
    plot_side_by_side('b_g', viekf.xB_G, viekf.xB_G + 3, tm, all_x_hat, cov=None, truth_t=tm, truth=all_b_w, labels=['x', 'y', 'z'])
    plot_side_by_side('b_a', viekf.xB_A, viekf.xB_A + 3, tm, all_x_hat, cov=None, truth_t=tm, truth=all_b_a, labels=['x', 'y', 'z'])
    plot_side_by_side('mu', viekf.xMU, viekf.xMU+1, tm, all_x_hat, cov=None, truth_t=None, truth=None, labels=['mu'])
    plot_side_by_side('z_alt', 0, 1, tm, all_alt_hat, truth_t=tm, truth=-all_pos[:,2], labels=['z_alt_'])
    plot_side_by_side('z_att', 0, 4, tm, all_att_hat, truth_t=tm, truth=all_att, labels='z_att')
    plot_side_by_side('z_acc', 0, 2, tm, all_acc_hat, truth_t=tm, truth=all_acc[:, :2], labels=['x', 'y'])
    plot_side_by_side('z_pos', 0, 2, tm, all_pos_hat, truth_t=tm, truth=all_pos, labels=['x', 'y', 'z'])
    plot_side_by_side('gyro', 0, 3, tm, all_gyro, labels=['x', 'y', 'z'])
    plot_side_by_side('acc', 0, 3, tm, all_acc, labels=['x', 'y', 'z'])

    for i in feature_ids:
        plot_side_by_side('z_feat_{}'.format(i), i*3, i*3+3, tm, zeta_hat_array, truth_t=tm, truth=zeta_array[:,i*3:i*3+3], labels=['x', 'y', 'z'])
        plot_side_by_side('z_rho_{}'.format(i), i, i+1, tm, depth_hat_array, truth_t=tm, truth=depth_array[:,i:i+1], labels=['1/rho'])
        # plot_side_by_side('zeta_{}'.format(i), 0, 4, time, est, cov=None, truth_t=time, truth=truth, labels=['zx', 'zy', 'zz', 'rho'])

    # plt.show()

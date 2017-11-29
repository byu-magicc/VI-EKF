import vi_ekf as viekf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from plot_helper import plot_cube, plot_side_by_side, init_plots
from data import FakeData, ETHData, ROSbagData
from pyquat import Quaternion, quat_arr_to_euler
import time
from math_helper import  norm

init_plots()

# data = ETHData(filename='data/V1_01_easy/mav0', start=5.0, end=120.0, sim_features=True, load_new=True)
data = ROSbagData(filename='data/truth_imu_flight.bag', start=30.0, end=120.0, sim_features=True, load_new=True)
data.__test__()

ekf = viekf.VI_EKF(data.x0)
ekf.x[viekf.xMU] = 0.15
# ekf.x[viekf.xB_A:viekf.xB_A+3] = np.array([[0.05, 0.1, -0.05]]).T
last = time.time()
history = {'state_t': [],
           'xhat': [],
           'P': [],
           'depth_hats': [],
           'zeta_hats': [],
           'alt_res': [],
           'att_res': [],
           'acc_res': [],
           'pos_res': [],
           'vel_res': [],
           'zeta_res': [],
           'depth_res': [],
           'pos': [],
           'vel': [],
           'att': [],
           'gyro': [],
           'acc': [],
           'depths': [],
           'zetas':[],
           'ids':[],
           'qzetas':[],
           'feat_t':[],
           'alt_t':[],
           'acc_t':[],
           'truth_t':[],
           'delta_pos_feat':[]
           }

print "running filter"
for i, (t, dt, pos, vel, att, gyro, acc, qzetas, depths, ids) in enumerate(tqdm(data)):
    if acc is not None and gyro is not None:
        x_hat, P = ekf.propagate(acc, gyro, dt)
    else:
        x_hat = ekf.x.copy()
        P = ekf.P.copy()

    # Save State information
    history['state_t'].append(t)
    history['xhat'].append(x_hat[:viekf.xZ])
    history['P'].append(P[:viekf.dxZ, :viekf.dxZ])

    # Save Inputs
    history['acc'].append(acc)
    history['gyro'].append(gyro)

    # sensor updates - save off residual information
    if pos is not None:
        history['alt_t'].append(t)
        history['alt_res'].append(ekf.update(-pos[2], 'alt', data.R['alt'], passive=True))

    if acc is not None:
        history['acc_t'].append(t)
        history['acc_res'].append(ekf.update(acc[:2], 'acc', data.R['acc'], passive=True))

    # Truth updates - save off residual information
    if pos is not None:
        history['truth_t'].append(t)
        history['pos'].append(pos)
        history['vel'].append(vel)
        history['att'].append(att)
        history['att_res'].append(ekf.update(att, 'att', data.R['att'], passive=False))
        history['pos_res'].append(ekf.update(pos, 'pos', data.R['pos'], passive=True))
        history['vel_res'].append(ekf.update(vel, 'vel', data.R['vel'], passive=False))

    # feature updates
    if ids is not None and len(ids) > 0:
        ekf.keep_only_features(ids)
        zeta_res, depth_res, zetas, zeta_hats, depth_hats = [], [], [], [], []
        for qzeta, depth, id in zip(qzetas, depths, ids):
            zeta_res.append(ekf.update(qzeta, 'feat', data.R['zeta'], passive=True, i=id, depth=depth))
            depth_res.append(ekf.update(depth, 'depth', data.R['depth'], passive=True, i=id))
            zetas.append(Quaternion(qzeta).rot(ekf.khat))
        zeta_hats.append(ekf.get_zetas().T)
        depth_hats.append(ekf.get_depths())

        # Save Feature History
        history['feat_t'].append(t)
        history['depths'].append(depths)
        history['ids'].append(ids)
        history['qzetas'].append(qzetas)
        history['zeta_res'].append(zeta_res)
        history['depth_res'].append(depth_res)
        history['zetas'].append(zetas)
        history['depth_hats'].append(depth_hats)
        history['zeta_hats'].append(zeta_hats)

# Convert the lists of numpy arrays into actual numpy arrays with nans in invalid regions
for key, item in history.iteritems():
    # Find the proper sized NaN array
    nan_array = None
    for val in item:
        if val is not None:
            nan_array = np.ones_like(np.array(val))*np.nan
            break
    item = [val if val is not None else nan_array for val in item]
    history[key] = np.array(item).squeeze()


# Massage Feature Data
feature_ids = np.unique(history['ids'][~np.isnan(history['ids'])]).astype('uint32')
feat_len = len(history['zetas'])
# Create a massive array to hold all features for all times, with Nans when the feature doesn't appear
zeta_hat_array = np.nan*np.ones((feat_len, 3*len(feature_ids)))
depth_hat_array = np.nan*np.ones((feat_len, len(feature_ids)))
zeta_array = np.nan*np.ones((feat_len, 3*len(feature_ids)))
depth_array = np.nan*np.ones((feat_len, len(feature_ids)))
zeta_res_array = np.nan*np.ones((feat_len, 2*len(feature_ids)))
depth_res_array = np.nan*np.ones((feat_len, len(feature_ids)))

for i in range(feat_len):
    for l in history['ids'][i]:
        if np.isnan(history['ids'][i]).all():
            continue
        l = l.astype('uint32')
        zeta_hat_array[i,3*l:3*l+3] = history['zeta_hats'][i,l,:]
        zeta_array[i,3*l:3*l+3] = history['zetas'][i,l,:]
        depth_hat_array[i,l] = history['depth_hats'][i,l]
        depth_array[i, l] = history['depths'][i,l]
        zeta_res_array[i, 2*l:2*l+2] = history['zeta_res'][i,l,:]
        depth_res_array[i,l] = history['depth_res'][i,l]

euler = quat_arr_to_euler(history['att'].T).T
euler_hat = quat_arr_to_euler(history['xhat'][:, viekf.xATT:viekf.xATT+4].T).T

# plot
if True:
    plot_side_by_side('x_pos', viekf.xPOS, viekf.xPOS+3, history['state_t'], history['xhat'], cov=history['P'], truth_t=history['truth_t'], truth=history['pos'], labels=['x', 'y', 'z'])
    plot_side_by_side('x_vel', viekf.xVEL, viekf.xVEL+3, history['state_t'], history['xhat'], cov=history['P'], truth_t=history['truth_t'], truth=history['vel'], labels=['x', 'y', 'z'])
    plot_side_by_side('x_att', viekf.xATT, viekf.xATT+4, history['state_t'], history['xhat'], truth_t=history['truth_t'], truth=history['att'], labels=['w', 'x', 'y', 'z'])
    plot_side_by_side('x_euler', 0, 3, history['state_t'], euler_hat, truth_t=history['truth_t'], truth=euler, labels=[r'$\phi$', r'$\rho$', r'$\psi$'])
    plot_side_by_side('x_b_g', viekf.xB_G, viekf.xB_G + 3, history['state_t'], history['xhat'], cov=history['P'], labels=['x', 'y', 'z'], cov_bounds=(viekf.dxB_G,viekf.dxB_G+3))
    plot_side_by_side('x_b_a', viekf.xB_A, viekf.xB_A + 3, history['state_t'], history['xhat'], cov=history['P'], labels=['x', 'y', 'z'], cov_bounds=(viekf.dxB_A,viekf.dxB_A+3))
    plot_side_by_side('x_mu', viekf.xMU, viekf.xMU+1, history['state_t'], history['xhat'], cov=history['P'], labels=['mu'], cov_bounds=(viekf.dxMU,viekf.dxMU+1))
    plot_side_by_side('z_alt_residual', 0, 1, history['alt_t'], history['alt_res'][:,None], labels=['z_alt_res'])
    plot_side_by_side('z_att_residual', 0, 3, history['truth_t'], history['att_res'], labels=['x', 'y', 'z'])
    plot_side_by_side('z_acc_residual', 0, 2, history['acc_t'], history['acc_res'], labels=['x', 'y'])
    plot_side_by_side('z_pos_residual', 0, 3, history['truth_t'], history['pos_res'], labels=['x', 'y', 'z'])
    plot_side_by_side('z_vel_residual', 0, 3, history['truth_t'], history['vel_res'], labels=['x', 'y', 'z'])
    plot_side_by_side('u_gyro', 0, 3, history['state_t'], history['gyro'], labels=['x', 'y', 'z'])
    plot_side_by_side('u_acc', 0, 3, history['state_t'], history['acc'], labels=['x', 'y', 'z'])

    for i in feature_ids:
        plot_side_by_side('x_feat_{}'.format(i), i*3, i*3+3, history['feat_t'], zeta_hat_array, truth_t=history['feat_t'], truth=zeta_array[:,i*3:i*3+3], labels=['x', 'y', 'z'])
        plot_side_by_side('x_rho_{}'.format(i), i, i+1, history['feat_t'], depth_hat_array, truth_t=history['feat_t'], truth=depth_array[:,i:i+1], labels=['1/rho'], cov_bounds=(3*i+2, 3*i+3))
        plot_side_by_side('z_zeta_{}_residual'.format(i), 0, 2, history['feat_t'], zeta_res_array, labels=['x', 'y'])
        plot_side_by_side('z_depth_{}_residual'.format(i), 0, 1, history['feat_t'], depth_res_array, labels=['rho'])

    # plt.show()

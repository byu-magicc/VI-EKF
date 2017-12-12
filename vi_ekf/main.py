import vi_ekf as viekf
from tqdm import tqdm
import numpy as np
from plot_helper import plot_cube, plot_side_by_side, init_plots
from data import FakeData, ETHData, ROSbagData, History
from pyquat import Quaternion, quat_arr_to_euler
from plot_helper import plot_3d_trajectory

# data = ETHData(filename='/mnt/pccfs/not_backed_up/eurocmav/V1_01_easy/mav0', start=5.0, end=120.0, sim_features=True, load_new=True)
# data = ROSbagData(filename='data/truth_imu_flight.bag', start=30.0, end=68.0, sim_features=True, load_new=True)
data = ROSbagData(filename='data/truth_imu_depth_mono.bag', start=8.0, end=np.inf, sim_features=True, load_new=True)
data.__test__()

ekf = viekf.VI_EKF(data.x0)
ekf.set_camera_to_IMU(data.data['p_b_c'], data.data['q_b_c'])
h = History()

for i, (t, pos, vel, att, gyro, acc, qzetas, depths, ids) in enumerate(tqdm(data)):
    h.store(t, pos=pos, vel=vel, att=att)
    h.store(t, gyro=gyro, acc=acc)
    h.store(t, qzetas=qzetas, depths=depths, ids=ids)

    # propagate if this tick contains data
    if acc is not None and gyro is not None:
        x_hat, P = ekf.propagate(acc, gyro, t)
        h.store(t, x_hat=x_hat[:viekf.xZ], P=P[:viekf.dxZ, :viekf.dxZ])

    # sensor updates - save off residual information
    if pos is not None:
        h.store(t, alt_res=ekf.update(-pos[2], 'alt', data.R['alt'], passive=True)[0])
        h.store(t, att_res=ekf.update(att, 'att', data.R['att'], passive=False)[0],
                pos_res=ekf.update(pos, 'pos', data.R['pos'], passive=True)[0],
                vel_res=ekf.update(vel, 'vel', data.R['vel'], passive=True)[0])

    if acc is not None:
        h.store(t, acc_res=ekf.update(acc[:2], 'acc', data.R['acc'], passive=False)[0])

    # feature updates
    if ids is not None and len(ids) > 0:
        ekf.keep_only_features(ids)
        for qzeta, depth, id in zip(qzetas, depths, ids):
            zeta_res, qzeta_hat = ekf.update(qzeta, 'feat', data.R['zeta'], passive=False, i=id)
            depth_res, depth_hat = ekf.update(depth, 'depth', data.R['depth'], passive=True, i=id)

            h.store(t, id, zeta_res=zeta_res, depth_res=depth_res, zeta=Quaternion(qzeta).rot(ekf.khat), Pfeat=ekf.P[viekf.dxZ:, viekf.dxZ:])
            h.store(t, id, zeta_hat=ekf.get_zeta(id), depth_hat=depth_hat, depth=depth)
            h.store(t, id, qzeta=qzeta, qzeta_hat=qzeta_hat)

# plot
if True:
    # convert all our linked lists to contiguous numpy arrays and initialize the plot parameters
    h.tonumpy()
    init_plots()

    plot_side_by_side('x_pos', viekf.xPOS, viekf.xPOS+3, h.t.x_hat, h.x_hat, cov=h.P, truth_t=h.t.pos, truth=h.pos, labels=['x', 'y', 'z'])
    plot_side_by_side('x_vel', viekf.xVEL, viekf.xVEL+3, h.t.x_hat, h.x_hat, cov=h.P, truth_t=h.t.vel, truth=h.vel, labels=['x', 'y', 'z'])
    # plot_side_by_side('x_att', viekf.xATT, viekf.xATT+4, h.t.x_hat, h.x_hat, truth_t=h.t.att, truth=h.att, labels=['w', 'x', 'y', 'z'])

    plot_side_by_side('x_euler', 0, 3, h.t.x_hat, quat_arr_to_euler(h.x_hat[:, viekf.xATT:viekf.xATT+4, 0].T).T, truth_t=h.t.att, truth=quat_arr_to_euler(h.att.T[0]).T, labels=[r'$\phi$', r'$\rho$', r'$\psi$'])
    plot_side_by_side('x_b_g', viekf.xB_G, viekf.xB_G + 3, h.t.x_hat, h.x_hat, cov=h.P, labels=['x', 'y', 'z'], cov_bounds=(viekf.dxB_G,viekf.dxB_G+3))
    plot_side_by_side('x_b_a', viekf.xB_A, viekf.xB_A + 3, h.t.x_hat, h.x_hat, cov=h.P, labels=['x', 'y', 'z'], cov_bounds=(viekf.dxB_A,viekf.dxB_A+3))
    plot_side_by_side('x_mu', viekf.xMU, viekf.xMU+1, h.t.x_hat, h.x_hat, cov=h.P, labels=['mu'], cov_bounds=(viekf.dxMU,viekf.dxMU+1))

    plot_side_by_side('z_alt_residual', 0, 1, h.t.alt_res, h.alt_res, labels=['z_alt_res'])
    plot_side_by_side('z_att_residual', 0, 3, h.t.att_res, h.att_res, labels=['x', 'y', 'z'])
    plot_side_by_side('z_acc_residual', 0, 2, h.t.acc_res, h.acc_res, labels=['x', 'y'])
    plot_side_by_side('z_pos_residual', 0, 3, h.t.pos_res, h.pos_res, labels=['x', 'y', 'z'])
    plot_side_by_side('z_vel_residual', 0, 3, h.t.vel_res, h.vel_res, labels=['x', 'y', 'z'])

    plot_side_by_side('u_gyro', 0, 3, h.t.gyro, h.gyro, labels=['x', 'y', 'z'])
    plot_side_by_side('u_acc', 0, 3, h.t.acc, h.acc, labels=['x', 'y', 'z'])

    ids = []
    for step_ids in h.ids:
        for id in step_ids:
            if id not in ids:
                ids.append(id)

    for i in ids:
        plot_side_by_side('x_feat_{}'.format(i), 0, 3, h.t.zeta_hat[i], h.zeta_hat[i], truth_t=h.t.zeta[i], truth=h.zeta[i], labels=['x', 'y', 'z'])
        plot_side_by_side('x_qfeat_{}'.format(i), 0, 4, h.t.qzeta_hat[i], h.qzeta_hat[i], truth_t=h.t.qzeta[i], truth=h.qzeta[i], labels=['w','x', 'y', 'z'])
        plot_side_by_side('x_rho_{}'.format(i), 0, 1, h.t.depth_hat[i], h.depth_hat[i], cov=h.Pfeat, truth_t=h.t.depth[i], truth=h.depth[i], labels=[r'$\frac{1}{\rho}$'], cov_bounds=(3*i+2, 3*i+3))


        plot_side_by_side('z_zeta_{}_residual'.format(i), 0, 2, h.t.zeta_res[i], h.zeta_res[i], labels=['x', 'y'])
        plot_side_by_side('z_depth_{}_residual'.format(i), 0, 1, h.t.depth_res[i], h.depth_res[i], labels=['rho'])















if True:
    depth_hats, qzeta_hats = [], []
    for i in np.unique(h.ids):
        qzeta_hats.append(h.qzeta_hat[i])
        depth_hats.append(h.depth_hat[i])
    depth_hats = np.array(depth_hats)
    qzeta_hats = np.array(qzeta_hats)


    plot_3d_trajectory(h.x_hat[:,viekf.xPOS:viekf.xPOS+3,0], h.x_hat[:,viekf.xATT:viekf.xATT+4, 0], qzetas=qzeta_hats, depths=depth_hats)

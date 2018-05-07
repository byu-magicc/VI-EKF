#!/usr/bin/env python

import os
import numpy as np
from data import History
from plot_helper import plot_side_by_side, init_plots, get_colors
from tqdm import tqdm
import scipy.signal
import matplotlib.pyplot as plt
from pyquat import Quaternion
import sys

def main():
    make_feature_plots = False
    if len(sys.argv) > 1  and sys.argv[1] == '-a':
        make_feature_plots = True

    # Shift truth timestamp
    # offset = -0.35
    offset = 0.0

    plot_cov = True
    pose_cov = True

    log_dir = os.path.dirname(os.path.realpath(__file__)) + "/../logs/"
    log_folders =  [int(name) for name in os.listdir(log_dir) if os.path.isdir(log_dir + name)]

    latest_folder = max(log_folders)

    prop_file = open(log_dir + str(latest_folder) + "/prop.txt")
    perf_file = open(log_dir + str(latest_folder) + "/perf.txt")
    meas_file = open(log_dir + str(latest_folder) + "/meas.txt")
    conf_file = open(log_dir + str(latest_folder) + "/conf.txt")
    input_file = open(log_dir + str(latest_folder) + "/input.txt")
    xdot_file = open(log_dir + str(latest_folder) + "/xdot.txt")
    fig_dir = log_dir + str(latest_folder) + "/plots"
    if not os.path.isdir(fig_dir): os.system("mkdir " + fig_dir)

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

    for i, line in enumerate(perf_file):
        if i == 0: continue
        line_arr = np.array([float(item) for item in line.split()])
        if len(line_arr) == 12:
            t = line_arr[0]
            h.store(t, prop_time=line_arr[1], acc_time=line_arr[2], pos_time=line_arr[5], feat_time=line_arr[8], depth_time=line_arr[10])


    ids = []
    for line in meas_file:
        try:
            meas_type = line.split()[0]
            line_arr = np.array([float(item) for item in line.split()[2:]])
            t = float(line.split()[1])
        except:
            continue


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
            if len(line_arr) < 7: continue
            id = line_arr[6]
            h.store(t, id, feat_hat=line_arr[0:2], feat=line_arr[2:4], feat_cov=np.diag(line_arr[4:6]))
            ids.append(id) if id not in ids else None
        elif meas_type == 'DEPTH':
            if len(line_arr) < 4: continue
            # Invert the covariance measurement
            p = 1.0/line_arr[0]
            s = line_arr[2]
            cov = 1./(p+s) - 1./p
            h.store(t, line_arr[3], depth=line_arr[1], depth_hat=line_arr[0], depth_cov=[[cov   ]])
        elif meas_type == 'ALT':
            if len(line_arr) < 3: continue
            h.store(t, alt=line_arr[0], alt_hat=line_arr[1])
        else:
            print("unsupported measurement type ", meas_type)

    for line in input_file:
        line_arr = np.array([float(item) for item in line.split()])
        if len(line_arr) < 6: continue
        h.store(line_arr[0], u_acc=line_arr[1:4], u_gyro=line_arr[4:])

    for line in xdot_file:
        line_arr = np.array([float(item) for item in line.split()])
        if len(line_arr) < 18: continue
        h.store(line_arr[0], dt=line_arr[1], xdot=line_arr[2:18])


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

    # PLOT Performance Results
    plt.hist(h.prop_time[h.prop_time > 0.001], bins=35, alpha=0.5, label='propagation')
    plt.hist(h.feat_time[h.feat_time > 0], bins=5, alpha=0.5, label='feature update')
    plt.legend()
    plt.savefig(fig_dir+"/perf.svg", bbox_inches='tight')
    plt.close()

    # PLOT STATES
    plot_side_by_side(r'$p_{b/n}^n$', 0, 3, h.t.xhat, h.xhat, cov=h.cov if pose_cov else None, truth_t=h.t.pos, truth=h.pos, labels=['p_x', 'p_y', 'p_z'], start_t=start, end_t=end, truth_offset=offset)
    plot_side_by_side(r'$p_{b/I}^I$', 0, 3, h.t.global_pos_hat, h.global_pos_hat, cov=h.cov if pose_cov else None, truth_t=h.t.global_pos, truth=h.global_pos, labels=['p_x', 'p_y', 'p_z'], start_t=start, end_t=end, truth_offset=offset)
    plot_side_by_side(r'$v_{b/I}^b$', 3, 6, h.t.xhat, h.xhat, cov=h.cov if plot_cov else None, truth_t=v_t, truth=vel_data, labels=['v_x', 'v_y', 'v_z'], start_t=start, end_t=end, truth_offset=offset)
    plot_side_by_side(r'$q_n^b$', 6, 10, h.t.xhat, h.xhat, cov=None, truth_t=h.t.att, truth=h.att, labels=['q_w','q_x', 'q_y', 'q_z'], start_t=start, end_t=end, truth_offset=offset)
    plot_side_by_side(r'$q_I^b$', 0, 4, h.t.global_att_hat, h.global_att_hat, cov=None, truth_t=h.t.global_att, truth=h.global_att, labels=['q_w','q_x', 'q_y', 'q_z'], start_t=start, end_t=end, truth_offset=offset)

    ## PLOT EULER ANGLES
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


    # PLOT INPUTS AND MEASUREMENTS
    b_acc, a_acc = scipy.signal.butter(6, 0.05)
    acc_smooth = scipy.signal.filtfilt(b_acc, a_acc, h.acc, axis=0)
    plot_side_by_side('$y_{a}$', 0, 2, h.t.acc, acc_smooth, truth=h.acc, truth_t=h.t.acc, labels=[r'y_{a,x}', r'y_{a,y}'], start_t=start, end_t=end, truth_offset=offset)
    # plot_side_by_side('$y_{alt}$', 0, 1, h.t.alt, h.alt_hat[:,None], truth=h.alt[:,None], truth_t=h.t.alt, labels=[r'-p_z'], start_t=start, end_t=end)
    plot_side_by_side(r'$\beta_{a}$', 10, 13, h.t.xhat, h.xhat, labels=[r'\beta_{a,x}', r'\beta_{a,y}', r'\beta_{a,z}'], start_t=start, end_t=end, cov=h.cov if plot_cov else None, cov_bounds=(9,12), truth_offset=offset)
    plot_side_by_side(r'$\beta_{\omega}$', 13, 16, h.t.xhat, h.xhat, labels=[r'\beta_{\omega,x}', r'\beta_{\omega,y}', r'\beta_{\omega,z}'], start_t=start, end_t=end, cov=h.cov if plot_cov else None, cov_bounds=(12,15), truth_offset=offset)
    plot_side_by_side('drag', 16, 17, h.t.xhat, h.xhat, labels=['b'], start_t=start, end_t=end, cov=h.cov if plot_cov else None, cov_bounds=(15,16), truth_offset=offset)
    plot_side_by_side('u_acc', 0, 3, h.t.u_acc, h.u_acc, labels=[r'u_{a,x}',r'u_{a,y}',r'u_{a,z}'], start_t=start, end_t=end)
    plot_side_by_side('u_gyro', 0, 3, h.t.u_gyro, h.u_gyro, labels=[r'u_{\omega,x}',r'u_{\omega,y}',r'u_{\omega,z}'], start_t=start, end_t=end)

    # PLOT DERIVATIVES
    plot_side_by_side(r'$\dot{p}_{b/n}^n$', 0, 3, h.t.xdot, h.xdot, labels=[r'\dot{p}_x', r'\dot{p}_y', r'\dot{p}_z'], start_t=start, end_t=end, truth_offset=offset)
    plot_side_by_side(r'$\dot{v}_{b/I}^b$', 3, 6, h.t.xdot, h.xdot, labels=[r'\dot{v}_x', r'\dot{v}_y', r'\dot{v}_z'], start_t=start, end_t=end, truth_offset=offset)
    plot_side_by_side(r'$\dot{q}_{b/I}$', 6, 9, h.t.xdot, h.xdot, labels=[r'\dot{q}_x', r'\dot{q}_y', r'\dot{q}_z'], start_t=start, end_t=end, truth_offset=offset)
    plot_side_by_side(r'$\dot{\beta}_{a}$', 9, 12, h.t.xdot, h.xdot, labels=[r'\dot{\beta}_{a,x}', r'\dot{\beta}_{a,y}', r'\dot{\beta}_{a,z}'], start_t=start, end_t=end, truth_offset=offset)
    plot_side_by_side(r'$\dot{\beta}_{\omega}$', 12, 15, h.t.xdot, h.xdot, labels=[r'\dot{\beta}_{\omega,x}', r'\dot{\beta}_{\omega,y}', r'\dot{\beta}_{\omega,z}'], start_t=start, end_t=end, truth_offset=offset)
    plot_side_by_side(r'$\dot{b}$', 15, 16, h.t.xdot, h.xdot, labels=[r'\dot{b}'], start_t=start, end_t=end, truth_offset=offset)

    # print Final States for baiases for tuning
    print "\nFinal bias States"
    print "Accel", h.xhat[-1, 10:13]
    print "Gyro", h.xhat[-1, 13:16]
    print "Drag", h.xhat[-1, 16]


    print "\n Average Inputs"
    print "acc", np.mean(h.u_acc, axis=0)
    print "gyro", np.mean(h.u_gyro, axis=0)

    plt.figure(figsize=(16, 10))
    colors = get_colors(35, plt.cm.jet)
    for i in ids:
        plt.subplot(211)
        plt.plot(h.t.feat_hat[i], h.feat_hat[i][:,0], color=colors[int(i%35)])
        plt.subplot(212)
        plt.plot(h.t.feat_hat[i], h.feat_hat[i][:,1], color=colors[int(i%35)])
    plt.savefig(fig_dir + "/features.svg", bbox_inches='tight')  

    plt.figure(figsize=(16, 10))
    plt.title('depth')
    for i in ids:
        plt.plot(h.t.depth_hat[i], h.depth_hat[i][:], color=colors[int(i%35)])
    plt.savefig(fig_dir + "/depth.svg", bbox_inches='tight')

    if make_feature_plots:
        for i in tqdm(ids):
            if i not in h.feat_hat:
                continue
            plot_side_by_side('x_{}'.format(i), 0, 2, h.t.feat_hat[i], h.feat_hat[i], truth_t=h.t.feat[i],
                              truth=h.feat[i], labels=['u', 'v'], start_t=start, end_t=end, subdir='lambda',
                              cov=h.feat_cov[i] if plot_cov else None)
            if hasattr(h, 'depth_hat'):
                plot_side_by_side('x_{}'.format(i), 0, 1, h.t.depth_hat[i], h.depth_hat[i][:, None], truth_t=h.t.depth[i] if hasattr(h, 'depth') else None,
                              truth=h.depth[i][:, None] if hasattr(h, 'depth') else None, labels=[r'\frac{1}{\rho}'], start_t=start, end_t=end,
                              cov=h.depth_cov[i] if plot_cov else None, subdir='rho')



if __name__ == '__main__':
    main()







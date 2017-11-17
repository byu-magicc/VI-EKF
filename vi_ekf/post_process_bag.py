import cPickle
from vi_ekf import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from plot_helper import plot_cube, plot_side_by_side
from generate_data import generate_data
from bag_loader import read_bag

# read_bag('data/simulated_waypoints.bag')
# data = cPickle.load(open('simulated_waypoints.pkl', 'rb'))

# generate_data()
data = cPickle.load(open('generated_data.pkl', 'rb'))

start = 1.0
end = 30.0
truth_start_index = np.argmax(data['truth_NED']['t'] > start)


# State: pos, vel, att, b_gyro, b_acc, mu
x0 = np.concatenate([data['truth_NED']['pos'][truth_start_index,:,None],
                     data['truth_NED']['vel'][truth_start_index,:,None],
                     data['truth_NED']['att'][truth_start_index,:,None],
                     np.zeros((3,1)),
                     np.zeros((3,1)),
                     0.2*np.ones((1,1))], axis=0)

ekf = VI_EKF(x0)

# Load Sensor Noise Matrices
# R_acc = data['R_acc']
R_alt = 0.01
R_acc = np.diag([0.01, 0.01])
R_att = np.diag([0.0001, 0.0001, 0.0001])
R_pos = np.diag([0.0001,0.0001, 0.0001])
R_feat = np.diag([0.0001, 0.0001])
R_depth = 0.01
# R_alt = data['R_alt']


for i in range(1):
    ekf.init_feature(data['features']['zeta'][truth_start_index,i,:,None], data['features']['depth'][truth_start_index,i,None,None])

prev_time = 0
estimate = []
cov = []
est_zeta = []
est_qzeta = []
est_depth = []
est_t = []


alt_index = 0
alt_zhat = []
alt_zhat_t = []
while data['truth_NED']['t'][alt_index] < start:
    alt_index += 1

acc_index = 0
acc_zhat = []
acc_zhat_t = []
while data['imu_data']['t'][acc_index] < start:
    acc_index += 1

att_index = 0
att_zhat = []
att_zhat_t = []
while data['truth_NED']['t'][att_index] < start:
    att_index += 1

pos_index = 0
pos_zhat = []
pos_zhat_t = []
while data['truth_NED']['t'][pos_index] < start:
    pos_index += 1

feat_index = 0
feat_zhat = [[] for i in range(ekf.len_features)]
feat_zhat_t = []
while data['features']['t'][feat_index] < start:
    feat_index += 1

depth_index = 0
depth_zhat = [[] for i in range(ekf.len_features)]
depth_zhat_t = []
while data['features']['t'][depth_index] < start:
    depth_index += 1

# estimate.append(ekf.x)
# est_zeta.append(ekf.get_zeta())
# est_depth.append(ekf.get_depth())
# est_qzeta.append(ekf.get_qzeta())


for i, t in enumerate(tqdm(data['imu_data']['t'])):
    if prev_time == 0 or t <= start:
        prev_time = t
        continue
    if t > end:
        break

    dt = t - prev_time
    prev_time = t

    # Propagation step
    xhat, P = ekf.propagate(data['imu_data']['acc'][i,:, None], data['imu_data']['gyro'][i,:, None], dt)
    estimate.append(xhat)
    cov.append(P)
    est_t.append(t)
    est_zeta.append(ekf.get_zeta())
    est_depth.append(ekf.get_depth())
    est_qzeta.append(ekf.get_qzeta())

    # Update Step
    # TODO: Measurement throttling, Delayed Update

    while data['truth_NED']['t'][alt_index] <= t:
        alt_zhat.append(ekf.update(data['truth_NED']['pos'][alt_index, 2, None], 'alt', R_alt, passive=True))
        alt_zhat_t.append(t)
        alt_index += 1

    while data['imu_data']['t'][acc_index] <= t:
        acc_zhat.append(ekf.update(data['imu_data']['acc'][acc_index, :2, None], 'acc', R_acc, passive=True))
        acc_zhat_t.append(t)
        acc_index += 1

    while data['truth_NED']['t'][att_index] <= t:
        att_zhat.append(ekf.update(data['truth_NED']['att'][att_index, :, None], 'att', R_att, passive=True))
        att_zhat_t.append(t)
        att_index += 1

    while data['truth_NED']['t'][pos_index] <= t:
        pos_zhat.append(ekf.update(data['truth_NED']['pos'][pos_index, :, None], 'pos', R_pos, passive=True))
        pos_zhat_t.append(t)
        pos_index += 1

    while data['features']['t'][feat_index] <= t:
        for k in range(ekf.len_features):
            feat_zhat[k].append(ekf.update(data['features']['zeta'][feat_index, k, :, None], 'feat', R_feat, passive=True, i=k))
        feat_zhat_t.append(t)
        feat_index += 1

    while data['features']['t'][depth_index] <= t:
        for k in range(ekf.len_features):
            depth_zhat[k].append(
                ekf.update(data['features']['depth'][depth_index, k, None], 'depth', R_depth, passive=True, i=k))
        depth_zhat_t.append(t)
        depth_index += 1


    if i % 30 == 0 and True:
        q_I_b = Quaternion(xhat[6:10])
        plot_cube(q_I_b, est_zeta[-1], data['features']['zeta'][i])

# convert lists to np arrays
est_t = np.array(est_t)
estimate = np.array(estimate)
cov = np.array(cov)
est_zeta = np.array(est_zeta)
est_depth = np.array(est_depth)
est_qzeta = np.array(est_qzeta)

alt_zhat = np.array(alt_zhat)
alt_zhat_t = np.array(alt_zhat_t)
att_zhat = np.array(att_zhat)
att_zhat_t = np.array(att_zhat_t)
acc_zhat = np.array(acc_zhat)
acc_zhat_t = np.array(acc_zhat_t)
pos_zhat = np.array(pos_zhat)
pos_zhat_t = np.array(pos_zhat_t)
feat_zhat = np.array(feat_zhat)
feat_zhat_t = np.array(feat_zhat_t)
depth_zhat = np.array(depth_zhat)
depth_zhat_t = np.array(depth_zhat_t)


# Plot States
plot_side_by_side('pos', xPOS, xPOS+3, est_t, estimate, cov=cov, truth_t=data['truth_NED']['t'], truth=data['truth_NED']['pos'], labels=['x','y','z'])
plot_side_by_side('vel', xVEL, xVEL+3, est_t, estimate, cov=cov, truth_t=data['truth_NED']['t'], truth=data['truth_NED']['vel'], labels=['x','y','z'])
plot_side_by_side('att', xATT, xATT+4, est_t, estimate, cov=None, truth_t=data['truth_NED']['t'], truth=data['truth_NED']['att'], labels=['w','x','y','z'])
# plot_side_by_side('gyro_bias', xB_G, xB_G+3, est_t, estimate, cov=cov, truth_t=data['imu_gyro_bias']['t'], truth=data['imu_gyro_bias']['v'], labels=['x','y','z'])
# plot_side_by_side('acc_bias', xB_A, xB_A+3, est_t, estimate, cov=cov, truth_t=data['imu_acc_bias']['t'], truth=data['imu_acc_bias']['v'], labels=['x','y','z'])
plot_side_by_side('mu', xMU, xMU+1, est_t, estimate, cov=cov, truth_t=None, truth=None, labels=['mu'])

# Plot features
for i in range(ekf.len_features):
    truth_feature = np.hstack((data['features']['zeta'][:,i,:], data['features']['depth'][:,i,None]))
    est = np.hstack((est_zeta[:,i,:], est_depth[:,i]))
    plot_side_by_side('zeta_'+str(i), 0, 4, est_t, est, cov=None, truth_t=data['features']['t'], truth=truth_feature, labels=['zx', 'zy', 'zz', 'rho'])

# Plot Inputs
plot_side_by_side('gyro', 0, 3, data['imu_data']['t'], data['imu_data']['gyro'], labels=['x','y','z'])
plot_side_by_side('acc', 0, 3, data['imu_data']['t'], data['imu_data']['acc'], labels=['x','y','z'])

# Plot Measurements
plot_side_by_side('z_alt', 0, 1, alt_zhat_t, alt_zhat, truth_t=data['truth_NED']['t'], truth=-data['truth_NED']['pos'][:,2,None], labels=['z_alt_'])
plot_side_by_side('z_att', 0, 4, att_zhat_t, att_zhat, truth_t=data['truth_NED']['t'], truth=data['truth_NED']['att'], labels='z_att')
plot_side_by_side('z_acc', 0, 2, acc_zhat_t, acc_zhat, truth_t=data['imu_data']['t'], truth=data['imu_data']['acc'][:,:2], labels=['x', 'y'])
plot_side_by_side('z_pos', 0, 2, pos_zhat_t, pos_zhat, truth_t=data['truth_NED']['t'], truth=data['truth_NED']['pos'], labels=['x', 'y', 'z'])
for i in range(ekf.len_features):
    plot_side_by_side('z_feat_'+str(i), 0, 3, feat_zhat_t, feat_zhat[i], truth_t=data['features']['t'], truth= data['features']['zeta'][:, i, :], labels=['x', 'y', 'z'])
    plot_side_by_side('z_rho_'+str(i), 0, 1, depth_zhat_t, depth_zhat[i], truth_t=data['features']['t'], truth= data['features']['depth'][:, i,None], labels=['1/rho'])

# plt.show()
debug = 1







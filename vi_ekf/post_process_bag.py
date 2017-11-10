import cPickle
from vi_ekf import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from plot_helper import plot_cube
from generate_data import generate_data
from bag_loader import read_bag

# read_bag('data/simulated_waypoints.bag')
# data = cPickle.load(open('simulated_waypoints.pkl', 'rb'))

# generate_data()
data = cPickle.load(open('generated_data.pkl', 'rb'))

# Find the true data closest to the first imu message
# t0 = data['imu_data']['t'][0]
# truth_start_index = np.argmax(data['truth_NED']['t'] > t0)
truth_start_index = 0


# State: pos, vel, att, b_gyro, b_acc, mu
x0 = np.concatenate([data['truth_NED']['pos'][truth_start_index,:,None],
                     data['truth_NED']['vel'][truth_start_index,:,None],
                     data['truth_NED']['att'][truth_start_index,:,None],
                     np.zeros((3,1)),
                     np.zeros((3,1)),
                     0.2*np.ones((1,1))], axis=0)

 

ekf = VI_EKF(x0)

for i in range(4):
    ekf.init_feature(data['features']['zeta'][truth_start_index,i,:,None], data['features']['depth'][truth_start_index,i])

prev_time = 0
estimate = []
est_zeta = []
est_qzeta = []
est_depth = []
end = 10.0

estimate.append(ekf.x)
est_zeta.append(ekf.get_zeta())
est_depth.append(ekf.get_depth())
est_qzeta.append(ekf.get_qzeta())
for i, t in enumerate(tqdm(data['imu_data']['t'])):
    if prev_time == 0:
        prev_time = t
        continue
    if t > end:
        break

    dt = t - prev_time
    prev_time = t

    # Propagation step
    xhat = ekf.propagate(data['imu_data']['acc'][i,:, None], data['imu_data']['gyro'][i,:, None], dt)
    estimate.append(xhat)
    est_zeta.append(ekf.get_zeta())
    est_depth.append(ekf.get_depth())
    est_qzeta.append(ekf.get_qzeta())

    # if i % 30 == 0 and True:
    #     q_I_b = Quaternion(xhat[6:10])
    #     plot_cube(q_I_b, est_zeta[-1], data['features']['zeta'][i])

# convert lists to np arrays
estimate = np.array(estimate)
est_zeta = np.array(est_zeta)
est_depth = np.array(est_depth)
est_qzeta = np.array(est_qzeta)

imu_t = data['imu_data']['t']
est_t = data['imu_data']['t'][imu_t < end]
gyro = data['imu_data']['gyro'][imu_t < end]
acc = data['imu_data']['acc'][imu_t < end]
# b_acc = [data['imu_acc_bias']['t'][data['imu_acc_bias']['t'] < end], data['imu_acc_bias']['v'][data['imu_acc_bias']['t'] < end]]
# b_gyro = [data['imu_gyro_bias']['t'][data['imu_gyro_bias']['t'] < end], data['imu_gyro_bias']['v'][data['imu_gyro_bias']['t'] < end]]

truth_t = data['truth_NED']['t']
truth_pos = data['truth_NED']['pos'][truth_t < end]
truth_vel = data['truth_NED']['vel'][truth_t < end]
truth_att = data['truth_NED']['att'][truth_t < end]
truth_t = truth_t[truth_t < end]

truth_feature_t = data['features']['t']
truth_zeta = data['features']['zeta'][truth_feature_t < end, :, :]
truth_depth = data['features']['depth'][truth_feature_t < end, :]
truth_feature_t = truth_feature_t[truth_feature_t < end]

# Plot
plt.figure(1)
plt.subplot(311)
plt.title('pos')
plt.plot(est_t, estimate[:,0], label='xhat')
plt.plot(truth_t, truth_pos[:,0], label='x')
plt.legend()
plt.subplot(312)
plt.plot(est_t, estimate[:,1], label='yhat')
plt.plot(truth_t, truth_pos[:,1], label='y')
plt.subplot(313)
plt.plot(est_t, estimate[:,2], label='zhat')
plt.plot(truth_t, truth_pos[:,2], label='z')

# plt.figure(2)
# plt.title('acc')
# plt.plot(est_t, acc[:,0], label='x')
# plt.plot(est_t, acc[:,1], label='y')
# plt.plot(est_t, acc[:,2], label='z')
#
# plt.figure(5)
# plt.title('gyro')
# plt.plot(est_t, gyro[:,0], label='x')
# plt.plot(est_t, gyro[:,1], label='y')
# plt.plot(est_t, gyro[:,2], label='z')

# plt.figure(4)
# plt.subplot(211)
# plt.title('acc bias')
# plt.plot(b_acc[0], b_acc[1][:,0], label='x')
# plt.plot(b_acc[0], b_acc[1][:,1], label='y')
# plt.plot(b_acc[0], b_acc[1][:,2], label='z')
# plt.subplot(212)
# plt.title('gyro bias')
# plt.plot(b_gyro[0], b_gyro[1][:,0], label='x')
# plt.plot(b_gyro[0], b_gyro[1][:,1], label='y')
# plt.plot(b_gyro[0], b_gyro[1][:,2], label='z')

plt.figure(3)
plt.subplot(311)
plt.title('vel')
plt.plot(est_t, estimate[:,3], label='xhat')
plt.plot(truth_t, truth_vel[:,0], label='x')
plt.legend()
plt.subplot(312)
plt.plot(est_t, estimate[:,4], label='yhat')
plt.plot(truth_t, truth_vel[:,1], label='y')
plt.subplot(313)
plt.plot(est_t, estimate[:,5], label='zhat')
plt.plot(truth_t, truth_vel[:,2], label='z')

plt.figure(6)
plt.subplot(411)
plt.title('att')
plt.plot(est_t, estimate[:,6], label='what')
plt.plot(truth_t, truth_att[:,0], label='w')
plt.legend()
plt.subplot(412)
plt.plot(est_t, estimate[:,7], label='xhat')
plt.plot(truth_t, truth_att[:,1], label='x')
plt.subplot(413)
plt.plot(est_t, estimate[:,8], label='yhat')
plt.plot(truth_t, truth_att[:,2], label='y')
plt.subplot(414)
plt.plot(est_t, estimate[:,9], label='zhat')
plt.plot(truth_t, truth_att[:,3], label='z')

# plt.figure(7, figsize=(20,13))
# for i in range(ekf.len_features):
#     for j in range(3):
#         plt.subplot(4,ekf.len_features,j*ekf.len_features+i+1)
#         plt.plot(est_t, est_zeta[:,i,0], label='xhat')
#         plt.plot(truth_feature_t, truth_zeta[:,i,0], label="x")
#     if i == 0 and j == 0:
#         plt.title('zeta')
#         plt.legend()
#     plt.subplot(4,ekf.len_features,3*ekf.len_features+i+1)
#     plt.title('depth')
#     plt.plot(est_t, est_depth[:,i], label='1/rho hat')
#     plt.plot(truth_feature_t, truth_depth[:,i], label="1/rho")
#     if i == 0:
#         plt.legend()


plt.show()
# debug = 1







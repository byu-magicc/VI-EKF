import cPickle
from vi_ekf import *
import matplotlib.pyplot as plt
from tqdm import tqdm


data = cPickle.load(open('simulated_waypoints.pkl', 'rb'))

# State: pos, vel, att, b_gyro, b_acc, mu
x0 = np.concatenate([data['truth_NED']['pos'][0,:],
                     data['truth_NED']['vel'][0,:],
                     data['truth_NED']['att'][0,:],
                     np.zeros(3),
                     np.zeros(3),
                     0.2*np.ones(1)], axis=0)
ekf = VI_EKF(x0)

prev_time = 0
estimate = []
end = 1000.
for i, t in tqdm(enumerate(data['imu_data']['t'])):
    if prev_time == 0:
        prev_time = t
        continue
    if t > end:
        break

    dt = t - prev_time
    prev_time = t

    # Propagation step
    xhat = ekf.propagate(data['imu_data']['acc'][i,:], data['imu_data']['gyro'][i,:], dt)
    estimate.append(xhat)

# convert lists to np arrays
estimate = np.array(estimate)

est_t = data['imu_data']['t'][:estimate.shape[0]]

gyro = data['imu_data']['gyro'][est_t < end]
acc = data['imu_data']['acc'][est_t < end]
b_acc = [data['imu_acc_bias']['t'][data['imu_acc_bias']['t'] < end], data['imu_acc_bias']['v'][data['imu_acc_bias']['t'] < end]]
b_gyro = [data['imu_gyro_bias']['t'][data['imu_gyro_bias']['t'] < end], data['imu_gyro_bias']['v'][data['imu_gyro_bias']['t'] < end]]

truth_pos = data['truth_NED']['pos']
truth_vel = data['truth_NED']['vel']
truth_att = data['truth_NED']['att']
truth_t = data['truth_NED']['t']

truth_t = truth_t[truth_t < end]
truth_pos = truth_pos[truth_t < end]
truth_vel = truth_vel[truth_t < end]


# Plot
plt.figure(1)
plt.subplot(311)
plt.title('pos')
plt.plot(est_t, estimate[:,0], label='xhat')
plt.plot(truth_t, truth_pos[:,0], label='x')
plt.subplot(312)
plt.plot(est_t, estimate[:,1], label='yhat')
plt.plot(truth_t, truth_pos[:,1], label='y')
plt.subplot(313)
plt.plot(est_t, estimate[:,2], label='zhat')
plt.plot(truth_t, truth_pos[:,2], label='z')

plt.figure(2)
plt.title('acc')
plt.plot(est_t, acc[:,0], label='x')
plt.plot(est_t, acc[:,1], label='y')
plt.plot(est_t, acc[:,2], label='z')

plt.figure(5)
plt.title('gyro')
plt.plot(est_t, gyro[:,0], label='x')
plt.plot(est_t, gyro[:,1], label='y')
plt.plot(est_t, gyro[:,2], label='z')

plt.figure(4)
plt.subplot(211)
plt.title('acc bias')
plt.plot(b_acc[0], b_acc[1][:,0], label='x')
plt.plot(b_acc[0], b_acc[1][:,1], label='y')
plt.plot(b_acc[0], b_acc[1][:,2], label='z')
plt.subplot(212)
plt.title('gyro bias')
plt.plot(b_gyro[0], b_gyro[1][:,0], label='x')
plt.plot(b_gyro[0], b_gyro[1][:,1], label='y')
plt.plot(b_gyro[0], b_gyro[1][:,2], label='z')

plt.figure(3)
plt.subplot(311)
plt.title('vel')
plt.plot(est_t, estimate[:,3], label='xhat')
plt.plot(truth_t, truth_vel[:,0], label='x')
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
plt.subplot(412)
plt.plot(est_t, estimate[:,7], label='xhat')
plt.plot(truth_t, truth_att[:,1], label='x')
plt.subplot(413)
plt.plot(est_t, estimate[:,8], label='yhat')
plt.plot(truth_t, truth_att[:,2], label='y')
plt.subplot(414)
plt.plot(est_t, estimate[:,9], label='zhat')
plt.plot(truth_t, truth_att[:,3], label='z')




plt.show()



debug = 1



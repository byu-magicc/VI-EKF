import cPickle
from vi_ekf import *
import matplotlib.pyplot as plt


data = cPickle.load(open('simulated_waypoints.pkl', 'rb'))

# State: pos, vel, att, b_gyro, b_acc, mu
x0 = np.concatenate([data['truth_NED']['pos'][0,:],
                     data['truth_NED']['vel'][0,:],
                     data['truth_NED']['att'][0,:],
                     np.zeros(3),
                     np.zeros(3),
                     np.zeros(1)], axis=0)
ekf = VI_EKF(x0)

prev_time = 0
estimate = []
for i, t in enumerate(data['perfect_imu_data']['t']):
    if prev_time == 0:
        prev_time = t
        continue

    dt = t - prev_time
    prev_time = t

    # Propagation step
    xhat = ekf.propagate(data['perfect_imu_data']['acc'][i,:], data['perfect_imu_data']['gyro'][i,:], dt)
    estimate.append(xhat)

# convert lists to np arrays
estimate = np.array(estimate)

# Plot
plt.figure(1)
plt.title('pos')
plt.plot(data['perfect_imu_data']['t'], estimate[:,0], label='x')
plt.plot(data['perfect_imu_data']['t'], estimate[:,1], label='y')
plt.plot(data['perfect_imu_data']['t'], estimate[:,2], label='z')
plt.show()



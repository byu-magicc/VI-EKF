import numpy as np
from quaternion import Quaternion
import scipy.linalg

# State: pos, vel, att, b_gyro, b_acc, mu
class VI_EKF():
    def __init__(self, x0):
        self.x = x0

        self.Q = np.diag([0.001, 0.001, 0.001,
                          0.01, 0.01, 0.01,
                          0.001, 0.001, 0.001,
                          0.0001, 0.0001, 0.0001,
                          0.0001, 0.0001, 0.0001,
                          0.001])

        self.P = self.Q

        self.gravity = np.array([[0, 0, 9.80665]]).T
        self.khat = np.array([[0, 0, 1]]).T

    def boxplus(self, x, dx):
        out = np.zeros(17)
        out[0:6] = x[0:6] + dx[0:6]
        quat = Quaternion(x[6:10])

        norm_delta = scipy.linalg.norm(dx[6:9])
        if norm_delta > 1e-4:
            dquat = Quaternion(scalar=np.cos(norm_delta/2.), vector=np.sin(norm_delta/2.)*dx[6:9]/norm_delta)
            out[6:10] = (quat * dquat).elements
        else:
            dquat = Quaternion(scalar = 1., vector=dx[6:9]/2.)
            out[6:10] = (quat * dquat).unit.elements
        out[10:] = x[10:] + dx[9:]
        return out

    def propagate(self, y_acc, y_gyro, dt):
        xdot = self.f(self.x, y_acc, y_gyro)
        self.x = self.boxplus(self.x, xdot*dt)
        return self.x.copy()

    def set_imu_bias(self, b_g, b_a):
        self.x[10:13] = b_g
        self.x[13:16] = b_a

    def f(self, x, y_acc, y_gyro):
        vel = x[3:6]
        q_I_b = Quaternion(x[6:10])

        omega = y_gyro - x[10:13]
        acc = y_acc - x[13:16]
        mu = x[16]

        pdot = np.atleast_2d(q_I_b.rotate(vel))
        vdot = np.cross(vel, omega) + np.atleast_2d(acc) + np.atleast_2d(q_I_b.inverse.rotate(self.gravity))
        qdot = np.array([omega])

        return np.concatenate((pdot, vdot, qdot, np.zeros((1,7))), axis=1).flatten()

if __name__ == '__main__':
    from data_loader import *
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    # Load Data
    # data = load_data('data/mav0', show_image=False, start=16, end=30)
    # save_to_file('data/mav0/data.npy', data)
    data = load_from_file('data/mav0/data.npy')

    imu_t = data['imu'][:,0]
    gyro = data['imu'][:,1:4]
    acc = data['imu'][:,4:7]

    truth = data['truth']

    x0 = np.concatenate([truth[0,1:4,None], truth[0,8:11,None], truth[0,4:8,None], np.zeros((7,1))], axis=0)

    ekf = VI_EKF(x0)

    estimate = []

    truth_index = 0
    prev_time = imu_t[0].copy()
    imu_t = np.delete(imu_t, 0)
    for i, t in enumerate(imu_t):
        while truth[truth_index,0] < t and truth[truth_index,0] > prev_time:
            ekf.set_imu_bias(truth[11:14], truth[14:17])
        x = ekf.propagate(acc[i,:,None], gyro[i,:,None], t - prev_time)
        estimate.append(x.copy().flatten())
        prev_time = t

    estimate = np.array(estimate)


    truth_t = truth[:,0]

    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(truth[:,1], truth[:,2], truth[:,3], '-c', label='truth')
    plt.plot(estimate[:, 0], estimate[:, 1], estimate[:, 2], '-m', label='estimate')

    # Plot coordinate frame at origin
    origin = np.tile(truth[0, 1:4], (3, 1))
    axes = np.array([origin, origin + np.eye(3)])
    plt.plot(axes[:,0,0], axes[:, 0, 1], axes[:, 0, 2], '-r', label="x")
    plt.plot(axes[:,1,0], axes[:, 1, 1], axes[:, 1, 2], '-g', label="y")
    plt.plot(axes[:,2,0], axes[:, 2, 1], axes[:, 2, 2], '-b', label="z")
    plt.axis('equal')

    # plot position states
    plt.figure(1)
    for i in xrange(3):
        plt.subplot(3,1,i+1)
        plt.plot(truth[:,0], truth[:,i+1])
        plt.plot(imu_t, estimate[:, i])

    # plot acc and gyro biases
    plt.figure(2)
    for i in xrange(3):
        plt.subplot(3,2,2*i+1)
        plt.plot(truth[:,0], truth[:,i+11])
        plt.plot(imu_t, estimate[:, i+10])
        plt.subplot(3, 2, 2*i + 2)
        plt.plot(truth[:, 0], truth[:, i + 14])
        plt.plot(imu_t, estimate[:, i + 13])


    plt.show()

    debug = 1
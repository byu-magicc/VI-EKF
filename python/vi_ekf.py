import numpy as np
from quaternion import Quaternion
import scipy.linalg
import cv2

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
        assert x.shape == (17, 1) and dx.shape == (16, 1)

        out = np.zeros((17, 1))
        out[0:6] = x[0:6] + dx[0:6]
        quat = Quaternion(x[6:10])
        norm_delta = scipy.linalg.norm(dx[6:9])

        if norm_delta > 1e-4:
            dquat = Quaternion(scalar=np.cos(norm_delta/2.), vector=np.sin(norm_delta/2.)*dx[6:9]/norm_delta)
            out[6:10, 0] = (quat * dquat).elements
        else:
            dquat = Quaternion(scalar=1., vector=dx[6:9]/2.)
            out[6:10, 0] = (quat * dquat).unit.elements

        out[10:] = x[10:] + dx[9:]
        return out

    def propagate(self, y_acc, y_gyro, dt):
        assert y_acc.shape == (3, 1) and y_gyro.shape == (3, 1)

        xdot = self.f(self.x, y_acc, y_gyro)
        self.x = self.boxplus(self.x, xdot*dt)

        return self.x.copy()

    def set_imu_bias(self, b_g, b_a):
        self.x[10:13] = b_g
        self.x[13:16] = b_a

    def f(self, x, y_acc, y_gyro):
        assert x.shape == (17, 1) and y_acc.shape == (3,1) and y_gyro.shape == (3,1)

        vel = x[3:6]
        q_I_b = Quaternion(x[6:10])

        omega = y_gyro - x[10:13]
        acc = y_acc - x[13:16]
        mu = x[16, None]

        vdot = np.cross(vel, omega, axis=0) + acc + q_I_b.inverse.rotate(self.gravity)[:, None]
        qdot = omega
        pdot = q_I_b.rotate(vel)[:, None]

        return np.vstack((pdot, vdot, qdot, np.zeros((7, 1))))

if __name__ == '__main__':
    from data_loader import *
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import tqdm

    # Load Data
    #data = load_data('/mnt/pccfs/not_backed_up/eurocmav/mav0', show_image=False, start=16, end=30)
    #save_to_file('/mnt/pccfs/not_backed_up/eurocmav/mav0/data.npy', data)
    data = load_from_file('/mnt/pccfs/not_backed_up/eurocmav/mav0/data.npy')

    imu_t = data['imu'][:, 0]
    gyro = data['imu'][:, 1:4]
    acc = data['imu'][:, 4:7]
    cam_time = data['cam_time']

    def load_cam0(filename):
        image = cv2.imread(filename, 0)
        return cam0_undistort(image)

    def load_cam1(filename):
        image = cv2.imread(filename, 0)
        return cam0_undistort(image)

    cam0_undistort = make_undistort_funtion(intrinsics=data['cam0_sensor']['intrinsics'], resolution=data['cam0_sensor']['resolution'], distortion_coefficients=data['cam0_sensor']['distortion_coefficients'])
    cam1_undistort = make_undistort_funtion(intrinsics=data['cam1_sensor']['intrinsics'], resolution=data['cam1_sensor']['resolution'], distortion_coefficients=data['cam1_sensor']['distortion_coefficients'])

    truth = data['truth']

    x0 = np.vstack([truth[0, 1:4, None], truth[0, 8:11, None], truth[0, 4:8, None], np.zeros((7, 1))])

    ekf = VI_EKF(x0)

    estimate = []
    measurement_index = 1 # skip the first image so that diffs are always possible

    for i, (t, dt) in enumerate(tqdm.tqdm(zip(imu_t, np.diff(np.concatenate([[0], imu_t]))))):

        if measurement_index < cam_time.shape[0] and t > cam_time[measurement_index]:
            left = load_cam0(data['cam0_frame_filenames'][measurement_index])
            left_previous = load_cam0(data['cam0_frame_filenames'][measurement_index - 1])

            # very slow dense flow calculation
            # dense_flow = cv2.calcOpticalFlowFarneback(left_previous, left, 0.5, 3, 15, 3, 5, 1.2, 0)

            measurement_index += 1

        #while truth[truth_index,0] < t and truth[truth_index,0] > prev_time:
        #    ekf.set_imu_bias(truth[i, 11:14, None], truth[i, 14:17, None])
        x = ekf.propagate(acc[i, :, None], gyro[i, :, None], dt)

        estimate.append(x[:, 0])

    estimate = np.array(estimate)

    truth_t = truth[:, 0]

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
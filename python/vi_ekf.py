import numpy as np
from quaternion import Quaternion
import scipy.linalg
import cv2

class VI_EKF():
    def __init__(self, x0):
        assert x0.shape == (17, 1)

        # 17 main states
        # pos, vel, att, b_gyro, b_acc, mu
        self.x = x0

        # Process noise matrix for the 16 main delta states
        self.Q = np.diag([0.001, 0.001, 0.001,
                          0.01, 0.01, 0.01,
                          0.001, 0.001, 0.001,
                          0.0001, 0.0001, 0.0001,
                          0.0001, 0.0001, 0.0001,
                          0.001])

        # process noise matrix for the features (assumed all the same) 3x3
        self.Q_feat = np.diag([0.001, 0.001, 0.01]) # x, y, and depth

        # State covariances.  Size is (16 + 3N) x (16 + 3N) where N is the number of
        # features currently being tracked
        self.P = self.Q

        # gravity vector
        self.gravity = np.array([[0, 0, 9.80665]]).T

        # Unit vectors in the x, y, and z directions (used a lot for projection functions)
        self.ihat = np.array([[0, 0, 1]]).T
        self.jhat = np.array([[0, 0, 1]]).T
        self.khat = np.array([[0, 0, 1]]).T

        # The number of features currently being tracked
        self.len_features = 0

        # The next feature id to be assigned to a feature
        self.next_feature_id = 0

        # A map which corresponds to which feature id is occupying which index in the self.features
        # np array
        self.feature_ids = []

        # A numpy array holding 5N items for each feature, indexed according to the
        # feature_ids list
        self.features = np.zeros((0, 1))  # qw, qx, qy, qz, rho


    # Creates 3x2 projection matrix onto the plane perpendicular to zeta
    def T_zeta(self, q_zeta):
        assert q_zeta.shape == (4,1)

        quat_zeta = Quaternion(q_zeta)
        return np.array([quat_zeta.rotate(np.array([1, 0, 0])), quat_zeta.rotate(np.array([0, 1, 0]))])

    # Determines the quaternion which describes the rotation from the camera z-axis to the
    # unit bearing vector zeta
    def quat_from_zeta(self, zeta):
        assert zeta.shape == (3,1)
        assert (1.0 - scipy.linalg.norm(zeta)) < 0.00001

        quat_unnorm = np.concatenate((np.array([[self.khat.dot(zeta)]]),  np.cross(self.khat, zeta)), axis=0)
        return scipy.linalg.normalize(quat_unnorm)

    # Adds the state with the delta state on the manifold
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

    # propagates all states, features and covariances
    def propagate(self, y_acc, y_gyro, dt):
        assert y_acc.shape == (3, 1) and y_gyro.shape == (3, 1)

        xdot = self.f(self.x, y_acc, y_gyro)
        self.x = self.boxplus(self.x, xdot*dt)

        return self.x.copy()

    # Used for overriding imu biases, Not to be used in real life
    def set_imu_bias(self, b_g, b_a):
        assert b_g.shape == (3,1) and b_a.shape(3,1)
        self.x[10:13] = b_g
        self.x[13:16] = b_a

    # Used to initialize a new feature.  Returns the feature id associated with this feature
    def init_feature(self, zeta, depth):
        assert zeta.shape == (3, 1)

        self.len_features += 1
        self.feature_ids.append(self.next_feature_id)
        self.next_feature_id += 1
        quat_0 = self.quat_from_zeta(zeta)
        self.features = np.concatenate((self.features, quat_0, np.array([[depth]])), axis=0)
        return self.next_feature_id - 1

    # Used to remove a feature from the EKF.  Removes the feature from the features array and
    # Clears the associated rows and columns from the covariance.  The covariance matrix will
    # now be 3x3 smaller than before and the feature array will be 5 smaller
    def clear_feature(self, id):
        feature_index = self.feature_ids.index(id)
        mask = np.ones(len(self.features), dtype=bool)
        mask[[feature_index+i for i in range(5)]] = False
        self.features = self.features[mask,...]
        self.P = self.P[mask, mask]
        del self.feature_ids[feature_index]
        self.len_features -= 1

    # Determines the derivative of state x given inputs y_acc and y_gyro
    # the returned value of f is a delta state, and therefore is a different
    # size than the state
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
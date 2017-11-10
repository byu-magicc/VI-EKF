import numpy as np
from pyquat import Quaternion
from math_helper import skew, T_zeta
import scipy.linalg
import cv2

dxPOS = 0
dxVEL = 3
dxATT = 6
dxB_A = 9
dxB_G = 12
dxMU = 15
dxZ = 16

xPOS = 0
xVEL = 3
xATT = 6
xB_A = 10
xB_G = 13
xMU = 16
xZ = 17

class VI_EKF():
    def __init__(self, x0):
        assert x0.shape == (17, 1)

        # 17 main states + 5N feature states
        # pos, vel, att, b_gyro, b_acc, mu, q_feat, rho_feat, q_feat, rho_feat ...
        self.x = x0

        # Process noise matrix for the 16 main delta states
        self.Qx = np.diag([0.000, 0.000, 0.000,     # pos
                           0.01, 0.01, 0.01,        # vel
                           0.000, 0.000, 0.000,     # att
                           0.0001, 0.0001, 0.0001,  # b_acc
                           0.0001, 0.0001, 0.0001,  # b_omega
                           0.001])                  # mu

        # process noise matrix for the features (assumed all the same) 3x3
        self.Qx_feat = np.diag([0.000, 0.000, 0.00]) # x, y, and 1/depth

        # Process noise assumed from inputs (mechanized sensors)
        self.Qu = np.diag([0.01,                    # y_acc
                           0.001, 0.001, 0.001])    # y_omega



        # State covariances.  Size is (16 + 3N) x (16 + 3N) where N is the number of
        # features currently being tracked
        self.P = np.diag([0.001, 0.001, 0.001,      # pos
                           0.001, 0.001, 0.001,        # vel
                           0.001, 0.001, 0.001,     # att
                           0.0001, 0.0001, 0.0001,  # b_acc
                           0.0001, 0.0001, 0.0001,  # b_omega
                           0.01])                  # mu

        # Initial Covariance estimate for new features
        self.P0_feat = np.diag([0.01, 0.01, 0.1]) # x, y, and 1/depth

        # gravity vector
        self.gravity = np.array([[0, 0, 9.80665]]).T

        # Unit vectors in the x, y, and z directions (used a lot for projection functions)
        self.ihat = np.array([[1, 0, 0]]).T
        self.jhat = np.array([[0, 1, 0]]).T
        self.khat = np.array([[0, 0, 1]]).T

        # The number of features currently being tracked
        self.len_features = 0

        # The next feature id to be assigned to a feature
        self.next_feature_id = 0

        # A map which corresponds to which feature id is occupying which index in the state vector np array
        self.feature_ids = []

        # Body-to-Camera transform
        self.q_b_c = Quaternion(np.array([[1, 0, 0, 0]]).T) # Rotation from body to camera
        self.t_b_c = np.array([[0, 0, 0]]).T # translation from body to camera (in body frame)

    # Returns the depth to all features
    def get_depth(self):
        return 1./self.x[17+4::5]

    # Returns the estimated bearing vector to all features
    def get_zeta(self):
        zetas = np.zeros((self.len_features, 3))
        for i in range(self.len_features):
            qzeta = self.x[17 + 5 * i:17 + 5 * i + 4, :]  # 4-vector quaternion
            zetas[i] = Quaternion(qzeta).R.T[:,2].copy()  # 3-vector pointed at the feature in the camera frame
        return zetas

    # Returns the quaternion which
    def get_qzeta(self):
        qzetas = np.zeros((self.len_features, 4))
        for i in range(self.len_features):
            qzetas[i,:,None] = self.x[17+5*i:17+5*i+4]   # 4-vector quaternion
        return qzetas

    # Adds the state with the delta state on the manifold
    def boxplus(self, x, dx):
        # assert x.shape == (17+5*self.len_features, 1) and dx.shape == (16+3*self.len_features, 1)

        out = np.zeros((17+5*self.len_features, 1))

        # Add position and velocity vector states
        out[xPOS:xPOS+3] = x[xPOS:xPOS+3] + dx[xPOS:xPOS+3]
        out[xVEL:xVEL+3] = x[xVEL:xVEL+3] + dx[xVEL:xVEL+3]

        # Add attitude quaternion state on the manifold
        out[xATT:xATT+4] = (Quaternion(x[xATT:xATT+4]) + dx[dxATT:dxATT+3]).elements

        # add bias and drag term vector states
        out[xB_A:xB_A+3] = x[xB_A:xB_A+3] + dx[dxB_A:dxB_A+3]
        out[xB_G:xB_G+3] = x[xB_G:xB_G+3] + dx[dxB_G:dxB_G+3]
        out[xMU] = x[xMU] + dx[dxMU]



        # add Feature quaternion states
        for i in range(self.len_features):
            xFEAT = xZ+i*5
            xRHO = xZ+i*5+4
            dxFEAT = dxZ+3*i
            dxRHO = dxZ+3*i+2
            dqzeta = dx[dxFEAT:dxRHO,:]  # 2-vector which is the derivative of qzeta
            qzeta = x[xFEAT:xRHO,:] # 4-vector quaternion

            # Feature Quaternion States (use manifold)
            out[xFEAT:xRHO,:] = (Quaternion(qzeta).inverse + T_zeta(qzeta).dot(dqzeta)).inverse.elements

            # Inverse Depth State
            out[xRHO,:] = x[xRHO] + dx[dxRHO]
        return out

    # propagates all states, features and covariances
    def propagate(self, y_acc, y_gyro, dt):
        # assert y_acc.shape == (3, 1) and y_gyro.shape == (3, 1)

        # Propagate State
        xdot = self.f(self.x, y_acc, y_gyro)
        self.x = self.boxplus(self.x, xdot*dt)

        # Propagate Uncertainty
        # A = self.dfdx(self.x, y_acc, y_gyro)
        # G = self.dfdu(self.x)

        ## TODO: Convert to proper noise introduction (instead of additive noise on all states)
        # Pdot = A.dot(self.P) + self.P.dot(A.T) + G.dot(self.Qu).dot(G.T) + self.Qx
        # self.P += Pdot*dt

        return self.x.copy()

    # Used for overriding imu biases, Not to be used in real life
    def set_imu_bias(self, b_g, b_a):
        # assert b_g.shape == (3,1) and b_a.shape(3,1)
        self.x[xB_A:xB_A+3] = b_g
        self.x[xB_G:xB_G+3] = b_a

    # Used to initialize a new feature.  Returns the feature id associated with this feature
    def init_feature(self, zeta, depth):
        # assert zeta.shape == (3, 1)

        self.len_features += 1
        self.feature_ids.append(self.next_feature_id)
        self.next_feature_id += 1
        quat_0 = Quaternion().from_two_unit_vectors(zeta, self.khat).elements
        self.x = np.concatenate((self.x, quat_0, np.array([[1./depth]])), axis=0) # add 5 states to the state vector

        # Add three states to the process noise matrix
        self.Qx = scipy.linalg.block_diag(self.Qx, self.Qx_feat)
        self.P = scipy.linalg.block_diag(self.P, self.P0_feat)

        # self.q_zeta = Quaternion().from_two_unit_vectors(zeta, self.khat)

        return self.next_feature_id - 1

    # Used to remove a feature from the EKF.  Removes the feature from the features array and
    # Clears the associated rows and columns from the covariance.  The covariance matrix will
    # now be 3x3 smaller than before and the feature array will be 5 smaller
    def clear_feature(self, feature_id):
        feature_index = self.feature_ids.index(feature_id)
        mask = np.ones(len(self.x), dtype=bool)
        mask[[xZ+feature_index+i for i in range(5)]] = False
        self.x = self.x[mask,...]
        self.P = self.P[mask, mask]
        del self.feature_ids[feature_index]
        self.len_features -= 1

    # Determines the derivative of state x given inputs y_acc and y_gyro
    # the returned value of f is a delta state, delta features, and therefore is a different
    # size than the state and features and needs to be applied with boxplus
    def f(self, x, y_acc, y_gyro):
        # assert x.shape == (17+5*self.len_features, 1) and y_acc.shape == (3,1) and y_gyro.shape == (3,1)

        vel = x[xVEL:xVEL+3]
        q_I_b = Quaternion(x[xATT:xATT+4])

        omega = y_gyro - x[xB_G:xB_G+3]
        acc = y_acc - x[xB_A:xB_A+3]
        mu = x[xMU, None]

        vdot = acc + q_I_b.rot(self.gravity)
        qdot = omega
        pdot = q_I_b.rot(vel)

        feat_dot = np.zeros((3*self.len_features, 1))
        for i in range(self.len_features):
            xZETA_i = xZ+i*5
            xRHO_i = xZ+5*i+4
            dxZETA_i = i*3
            dxRHO_i = i*3+2

            q_zeta = x[xZETA_i:xRHO_i,:]
            rho = x[xRHO_i,0]
            zeta = Quaternion(q_zeta).rot(self.khat)
            vel_c_i = self.q_b_c.invrot(vel + skew(omega).dot(self.t_b_c))
            omega_c_i = omega

            # feature bearing vector dynamics
            feat_dot[dxZETA_i:dxRHO_i,:] = T_zeta(q_zeta).T.dot(rho*skew(zeta).dot(vel_c_i) + omega_c_i)

            # feature inverse depth dynamics
            feat_dot[dxRHO_i,:] = rho*rho*zeta.T.dot(vel_c_i)

        xdot = np.vstack((pdot, vdot, qdot, np.zeros((7, 1)), feat_dot))

        return xdot

    # Calculates the jacobian of the state dynamics with respect to the state.
    # this is used in propagating the state, and will return a matrix of size 16+3N x 16+3N
    def dfdx(self, x, y_acc, y_gyro):
        # assert x.shape == (17+5*self.len_features, 1) and y_acc.shape == (3,1) and y_gyro.shape == (3,1)

        vel = x[xVEL:xVEL+6]
        q_I_b = Quaternion(x[dxATT:dxATT+4])

        omega = y_gyro - x[dxB_G:dxB_G+3]
        acc = y_acc - x[dxB_A:dxB_A+3]
        mu = x[dxMU, None]

        A = np.zeros((dxZ+3*self.len_features, dxZ+3*self.len_features))

        # Position Partials
        A[dxPOS:dxPOS+3, dxVEL:dxVEL+3] = q_I_b.R.T
        A[dxPOS:dxPOS+3, dxATT:dxATT+3] = -skew(q_I_b.invrot(vel))

        # Velocity Partials
        A[dxVEL:dxVEL+3, dxVEL:dxVEL+3] = -skew(omega) # - mu * np.eye(3)
        A[dxVEL:dxVEL+3, dxATT:dxATT+3] = -skew(q_I_b.invrot(self.gravity))
        A[dxVEL:dxVEL+3, dxB_A:dxB_A+3] = -self.khat.dot(self.khat.T)
        A[dxVEL:dxVEL+3, dxB_G:dxB_G+3] = -skew(vel)
        A[dxVEL:dxVEL+3, dxMU, None] = vel

        # Attitude Partials
        A[dxATT:dxATT+3, dxB_G:dxB_G+3] = -np.eye(3)

        # Accel Bias Partials (constant)
        # Gyro Bias Partials (constant)
        # Drag Term Partials (constant)

        # Feature Terms Partials
        for i in range(self.len_features):
            dxZETA_i = dxZ + i * 3
            dxRHO_i = dxZ + i * 3 + 2

            q_zeta = x[i * 5 + xZ:i * 5 + 4 + xZ, :]
            rho = x[i * 5 + 4 + xZ, 0]

            zeta = Quaternion(q_zeta).invrot(self.khat)
            vel_c_i = self.q_b_c.invrot(vel + skew(omega).dot(self.t_b_c))
            omega_c_i = self.q_b_c.invrot(omega)
            T_z = T_zeta(q_zeta)
            skew_zeta = skew(zeta)
            R_b_c = self.q_b_c.R

            # Bearing Quaternion Partials
            A[dxZETA_i:dxZETA_i+3, dxVEL:dxVEL+3] = rho*T_z.T.dot(skew_zeta).dot(R_b_c)
            A[dxZETA_i:dxZETA_i+3, dxB_G:dxB_G+3] = T_z.T.dot(rho*skew_zeta.dot(R_b_c).dot(skew(self.t_b_c)) - R_b_c)
            A[dxZETA_i:dxZETA_i+3, dxZETA_i:dxZETA_i+3] = -T_z.T.dot(skew(omega_c_i - rho*skew(vel_c_i).dot(zeta)) + rho*skew(vel_c_i).dot(skew_zeta)).dot(T_z)
            A[dxZETA_i:dxZETA_i+3, dxRHO_i,None] = T_z.T.dot(skew_zeta).dot(vel_c_i)

            # Inverse Depth Partials
            A[dxRHO_i, dxVEL:dxVEL+3] = rho*rho*zeta.T.dot(R_b_c)
            A[dxRHO_i, dxB_G:dxB_G+3] = rho*rho*zeta.T.dot(R_b_c).dot(skew(self.t_b_c))
            A[dxRHO_i, dxZETA_i:dxZETA_i+3] = rho*rho*vel_c_i.T.dot(skew_zeta).dot(T_z)
            A[dxRHO_i, dxRHO_i] = 2*rho*zeta.T.dot(vel_c_i).squeeze()

        return A

    # Calculates the jacobian of the state dynamics with respect to the input noise.
    # this is used in propagating the state, and will return a matrix of size 16+3N x 4
    def dfdu(self, x):
        # assert x.shape == (17+5*self.len_features, 1)

        # Input indexes to make code easier to read
        Y_A = 0
        Y_W = 1

        G = np.zeros((16+3*self.len_features, 4))

        vel = x[3:6]
        q_I_b = Quaternion(x[6:10])

        # State partials
        G[dxVEL:dxATT, Y_A,None] = self.khat
        G[dxVEL:dxATT, Y_W:] = skew(vel)

        # Feature Partials
        for i in range(self.len_features):
            dxZETA_i = dxZ + i * 3
            dxRHO_i = dxZ + i * 3 + 2

            q_zeta = x[i * 5 + 17:i * 5 + 4 + 17, :]
            rho = x[i * 5 + 4 + 17, 0]
            zeta = Quaternion(q_zeta).rot(self.khat)

            skew_zeta = skew(zeta)
            R_b_c = self.q_b_c.R
            skew_t_b_c = skew(self.t_b_c)

            G[dxZETA_i:dxRHO_i, Y_W:] = rho*T_zeta(q_zeta).T.dot(-skew_zeta.dot(R_b_c).dot(skew_t_b_c) + R_b_c)
            G[dxRHO_i, Y_W:] = rho*rho*zeta.T.dot(R_b_c).dot(skew_t_b_c)

        return G

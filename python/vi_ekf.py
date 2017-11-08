import numpy as np
from quaternion import Quaternion
import scipy.linalg
import cv2

class VI_EKF():
    def __init__(self, x0, debug=False):
        self.debug = debug
        if self.debug:
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

    # Creates the skew-symmetric matrix from v
    def skew(self, v):
        if self.debug:
            assert v.shape == (3,1)
        return np.array([[0, -v[2,0], v[1,0]],
                         [v[2,0], 0, -v[0,0]],
                         [-v[2,0], v[0,0], 0]])

    # Creates 3x2 projection matrix onto the plane perpendicular to zeta
    def T_zeta(self, q_zeta):
        if self.debug:
            assert q_zeta.shape == (4,1)

        quat_zeta = Quaternion(q_zeta)
        return quat_zeta.R.T[:,0:2]

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
        if self.debug:
            assert x.shape == (17+5*self.len_features, 1) and dx.shape == (16+3*self.len_features, 1)

        out = np.zeros((17+5*self.len_features, 1))

        # Add position and velocity vector states
        out[0:6] = x[0:6] + dx[0:6]

        # Add attitude quaternion state on the manifold
        out[6:10] = (Quaternion(x[6:10]) + dx[6:9]).elements

        # add bias and drag term vector states
        out[10:17] = x[10:17] + dx[9:16]

        # add Feature quaternion states
        for i in range(self.len_features):
            xFEAT = 17+i*5
            xRHO = 17+i*5+4
            dxFEAT = 16+3*i
            dxRHO = 16+3*i+2
            dqzeta = dx[dxFEAT:dxRHO,:]  # 2-vector which is the derivative of qzeta
            qzeta = x[xFEAT:xRHO,:] # 4-vector quaternion

            # Feature Quaternion States
            out[xFEAT:xRHO,:] = (Quaternion(qzeta).inverse + self.T_zeta(qzeta).dot(dqzeta)).inverse.elements

            # Inverse Depth State
            out[xRHO,:] = x[xRHO] + dx[dxRHO]


        return out

    # propagates all states, features and covariances
    def propagate(self, y_acc, y_gyro, dt, truth):
        if self.debug:
            assert y_acc.shape == (3, 1) and y_gyro.shape == (3, 1)

        # Propagate State
        xdot = self.f(self.x, y_acc, y_gyro, truth)
        self.x = self.boxplus(self.x, xdot*dt)

        # Propagate Uncertainty
        # A = self.dfdx(self.x, y_acc, y_gyro)
        # G = self.dfdu(self.x)

        ## TOTO: Convert to proper noise introduction (instead of additive noise on all states)
        # Pdot = A.dot(self.P) + self.P.dot(A.T) + G.dot(self.Qu).dot(G.T) + self.Qx
        # self.P += Pdot*dt

        return self.x.copy()

    # Used for overriding imu biases, Not to be used in real life
    def set_imu_bias(self, b_g, b_a):
        if self.debug:
            assert b_g.shape == (3,1) and b_a.shape(3,1)
        self.x[10:13] = b_g
        self.x[13:16] = b_a

    # Used to initialize a new feature.  Returns the feature id associated with this feature
    def init_feature(self, zeta, depth):
        if self.debug:
            assert zeta.shape == (3, 1)

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
        mask[[17+feature_index+i for i in range(5)]] = False
        self.x = self.x[mask,...]
        self.P = self.P[mask, mask]
        del self.feature_ids[feature_index]
        self.len_features -= 1

    # Determines the derivative of state x given inputs y_acc and y_gyro
    # the returned value of f is a delta state, delta features, and therefore is a different
    # size than the state and features and needs to be applied with boxplus
    def f(self, x, y_acc, y_gyro, truth):
        if self.debug:
            assert x.shape == (17+5*self.len_features, 1) and y_acc.shape == (3,1) and y_gyro.shape == (3,1)

        vel = x[3:6]
        q_I_b = Quaternion(x[6:10])

        # q_I_b = Quaternion(truth)

        omega = y_gyro - x[10:13]
        acc = y_acc - x[13:16]
        mu = x[16, None]

        vdot = np.zeros((3,1)) #acc + q_I_b.invrot(self.gravity)
        qdot = omega
        pdot = q_I_b.invrot(vel)

        feat_dot = np.zeros((3*self.len_features, 1))
        for i in range(self.len_features):
            xZETA_i = 17+i*5
            xRHO_i = 17+5*i+4
            ftZETA_i = i*3
            ftRHO_i = i*3+2

            q_zeta = x[xZETA_i:xRHO_i,:]
            rho = x[xRHO_i,0]
            zeta = Quaternion(q_zeta).rot(self.khat)
            vel_c_i = self.q_b_c.invrot(vel + self.skew(omega).dot(self.t_b_c))
            omega_c_i = omega

            # feature bearing vector dynamics
            feat_dot[ftZETA_i:ftRHO_i,:] = self.T_zeta(q_zeta).T.dot(rho*self.skew(zeta).dot(vel_c_i) + omega_c_i)

            # feature inverse depth dynamics
            feat_dot[ftRHO_i,:] = rho*rho*zeta.T.dot(vel_c_i)

        xdot = np.vstack((pdot, vdot, qdot, np.zeros((7, 1)), feat_dot))

        return xdot

    # Calculates the jacobian of the state dynamics with respect to the state.
    # this is used in propagating the state, and will return a matrix of size 16+3N x 16+3N
    def dfdx(self, x, y_acc, y_gyro):
        if self.debug:
            assert x.shape == (17+5*self.len_features, 1) and y_acc.shape == (3,1) and y_gyro.shape == (3,1)

        POS = 0
        VEL = 3
        ATT = 6
        B_A = 9
        B_G = 12
        MU = 15
        Z = 16

        vel = x[3:6]
        q_I_b = Quaternion(x[6:10])

        omega = y_gyro - x[10:13]
        acc = y_acc - x[13:16]
        mu = x[16, None]

        A = np.zeros((16+3*self.len_features, 16+3*self.len_features))

        # Position Partials
        A[POS:VEL, VEL:ATT] = q_I_b.R.T
        A[POS:VEL, ATT:B_A] = -self.skew(q_I_b.invrot(vel))

        # Velocity Partials
        A[VEL:ATT, VEL:ATT] = -self.skew(omega) # - mu * np.eye(3)
        A[VEL:ATT, ATT:B_A] = -self.skew(q_I_b.invrot(self.gravity))
        A[VEL:ATT, B_A:B_G] = -self.khat.dot(self.khat.T)
        A[VEL:ATT, B_G:MU] = -self.skew(vel)
        A[VEL:ATT, MU, None] = vel

        # Attitude Partials
        A[ATT:B_A, B_G:MU] = -np.eye(3)

        # Accel Bias Partials (constant)
        # Gyro Bias Partials (constant)
        # Drag Term Partials (constant)

        # Feature Terms Partials
        for i in range(self.len_features):
            q_zeta = x[i * 5 + 17:i * 5 + 4 + 17, :]
            rho = x[i * 5 + 4 + 17, 0]
            zeta = Quaternion(q_zeta).invrot(self.khat)
            vel_c_i = self.q_b_c.invrot(vel + self.skew(omega).dot(self.t_b_c))
            omega_c_i = self.q_b_c.invrot(omega)
            ZETA_i = Z+i*3
            RHO_i = Z+i*3+2
            T_z = self.T_zeta(q_zeta)
            skew_zeta = self.skew(zeta)
            R_b_c = self.q_b_c.R

            # Bearing Quaternion Partials
            A[ZETA_i:RHO_i, VEL:ATT] = rho*T_z.T.dot(skew_zeta).dot(R_b_c)
            A[ZETA_i:RHO_i, B_G:MU] = T_z.T.dot(rho*skew_zeta.dot(R_b_c).dot(self.skew(self.t_b_c)) - R_b_c)
            A[ZETA_i:RHO_i, ZETA_i:RHO_i] = -T_z.T.dot(self.skew(omega_c_i - rho*self.skew(vel_c_i).dot(zeta)) + rho*self.skew(vel_c_i).dot(skew_zeta)).dot(T_z)
            A[ZETA_i:RHO_i, RHO_i,None] = T_z.T.dot(skew_zeta).dot(vel_c_i)

            # Inverse Depth Partials
            A[RHO_i, VEL:ATT] = rho*rho*zeta.T.dot(R_b_c)
            A[RHO_i, B_G:MU] = rho*rho*zeta.T.dot(R_b_c).dot(self.skew(self.t_b_c))
            A[RHO_i, ZETA_i:RHO_i] = rho*rho*vel_c_i.T.dot(skew_zeta).dot(T_z)
            A[RHO_i, RHO_i] = 2*rho*zeta.T.dot(vel_c_i).squeeze()

        return A

    # Calculates the jacobian of the state dynamics with respect to the input noise.
    # this is used in propagating the state, and will return a matrix of size 16+3N x 4
    def dfdu(self, x):
        if self.debug:
            assert x.shape == (17+5*self.len_features, 1)

        # State indexes to make code easier to read
        POS = 0
        VEL = 3
        ATT = 6
        B_A = 9
        B_G = 12
        MU = 15
        Z = 16

        # Input indexes to make code easier to read
        Y_A = 0
        Y_W = 1

        G = np.zeros((16+3*self.len_features, 4))

        vel = x[3:6]
        q_I_b = Quaternion(x[6:10])

        # State partials
        G[VEL:ATT, Y_A,None] = self.khat
        G[VEL:ATT, Y_W:] = self.skew(vel)

        # Feature Partials
        for i in range(self.len_features):
            q_zeta = x[i * 5 + 17:i * 5 + 4 + 17, :]
            rho = x[i * 5 + 4 + 17, 0]
            zeta = Quaternion(q_zeta).rot(self.khat)
            ZETA_i = Z+i*3
            RHO_i = Z+i*3+2
            skew_zeta = self.skew(zeta)
            R_b_c = self.q_b_c.R
            skew_t_b_c = self.skew(self.t_b_c)

            G[ZETA_i:RHO_i, Y_W:] = rho*self.T_zeta(q_zeta).T.dot(-skew_zeta.dot(R_b_c).dot(skew_t_b_c) + R_b_c)
            G[RHO_i, Y_W:] = rho*rho*zeta.T.dot(R_b_c).dot(skew_t_b_c)

        return G






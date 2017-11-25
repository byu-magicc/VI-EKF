import numpy as np
from pyquat import Quaternion
from math_helper import skew, T_zeta, norm
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

uA = 0
uG = 1

I_2x3 = np.array([[1, 0, 0],
                  [0, 1, 0]])
I_3x3 = np.eye(3)
I_2x2 = np.eye(2)

class VI_EKF():
    def __init__(self, x0):
        assert x0.shape == (xZ, 1)

        # 17 main states + 5N feature states
        # pos, vel, att, b_gyro, b_acc, mu, q_feat, rho_feat, q_feat, rho_feat ...
        self.x = x0

        # Process noise matrix for the 16 main delta states
        self.Qx = np.diag([0.000, 0.000, 0.000,     # pos
                           0.00, 0.00, 0.00,        # vel
                           0.000, 0.000, 0.000,     # att
                           0.0001, 0.0001, 0.0001,  # b_acc
                           0.0001, 0.0001, 0.0001,  # b_omega
                           0.001])                  # mu

        # process noise matrix for the features (assumed all the same) 3x3
        self.Qx_feat = np.diag([0.000, 0.000, 0.00]) # x, y, and 1/depth

        # Process noise assumed from inputs (mechanized sensors)
        self.Qu = np.diag([0.001,                    # y_acc
                           0.0001, 0.0001, 0.0001])    # y_omega



        # State covariances.  Size is (16 + 3N) x (16 + 3N) where N is the number of
        # features currently being tracked
        self.P = np.diag([0.001, 0.001, 0.001,      # pos
                           0.001, 0.001, 0.001,        # vel
                           0.001, 0.001, 0.001,     # att
                           0.000001, 0.000001, 0.000001,  # b_acc
                           0.0000001, 0.0000001, 0.0000001,  # b_omega
                           0.0001])                  # mu

        # Initial Covariance estimate for new features
        self.P0_feat = np.diag([0.01, 0.01, 0.1]) # x, y, and 1/depth

        # gravity vector (NED)
        self.gravity = np.array([[0, 0, 9.80665]]).T

        # Unit vectors in the x, y, and z directions (used a lot for projection functions)
        self.ihat = np.array([[1, 0, 0]]).T
        self.jhat = np.array([[0, 1, 0]]).T
        self.khat = np.array([[0, 0, 1]]).T

        # The number of features currently being tracked
        self.len_features = 0

        # The next feature id to be assigned to a feature
        self.next_feature_id = 0

        # Set of initialized feature ids
        self.initialized_features = set()
        self.global_to_local_feature_id = {}

        # A map which corresponds to which feature id is occupying which index in the state vector np array
        self.feature_ids = []

        # Body-to-Camera transform
        self.q_b_c = Quaternion(np.array([[1, 0, 0, 0]]).T) # Rotation from body to camera
        self.p_b_c = np.array([[0, 0, 0]]).T # translation from body to camera (in body frame)

        # Camera Parameters
        self.focal_len = 258 # made up

        self.measurement_functions = dict()
        self.measurement_functions['acc'] = self.h_acc
        self.measurement_functions['alt'] = self.h_alt
        self.measurement_functions['att'] = self.h_att
        self.measurement_functions['pos'] = self.h_pos
        self.measurement_functions['feat'] = self.h_feat
        self.measurement_functions['pixel_vel'] = self.h_pixel_vel
        self.measurement_functions['depth'] = self.h_depth
        self.measurement_functions['inv_depth'] = self.h_inv_depth

        # Matrix Workspace
        self.A = np.zeros((dxZ, dxZ))
        self.G = np.zeros((dxZ, 4))
        self.I_big = np.eye(dxZ)

    # Returns the depth to all features
    def get_depth(self):
        return 1./self.x[xZ+4::5]

    # Returns the estimated bearing vector to all features
    def get_zeta(self):
        zetas = np.zeros((self.len_features, 3))
        for i in range(self.len_features):
            qzeta = self.x[xZ + 5 * i:xZ + 5 * i + 4, :]  # 4-vector quaternion
            zetas[i] = Quaternion(qzeta).rot(self.khat)  # 3-vector pointed at the feature in the camera frame
        return zetas

    # Returns the quaternion which
    def get_qzeta(self):
        qzetas = np.zeros((self.len_features, 4))
        for i in range(self.len_features):
            qzetas[i,:,None] = self.x[xZ+5*i:xZ+5*i+4]   # 4-vector quaternion
        return qzetas

    # Adds the state with the delta state on the manifold
    def boxplus(self, x, dx):
        assert  x.shape == (xZ+5*self.len_features, 1) and dx.shape == (dxZ+3*self.len_features, 1)

        out = np.zeros((xZ+5*self.len_features, 1))

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
            out[xFEAT:xRHO,:] = (Quaternion(qzeta).inverse + (T_zeta(qzeta).dot(dqzeta))).inverse.elements

            # Inverse Depth State
            out[xRHO,:] = x[xRHO] + dx[dxRHO]
        return out

    # propagates all states, features and covariances
    def propagate(self, y_acc, y_gyro, dt):
        assert y_acc.shape == (3, 1) and y_gyro.shape == (3, 1) and isinstance(dt, float)

        # Propagate State
        u = np.vstack((y_acc[2], y_gyro))
        xdot = self.f(self.x, u)
        self.x = self.boxplus(self.x, xdot*dt)

        # Propagate Uncertainty
        A = self.dfdx(self.x, u)
        G = self.dfdu(self.x)

        # TODO: Convert to proper noise introduction (instead of additive noise on all states)
        Pdot = A.dot(self.P) + self.P.dot(A.T) + G.dot(self.Qu).dot(G.T) + self.Qx
        self.P += Pdot*dt

        if np.isnan(self.P).any():
            debug = 1

        return self.x.copy(), self.P.copy()

    def update(self, z, measurement_type, R, passive=False, **kwargs):
        assert measurement_type in self.measurement_functions.keys(), "Unknown Measurement Type"

        passive_update = passive

        if measurement_type == 'feat':
            if kwargs['i'] not in self.initialized_features:
                self.init_feature(z, id=kwargs['i'], depth=(kwargs['depth'] if 'depth' in kwargs else np.array([[1.0]])))

        # Feature Points need a slightly modified update process because of the non-vectorness of the measurement
        zhat, H = self.measurement_functions[measurement_type](self.x, **kwargs)

        if measurement_type == 'feat':
            # For features, we have to do boxminus on the weird space between quaternions
            i = self.global_to_local_feature_id[kwargs['i']]
            xZETA_i = xZ + 5 * i
            T_z = T_zeta(self.x[xZETA_i:xZETA_i + 4])
            zhat_x_z = skew(zhat).dot(z)
            residual = T_z.T.dot(np.arccos(zhat.T.dot(z)) * zhat_x_z / norm(zhat_x_z))

        elif measurement_type == 'att':
            residual = Quaternion(z) - Quaternion(zhat)
        else:
            residual = z - zhat

        # Residual Saturation
        # residual[residual > 0.1] = 0.1
        # residual[residual < -0.1] = -0.1
        if not passive_update:
            try:
                K = self.P.dot(H.T).dot(scipy.linalg.inv(R + H.dot(self.P).dot(H.T)))
            except:
                debug = 1
            self.P = (self.I_big - K.dot(H)).dot(self.P)
            self.x = self.boxplus(self.x, K.dot(residual))
        return zhat

    # Used for overriding imu biases, Not to be used in real life
    def set_imu_bias(self, b_g, b_a):
        assert b_g.shape == (3,1) and b_a.shape == (3,1)
        self.x[xB_G:xB_G+3] = b_g
        self.x[xB_A:xB_A+3] = b_a

    # Used to initialize a new feature.  Returns the feature id associated with this feature
    def init_feature(self, zeta, id, depth=None):
        assert zeta.shape == (3, 1) and abs(1.0 - norm(zeta)) < 1e-3
        assert depth.shape == (1, 1)

        self.len_features += 1
        self.feature_ids.append(self.next_feature_id)
        self.next_feature_id += 1
        quat_0 = Quaternion.from_two_unit_vectors(self.khat,zeta).elements
        self.x = np.vstack((self.x, quat_0, 1./depth)) # add 5 states to the state vector

        # Add three states to the process noise matrix
        self.Qx = scipy.linalg.block_diag(self.Qx, self.Qx_feat)
        self.P = scipy.linalg.block_diag(self.P, self.P0_feat)

        # Set up the matrices to work with
        self.A = np.zeros((dxZ + 3 * self.len_features, dxZ + 3 * self.len_features))
        self.G = np.zeros((dxZ + 3 * self.len_features, 4))
        self.I_big = np.eye(dxZ+3*self.len_features)

        self.initialized_features.add(id)
        self.global_to_local_feature_id[id] = self.next_feature_id - 1

        return self.next_feature_id - 1

    # Used to remove a feature from the EKF.  Removes the feature from the features array and
    # Clears the associated rows and columns from the covariance.  The covariance matrix will
    # now be 3x3 smaller than before and the feature array will be 5 smaller
    def clear_feature(self, id):
        self.initialized_features.remove(id)
        feature_id = self.global_to_local_feature_id[id]
        feature_index = self.feature_ids.index(feature_id)
        mask = np.ones(len(self.x), dtype=bool)
        mask[[xZ+feature_index+i for i in range(5)]] = False
        self.x = self.x[mask,...]
        self.P = self.P[mask, mask]
        del self.feature_ids[feature_index]
        self.len_features -= 1

        # Matrix Workspace Modifications
        self.A = np.zeros((dxZ + 3 * self.len_features, dxZ + 3 * self.len_features))
        self.G = np.zeros((dxZ+3*self.len_features, 4))
        self.I_big = np.eye(dxZ + 3 * self.len_features)


    def keep_only_features(self, features):
        features_to_clear = self.initialized_features.difference(set(features))
        for f in features_to_clear:
            self.clear_feature(f)

    # Determines the derivative of state x given inputs u
    # the returned value of f is a delta state, delta features, and therefore is a different
    # size than the state and features and needs to be applied with boxplus
    def f(self, x, u):
        assert x.shape == (xZ+5*self.len_features, 1) and u.shape == (4,1)

        vel = x[xVEL:xVEL+3]
        q_I_b = Quaternion(x[xATT:xATT+4])

        omega = u[uG:uG+3] - x[xB_G:xB_G+3]
        y_acc_z = u[uA, 0] - x[xB_A+2, 0]
        acc_z = np.array([[0, 0, y_acc_z]]).T
        mu = x[xMU, 0]

        pdot = q_I_b.invrot(vel)
        vdot = skew(vel).dot(omega) - mu*I_2x3.T.dot(I_2x3).dot(vel) + acc_z + q_I_b.rot(self.gravity)
        # pdot = np.zeros((3,1))
        # vdot = np.zeros((3, 1))
        qdot = omega

        feat_dot = np.zeros((3*self.len_features, 1))
        for i in range(self.len_features):
            xZETA_i = xZ+i*5
            xRHO_i = xZ+5*i+4
            dxZETA_i = i*3
            dxRHO_i = i*3+2

            q_zeta = x[xZETA_i:xZETA_i+4,:]
            rho = x[xRHO_i,0]
            zeta = Quaternion(q_zeta).rot(self.khat)
            vel_c_i = self.q_b_c.rot(vel + skew(omega).dot(self.p_b_c))
            omega_c_i = self.q_b_c.rot(omega)

            # feature bearing vector dynamics
            feat_dot[dxZETA_i:dxZETA_i+2,:] = T_zeta(q_zeta).T.dot(rho*skew(zeta).dot(vel_c_i) + omega_c_i)
            # feat_dot[dxZETA_i:dxRHO_i,:] = T_zeta(q_zeta).T.dot(omega)

            # feature inverse depth dynamics
            feat_dot[dxRHO_i,:] = rho*rho*zeta.T.dot(vel_c_i)

        xdot = np.vstack((pdot, vdot, qdot, np.zeros((7, 1)), feat_dot))

        return xdot

    # Calculates the jacobian of the state dynamics with respect to the state.
    # this is used in propagating the state, and will return a matrix of size 16+3N x 16+3N
    def dfdx(self, x, u):
        assert x.shape == (xZ+5*self.len_features, 1) and u.shape == (4,1)

        vel = x[xVEL:xVEL+3]
        q_I_b = Quaternion(x[xATT:xATT+4])

        omega = u[uG:uG+3] - x[xB_G:xB_G+3]
        mu = x[xMU, None]

        self.A *= 0.0

        # Position Partials
        self.A[dxPOS:dxPOS+3, dxVEL:dxVEL+3] = q_I_b.R
        self.A[dxPOS:dxPOS+3, dxATT:dxATT+3] = skew(q_I_b.invrot(vel))

        # Velocity Partials
        self.A[dxVEL:dxVEL+3, dxVEL:dxVEL+3] = -skew(omega) - mu*I_2x3.T.dot(I_2x3)
        self.A[dxVEL:dxVEL+3, dxATT:dxATT+3] = -q_I_b.R.T.dot(skew(self.gravity))
        self.A[dxVEL:dxVEL+3, dxB_A:dxB_A+3] = -self.khat.dot(self.khat.T)
        self.A[dxVEL:dxVEL+3, dxB_G:dxB_G+3] = -skew(vel)
        self.A[dxVEL:dxVEL+3, dxMU, None] = -I_2x3.T.dot(I_2x3).dot(vel)

        # Attitude Partials
        self.A[dxATT:dxATT+3, dxB_G:dxB_G+3] = -I_3x3

        # Accel Bias Partials (constant)
        # Gyro Bias Partials (constant)
        # Drag Term Partials (constant)

        # Feature Terms Partials
        for i in range(self.len_features):
            dxZETA_i = dxZ + i * 3
            dxRHO_i = dxZ + i * 3 + 2
            xZETA_i = xZ + i * 5
            xRHO_i = xZ + 5 * i + 4

            q_zeta = x[xZETA_i:xZETA_i+4, :]
            rho = x[xRHO_i, 0]

            zeta = Quaternion(q_zeta).rot(self.khat)
            vel_c_i = self.q_b_c.invrot(vel + skew(omega).dot(self.p_b_c))
            omega_c_i = self.q_b_c.invrot(omega)
            T_z = T_zeta(q_zeta)
            skew_zeta = skew(zeta)
            skew_vel_c = skew(vel_c_i)
            R_b_c = self.q_b_c.R

            # Bearing Quaternion Partials
            self.A[dxZETA_i:dxZETA_i+2, dxVEL:dxVEL+3] = rho*T_z.T.dot(skew_zeta).dot(R_b_c)
            self.A[dxZETA_i:dxZETA_i+2, dxB_G:dxB_G+3] = T_z.T.dot(rho*skew_zeta.dot(R_b_c).dot(skew(self.p_b_c)) - R_b_c)
            self.A[dxZETA_i:dxZETA_i+2, dxZETA_i:dxZETA_i+2] = -T_z.T.dot(skew(rho*skew_vel_c.dot(zeta) + omega_c_i) + (rho*skew_vel_c.dot(skew_zeta))).dot(T_z)
            self.A[dxZETA_i:dxZETA_i+2, dxRHO_i,None] = T_z.T.dot(skew_zeta).dot(vel_c_i)

            # Inverse Depth Partials
            rho2 = rho*rho
            self.A[dxRHO_i, dxVEL:dxVEL+3] = rho2*zeta.T.dot(R_b_c)
            self.A[dxRHO_i, dxB_G:dxB_G+3] = rho2*zeta.T.dot(R_b_c).dot(skew(self.p_b_c))
            self.A[dxRHO_i, dxZETA_i:dxZETA_i+2] = rho2*vel_c_i.T.dot(skew_zeta).dot(T_z)
            self.A[dxRHO_i, dxRHO_i] = 2*rho*zeta.T.dot(vel_c_i).squeeze()

        if np.isnan(self.A).any() or (self.A > 1e50).any():
            debug = 1


        return self.A

    # Calculates the jacobian of the state dynamics with respect to the input noise.
    # this is used in propagating the state, and will return a matrix of size 16+3N x 4
    def dfdu(self, x):
        assert x.shape == (xZ+5*self.len_features, 1)


        vel = x[xVEL:xVEL+3]
        q_I_b = Quaternion(x[xATT:xATT+4])

        self.G *= 0.0

        # State partials
        self.G[dxVEL:dxVEL+3, uA, None] = self.khat
        self.G[dxVEL:dxVEL+3, uG:uG+3] = skew(vel)
        self.G[dxATT:dxATT+3, uG:uG+3] = I_3x3

        # Feature Partials
        for i in range(self.len_features):
            dxZETA_i = dxZ + i * 3
            dxRHO_i = dxZ + i * 3 + 2

            q_zeta = x[i * 5 + xZ:i * 5 + 4 + xZ, :]
            rho = x[i * 5 + 4 + xZ, 0]
            zeta = Quaternion(q_zeta).rot(self.khat)

            skew_zeta = skew(zeta)
            R_b_c = self.q_b_c.R
            skew_p_b_c = skew(self.p_b_c)

            self.G[dxZETA_i:dxZETA_i+2, uG:uG+3] = T_zeta(q_zeta).T.dot(R_b_c - rho*skew_zeta.dot(R_b_c).dot(skew_p_b_c))
            self.G[dxRHO_i, uG:] = rho*rho*zeta.T.dot(R_b_c).dot(skew_p_b_c)

        if np.isnan(self.G).any():
            debug = 1

        return self.G

    # Accelerometer model
    # Returns estimated measurement (2 x 1) and Jacobian (2 x 16+3N)
    def h_acc(self, x):
        assert x.shape==(xZ+5*self.len_features,1)

        vel = x[xVEL:xVEL + 3]
        b_a = x[xB_A:xB_A + 3]
        mu = x[xMU, 0]

        h = I_2x3.dot(-mu*vel + b_a)

        dhdx = np.zeros((2, dxZ+3*self.len_features))
        dhdx[:,dxVEL:dxVEL+3] = -mu * I_2x3
        dhdx[:,dxB_A:dxB_A+3] = I_2x3
        dhdx[:,dxMU,None] = I_2x3.dot(-vel)

        return h, dhdx

    # Altimeter model
    # Returns estimated measurement (1x1) and Jacobian (1 x 16+3N)
    def h_alt(self, x):
        assert x.shape == (xZ + 5 * self.len_features, 1)

        pos = x[xPOS:xPOS + 3]

        h = -self.khat.T.dot(pos)

        dhdx = np.zeros((1, dxZ+3*self.len_features))
        dhdx[0,dxPOS:dxPOS+3] = -self.khat.T

        return h, dhdx

    # Attitude Model
    # Returns the estimated attitude measurement (4x1) and Jacobian (3 x 16+3N)
    def h_att(self, x):
        assert x.shape == (xZ + 5 * self.len_features, 1)

        h = x[xATT:xATT + 4]

        dhdx = np.zeros((3, dxZ + 3 *self.len_features))
        dhdx[:,dxATT:dxATT+3] = I_3x3

        return h, dhdx

    # Position Model
    # Returns the estimated Position measurement (3x1) and Jacobian (3 x 16+3N)
    def h_pos(self, x):
        assert x.shape == (xZ + 5 * self.len_features, 1)

        h = x[xPOS:xPOS + 3]

        dhdx = np.zeros((3, dxZ + 3 * self.len_features))
        dhdx[:, dxPOS:dxPOS + 3] = I_3x3

        return h, dhdx

    # Feature model for feature index i
    # Returns estimated measurement (3x1) and Jacobian (3 x 16+3N)
    def h_feat(self, x, **kwargs):
        i = self.global_to_local_feature_id[kwargs['i']]
        assert x.shape == (xZ + 5 * self.len_features, 1) and isinstance(i, int)
        dxZETA_i = dxZ + i * 3
        q_c_z = x[xZ+i*5:xZ+i*5+4]

        h = q_c_z

        dhdx = np.zeros((2, dxZ+3*self.len_features))
        dhdx[:, dxZETA_i:dxZETA_i+2] = I_2x2

        return h, dhdx

    # Feature depth measurement
    # Returns estimated measurement (1x1) and Jacobian (1 x 16+3N)
    def h_depth(self, x, i):
        assert x.shape == (xZ + 5 * self.len_features, 1) and isinstance(i, int)
        rho = x[xZ+i*5+4,0]

        h = np.array([[1.0/rho]])

        dhdx = np.zeros((1, dxZ+3*self.len_features))
        dhdx[0, dxZ+3*i+2,None] = -1/(rho*rho)

        return h, dhdx

    # Feature inverse depth measurement
    # Returns estimated measurement (1x1) and Jacobian (1 x 16+3N)
    def h_inv_depth(self, x, i):
        assert x.shape == (xZ + 5 * self.len_features, 1) and isinstance(i, int)
        h = x[xZ+i*5+4,None]

        dhdx = np.zeros((1, dxZ+3*self.len_features))
        dhdx[0, dxZ+3*i+2] = 1

        return h, dhdx

    # Feature pixel velocity measurement
    # Returns estimated measurement (2x1) and Jacobian (2 x 16+3N)
    def h_pixel_vel(self, x, i, u):
        assert x.shape == (xZ + 5 * self.len_features, 1) and isinstance(i, int) and u.shape == (4, 1)

        vel = x[xVEL:xVEL + 3]
        omega = u[uG:uG+3] - x[xB_G:xB_G+3]
        q_c_z = x[xZ+i*5:xZ+i*5+4]
        rho = x[xZ+i*5+4]
        zeta = Quaternion(q_c_z).rot(self.khat)

        sk_vel = skew(vel)
        sk_ez = skew(self.khat)
        sk_zeta = skew(zeta)
        R_b_c = self.q_b_c.R

        # TODO: Need to convert to camera dynamics

        h = -self.focal_len*I_2x3.dot(sk_ez).dot(rho*(sk_zeta.dot(vel)) + omega)

        ZETA_i = dxZ+3*i
        RHO_i = dxZ+3*i+2
        dhdx = np.zeros((2,dxZ+3*self.len_features))
        dhdx[:,dxVEL:dxVEL+3] = -self.focal_len*rho*I_2x3.dot(sk_ez).dot(sk_zeta)
        dhdx[:,ZETA_i:ZETA_i+2] = self.focal_len*rho*I_2x3.dot(sk_ez).dot(sk_vel).dot(sk_zeta).dot(T_zeta(q_c_z))
        dhdx[:,RHO_i,None] = -self.focal_len*I_2x3.dot(sk_ez).dot(sk_zeta).dot(vel)
        dhdx[:,dxB_G:dxB_G+3] = self.focal_len*I_2x3.dot(sk_ez).dot(R_b_c - rho*sk_zeta.dot(R_b_c).dot(skew(self.p_b_c)))
        # dhdx[:, dxB_G:dxB_G + 3] = self.focal_len * I_2x3.dot(sk_ez).dot(I_zz)


        return h, dhdx



from pyquat import Quaternion
import numpy as np

def norm(v, axis=None):
    return np.sqrt(np.sum(v*v, axis=axis))

cross_matrix = np.array([[[0, 0, 0],
                          [0, 0, -1.0],
                          [0, 1.0, 0]],
                         [[0, 0, 1.0],
                          [0, 0, 0],
                          [-1.0, 0, 0]],
                         [[0, -1.0, 0],
                          [1.0, 0, 0],
                          [0, 0, 0]]])

e_z = np.array([[0, 0, 1.0]]).T

# Creates the skew-symmetric matrix from v
def skew(v):
    assert v.shape == (3, 1)
    return cross_matrix.dot(v).squeeze()

# Creates 3x2 projection matrix onto the plane perpendicular to zeta
e_x_e_y = np.array([[1., 0, 0], [0, 1., 0]]).T
def T_zeta(q_zeta):
    assert q_zeta.shape == (4,1)
    # Rotate onto the coordinate frame aligned with the feature vector
    # (This is where the Lie Group is linearized about)
    return Quaternion(q_zeta).rot(e_x_e_y)

def q_feat_boxminus(q0, q1):
    assert q0.shape == (4,1) and q1.shape == (4,1)
    T_z0 = T_zeta(q0)
    zeta0 = Quaternion(q0).rot(e_z)
    zeta1 = Quaternion(q1).rot(e_z)

    if norm(zeta0 - zeta1) > 1e-16:
        z0_x_z1 = skew(zeta0).dot(zeta1)
        theta = np.arccos(zeta0.T.dot(zeta1)) * z0_x_z1/norm(z0_x_z1)
        return T_z0.T.dot(theta)
    else:
        return np.zeros((2,1))


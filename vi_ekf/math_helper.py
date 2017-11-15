from pyquat import Quaternion
import numpy as np

def norm(v):
    return np.sqrt(np.sum(v*v))

cross_matrix = np.array([[[0, 0, 0],
                          [0, 0, -1.0],
                          [0, 1.0, 0]],
                         [[0, 0, 1.0],
                          [0, 0, 0],
                          [-1.0, 0, 0]],
                         [[0, -1.0, 0],
                          [1, 0, 0],
                          [0, 0, 0]]])

# Creates the skew-symmetric matrix from v
def skew(v):
    assert v.shape == (3, 1)
    return cross_matrix.dot(v).squeeze()

# Creates 3x2 projection matrix onto the plane perpendicular to zeta
e_x_e_y = np.array([[1., 0, 0], [0, 1., 0]]).T
def T_zeta(q_zeta):
    assert q_zeta.shape == (4,1)
    return Quaternion(q_zeta).invrot(e_x_e_y)


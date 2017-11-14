from pyquat import Quaternion
import numpy as np

def norm(v):
    return np.sqrt(np.sum(v*v))

# Creates the skew-symmetric matrix from v
def skew(v):
    assert v.shape == (3,1)
    return np.array([[0, -v[2,0], v[1,0]],
                     [v[2,0], 0, -v[0,0]],
                     [-v[1,0], v[0,0], 0]])

# Creates 3x2 projection matrix onto the plane perpendicular to zeta
def T_zeta(q_zeta):
    assert q_zeta.shape == (4,1)

    quat_zeta = Quaternion(q_zeta)
    return quat_zeta.R.T[:,0:2]


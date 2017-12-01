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
    assert v.shape[0] == 3
    return cross_matrix.dot(v).squeeze()

# Creates 3x2 projection matrix onto the plane perpendicular to zeta
e_x_e_y = np.array([[1., 0, 0], [0, 1., 0]]).T
def T_zeta(q_zeta):
    assert q_zeta.shape == (4,1)
    # The coordinate basis normal to the feature vector, expressed in the camera frame (hence the active rotation)
    # (This is where the Lie Group is linearized about)
    return Quaternion(q_zeta).rot(e_x_e_y)

def q_feat_boxminus(q0, q1):
    assert q0.shape == (4,1) and q1.shape == (4,1)
    zeta0 = Quaternion(q0).rot(e_z)
    zeta1 = Quaternion(q1).rot(e_z)

    if norm(zeta0 - zeta1) > 1e-16:
        z0_x_z1 = skew(zeta1).dot(zeta0)
        v = z0_x_z1 / norm(z0_x_z1) # The vector about which rotation occurs (normalized)
        theta = np.arccos(zeta1.T.dot(zeta0)) # the magnitude of the rotation
        # The rotation vector exists in the plane normal to the feature vector.  Therefore if we rotate to this
        # basis, then all the information is stored in the x and y components only.  This reduces the dimensionality
        # of the delta-feature quaternion
        return theta * T_zeta(q1).T.dot(v)
    else:
        return np.zeros((2,1))

def q_feat_boxplus(q, delta):
    assert q.shape == (4,1) and delta.shape == (2,1)
    # the delta vector is expressed in the plane normal to the feature vector described by q.  Therefore we have to
    # rotate back into the camera frame and add this delta-q.  What we will do is create this delta quaternion from the
    # axis-angle approximation encoded by delta and add it
    v = T_zeta(Quaternion(q).inverse.elements).dot(delta)
    angle = norm(v)
    if angle < 1e-16:
        return q
    else:
        axis = v/angle
        dq = Quaternion.from_axis_angle(axis, angle)
    return (Quaternion(q) * dq).elements

# Calculates the quaternion which rotates u into v.
# That is, if q = q_from_two_unit_vectors(u,v)
# q.rot(u) = v and q.invrot(v) = u
# This is a vectorized version which returns multiple quaternions for multiple v's from one u
def q_array_from_two_unit_vectors(u, v):
    assert u.shape[0] == 3
    assert v.shape[0] == 3
    u = u.copy()
    v = v.copy()

    num_arrays = v.shape[1]
    arr = np.vstack((np.ones((1, num_arrays)), np.zeros((3, num_arrays))))

    d = u.T.dot(v)

    invs = (2.0*(1.0+d))**-0.5
    xyz = skew(u).dot(v)*invs
    arr[0, :, None] = 0.5 / invs.T
    arr[1:,:] = xyz
    arr /= norm(arr, axis=0)
    return arr

if __name__ == '__main__':
    # run some math helper tests

    # Test vectorized quat from two unit vectors
    v1 = np.random.uniform(-1, 1, (3, 100))
    v2 = np.random.uniform(-1, 1, (3, 1))
    v3 = np.random.uniform(-1, 1, (3, 1))
    v1 /= norm(v1, axis=0)
    v2 /= norm(v2)
    v3 /= norm(v3)
    # On a single vector
    assert norm(Quaternion(q_array_from_two_unit_vectors(v3, v2)).rot(v3) - v2) < 1e-8
    # on a bunch of vectors
    quat_array = q_array_from_two_unit_vectors(v2, v1)
    for q, v in zip(quat_array.T, v1.T):
        Quaternion(q[:,None]).rot(v2) - v[:,None]

    # Test T_zeta
    assert norm(T_zeta(q_array_from_two_unit_vectors(e_z, v2)).T.dot(v2)) < 1e-8


    zeta = np.random.uniform(-1, 1, (3,1))
    zeta2 = np.random.uniform(-1, 1, (3, 1))
    zeta[2] *= np.sign(zeta[2])  # Make sure it is feasible that this zeta is in the camera frame
    zeta2[2] *= np.sign(zeta2[2])  # Make sure it is feasible that this zeta is in the camera frame
    zeta /= norm(zeta)
    zeta2 /= norm(zeta2)
    qzeta = q_array_from_two_unit_vectors(e_z, zeta)
    qzeta2 = q_array_from_two_unit_vectors(e_z, zeta2)

    dqzeta = np.random.normal(-0.25, 0.25, (2,1))

    assert norm( q_feat_boxplus(qzeta, np.zeros((2,1))) - qzeta) < 1e-8
    # print q_feat_boxplus(qzeta, q_feat_boxminus(qzeta2, qzeta)).T
    print qzeta2.T
    assert norm( q_feat_boxplus(qzeta, q_feat_boxminus(qzeta2, qzeta)) - qzeta2) < 1e-8
    # assert norm( q_feat_boxminus(q_feat_boxplus(qzeta, dqzeta), qzeta) - dqzeta) < 1e-8





    print "math helper test: [PASSED]"




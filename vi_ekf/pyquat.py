import numpy as np

cross_matrix = np.array([[[0, 0, 0],
                          [0, 0, -1.0],
                          [0, 1.0, 0]],
                         [[0, 0, 1.0],
                          [0, 0, 0],
                          [-1.0, 0, 0]],
                         [[0, -1.0, 0],
                          [1, 0, 0],
                          [0, 0, 0]]])

qmat_matrix = np.array([[[1.0, 0, 0, 0],
                         [0, -1.0, 0, 0],
                         [0, 0, -1.0, 0],
                         [0, 0, 0, -1.0]],
                        [[0, 1.0, 0, 0],
                         [1.0, 0, 0, 0],
                         [0, 0, 0, 1.0],
                         [0, 0, -1.0, 0]],
                        [[0, 0, 1.0, 0],
                         [0, 0, 0, -1.0],
                         [1.0, 0, 0, 0],
                         [0, 1.0, 0, 0]],
                        [[0, 0, 0, 1.0],
                         [0, 0, 1.0, 0],
                         [0, -1.0, 0, 0],
                         [1.0, 0, 0, 0]]])

def quat_arr_to_euler(array):
    assert array.shape[0] == 4
    w = array[0,:]
    x = array[1,:]
    y = array[2,:]
    z = array[3,:]
    return np.array([np.arctan2(2.0 * (w * x + y * z), 1. - 2. * (x * x + y * y)),
                     np.arcsin(2.0 * (w * y - z * x)),
                     np.arctan2(2.0 * (w * z + x * y), 1. - 2. * (y * y + z * z))])



def skew(v):
    assert v.shape == (3, 1)
    return cross_matrix.dot(v).squeeze()

def norm(v, axis=None):
    return np.sqrt(np.sum(v*v, axis=axis))

class Quaternion():
    def __init__(self, v):
        assert isinstance(v, np.ndarray)
        assert v.shape == (4,1)
        self.arr = v

    def __str__(self):
        return "[ " + str(self.arr[0,0]) + ", " + str(self.arr[1,0]) + "i, " \
               + str(self.arr[2,0]) + "j, " + str(self.arr[3,0]) + "k ]"

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        return self.otimes(other)

    def __imul__(self, other):
        self.arr = qmat_matrix.dot(other.arr).squeeze().dot(self.arr)
        return self

    def __add__(self, other):
        return self.boxplus(other)

    def __iadd__(self, other):
        assert other.shape == (3, 1)
        delta = other.copy()

        # norm_delta = norm(delta)

        # If we aren't going to run into numerical issues
        # if norm_delta > 1e-4:
        #     v = np.sin(norm_delta / 2.) * (delta / norm_delta)
        #     self.arr = qmat_matrix.dot(np.vstack((np.cos(norm_delta / 2.0), v))).squeeze().dot(self.arr)
        # else:
        arrdot = 0.5*qmat_matrix.dot(np.vstack((np.zeros((1, 1)), delta))).squeeze().dot(self.arr)
        self.arr += arrdot
        self.arr /= norm(self.arr)
        return self

    def __sub__(self, other):
        return self.log(other.inverse.otimes(self))

    @property
    def w(self):
        return self.arr[0,0]

    @property
    def x(self):
        return self.arr[1, 0]

    @property
    def y(self):
        return self.arr[2, 0]

    @property
    def z(self):
        return self.arr[3, 0]

    @property
    def elements(self):
        return self.arr

    @property
    def euler(self):
        w = self.arr[0, 0]
        x = self.arr[1, 0]
        y = self.arr[2, 0]
        z = self.arr[3, 0]
        return np.array([[np.arctan2(2.0*(w*x + y*z), 1. - 2.*(x*x + y*y))],
                         [np.arcsin(2.0*(w*y - z*x))],
                         [np.arctan2(2.0*(w*z + x*y), 1. - 2.*(y*y + z*z))]])


    # Calculates the rotation matrix equivalent.  If you are performing a rotation,
    # is is much faster to use rot or invrot
    @property
    def R(self):
        w = self.arr[0,0]
        x = self.arr[1,0]
        y = self.arr[2,0]
        z = self.arr[3,0]

        wx = w*x
        wy = w*y
        wz = w*z
        xx = x*x
        xy = x*y
        xz = x*z
        yy = y*y
        yz = y*z
        zz = z*z

        return np.array([[1. - 2.*yy - 2.*zz, 2.*xy - 2.*wz, 2.*xz + 2.*wy],
                         [2.*xy + 2.*wz, 1. - 2.*xx - 2.*zz, 2.*yz - 2.*wx],
                         [2.*xz - 2.*wy, 2.*yz + 2.*wx, 1. - 2.*xx - 2.*yy]])

    # Calculates the quaternion exponential map for a 3-vector.
    # Returns a quaternion
    @staticmethod
    def qexp(v):
        assert v.shape == (3,1)
        v = v.copy()

        norm_v = norm(v)
        # If we aren't going to run into numerical issues
        if norm_v > 1e-4:
            v = np.sin(norm_v / 2.) * v / norm_v
            exp_quat = Quaternion(np.array([[np.cos(norm_v / 2.0), v[0, 0], v[1, 0], v[2, 0]]]).T)
        else:
            v /= 2.0
            exp_quat = Quaternion(np.array([[1.0, v[0, 0], v[1, 0], v[2, 0]]]).T)
            exp_quat.normalize()
        return exp_quat

    @staticmethod
    def log(q):
        assert isinstance(q, Quaternion)

        v = q.arr[1:]
        w = q.arr[0,0]
        norm_v = norm(v)

        return 2.0*np.arctan2(norm_v, w)*v/norm_v

    def copy(self):
        q_copy = Quaternion(self.arr.copy())
        return q_copy

    def normalize(self):
        self.arr /= norm(self.arr)

    # Perform a passive rotation on v (same as q.R.dot(v), but faster) CONFIRMED
    def rot(self, v):
        assert v.shape[0] == 3
        start_norm = norm(v, axis=0)
        skew_xyz = skew(self.arr[1:])
        t = 2.0 * skew_xyz.dot(v)
        out = v + self.arr[0,0] * t + skew_xyz.dot(t)
        out_norm = norm(out, axis=0)
        if ((out_norm - start_norm) > 1e-3*start_norm).any():
            debug = 1
        return out

    # Perform an active rotation on v (same as q.R.T.dot(v), but faster) CONFIRMED
    def invrot(self, v):
        assert v.shape[0] == 3
        skew_xyz = skew(self.arr[1:])
        t = 2.0 * skew_xyz.dot(v)
        return v - self.arr[0,0] * t + skew_xyz.dot(t)

    def inv(self):
        self.arr[1:] *= -1.0

    @property
    def inverse(self):
        inverted = self.arr.copy()
        inverted[1:] *= -1.0
        return Quaternion(inverted)

    @staticmethod
    def from_two_unit_vectors(u, v):
        assert u.shape == (3,1)
        assert v.shape == (3,1)
        u = u.copy()
        v = v.copy()

        arr = np.array([[1., 0., 0., 0.]]).T

        d = u.T.dot(v).squeeze()
        if d < 1.0:
            invs = (2.0*(1.0+d))**-0.5
            xyz = skew(u).dot(v)*invs.squeeze()
            arr[0,0]=0.5/invs
            arr[1:,:] = xyz
            arr /= norm(arr)
        return Quaternion(arr)

    @staticmethod
    def from_R(m):
        q = np.zeros((4,1))
        tr = np.trace(m)

        if tr > 0:
            S = np.sqrt(tr+1.0) * 2.
            q[0] = 0.25 * S
            q[1] = (m[2,1] - m[1,2]) / S
            q[2] = (m[0,2] - m[2,0]) / S
            q[3] = (m[1,0] - m[0,1]) / S
        elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.
            q[0] = (m[2,1] - m[1,2]) / S
            q[1] = 0.25 * S
            q[2] = (m[0,1] + m[1,0]) / S
            q[3] = (m[0,2] + m[2,0]) / S
        elif m[1,1] > m[2,2]:
            S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.
            q[0] = (m[0,2] - m[2,0]) / S
            q[1] = (m[0,1] + m[1,0]) / S
            q[2] = 0.25 * S
            q[3] = (m[1,2] + m[2,1]) / S
        else:
            S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.
            q[0] = (m[1,0] - m[0,1]) / S
            q[1] = (m[0,2] + m[2,0]) / S
            q[2] = (m[1,2] + m[2,1]) / S
            q[3] = 0.25 * S
        return Quaternion(q)

    def otimes(self, q):
        q_new = Quaternion(qmat_matrix.dot(q.arr).squeeze().dot(self.arr).copy())
        return q_new

    def boxplus(self, delta):
        assert delta.shape == (3,1)
        delta = delta.copy()

        # norm_delta = norm(delta)

        # If we aren't going to run into numerical issues
        # if norm_delta > 1e-4:
        #     v = np.sin(norm_delta / 2.) * (delta / norm_delta)
        #     out_arr = qmat_matrix.dot(np.vstack((np.cos(norm_delta/2.0), v))).squeeze().dot(self.arr)
        # else:
        arrdot = 0.5*qmat_matrix.dot(np.vstack((np.zeros((1,1)), delta))).squeeze().dot(self.arr)
        out_arr = self.arr.copy() + arrdot
        out_arr /= norm(out_arr)
        return Quaternion(out_arr)
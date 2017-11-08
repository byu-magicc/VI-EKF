import numpy as np
import scipy.linalg

class Quaternion():
    def __init__(self, *args):
        if len(args) == 0:
            self.arr = np.array([[1., 0., 0., 0.]]).T
        elif isinstance(args[0], np.ndarray):
            assert args[0].shape == (4,1)
            self.arr = args[0].copy()
        elif len(args[0]) == 4:
            self.arr = np.array([args[0]]).T

    def __str__(self):
        return "[ " + str(self.arr[0,0]) + ", " + str(self.arr[1,0]) + "i, " \
               + str(self.arr[2,0]) + "j, " + str(self.arr[3,0]) + "k ]"

    def __mul__(self, other):
        return self.otimes(other)

    def __add__(self, other):
        return self.boxplus(other)

    def __float__(self):
        return self.arr

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
        return self.arr.copy()

    @property
    def R(self):
        w = self.arr[0,0]
        x = self.arr[1,0]
        y = self.arr[2,0]
        z = self.arr[3,0]

        return np.array([[[1. - 2.*y*y - 2.*z*z], [2.*x*y - 2.*z*w], [2.*x*z + 2.*y*w]],
                         [[2.*x*y + 2.*z*w], [1. - 2.*x*x - 2.*z*z], [2.*y*z - 2.*x*w]],
                         [[2.*x*z - 2.*y*w], [2.*y*z + 2.*x*w], [1. - 2.*x*x - 2.*y*y]]]).squeeze()

    @staticmethod
    def qexp(v):
        # assert v.shape == (3,1)
        v = v.copy()

        norm_v = scipy.linalg.norm(v)
        # If we aren't going to run into numerical issues
        if norm_v > 1e-4:
            v = np.sin(norm_v / 2.) * v / norm_v
            exp_quat = Quaternion([np.cos(norm_v / 2.0), v[0, 0], v[1, 0], v[2, 0]])
        else:
            v = v / 2.0
            exp_quat = Quaternion([1.0, v[0, 0], v[1, 0], v[2, 0]])
            exp_quat.normalize()
        return exp_quat

    def copy(self):
        q_copy = Quaternion(self.arr.copy())
        return q_copy

    def normalize(self):
        self.arr /= scipy.linalg.norm(self.arr)

    def rot(self, v):
        # assert v.shape == (3,1)
        v = v.copy()

        w = self.arr[0,0]
        x = self.arr[1,0]
        y = self.arr[2,0]
        z = self.arr[3,0]

        return np.array([[(1.0 - 2.0*y*y - 2.0*z*z) * v[0,0] + (2.0*(x*y + w*z))*v[1,0] + 2.0*(x*z - w*y)*v[2,0]],
                         [(2.0*(x*y - w*z)) * v[0,0] + (1.0 - 2.0*x*x - 2.0*z*z) * v[1,0] + 2.0*(y*z + w*x)*v[2,0]],
                         [(2.0*(x*z + w*y)) * v[0,0] + 2.0*(y*z - w*x)*v[1,0] + (1.0 - 2.0*x*x - 2.0*y*y)*v[2,0]]])

    def invrot(self, v):
        # assert v.shape == (3,1)
        v = v.copy()

        w = self.arr[0,0]
        x = -self.arr[1,0]
        y = -self.arr[2,0]
        z = -self.arr[3,0]

        return np.array([[(1.0 - 2.0*y*y - 2.0*z*z) * v[0,0] + (2.0*(x*y + w*z))*v[1,0] + 2.0*(x*z - w*y)*v[2,0]],
                         [(2.0*(x*y - w*z)) * v[0,0] + (1.0 - 2.0*x*x - 2.0*z*z) * v[1,0] + 2.0*(y*z + w*x)*v[2,0]],
                         [(2.0*(x*z + w*y)) * v[0,0] + 2.0*(y*z - w*x)*v[1,0] + (1.0 - 2.0*x*x - 2.0*y*y)*v[2,0]]])

    def invert(self):
        self.arr[1:] *= -1.0

    @property
    def inverse(self):
        inverted = self.arr.copy()
        inverted[1:] *= -1.0
        return Quaternion(inverted)

    def from_two_unit_vectors(self, u, v):
        # assert u.shape == (3,1)
        # assert v.shape == (3,1)
        u = u.copy()
        v = v.copy()

        d = u.T.dot(v).squeeze()
        if d >= 1.0:
            self.arr = np.array([[1., 0., 0., 0.]]).T
        else:
            invs = (2.0*(1.0+d))**-0.5
            xyz = np.cross(u, v, axis=0)*invs.squeeze()
            self.arr[0,0]=0.5/invs
            self.arr[1:,:] = xyz
            self.normalize()
        return self

    def otimes(self, q):
        # assert isinstance(q, Quaternion)
        # q = q.copy()

        w = self.arr[0,0]
        x = self.arr[1,0]
        y = self.arr[2,0]
        z = self.arr[3,0]
        return Quaternion([w * q.w - x * q.x - y * q.y - z * q.z,
                           w * q.x + x * q.w - y * q.z + z * q.y,
                           w * q.y + x * q.z + y * q.w - z * q.x,
                           w * q.z - x * q.y + y * q.x + z * q.w])

    def boxplus(self, delta):
        # assert delta.shape == (3,1)
        delta = delta.copy()

        norm_delta = scipy.linalg.norm(delta)

        # If we aren't going to run into numerical issues
        if norm_delta > 1e-4:
            v = np.sin(norm_delta / 2.) * (delta / norm_delta)
            dquat = Quaternion([np.cos(norm_delta/2.0), v[0,0], v[1,0], v[2,0]])
            self.arr = (self * dquat).elements
        else:
            v = (delta / 2.0)
            dquat = Quaternion([1.0, v[0,0], v[1,0], v[2,0]])
            self.arr = (self * dquat).elements
            self.normalize()
        return self

if __name__ == '__main__':
    q = Quaternion(np.array([[1, 0, 0, 0]]).T)
    print q
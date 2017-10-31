import numpy as np
import scipy.linalg

class Quaternion():
    def __init__(self, *args):
        if len(args) == 0:
            self.arr = np.array([[1, 0, 0, 0]]).T
        else:
            arr = args[0]
            assert arr.shape == (4,1)
            self.arr = arr

    def __str__(self):
        return str(self.arr[0,0]) + ", " + str(self.arr[1,0]) + "i, " \
               + str(self.arr[2,0]) + "j, " + str(self.arr[3,0]) + "k"

    def __mul__(self, other):
        return self.otimes(other)

    def __add__(self, other):
        return self.boxplus(other)

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

    def normalize(self):
        self.arr /= scipy.linalg.norm(self.arr)

    def rotate(self, v):
        assert v.shape == (3,1)

        w = self.arr[0,0]
        x = self.arr[1,0]
        y = self.arr[2,0]
        z = self.arr[3,0]

        return np.array([[(1.0 - 2.0*y*y - 2.0*z*z) * v[0,0] + (2.0*(x*y + w*z))*v[1,0] + 2.0*(x*z - w*y)*v[2,0]],
                         [(2.0*(x*y - w*z)) * v[0,0] + (1.0 - 2.0*x*x - 2.0*z*z) * v[1,0] + 2.0*(y*z + w*x)*v[2,0]],
                         [(2.0*(x*z + w*y)) * v[0,0] + 2.0*(y*z - w*x)*v[1,0] + (1.0 - 2.0*x*x - 2.0*y*y)*v[2,0]]])

    def invert(self):
        self.arr[1:] *= -1.0

    @property
    def inverse(self):
        inverted = self.arr
        inverted[1:] *= -1.0
        return Quaternion(inverted)

    def from_two_unit_vectors(self, u, v):
        assert u.shape == (3,1)
        assert v.shape == (3,1)
        d = u.T.dot(v).squeeze()
        if d >= 1.0:
            self.arr = np.array([[1, 0, 0, 0]]).T
        else:
            invs = (2.0*(1.0+d))**-0.5
            xyz = np.cross(v, u, axis=0)*invs.squeeze()
            self.arr = np.array([[0.5/invs, xyz[0], xyz[1], xyz[2]]]).T
            self.normalize()
        return self

    def otimes(self, q):
        assert isinstance(q, Quaternion)

        w = self.arr[0,0]
        x = self.arr[1,0]
        y = self.arr[2,0]
        z = self.arr[3,0]
        return Quaternion(np.array([[w * q.w - x * q.x - y * q.y - z * q.z],
                                    [w * q.x + x * q.w - y * q.z + z * q.y],
                                    [w * q.y + x * q.z + y * q.w - z * q.x],
                                    [w * q.z - x * q.y + y * q.x + z * q.w]]))

    def boxplus(self, delta):
        assert delta.shape == (3,1)

        norm_delta = scipy.linalg.norm(delta)

        # If we aren't going to run into numerical issues
        if norm_delta > 1e-4:
            v = np.sin(norm_delta / 2.) * delta / norm_delta
            dquat = Quaternion(np.array([[np.cos(norm_delta/2.0), v[0,0], v[1,0], v[2,0]]]).T)
            self.arr = (self * dquat).elements
        else:
            v = delta / 2.0
            dquat = Quaternion(np.array([[1.0, v[0,0], v[1,0], v[2,0]]]).T)
            self.arr = (self * dquat).elements
            self.normalize()
        return self








if __name__ == '__main__':
    q = Quaternion(np.array([[1, 0, 0, 0]]).T)
    print q
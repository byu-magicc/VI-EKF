import numpy as np
from pyquaternion import Quaternion
import scipy.linalg
from sklearn.preprocessing import normalize


class QuadcopterSim():
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.quat = Quaternion()
        self.beta_g = np.zeros(3)
        self.beta_a = np.zeros(3)
        self.omega = np.zeros(3)
        self.acc = np.zeros(3)

        self.gravity = np.array([0, 0, 9.80665])
        self.khat = np.array([0, 0, 1])

        self.mu = 0.1
        self.ang_mu = 0.1

        self.mass = 1.0
        self.Inertia = 0.3*np.eye(3)
        self.invInertia = scipy.linalg.inverse(self.Inertia)

        self.eta_vel = 0.01
        self.eta_b_g = 0.0001
        self.eta_b_a = 0.0001
        self.eta_omega = 0.01

        self.L = np.array([[4., 4., -10.],
                           [10., 15., -3.],
                           [25., 17., -9.],
                           [-25., -12., -20.],
                           [-16., -34., -10.]])

        self.x = np.concatenate((self.pos, self.vel, self.quat.elements, self.beta_g, self.beta_a, self.omega))

    def boxplus(self, x, deltax):
        out = x.copy()
        out[0:6] += deltax[0:6]
        out[6:11] = Quaternion.exp_map(Quaternion(x[6:11]), deltax[6:10]).elements
        out[11:] += deltax[10:]
        return out

    def dynamics(self, x, force, torques):
        vel = x[3:6]
        omega = x[17:20]
        quat = Quaternion(x[6:10])
        posdot = quat.inverse.rotate(vel)
        veldot = vel.cross(omega) + quat.rotate(self.gravity) - force * self.khat - vel*self.mu/self.mass + self.eta_vel*np.random.randn(3)
        qdot = omega
        beta_gdot = self.eta_b_g*np.random.randn(3)
        beta_adot = self.eta_b_g*np.random.randn(3)
        omegadot = (torques- omega*self.ang_mu).dot(self.invInertia) + self.eta_omega*np.random.randn(3)

        xdot = np.concatenate((posdot, veldot, qdot, beta_gdot, beta_adot, omegadot))
        return xdot

    def propagate(self, dt, force, torques):
        k1 = self.dynamics(self.x, force, torques)
        k2 = self.dynamics(self.boxplus(self.x + k1*dt/2.0), force, torques)
        k3 = self.dynamics(self.boxplus(self.x + k2*dt/2.0), force, torques)
        k4 = self.dynamics(self.boxplus(self.x + k3*dt), force, torques)

        xdot = 1./6. * (k1  + 2.0*k2 + 2.0*k3 + k4)
        self.acc = xdot[3:6]
        self.x = self.boxplus(self.x, xdot)

    def get_state(self):
        return self.x

    def get_acc(self):
        return self.acc

    def get_gyro(self):
        return self.omega

    def get_bearing_vectors(self):
        num_landmarks = self.l.shape[0]
        delta = normalize(self.l - np.tile(self.x[0:3], (num_landmarks, 1)), axis=1)
        return delta
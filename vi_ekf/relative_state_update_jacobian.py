from pyquat import Quaternion, norm, skew
import numpy as np

e = 1e-6

## Scratch remove axis of rotation from quaternion
u = np.array([[0, 0, 1.]]).T
# u = np.random.random((3,1))
u /= norm(u)
qm0 = Quaternion.from_euler(270.0 * np.pi / 180.0, 85.0 * np.pi / 180.0, 90.0 * np.pi / 180.0)

w = qm0.rot(u)
th = u.T.dot(w)
ve = skew(u).dot(w)
qp0 = Quaternion.exp(th * ve)

# RN Paper
phi = qm0.euler[0]
theta = qm0.euler[1]
psi = qm0.euler[2]

N = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
              [0, np.cos(phi) ** 2, -np.cos(phi) * np.sin(phi)],
              [0, -np.cos(phi) * np.sin(phi), np.sin(phi) ** 2]])

epsilon = np.eye(3) * e

t = u.T.dot(qm0.rot(u))
v = skew(u).dot(qm0.rot(u))

tv0 = u.T.dot(qm0.rot(u)) * (skew(u).dot(qm0.rot(u)))
a_dtvdq = (skew(u).dot(qm0.R).dot(u)).dot(v.T) + t * (skew(u).dot(qm0.R).dot(skew(u).T))
d_dtvdq = np.zeros_like(a_dtvdq)

nd = norm(t * v)
d0 = t * v
qd0 = Quaternion.exp(d0)
skd = skew(t * v)
Tau = np.eye(3) + ((1. - np.cos(nd)) * skd) / (nd * nd) + ((nd - np.sin(nd)) * skd.dot(skd)) / (nd * nd * nd)
Tau_approx = np.eye(3) + 1. / 2. * skd
a_dqdq = a_dtvdq.T.dot(Tau)
a_dqdq_approx = a_dtvdq.dot(Tau_approx)
d_dqdq = np.zeros_like(a_dqdq)

a_dexpdd = Tau
d_dexpdd = np.zeros_like(a_dexpdd)

for i in range(3):
    qmi = qm0 + epsilon[:, i, None]
    w = qmi.rot(u)
    th = u.T.dot(w)
    ve = skew(u).dot(w)
    qpi = Quaternion.exp(th * ve)
    d_dqdq[i, :, None] = (qpi - qp0) / e

    qdi = Quaternion.exp(d0 + epsilon[:, i, None])
    d_dexpdd[i, :, None] = (qdi - qd0) / e

    tvi = u.T.dot(qmi.rot(u)) * (skew(u).dot(qmi.rot(u)))
    d_dtvdq[i, :, None] = (tvi - tv0) / e

print "analytical:\n", np.around(a_dqdq, 5)
# print "approx:\n", np.around(a_dqdq_approx, 5)
print "finite difference:\n", np.around(d_dqdq, 5)
# print "dan:\n", np.around(N,5)

# print "bonus A:\n", np.around(a_dtvdq, 5)
# print "bonus FD:\n", np.around(d_dtvdq, 5)
print "bonus diff:\n", np.sum(a_dtvdq - d_dtvdq)
#
# print "magic A:\n", np.around(Tau, 5)
# print "magic FD:\n", np.around(d_dexpdd, 5)
print "magic diff:\n", np.sum(Tau - d_dexpdd)
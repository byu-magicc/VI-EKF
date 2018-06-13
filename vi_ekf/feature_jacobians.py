#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from pyquat import *
import scipy.linalg
import sys
from math_helper import T_zeta, q_feat_boxminus, q_feat_boxplus

RST   = "\x1B[0m"
KRED  = "\x1B[31m"
KGRN  = "\x1B[32m"
KYEL  = "\x1B[33m"
KBLU  = "\x1B[34m"
KMAG  = "\x1B[35m"
KCYN  = "\x1B[36m"
KWHT  = "\x1B[37m"
def FRED(x): return KRED + x + RST
def FGRN(x): return KGRN + x + RST
def FYEL(x): return KYEL + x + RST
def FBLU(x): return KBLU + x + RST
def FMAG(x): return KMAG + x + RST
def FCYN(x): return KCYN + x + RST
def FWHT(x): return KWHT + x + RST
def BOLD(x): return "\x1B[1m" + x + RST
def UNDL(x): return "\x1B[4m" + x + RST


khat = np.array([[0, 0, 1.]]).T
# Sample calibration matrix




def f_pi(zeta, K):
    return 1.0/zeta[2,0]*K.dot(zeta)[:2,:]

def J_pi(zeta, K):
    kz = zeta[2,0]
    return (-K.dot(zeta).dot((khat.T))/(kz*kz) + 1/kz * K)[:2,:]

def f_pi_inv(eta, Kinv):
    v = Kinv.dot(np.vstack((eta, np.array([[1]]))))
    v /= norm(v)
    return v

def J_pi_inv(eta, Kinv):
    v = Kinv.dot(np.vstack((eta, np.array([[1]]))))
    nv = norm(v)
    I = np.eye(3)
    return ((I * nv - v.dot(v.T)/nv)/(nv*nv)).dot(Kinv).dot(I[:,:2])

def qz(zeta):
    skk = skew(khat)
    return Quaternion.exp(np.arccos(khat.T.dot(zeta)) * skk.dot(zeta)/norm(skk.dot(zeta)))

def f_dyn(zeta, w, p, v, dt):
    qzeta = qz(zeta)
    return (q_feat_boxplus(qzeta, (-T_zeta(qzeta).T.dot(w + p*skew(zeta).dot(v))*dt))).R.T.dot(khat)

def jac_dyn(zeta, w, p, v, dt):
    return -skew(f_r(zeta, w, p, v, dt)) #.dot(jac_m(zeta)).dot(jac_r(zeta, w, p, v, dt))
    # return qz(zeta).R.dot(jac_r(zeta, w, p, v, dt))

def f_l(zeta):
    return (skew(khat).dot(zeta))/norm(skew(khat).dot(zeta))

def jac_l(zeta):
    v = skew(khat).dot(zeta)
    nv = norm(v)
    return ((np.eye(3) * nv - v.dot(v.T)/nv)/(nv*nv)).dot(skew(khat))

def f_s(zeta):
    return khat.T.dot(zeta)

def jac_s(zeta):
    return khat.T.dot(np.eye(3))

def f_f(zeta, w, rho, v, dt):
    return f_dyn(zeta, w, rho, v, dt)

def jac_f(zeta, w, rho, v, dt):
    z1 = f_f(zeta, w, rho, v, dt)
    return zeta.dot(scipy.linalg.pinv(z1))

def f_m(zeta):
    return np.arccos(f_s(zeta)) * f_l(zeta)

def jac_m(zeta):
    s = f_s(zeta)
    return -1.0/(1.-(s*s))**0.5 * f_l(zeta).dot(jac_s(zeta)) + np.arccos(s)*jac_l(zeta)

def f_delta(zeta, w, rho, v, dt):
    p = f_p(zeta, w)
    return (p - (w + rho*skew(zeta).dot(v)))*dt

def jac_delta(zeta, w, p, v, dt):
    dpdz = jac_p(zeta,w)
    return (dpdz + p * skew(v))*dt

def f_p(zeta, w):
    return zeta.dot(zeta.T).dot(w)

def f_r(zeta, w, rho, v, dt):
    return Quaternion.exp(f_delta(zeta, w, rho, v, dt)).invrot(khat)

def jac_r(zeta, w, rho, v, dt):
    return skew(khat).dot(jac_delta(zeta, w, rho, v, dt))

def f_dist(eta, w, rho, v, dt, K, Kinv):
    return f_pi(f_dyn(f_pi_inv(eta, Kinv), w, rho, v, dt), K)

def jac_p(zeta, w):
    out = np.zeros((3,3))
    for i in range(3):
        V = np.zeros((3,3))
        V[i,:] = zeta[:,0]
        V[:,i] += zeta[:,0]
        out[:,i,None] = V.dot(w)
    return out


def eq_test(name, x, y, dec=8):
    print(FCYN("[ " + name + " ]: "), end='')
    np.testing.assert_almost_equal(x, y, decimal=dec)
    print(FGRN("PASSED"))
    sys.stdout.flush()

def jac_test(name, fun, in_dim, x0, analytical, args=[], e=1e-8, dec=5):
    print(FCYN("[ " + name + " ]: "), end='')
    fd = np.zeros_like(analytical)
    I = np.eye(in_dim)
    y0 = fun(x0, *args)

    for i in range(in_dim):
        fd[:,i,None] = (fun(x0 + I[:,i,None]*e, *args) - y0)/e

    np.testing.assert_almost_equal(analytical, fd, decimal=dec, err_msg=name)
    print(FGRN("PASSED"))
    sys.stdout.flush()


if __name__ == '__main__':
    # Check the projection and inverse projection
    K = np.array([[255.0, 0, 320.0], # Nominal calibration matrix
                  [0.0, 255.0, 240.0],
                  [0.0, 0.0, 1.0]])
    K[:2,:] *= np.random.uniform(0.9, 1.1, (2,3)) # Noise-up calibration matrix
    Kinv = scipy.linalg.inv(K) # Inverse calibration matrix
    zeta = np.random.uniform(np.array([[-0.3, -0.3, 0.5]]).T, np.array([[0.3, 0.3, 1.0]]).T, (3,1)) # A bearing vector
    zeta /= norm(zeta) # normalize
    eta = np.random.uniform(np.array([[0, 0]]).T, np.array([[640.0, 480.0]]).T, (2,1)) # A pixel location
    eq_test("Proj Function", f_pi_inv(f_pi(zeta,K), Kinv), zeta)
    eq_test("Inv Proj Function", f_pi(f_pi_inv(eta,Kinv), K), eta)

    # Check jacobians of projection functions
    jac_test("Proj Jacobian", f_pi, 3, zeta, J_pi(zeta,K), args=[K])
    jac_test("Inv Proj Jacobian", f_pi_inv, 2, eta, J_pi_inv(eta,K), args=[K])

    # Check that with no inputs the dynamics are static
    w = v = np.zeros((3,1))
    p = 1.0
    dt = 0.001
    eq_test("dynamics - freeze", f_dyn(zeta, w, p, v, dt), zeta)

    jac_test("l(zeta)", f_l, 3, zeta, jac_l(zeta))
    jac_test("s(zeta)", f_s, 3, zeta, jac_s(zeta))
    jac_test("m(zeta)", f_m, 3, zeta, jac_m(zeta))

    v = np.random.random((3,1))
    w = np.random.random((3,1))
    jac_test("p(zeta)", f_p, 3, zeta, jac_p(zeta, w), args=[w])
    jac_test("del(zeta)", f_delta, 3, zeta, jac_delta(zeta, w, p, v, dt), args=[w, p, v, dt])
    jac_test("r(zeta)", f_r, 3, zeta, jac_r(zeta, w, p, v, dt), args=[w, p, v, dt])

    eq_test("r_eq", f_r(zeta, w, p, v, dt), Quaternion.exp(-(np.eye(3) - zeta.dot(zeta.T)).dot(w + p*skew(zeta).dot(v)) * dt).R.dot(khat))
    eq_test("dyn_eq", f_dyn(zeta, w, p, v, dt), qz(zeta).R.T.dot(f_r(zeta, w, p, v, dt)), dec=2)
    jac_test("f(zeta)", f_f, 3, zeta, jac_f(zeta, w, p, v, dt), args=[w, p, v, dt])


    # jac_test("D", f_dist, 2, eta, np.zeros((2,2)), args=[w, p, v, dt, K, Kinv])


    z0 = zeta.copy()
    z1 = f_dyn(zeta, w, p, v, dt)
    F = z0.dot(scipy.linalg.pinv(z1))


    print(BOLD(FGRN("[ALL TESTS PASSED]")))

#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from pyquat import *
import scipy.linalg

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

def eq_test(name, x, y, dec=8):
    print(FGRN("[" + name + "]: "), end='')
    np.testing.assert_almost_equal(x, y, decimal=dec)
    print(FGRN("PASSED"))

def jac_test(name, fun, in_dim, x0, analytical, args=[], e=1e-8, dec=5):
    print(FGRN("[" + name + "]: "), end='')
    fd = np.zeros_like(analytical)
    I = np.eye(in_dim)
    y0 = fun(x0, *args)

    for i in range(in_dim):
        fd[:,i,None] = (fun(x0 + I[:,i,None]*e, *args) - y0)/e

    np.testing.assert_almost_equal(analytical, fd, decimal=dec, err_msg=name)
    print(FGRN("PASSED"))


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


    jac_test("Proj Jacobian", f_pi, 3, zeta, J_pi(zeta,K), args=[K])
    jac_test("Inv Proj Jacobian", f_pi_inv, 2, eta, J_pi_inv(eta,K), args=[K])



    print(BOLD(FGRN("[ALL TESTS PASSED]")))

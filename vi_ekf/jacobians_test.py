from vi_ekf import *
import numpy as np

x0 = np.zeros((xZ, 1))
x0[xATT] = 1
x0[xMU] = 0.2

ekf = VI_EKF(x0)
zeta = np.random.randn(3)[:, None]
zeta /= scipy.linalg.norm(zeta)
ekf.init_feature(zeta, np.abs(np.random.randn(1))[0] * 10)

np.set_printoptions(suppress=True, linewidth=300, threshold=1000)
acc = np.array([[  1.04120597e-02],
               [ -6.93710705e-03],
               [ -9.80664202e+00]])
gyro = np.array([[ 0.03629203],
                   [ 0.0544231 ],
                   [-0.03629203]])
# acc = np.array([[ 0],
#                [ 0],
#                [ -9.80665]])
# gyro = np.array([[ 0.0],
#                    [ 0.0 ],
#                    [ 0.0]])
x = ekf.x
# x = ekf.propagate(acc, gyro, 0.01)
u = np.vstack([acc[2], gyro])

a_dfdx = ekf.dfdx(x, u)
d_dfdx = np.zeros_like(a_dfdx)
I = np.eye(d_dfdx.shape[0])
epsilon = 1e-8

for i in range(d_dfdx.shape[0]):
    x_prime = ekf.boxplus(x, (I[i] * epsilon)[:, None])
    d_dfdx[:, i] = ((ekf.f(x_prime, u) - ekf.f(x, u)) / epsilon)[:, 0]

if scipy.linalg.norm(a_dfdx - d_dfdx) > 1e-3:
    print 'Error in State Jacobians'
    print '\ndxPOS:dxVEL error\n', d_dfdx[dxPOS:dxPOS+3, dxVEL:dxVEL+3] - a_dfdx[dxPOS:dxPOS+3, dxVEL:dxVEL+3]
    print '\ndxPOS:dxATT error\n', d_dfdx[dxPOS:dxPOS+3, dxATT:dxATT+3] - a_dfdx[dxPOS:dxPOS+3, dxATT:dxATT+3]
    print '\ndxVEL:dxVEL error\n', d_dfdx[dxVEL:dxVEL+3, dxVEL:dxVEL+3] - a_dfdx[dxVEL:dxVEL+3, dxVEL:dxVEL+3]
    print '\ndxVEL:dxATT error\n', d_dfdx[dxVEL:dxVEL+3, dxATT:dxATT+3] - a_dfdx[dxVEL:dxVEL+3, dxATT:dxATT+3]
    print '\ndxVEL:dxB_A error\n', d_dfdx[dxVEL:dxVEL+3, dxB_A:dxB_A+3] - a_dfdx[dxVEL:dxVEL+3, dxB_A:dxB_A+3]
    print '\ndxVEL:dxB_G error\n', d_dfdx[dxVEL:dxVEL+3, dxB_G:dxB_G+3] - a_dfdx[dxVEL:dxVEL+3, dxB_G:dxB_G+3]
    print '\ndxVEL:dxMU error\n', d_dfdx[dxVEL:dxVEL+3, dxMU, None]    - a_dfdx[dxVEL:dxVEL+3, dxB_G:dxB_G+3]

    i = 0
    dxZETA_i = dxZ + i * 3
    dxRHO_i = dxZ + i * 3 + 2

    print '\ndZETA_i:dxVEL\n', d_dfdx[dxZETA_i:dxZETA_i+2, dxVEL:dxVEL+3] - a_dfdx[dxZETA_i:dxZETA_i+2, dxVEL:dxVEL+3]
    print '\ndZETA:dB_G\n', d_dfdx[dxZETA_i:dxZETA_i+2, dxB_G:dxB_G+3] - a_dfdx[dxZETA_i:dxZETA_i+2, dxB_G:dxB_G+3]
    print '\ndZETA:dZETA\n', d_dfdx[dxZETA_i:dxZETA_i+2, dxZETA_i:dxZETA_i+2] - a_dfdx[dxZETA_i:dxZETA_i+2, dxZETA_i:dxZETA_i+2]
    print '\ndZETA:dRHO\n', d_dfdx[dxZETA_i:dxZETA_i+2, dxRHO_i,None] - a_dfdx[dxZETA_i:dxZETA_i+2, dxRHO_i,None]
    print '\ndRHO:dVEL\n', d_dfdx[dxRHO_i, dxVEL:dxVEL+3] - a_dfdx[dxRHO_i, dxVEL:dxVEL+3]
    print '\ndRHO:dB_G\n', d_dfdx[dxRHO_i, dxB_G:dxB_G+3] - a_dfdx[dxRHO_i, dxB_G:dxB_G+3]
    print '\ndRHO:dZETA\n', d_dfdx[dxRHO_i, dxZETA_i:dxZETA_i+2] - a_dfdx[dxRHO_i, dxZETA_i:dxZETA_i+2]
    print '\ndRHO:dRHO\n', d_dfdx[dxRHO_i, dxRHO_i] - a_dfdx[dxRHO_i, dxRHO_i]

print '\nOverall dfdx Test\n', d_dfdx - a_dfdx

a_dfdu = ekf.dfdu(x)
d_dfdu = np.zeros_like(a_dfdu)
I = np.eye(d_dfdu.shape[1])
epsilon = 1e-15
for i in range(d_dfdu.shape[1]):
    u_prime = u + (I[i] * epsilon)[:, None]
    d_dfdu[:, i] = ((ekf.f(x, u_prime) - ekf.f(x, u)) / epsilon)[:, 0]

print '\ndxVEL:uA error\n', a_dfdu[dxVEL:dxATT, uA] - d_dfdu[dxVEL:dxATT, uA]
print '\ndxVEL:uG error\n', a_dfdu[dxVEL:dxATT, uG:] - d_dfdu[dxVEL:dxATT, uG:]

print '\nZETA:uG\n', d_dfdu[dxZETA_i:dxRHO_i, uG:] - a_dfdu[dxZETA_i:dxRHO_i, uG:]
print '\nRHO:uG\n', d_dfdu[dxRHO_i, uG:] - a_dfdu[dxRHO_i, uG:]


print '\nOverall dfdu Test\n', d_dfdu - a_dfdu

def htest(fn, **kwargs):
    print '\nTesting ', fn.__name__
    try:
        analytical = fn(x, **kwargs)[1]
        finite_difference = np.zeros_like(analytical)
        I = np.eye(finite_difference.shape[1])
        epsilon = 1e-5
        for i in range(finite_difference.shape[1]):
            x_prime = ekf.boxplus(x, (I[i] * epsilon)[:, None])
            finite_difference[:, i] = ((fn(x_prime, **kwargs)[0] - fn(x, **kwargs)[0]) / epsilon)[:, 0]

        print analytical - finite_difference
    except Exception as e:
        print 'error:', e


htest(ekf.h_acc)
htest(ekf.h_alt)
htest(ekf.h_feat, i=0)
htest(ekf.h_depth, i=0)
htest(ekf.h_inv_depth, i=0)
htest(ekf.h_pixel_vel, i=0, u=u)

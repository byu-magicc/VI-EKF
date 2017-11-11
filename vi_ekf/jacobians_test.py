from vi_ekf import *
import numpy as np

x0 = np.zeros((xZ, 1))
x0[xATT] = 1
x0[xMU] = 0.2

ekf = VI_EKF(x0)
# zeta = np.random.randn(3)[:, None]
zeta = np.array([[1., 1., 0.]]).T
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

dfdx_err = d_dfdx - a_dfdx

def print_error(rowname, colname, rowdim, coldim, matrix):
    row = 0
    col = 0
    exec("row = %s" % rowname)
    exec("col = %s" % colname)
    if scipy.linalg.norm(matrix[row:row+rowdim, col:col+coldim]) > 1e-3:
        print 'Error in Jacobian', rowname, colname
        print matrix[row:row+rowdim, col:col+coldim]

print_error('dxPOS', 'dxVEL', 3, 3, dfdx_err)
print_error('dxPOS', 'dxATT', 3, 3, dfdx_err)
print_error('dxVEL', 'dxATT', 3, 3, dfdx_err)
print_error('dxVEL', 'dxB_A', 3, 3, dfdx_err)
print_error('dxVEL', 'dxB_G', 3, 3, dfdx_err)
print_error('dxVEL', 'dxMU', 3, 3, dfdx_err)

i = 0
dxZETA_i = dxZ + i * 3
dxRHO_i = dxZ + i * 3 + 2

print_error('dxZETA_i','dxVEL', 2, 3, dfdx_err)
print_error('dxZETA_i','dxB_G', 2, 3, dfdx_err)
print_error('dxZETA_i','dxZETA_i', 2, 2, dfdx_err)
print_error('dxZETA_i','dxRHO_i', 2, 1, dfdx_err)
print_error('dxRHO_i','dxVEL', 1, 3, dfdx_err)
print_error('dxRHO_i','dxB_G', 1, 3, dfdx_err)
print_error('dxRHO_i','dxZETA_i', 1, 2, dfdx_err)
print_error('dxRHO_i','dxRHO_i', 1, 1, dfdx_err)

a_dfdu = ekf.dfdu(x)
d_dfdu = np.zeros_like(a_dfdu)
I = np.eye(d_dfdu.shape[1])
epsilon = 1e-8
for i in range(d_dfdu.shape[1]):
    u_prime = u + (I[i] * epsilon)[:, None]
    d_dfdu[:, i] = ((ekf.f(x, u_prime) - ekf.f(x, u)) / epsilon)[:, 0]
    dfdu_err = d_dfdu - a_dfdu

print_error('dxVEL','uA', 3, 1, dfdu_err)
print_error('dxVEL','uG', 3, 3, dfdu_err)
print_error('dxZETA_i','uG', 2, 3, dfdu_err)
print_error('dxRHO_i','uG', 1, 3, dfdu_err)

def htest(fn, **kwargs):

    # try:
    analytical = fn(x, **kwargs)[1]
    finite_difference = np.zeros_like(analytical)
    I = np.eye(finite_difference.shape[1])
    epsilon = 1e-8
    for i in range(finite_difference.shape[1]):
        x_prime = ekf.boxplus(x, (I[i] * epsilon)[:, None])
        finite_difference[:, i] = ((fn(x_prime, **kwargs)[0] - fn(x, **kwargs)[0]) / epsilon)[:, 0]

    error = analytical - finite_difference
    if np.max(error) > 1e-4:
        print '\nError in ', fn.__name__
        print error
    # except Exception as e:
    #     print 'error:', e


htest(ekf.h_acc)
htest(ekf.h_alt)
htest(ekf.h_feat, i=0)
htest(ekf.h_depth, i=0)
htest(ekf.h_inv_depth, i=0)
htest(ekf.h_pixel_vel, i=0, u=u)

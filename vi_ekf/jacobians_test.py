from vi_ekf import *
import numpy as np

x0 = np.zeros((xZ, 1))
x0[xATT] = 1
x0[xMU] = 0.2

ekf = VI_EKF(x0)

np.set_printoptions(suppress=True, linewidth=300, threshold=1000)
acc = np.array([[  1.04120597e-02],
               [ -6.93710705e-03],
               [ -9.80664202e+00]])
gyro = np.array([[ 0.03629203],
                   [ 0.0544231 ],
                   [-0.03629203]])

x = ekf.propagate(acc, gyro, 0.01)
u = np.vstack([acc[2], gyro])
a_dfdx = ekf.dfdx(x, u)
d_dfdx = np.zeros_like(a_dfdx)
I = np.eye(d_dfdx.shape[0])
epsilon = 1e-15
for i in range(d_dfdx.shape[0]):
    x_prime = ekf.boxplus(x, (I[i] * epsilon)[:, None])
    d_dfdx[:, i] = ((ekf.f(x_prime, u) - ekf.f(x, u)) / epsilon)[:, 0]

# print '\ndxPOS:dxVEL error\n', d_dfdx[dxPOS:dxPOS+3, dxVEL:dxVEL+3] - a_dfdx[dxPOS:dxPOS+3, dxVEL:dxVEL+3]
# print '\ndxPOS:dxATT error\n', d_dfdx[dxPOS:dxPOS+3, dxATT:dxATT+3] - a_dfdx[dxPOS:dxPOS+3, dxATT:dxATT+3]
print '\ndxVEL:dxVEL error\n', d_dfdx[dxVEL:dxVEL+3, dxVEL:dxVEL+3] - a_dfdx[dxVEL:dxVEL+3, dxVEL:dxVEL+3]
# print '\ndxVEL:dxATT error\n', d_dfdx[dxVEL:dxVEL+3, dxATT:dxATT+3] - a_dfdx[dxVEL:dxVEL+3, dxVEL:dxVEL+3]
# print '\ndxVEL:dxB_A error\n', d_dfdx[dxVEL:dxVEL+3, dxB_A:dxB_A+3] - a_dfdx[dxVEL:dxVEL+3, dxATT:dxATT+3]
# print '\ndxVEL:dxB_G error\n', d_dfdx[dxVEL:dxVEL+3, dxB_G:dxB_G+3] - a_dfdx[dxVEL:dxVEL+3, dxB_A:dxB_A+3]
# print '\ndxVEL:dxMU error\n', d_dfdx[dxVEL:dxVEL+3, dxMU, None]    - a_dfdx[dxVEL:dxVEL+3, dxB_G:dxB_G+3]


a_dfdu = ekf.dfdu(x)
d_dfdu = np.zeros_like(a_dfdu)
I = np.eye(d_dfdu.shape[1])
epsilon = 1e-15
for i in range(d_dfdu.shape[1]):
    u_prime = u + (I[i] * epsilon)[:, None]
    d_dfdu[:, i] = ((ekf.f(x, u_prime) - ekf.f(x, u)) / epsilon)[:, 0]

print '\ndxVEL:uA error\n', a_dfdu[dxVEL:dxATT, uA] - d_dfdu[dxVEL:dxATT, uA]
print '\ndxVEL:uG error\n', a_dfdu[dxVEL:dxATT, uG:] - d_dfdu[dxVEL:dxATT, uG:]

quit()
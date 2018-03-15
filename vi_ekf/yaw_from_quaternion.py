from pyquat import Quaternion
from math_helper import norm
import numpy as np

# for i in range(100):
# q = Quaternion.random()
q = Quaternion.from_euler(np.pi/4., 0, np.pi/8.)
yaw_true = q.yaw
v = np.array([[0,0,1]]).T


u = Quaternion.log(q)
alpha = norm(u)
beta = u/norm(u)
# u = q.elements[0,0] * q.elements[1:]
# alpha = np.arccos(q.w)*2
# beta = np.arccos(q.elements[1:]/np.sin(alpha/2.0))*2.0
# beta /= norm(beta)

# From stack overflow
# yaw_test = np.arctan(beta.T.dot(v)*np.tan(alpha/2.0))

yaw_test = alpha * beta.T.dot(v)
print q
print yaw_test
print yaw_true

    # assert abs(yaw_true - yaw_test) < 1e-8, "wrong: true = %f, test = %f" % (yaw_true, yaw_test)
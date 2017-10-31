import numpy as np
from vi_ekf import VI_EKF
import scipy.linalg
from quaternion import Quaternion
from tqdm import tqdm
from add_landmark import add_landmark
import cPickle

# This file just rotates the body around all randomly, so I can check rotations

def q_boxplus(q, dq):
    q_new = np.zeros((4,1))
    quat = Quaternion(q)
    norm_delta = scipy.linalg.norm(dq)
    if norm_delta > 1e-4:
        dquat = Quaternion(scalar=np.cos(norm_delta/2.), vector=np.sin(norm_delta/2.)*dq/norm_delta)
        q_new[:, 0] = (quat * dquat).elements
    else:
        dquat = Quaternion(scalar=1., vector=dq/2.)
        q_new[:, 0] = (quat * dquat).unit.elements
    return q_new

def generate_data():
    dt = 0.001
    t = np.arange(0.0, 3.01, dt)

    g = np.array([[0, 0, 9.80665]]).T

    q = np.zeros((len(t), 4))
    q[0,0] = 1.0

    frequencies = np.array([[1., 0.5, 1.1]]).T
    amplitudes = np.array([[0.1, 0.3, 0.0]]).T

    omega = amplitudes*np.sin(frequencies*t)

    acc = np.zeros([3, len(t)])

    for i in tqdm(range(len(t))):
        if i == 0.0:
            continue

        q[i,:,None] = q_boxplus(q[i-1,:,None], omega[:,i]*dt)

        acc[:,i] = -Quaternion(q[i,:]).inverse.rotate(g)

    data = dict()
    data['truth_NED'] = dict()
    data['truth_NED']['pos'] = np.zeros((len(t), 3))
    data['truth_NED']['vel'] = np.zeros((len(t), 3))
    data['truth_NED']['att'] = q
    data['truth_NED']['t'] = t

    data['imu_data'] = dict()
    data['imu_data']['t'] = t
    data['imu_data']['acc'] = acc.T
    data['imu_data']['gyro'] = omega.T

    landmarks = np.array([[0.1, 0, 1],
                          [0, 0.1, 1]])

    data['features'] = dict()
    data['features']['t'] = data['truth_NED']['t']
    data['features']['zeta'], data['features']['depth'] = add_landmark(data['truth_NED']['pos'],
                                                                       data['truth_NED']['att'], landmarks)

    cPickle.dump(data, open('generated_data.pkl', 'wb'))

if __name__ == '__main__':
    generate_data()



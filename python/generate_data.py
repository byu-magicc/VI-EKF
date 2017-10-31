import numpy as np
from vi_ekf import VI_EKF
import scipy.linalg
from quaternion import Quaternion
from tqdm import tqdm
from add_landmark import add_landmark
import cPickle

# This file just rotates the body around all randomly, so I can check rotations

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
        quat = Quaternion(q[i-1,:,None])

        q[i,:,None] = (quat + omega[:,i,None]*dt).elements

        acc[:,i,None] = -quat.inverse.rotate(g)

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
                          [2, 3, 5]])

    data['features'] = dict()
    data['features']['t'] = data['truth_NED']['t']
    data['features']['zeta'], data['features']['depth'] = add_landmark(data['truth_NED']['pos'],
                                                                       data['truth_NED']['att'], landmarks)

    cPickle.dump(data, open('generated_data.pkl', 'wb'))

if __name__ == '__main__':
    generate_data()



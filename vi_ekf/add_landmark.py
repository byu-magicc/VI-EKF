import numpy as np
from pyquat import Quaternion
import scipy.linalg
from tqdm import tqdm
from math_helper import norm, q_array_from_two_unit_vectors

def add_landmark(truth, landmarks, p_b_c, q_b_c):
    assert truth.shape[1] > 7 and landmarks.shape[1] == 5

    feature_array = np.zeros((truth.shape[0], 1 + 5*landmarks.shape[0]))
    # bearing = np.zeros((truth_pos.shape[0], len(landmarks), 3))
    # depth = np.zeros((len(truth_pos), len(landmarks)))

    bearing_mask = np.tile(np.array([[True, True, True, True, False]]), (1, len(landmarks)))
    depth_mask = np.tile(np.array([[False, False, False, False, True]]), (1, len(landmarks)))
    bearing_mask = np.hstack((np.array([[False]]), bearing_mask)).squeeze()
    depth_mask = np.hstack((np.array([[False]]), depth_mask)).squeeze()

    khat = np.array([[0, 0, 1.]]).T

    for i in range(len(truth)):
        q = Quaternion(truth[i, 4:8, None])
        delta_pose = landmarks[:,:3] - (truth[i, 1:4] + q.invrot(p_b_c).T)
        dist = norm(delta_pose, axis=1)
        q = Quaternion(truth[i,4:8,None])
        zetas = q_b_c.invrot(q.rot((delta_pose/dist[:,None]).T))
        q_zetas = q_array_from_two_unit_vectors(khat, zetas)
        feature_array[i,bearing_mask] = np.reshape(q_zetas, (1, -1), order='f')
        # if abs(1. - norm(np.reshape(feature_array[i,bearing_mask], (3, -1), order='f'), axis=0) > 1e-5).any():
        #     debug = 1
        feature_array[i,depth_mask] = dist
    feature_array[:,0] = truth[:,0]

    zetas = []
    depths = []
    t = truth[:,0]
    ids = [[] for i in t]
    for l, landmark in enumerate(landmarks):
        start = landmark[3]
        end = landmark[4]
        # NaN out the features that aren't visible
        feature_array[(start > t) | (t > end), l * 5 + 1:l * 5 + 5] = np.nan
        feature_array[(start > t) | (t > end), l * 5 + 5:l * 5 + 6] = np.nan
        # Append to the zetas list
        zetas.append(feature_array[:, l * 5 + 1:l * 5 + 5])
        depths.append(feature_array[:, l * 5 + 5:l * 5 + 6])
        for i in range(len(t)):
            if np.isfinite(feature_array[i, l*5+1]):
                ids[i].append(l)

    return t, zetas, depths, ids

def test():
    landmarks = np.random.uniform(-100, 100, (3,10))
    truth = []
    position = np.zeros((3,1))
    orientation = Quaternion.Identity()
    for i in range(1000):
        position += np.random.normal(0.0, 0.025, (3,1))
        orientation += np.random.normal(0.0, 0.025, (3,1))

        truth.append(np.hstack(np.array([[i]]),
                     position.T,
                     orientation.elements.T))
    truth = np.array(truth).squeeze()

    feature_time, zetas, depths, ids = add_landmark(truth, landmarks)






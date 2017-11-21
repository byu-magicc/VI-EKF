import numpy as np
from pyquat import Quaternion
import scipy.linalg
from tqdm import tqdm
from math_helper import norm

def add_landmark(truth, landmarks):
    assert truth.shape[1] == 17 and landmarks.shape[1] == 3

    feature_array = np.zeros((truth.shape[0], 1 + 4*len(landmarks)))
    # bearing = np.zeros((truth_pos.shape[0], len(landmarks), 3))
    # depth = np.zeros((len(truth_pos), len(landmarks)))

    bearing_mask = np.tile(np.array([[True, True, True, False]]), (1, len(landmarks)))
    depth_mask = np.tile(np.array([[False, False, False, True]]), (1, len(landmarks)))
    bearing_mask = np.hstack((np.array([[False]]), bearing_mask)).squeeze()
    depth_mask = np.hstack((np.array([[False]]), depth_mask)).squeeze()

    print "adding landmarks"
    for i in tqdm(range(len(truth))):
        delta_pose = landmarks - truth[i, 1:4]
        dist = norm(delta_pose, axis=1)
        q = Quaternion(truth[i,4:8,None])

        feature_array[i,bearing_mask] = np.reshape((q.R.dot((delta_pose/dist[:,None]).T)), (1, -1), order='f')
        # if abs(1. - norm(np.reshape(feature_array[i,bearing_mask], (3, -1), order='f'), axis=0) > 1e-5).any():
        #     debug = 1
        feature_array[i,depth_mask] = dist
    feature_array[:,0] = truth[:,0]

    zetas = []
    depths = []
    feature_time = feature_array[:,0]
    for l in range(len(landmarks)):
        zetas.append(feature_array[:, l * 4 + 1:l * 4 + 4])
        depths.append(feature_array[:, l * 4 + 4:l * 4 + 5])
    ids = np.tile(np.arange(0, len(landmarks), 1), (len(truth), 1))

    return feature_time, zetas, depths, ids




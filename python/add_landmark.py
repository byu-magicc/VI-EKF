import numpy as np
from quaternion import Quaternion
import scipy.linalg

def add_landmark(truth_pos, truth_att, landmarks):
    assert truth_pos.shape[1] == 3 and truth_att.shape[1] == 4 and landmarks.shape[1] == 3

    bearing = np.zeros((len(landmarks), truth_pos.shape[0], truth_pos.shape[1]))
    depth = np.zeros((len(landmarks), len(truth_pos)))

    for i in range(len(truth_pos)):
        delta_pose = np.tile(truth_pos[i,:], (len(landmarks), 1)) - landmarks
        dist = scipy.linalg.norm(delta_pose, axis=1)
        zeta = Quaternion(truth_att[i,:]).rotate(delta_pose.T/dist)
        bearing[:, i, :] = zeta
        depth[:,i] = dist

    return bearing, depth




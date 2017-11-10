import numpy as np
from quaternion import Quaternion
import scipy.linalg
from tqdm import tqdm

def add_landmark(truth_pos, truth_att, landmarks):
    assert truth_pos.shape[1] == 3 and truth_att.shape[1] == 4 and landmarks.shape[1] == 3

    bearing = np.zeros((truth_pos.shape[0], len(landmarks), 3))
    depth = np.zeros((len(truth_pos), len(landmarks)))

    print "adding landmarks"
    for i in tqdm(range(len(truth_pos))):
        delta_pose = landmarks - truth_pos[i]
        dist = scipy.linalg.norm(delta_pose, axis=1)
        q = Quaternion(truth_att[i,:,None])
        for l in range(len(landmarks)):
            bearing[i, l, :,None] = q.invrot(delta_pose[l,None].T/dist[l])

        depth[i,:] = dist


    return bearing, depth




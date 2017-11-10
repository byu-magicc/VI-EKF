import matplotlib.pyplot as plt
from quaternion import Quaternion
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def plot_cube(q_I_b, zetas, zeta_truth):

    cube = np.array([[-1., -1., -1.],
                     [1., -1., -1. ],
                     [1., 1., -1.],
                     [-1., 1., -1.],
                     [-1., -1., 1.],
                     [1., -1., 1. ],
                     [1., 1., 1.],
                     [-1., 1., 1.]])
    for i in range(8):
        cube[i,:,None] = q_I_b.R.dot(0.3*cube[i,:,None])

    right = [[cube[2],cube[3],cube[7],cube[6]]]
    front = [[cube[1],cube[2],cube[6],cube[5]]]
    bottom = [[cube[4],cube[5],cube[6],cube[7]]]
    rest = [[cube[0],cube[1],cube[2],cube[3]],
             [cube[0],cube[1],cube[5],cube[4]],
             [cube[4],cube[7],cube[3],cube[0]]]
    fig = plt.figure(10, figsize=(10,10))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(front,
     facecolors='red', linewidths=1, edgecolors='r', alpha=.25))
    ax.add_collection3d(Poly3DCollection(right,
     facecolors='green', linewidths=1, edgecolors='r', alpha=.25))
    ax.add_collection3d(Poly3DCollection(bottom,
     facecolors='blue', linewidths=1, edgecolors='r', alpha=.25))
    ax.add_collection3d(Poly3DCollection(rest,
     facecolors='grey', linewidths=1, edgecolors='r', alpha=.25))

    for i, (z, z_t) in enumerate(zip(zetas, zeta_truth)):
        z_NWU = q_I_b.R.T.dot(z[:, None])
        ax.plot([0, z_NWU[0]], [0, z_NWU[1]], [0, z_NWU[2]], '-b', label="est")
        zt_NWU = q_I_b.R.T.dot(z_t[:, None])
        ax.plot([0, zt_NWU[0]], [0, zt_NWU[1]], [0, zt_NWU[2]], '-r', label="truth")
        if i == 0:
            plt.legend()

    plt.ion()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    plt.show()
    plt.ioff()
    plt.pause(0.001)
import scipy.linalg
import numpy as np
from quaternion import *

def add_landmarks(landmarks, truth, image_time):
    bearing = np.zeros([len(landmarks), 3, len(image_time)])
    depth = np.zeros([len(landmarks), len(image_time)])
    prev_time = 0
    image_index = 0

    for row in tqdm(truth):
        t = row[0]
        pose = row[1:4]
        quat_i_b = Quaternion(row[4:8])
        # See if there was a picture during this update
        while image_time[image_index] > prev_time and image_time[image_index] <= t:
            # calculate the range and bearing to each landmark
            for i, l in enumerate(landmarks):
                i_vector_l_b = l - pose
                dist = scipy.linalg.norm(i_vector_l_b)
                i_dir_l_b = i_vector_l_b/dist
                b_dir_l_b = quat_i_b.inverse.rotate(i_dir_l_b)
                bearing[i, :, image_index] = b_dir_l_b
                depth[i, image_index] = dist
            image_index = (image_index + 1) % len(image_time)
        prev_time = t

    out_dict = dict()
    out_dict['bearing'] = bearing
    out_dict['range'] = depth
    out_dict['t'] = image_time

    return out_dict

if __name__ == '__main__':
    from data_loader import *
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt



    # Data for mav0 starts at 16 seconds
    data = load_data('data/mav0', show_image=False, start=16, end=-1)
    save_to_file('data/mav0/data.npy', data)
    data = load_from_file('data/mav0/data.npy')
    print "loaded data"

    landmarks = np.array([[0, 0, 0],
                          [5, 12, -1],
                          [10, -12, -3],
                          [7, -15, -2],
                          [5, 5, -1]])

    truth = data['truth']

    print "adding landmarks"
    landmarks_dict = add_landmarks(landmarks, truth, data['cam_time'])

    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(truth[:,1], truth[:,2], truth[:,3], '-c', label='truth')

    # Plot coordinate frame at origin
    origin = np.tile(truth[0, 1:4], (3, 1))
    axes = np.array([origin, origin + np.eye(3)])
    plt.plot(axes[:,0,0], axes[:, 0, 1], axes[:, 0, 2], '-r', label="x")
    plt.plot(axes[:,1,0], axes[:, 1, 1], axes[:, 1, 2], '-g', label="y")
    plt.plot(axes[:,2,0], axes[:, 2, 1], axes[:, 2, 2], '-b', label="z")

    # Plot Landmarks
    plt.plot(landmarks[:,0], landmarks[:,1], landmarks[:,2], 'xk', label='landmarks')

    plt.axis('equal')
    # plt.legend()
    plt.show()
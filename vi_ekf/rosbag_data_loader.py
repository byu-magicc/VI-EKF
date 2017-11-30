import csv
import numpy as np
import glob, os, sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from tqdm import tqdm
from pyquat import Quaternion
from add_landmark import add_landmark
import yaml
import matplotlib.pyplot as plt
import rosbag
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
from plot_helper import plot_3d_trajectory

def to_list(vector3):
    return [vector3.x, vector3.y, vector3.z]

def to_list4(quat):
    return [quat.w, quat.x, quat.y, quat.z]

def load_from_file(filename):
    data = np.load(filename)
    return data.item()

def save_to_file(filename, data):
    np.save(filename, data)

def make_undistort_funtion(intrinsics, resolution, distortion_coefficients):
    A = np.array([[float(intrinsics[0]), 0., float(intrinsics[2])], [0., float(intrinsics[1]), float(intrinsics[3])], [0., 0., 1.]])
    Ap, _ = cv2.getOptimalNewCameraMatrix(A, distortion_coefficients, (resolution[0], resolution[1]), 1.0)

    def undistort(image):
        return cv2.undistort(image, A, distortion_coefficients, None, Ap)

    return undistort, Ap

def calculate_velocity_from_position(t, position, orientation):
    # Calculate body-fixed velocity by differentiating position and rotating
    # into the body frame
    b, a = scipy.signal.butter(8, 0.03)  # Create a Butterworth Filter
    # differentiate Position
    delta_x = np.diff(position, axis=0)
    delta_t = np.diff(t)
    unfiltered_inertial_velocity = np.vstack((np.zeros((1, 3)), delta_x / delta_t[:, None]))
    # Filter
    v_inertial = scipy.signal.filtfilt(b, a, unfiltered_inertial_velocity, axis=0)
    # Rotate into Body Frame
    vel_data = []
    for i in range(len(t)):
        q_I_b = Quaternion(orientation[i, :, None])
        vel_data.append(q_I_b.rot(v_inertial[i, None].T).T)

    vel_data = np.array(vel_data).squeeze()
    return vel_data


def load_data(filename, start=0, end=np.inf, sim_features=False, show_image=False, plot_trajectory=False):
    print "loading rosbag", filename
    # First, load IMU data
    bag = rosbag.Bag(filename)
    imu_data = []
    truth_pose_data = []

    for topic, msg, t in bag.read_messages(topics=['/imu/data',
                                                   '/vrpn_client_node/Leo/pose',
                                                   '/baro/data',
                                                   '/sonar/data',
                                                   '/is_flying',
                                                   '/gps/data',
                                                   '/mag/data']):

        if topic == '/imu/data':
            imu_meas = [msg.header.stamp.to_sec(),
                        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            imu_data.append(imu_meas)

        if topic == '/vrpn_client_node/Leo/pose':
            truth_meas = [msg.header.stamp.to_sec(),
                          msg.pose.position.z, -msg.pose.position.x, -msg.pose.position.y,
                          -msg.pose.orientation.w, -msg.pose.orientation.z, msg.pose.orientation.x, msg.pose.orientation.y]
            truth_pose_data.append(truth_meas)

    imu_data = np.array(imu_data)
    truth_pose_data = np.array(truth_pose_data)

    # Remove Bad Truth Measurements
    good_indexes = np.hstack((True, np.diff(truth_pose_data[:,0]) > 1e-3))
    truth_pose_data = truth_pose_data[good_indexes]

    vel_data = calculate_velocity_from_position(truth_pose_data[:,0], truth_pose_data[:,1:4], truth_pose_data[:,4:8])

    ground_truth = np.hstack((truth_pose_data, vel_data))


    # Simulate Landmark Measurements
    # landmarks = np.random.uniform(-25, 25, (2,3))
    landmarks = np.vstack([np.eye(3), np.array([[0, 0, 0]])])
    # landmarks = np.zeros((3,3))
    feat_time, zetas, depths, ids = add_landmark(ground_truth, landmarks)

    if plot_trajectory:
        plot_3d_trajectory(ground_truth[:,1:4], ground_truth[:,4:8], qzetas=zetas, depths=depths)

    # Adjust timestamp
    t0 = imu_data[0,0]
    imu_data[:,0] -= t0
    ground_truth[:,0] -= t0
    feat_time[:] -= t0

    # Chop Data
    imu_data = imu_data[(imu_data[:,0] > start) & (imu_data[:,0] < end), :]
    # if sim_features:
    for l in range(len(landmarks)):
        zetas[l] = zetas[l][(feat_time > start) & (feat_time < end)]
        depths[l] = depths[l][(feat_time > start) & (feat_time < end)]
    ids = ids[(feat_time > start) & (feat_time < end)]
    # else:
    #     images0 = [f for f, t in zip(images0, (image_time > start) & (image_time < end)) if t]
    #     images1 = [f for f, t in zip(images1, (image_time > start) & (image_time < end)) if t]
    #     image_time = image_time[(image_time > start) & (image_time < end)]
    ground_truth = ground_truth[(ground_truth[:, 0] > start) & (ground_truth[:, 0] < end), :]

    out_dict = dict()
    out_dict['imu'] = imu_data
    out_dict['truth'] = ground_truth
    # if sim_features:
    out_dict['feat_time'] = feat_time[(feat_time > start) & (feat_time < end)]
    out_dict['zetas'] = zetas
    out_dict['depths'] = depths
    out_dict['ids'] = ids
    # else:
    #     out_dict['cam0_sensor'] = cam0_sensor
    #     out_dict['cam1_sensor'] = cam1_sensor
    #     out_dict['cam0_frame_filenames'] = images0
    #     out_dict['cam1_frame_filenames'] = images1
    #     out_dict['cam_time'] = image_time

    return out_dict



if __name__ == '__main__':
    data = load_data('data/truth_imu_flight.bag')
    print "done"

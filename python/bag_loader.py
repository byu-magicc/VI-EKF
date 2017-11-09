import rosbag
import rospy
import numpy as np
from add_landmark import add_landmark

import cPickle

def to_list(vector3):
    return [vector3.x, vector3.y, vector3.z]

def to_list4(quat):
    return [quat.w, quat.x, quat.y, quat.z]

def read_bag(filename):
    bag = rosbag.Bag(filename)

    data = dict()
    data['imu_data'] = dict()
    data['imu_data']['t'] = []
    data['imu_data']['acc'] = []
    data['imu_data']['gyro'] = []

    data['perfect_imu_data'] = dict()
    data['perfect_imu_data']['t'] = []
    data['perfect_imu_data']['acc'] = []
    data['perfect_imu_data']['gyro'] = []

    data['imu_acc_bias'] = {'t': [], 'v': []}
    data['imu_gyro_bias'] = {'t': [], 'v': []}

    data['truth_NED'] = dict()
    data['truth_NED']['t'] = []
    data['truth_NED']['pos'] = []
    data['truth_NED']['vel'] = []
    data['truth_NED']['att'] = []
    data['truth_NED']['omega'] = []

    data['baro'] = {'t': [], 'alt': []}
    data['sonar'] = {'t': [], 'alt': []}
    data['gps'] = {'t': [], 'lla': [], 'speed': [], 'course': [], 'fix': []}
    data['mag'] = {'t': [], 'vec': []}
    data['is_flying'] = {'t': [], 'dat': []}

    data['features'] = {'t': [], 'zeta': [], 'depth': []}

    for topic, msg, t in bag.read_messages(topics=['/multirotor/imu/data',
                                                   '/multirotor/imu/acc_bias',
                                                   '/multirotor/imu/gyro_bias',
                                                   '/multirotor/imu/data',
                                                   '/multirotor/ground_truth/odometry/NED',
                                                   '/multirotor/baro/data',
                                                   '/multirotor/sonar/data',
                                                   '/multirotor/is_flying',
                                                   '/multirotor/gps/data',
                                                   '/multirotor/mag/data']):
        if topic == '/multirotor/imu/data':
            data['imu_data']['t'].append(msg.header.stamp.to_sec())
            data['imu_data']['acc'].append(to_list(msg.linear_acceleration))
            data['imu_data']['gyro'].append(to_list(msg.angular_velocity))

        if topic == '/multirotor/imu/acc_bias':
            data['imu_acc_bias']['t'].append(msg.header.stamp.to_sec())
            data['imu_acc_bias']['v'].append(to_list(msg.vector))

        if topic == '/multirotor/imu/gyro_bias':
            data['imu_gyro_bias']['t'].append(msg.header.stamp.to_sec())
            data['imu_gyro_bias']['v'].append(to_list(msg.vector))

        if topic == '/multirotor/ground_truth/odometry/NED':
            data['truth_NED']['t'].append(msg.header.stamp.to_sec())
            data['truth_NED']['pos'].append(to_list(msg.pose.pose.position))
            data['truth_NED']['vel'].append(to_list(msg.twist.twist.linear))
            data['truth_NED']['att'].append(to_list4(msg.pose.pose.orientation))
            data['truth_NED']['omega'].append(to_list(msg.twist.twist.angular))

        if topic == '/multirotor/baro/data':
            data['baro']['t'].append(msg.header.stamp.to_sec())
            data['baro']['alt'].append(msg.altitude)

        if topic == '/multirotor/sonar/data':
            data['sonar']['t'].append(msg.header.stamp.to_sec())
            data['sonar']['alt'].append(msg.range)

        if topic == '/multirotor/is_flying':
            data['is_flying']['t'].append(t)
            data['is_flying']['dat'].append(msg.data)

        if topic == '/multirotor/gps/data':
            data['gps']['t'].append(msg.header.stamp.to_sec())
            data['gps']['lla'].append([msg.latitude, msg.longitude, msg.altitude])
            data['gps']['speed'].append(msg.speed)
            data['gps']['course'].append(msg.ground_course)
            data['gps']['fix'].append(msg.fix)

        if topic == '/multirotor/mag/data':
            data['mag']['t'].append(msg.header.stamp.to_sec())
            data['mag']['vec'].append(to_list(msg.magnetic_field))

    for key in data.iterkeys():
        for key2, item in data[key].iteritems():
            data[key][key2] = np.array(item)

    # Simulate landmarks in the bag
    landmarks = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    data['features']['t'] = data['truth_NED']['t']
    data['features']['zeta'], data['features']['depth'] = add_landmark(data['truth_NED']['pos'],
                                                                       data['truth_NED']['att'], landmarks)
    data['landmarks'] = landmarks

    cPickle.dump(data, open(filename.split('/')[-1].split('.')[0] + '.pkl', 'wb'))



if __name__ == '__main__':
    read_bag('data/simulated_waypoints.bag')
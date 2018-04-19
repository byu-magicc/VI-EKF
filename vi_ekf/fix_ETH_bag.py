import rosbag
import rospy
from pyquat import Quaternion
import numpy as np
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3
import scipy.signal
import sys
from tqdm import tqdm


inputfile = '/home/superjax/rosbag/EuRoC/V2_01_easy.bag'
outputfile = '/home/superjax/rosbag/EuRoC/V2_01_easy_NED.bag'

# print console header
print("fix ETH Bag")
print("input file:  ", inputfile)
print('output file: ', outputfile)

outbag = rosbag.Bag(outputfile, 'w')
messageCounter = 0
kPrintDotReductionFactor = 1000

q_NED_NWU = Quaternion(np.array([[0., 1., 0., 0.]]).T)
q_IMU_NWU = Quaternion.from_R(np.array([[0.33638, -0.01749,  0.94156],
                                        [-0.02078, -0.99972, -0.01114],
                                        [0.94150, -0.01582, -0.33665]]))
q_IMU_NED = q_IMU_NWU * q_NED_NWU

T_IMU_C = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                    [ 0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                    [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                    [ 0.0, 0.0, 0.0, 1.0]])
q_IMU_C = Quaternion.from_R(T_IMU_C[:3,:3])
p_IMU_C_IMU = T_IMU_C[:3,3,None]

print ("q_IMU_C = ", q_IMU_C)
print ("p_IMU_C_IMU", p_IMU_C_IMU.T)

# rotate camera transform into NED body frame
q_NED_C = (q_IMU_C * q_IMU_NED).inverse
p_IMU_C_NED = q_NED_C.rot(p_IMU_C_IMU)

print ("q_b_c", q_NED_C)
print ("p_b_c_b ", p_IMU_C_NED.T)

a = []
w = []
stamps = []
for topic, msg, t in rosbag.Bag(inputfile).read_messages():
    if topic == '/imu0':
        a.append([msg.linear_acceleration.x,
                  msg.linear_acceleration.y,
                  msg.linear_acceleration.z])
        w.append([msg.angular_velocity.x,
                  msg.angular_velocity.y,
                  msg.angular_velocity.z])
        stamps.append(msg.header.stamp)

a = np.array(a)
w = np.array(w)

b_acc, a_acc = scipy.signal.butter(6, 0.08)  # Smooth Differentiated Truth
a_smooth = scipy.signal.filtfilt(b_acc, a_acc, a, axis=0)
b_w, a_w = scipy.signal.butter(6, 0.08)  # Smooth Differentiated Truth
w_smooth = scipy.signal.filtfilt(b_w, a_w, w, axis=0)


q0 = None
p0 = None
i = 0
truth_offset = 0.0

bag = rosbag.Bag(inputfile)

for topic, msg, t in tqdm(bag.read_messages(), total=bag.get_message_count()):
    if topic == '/imu0':
        # acc = np.array([[msg.linear_acceleration.x],
        #                 [msg.linear_acceleration.y],
        #                 [msg.linear_acceleration.z]])
        # w = np.array([[msg.angular_velocity.x],
        #               [msg.angular_velocity.y],
        #               [msg.angular_velocity.z]])

        acc = a_smooth[i,:,None]
        w = w_smooth[i,:,None]
        i += 1

        acc = q_IMU_NED.invrot(acc)
        w = q_IMU_NED.invrot(w)

        msg.linear_acceleration.x = acc[0,0]
        msg.linear_acceleration.y = acc[1,0]
        msg.linear_acceleration.z = acc[2,0]
        msg.angular_velocity.x = w[0, 0]
        msg.angular_velocity.y = w[1, 0]
        msg.angular_velocity.z = w[2, 0]
        outbag.write('imu', msg, msg.header.stamp)

    elif topic == 'vicon/firefly_sbx/firefly_sbx':
        att = Quaternion(np.array([[msg.transform.rotation.w],
                                   [msg.transform.rotation.x],
                                   [msg.transform.rotation.y],
                                   [msg.transform.rotation.z]]))
        pos = np.array([[msg.transform.translation.x],
                        [msg.transform.translation.y],
                        [msg.transform.translation.z]])
        if q0 is None:
            q0 = att.copy()
            p0 = pos.copy()

        att = q0.inverse * att
        pos = q0.invrot(pos - p0)

        att_NED = q_NED_NWU.qinvrot(att)
        pos_NED = q_NED_NWU.invrot(pos)

        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose.position.x = pos_NED[0,0]
        pose_msg.pose.position.y = pos_NED[1,0]
        pose_msg.pose.position.z = pos_NED[2,0]
        pose_msg.pose.orientation.w = att_NED.w
        pose_msg.pose.orientation.x = att_NED.x
        pose_msg.pose.orientation.y = att_NED.y
        pose_msg.pose.orientation.z = att_NED.z

        euler_msg = Vector3()
        euler = att_NED.euler
        euler_msg.x = euler[0, 0]
        euler_msg.y = euler[1, 0]
        euler_msg.z = euler[2, 0]
        outbag.write('truth/pose', pose_msg, msg.header.stamp + rospy.Duration(truth_offset))
        outbag.write('truth/euler', euler_msg, msg.header.stamp + rospy.Duration(truth_offset))

    elif topic == "/cam0/image_raw":
        outbag.write('color', msg, msg.header.stamp)

    else:
        pass
sys.stdout.flush()
print ("done")
outbag.close()


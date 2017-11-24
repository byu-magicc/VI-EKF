import rospy
from sensor_msgs.msg import Imu
import geometry_msgs
from pyquat import Quaternion
import numpy as np

q = Quaternion(np.array([[1.0, 0, 0, 0]]).T)
rospy.init_node('integrator')
prev_t = rospy.get_time()
att_pub = rospy.Publisher('att', geometry_msgs.msg.Quaternion)
eul_pub = rospy.Publisher('euler', geometry_msgs.msg.Vector3)

def to_numpy(v):
    return np.array([[v.x, v.y, v.z]]).T

def q_to_ros(q):
    q_msg = geometry_msgs.msg.Quaternion()
    q_msg.w = q.w
    q_msg.x = q.x
    q_msg.y = q.y
    q_msg.z = q.z
    return q_msg

def v_to_ros(v):
    v_msg = geometry_msgs.msg.Vector3()
    v_msg.x = v[0,0]
    v_msg.y = v[1,0]
    v_msg.z = v[2,0]
    return v_msg

def integrate(msg):
    global q
    global prev_t
    t = msg.header.stamp.to_sec()
    dt = t - prev_t
    prev_t = t
    # dt = 0.01
    q += to_numpy(msg.angular_velocity)*dt
    q_msg = q_to_ros(q)
    att_pub.publish(q_msg)

    v_msg = v_to_ros(q.euler)
    eul_pub.publish(v_msg)

gyro_sub = rospy.Subscriber('multirotor/imu/data', Imu, integrate)

while not rospy.is_shutdown():
    rospy.spin()


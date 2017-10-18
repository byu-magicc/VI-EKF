from quad_sim import *
import rospy
from visualization_msgs.msg import Marker
import time

class ROSQuadSim():
    def __init__(self):
        self.quadcopter = QuadcopterSim()

        rospy.init_node('quadcopter_sim')
        self.marker_pub = rospy.Publisher('quadcopter', Marker)

        while not rospy.is_shutdown():
            time.sleep(0.1)

    def publish(self):
        x = self.quadcopter.get_state()
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.type = Marker.ARROW
        quat = Quaternion(x[6:10])
        scale = 0.5
        x = np.array([scale, 0, 0])
        y = np.array([0, scale, 0])
        z = np.array([0, 0, scale])
        marker.points[0].x = x[0]
        marker.points[0].y = x[1]
        marker.points[0].z = x[2]
        marker.color.a = 1.0

        # Draw X-axis
        x_end = x[0:3] + quat.rotate(x)
        marker.color.r = 1.0
        marker.points[1].x = x[0] + x_end[0]
        marker.points[1].y = x[1] + x_end[1]
        marker.points[1].z = x[2] + x_end[2]
        self.marker_pub.publish(marker)

        # Draw Y-axis
        y_end = x[0:3] + quat.rotate(y)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.points[1].x = x[0] + y_end[0]
        marker.points[1].y = x[1] + y_end[1]
        marker.points[1].z = x[2] + y_end[2]
        self.marker_pub.publish(marker)

        # Draw Z-axis
        z_end = x[0:3] + quat.rotate(z)
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.points[1].x = x[0] + z_end[0]
        marker.points[1].y = x[1] + z_end[1]
        marker.points[1].z = x[2] + z_end[2]
        self.marker_pub.publish(marker)

if __name__ == '__main__':
    thing = ROSQuadSim()
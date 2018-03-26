#!/usr/bin/env python

import rosbag
import rospy
from tqdm import tqdm

offset = -0.35
outbag = rosbag.Bag('/home/superjax/rosbag/no_yaw_inertialsense_shifted2.bag', 'w')
inbag = rosbag.Bag('/home/superjax/rosbag/no_yaw_inertialsense2.bag')
for topic, msg, t in tqdm(inbag.read_messages(), total=inbag.get_message_count()):
    if topic == '/vrpn/Leo/pose':
        outbag.write(topic, msg, t - rospy.Duration.from_sec(offset))
    else:
        outbag.write(topic, msg, t)
outbag.reindex()
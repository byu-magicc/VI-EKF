#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Function : restamp ros bagfile (using header timestamps)
# Project  : IJRR MAV Datasets
# Author   : www.asl.ethz.ch
# Version  : V01  21JAN2016 Initial version.
# Comment  :
# Status   : under review
#
# Usage    : python restamp_bag.py -i inbag.bag -o outbag.bag
# ------------------------------------------------------------------------------

import roslib
import rosbag
import rospy
import sys
import getopt
from   std_msgs.msg import String
from pyquat import Quaternion
import numpy as np
from geometry_msgs.msg import PoseStamped
import scipy.signal

def main(argv):

    inputfile = ''
    outputfile = ''

    # parse arguments
    # try:
    #     opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    # except getopt.GetoptError:
    #     print 'usage: restamp_bag.py -i <inputfile> -o <outputfile>'
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt == '-h':
    #         print 'usage: python restamp_bag.py -i <inputfile> -o <outputfile>'
    #         sys.exit()
    #     elif opt in ("-i", "--ifile"):
    #         inputfile = arg
    #     elif opt in ("-o", "--ofile"):
    #         outputfile = arg
    inputfile = '/home/superjax/rosbag/EuRoC/V2_01_easy.bag'
    outputfile = '/home/superjax/rosbag/EuRoC/V2_01_easy_NED.bag'

    # print console header
    print ""
    print "restamp_bag"
    print ""
    print 'input file:  ', inputfile
    print 'output file: ', outputfile
    print ""
    print "starting restamping (may take a while)"
    print ""

    outbag = rosbag.Bag(outputfile, 'w')
    messageCounter = 0
    kPrintDotReductionFactor = 1000

    q_NED_NWU = Quaternion(np.array([[0, 1, 0, 0]]).T)
    q_IMU_NWU = Quaternion.from_R(np.array([[0.33638, -0.01749,  0.94156],
                                            [-0.02078, -0.99972, -0.01114],
                                            [0.94150, -0.01582, -0.33665]]))

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
    try:
        for topic, msg, t in rosbag.Bag(inputfile).read_messages():

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

                acc = q_IMU_NWU.invrot(acc)
                w = q_IMU_NWU.invrot(w)
                acc = q_NED_NWU.invrot(acc)
                w = q_NED_NWU.invrot(w)

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

                att *= q0.inverse
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

                outbag.write('truth/pose', pose_msg, msg.header.stamp + rospy.Duration(truth_offset))

            elif topic == "/cam0/image_raw":
                outbag.write('color', msg, msg.header.stamp)

            else:
                pass

            # Write message in output bag with input message header stamp
            if (messageCounter % kPrintDotReductionFactor) == 0:
                    #print '.',
                    sys.stdout.write('.')
                    sys.stdout.flush()
            messageCounter = messageCounter + 1

    # print console footer
    finally:
        print ""
        print ""
        print "finished iterating through input bag"
        print "output bag written"
        print ""
        outbag.close()

if __name__ == "__main__":
   main(sys.argv[1:])


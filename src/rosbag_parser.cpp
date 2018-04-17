#include <iostream>
#include <vector>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>

#include "vi_ekf_ros.h"

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;

int main(int argc, char * argv[])
{
  if (argc < 2)
  {
    ROS_ERROR("no bag file supplied, please supply bag file to parse");
    return -1;
  }
  ROS_INFO("parsing bagfile: %s", argv[1]);

  ros::init(argc, argv, "vi_ekf_rosbag");  
  
  // Create the VIEKF_ROS object
  VIEKF_ROS node;
  
  rosbag::Bag bag;
  try
  {
    bag.open(argv[1], rosbag::bagmode::Read);
  }
  catch(rosbag::BagIOException e)
  {
    ROS_ERROR("unable to load rosbag %s, %s", argv[1], e.what());
    return -1;
  }
  rosbag::View view(bag);
  
  // Get list of topics and print to screen - https://answers.ros.org/question/39345/rosbag-info-in-c/
  vector<const rosbag::ConnectionInfo *> connections = view.getConnections();
  vector<string> topics;
  vector<string> types;
  cout << "loaded bagfile: \n===================================\n";
  cout << "Topics\t\tTypes\n----------------------------" << endl;
  foreach(const rosbag::ConnectionInfo *info, connections) {
    topics.push_back(info->topic);
    types.push_back(info->datatype);
    cout << info->topic << "\t\t" << info->datatype << endl;
  }
  
  // Call all the callbacks
  foreach (rosbag::MessageInstance const m, view)
  {
    // break if Ctrl+C
    if (!ros::ok())
      break;
    
    // Cast datatype into proper format and call the appropriate callback
    string datatype = m.getDataType();
    
    if (datatype.compare("sensor_msgs/Imu") == 0)
    {
      const sensor_msgs::ImuConstPtr imu(m.instantiate<sensor_msgs::Imu>());
      node.imu_callback(imu);
    }
    
    else if (datatype.compare("sensor_msgs/Image") == 0)
    {
      const sensor_msgs::ImageConstPtr image(m.instantiate<sensor_msgs::Image>());
      if (m.getTopic().compare("depth") == 0)
      {
        node.depth_image_callback(image);
      }
      else
      {
        node.color_image_callback(image);
      }
    }
    
    else if (datatype.compare("geometry_msgs/PoseStamped") == 0)
    {
      const geometry_msgs::PoseStampedConstPtr pose(m.instantiate<geometry_msgs::PoseStamped>());
      node.pose_truth_callback(pose);
    }
    
    else if (datatype.compare("geometry_msgs/TransformStamped") == 0)
    {
      const geometry_msgs::TransformStampedConstPtr pose(m.instantiate<geometry_msgs::TransformStamped>());
      node.transform_truth_callback(pose);
    }
  }
}

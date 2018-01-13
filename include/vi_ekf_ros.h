#include "vi_ekf.h"
#include "klt_tracker.h"

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>


class VIEKF_ROS
{
public:

  VIEKF_ROS();
  ~VIEKF_ROS();
  void image_callback(const sensor_msgs::ImageConstPtr& msg);
  void imu_callback(const sensor_msgs::ImuConstPtr& msg);

private:

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  image_transport::ImageTransport it_;
  image_transport::Publisher output_pub_;
  image_transport::Subscriber image_sub_;
  ros::Subscriber imu_sub_;
  ros::Publisher odometry_pub_;

  vi_ekf::VIEKF ekf_;
  KLT_Tracker klt_tracker_;



};

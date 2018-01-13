#include "vi_ekf_ros.h"

VIEKF_ROS::VIEKF_ROS() :
  nh_private_("~"),
  it_(nh_)
{
  imu_sub_ = nh_.subscribe("imu/data", 500, &VIEKF_ROS::imu_callback, this);
  odometry_pub_ = nh_.advertise<nav_msgs::Odometry>("odom", 1);
  image_sub_ = it_.subscribe("cv_camera/image_raw", 1, &VIEKF_ROS::image_callback, this);
  output_pub_ = it_.advertise("tracked", 1);
}

void VIEKF_ROS::imu_callback(const sensor_msgs::ImuConstPtr &msg)
{
  Eigen::Matrix<double, 6, 1> u;
  u(0) = msg->linear_acceleration.x;
  u(1) = msg->linear_acceleration.y;
  u(2) = msg->linear_acceleration.z;
  u(3) = msg->angular_velocity.x;
  u(4) = msg->angular_velocity.y;
  u(5) = msg->angular_velocity.z;
  ekf_.step(u, msg->header.stamp.toSec());
}

void VIEKF_ROS::image_callback(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_FATAL("cv_bridge exception: %s", e.what());
    return;
  }

  std::vector<Point2f> features;
  std::vector<int> ids;
  klt_tracker_.load_image(cv_ptr->image, msg->header.stamp.toSec(), features, ids);

  ekf_.keep_only_features(ids);
}





int main(int argc, char* argv[])
{

}

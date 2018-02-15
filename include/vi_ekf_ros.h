#include "vi_ekf.h"
#include "klt_tracker.h"

#include <mutex>
#include <deque>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>


using namespace Eigen;

typedef Matrix<double, 1, 1> Matrix1d;
typedef Matrix<double, 6, 1> Vector6d;


class VIEKF_ROS
{
public:

  VIEKF_ROS();
  ~VIEKF_ROS();
  void color_image_callback(const sensor_msgs::ImageConstPtr &msg);
  void depth_image_callback(const sensor_msgs::ImageConstPtr& msg);
  void truth_callback(const geometry_msgs::PoseStampedConstPtr &msg);
  void imu_callback(const sensor_msgs::ImuConstPtr& msg);

private:

  int num_features_;
  Vector6d u_prev_;
  ros::Time last_imu_update_;


  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  image_transport::ImageTransport it_;
  image_transport::Publisher output_pub_;
  image_transport::Subscriber image_sub_;
  image_transport::Subscriber depth_sub_;
  ros::Subscriber imu_sub_;
  ros::Subscriber truth_sub_;
  ros::Publisher odometry_pub_;
  nav_msgs::Odometry odom_msg_;

  std::mutex ekf_mtx_;
  vi_ekf::VIEKF ekf_;
  KLT_Tracker klt_tracker_;

  cv::Mat depth_image_;
  bool got_depth_;

  bool initialized_ = false;

  bool use_truth_;
  bool use_depth_;
  bool use_features_;
  bool use_acc_;
  bool use_imu_att_;
  bool use_alt_;
  double imu_LPF_;
  double imu_skip_;

  Vector6d imu_;

  Matrix2d feat_R_;
  Matrix2d acc_R_;
  Matrix3d att_R_;
  Matrix<double, 1, 1> alt_R_;
  Matrix3d pos_R_;
  Matrix3d vel_R_;
  Matrix1d depth_R_;
};




#include "vi_ekf_ros.h"

VIEKF_ROS::VIEKF_ROS() :
  nh_private_("~"),
  it_(nh_)
{
  imu_sub_ = nh_.subscribe("imu/data", 500, &VIEKF_ROS::imu_callback, this);
  truth_sub_ = nh_.subscribe("vrpn/Leo/pose", 10, &VIEKF_ROS::truth_callback, this);
  odometry_pub_ = nh_.advertise<nav_msgs::Odometry>("odom", 1);

  image_sub_ = it_.subscribe("camera/rgb/image_rect_mono", 10, &VIEKF_ROS::color_image_callback, this);
  depth_sub_ = it_.subscribe("camera/depth/image_rect", 10, &VIEKF_ROS::depth_image_callback, this);
  output_pub_ = it_.advertise("tracked", 1);

  std::string log_directory;
  std::string default_log_folder = ros::package::getPath("vi_ekf") + "/logs/" + to_string(ros::Time::now().sec) + "/";
  nh_private_.param<std::string>("log_directory", log_directory, default_log_folder );

  ekf_mtx_.lock();
  ekf_.init(ekf_.get_state(), log_directory, true);
  ekf_mtx_.unlock();
  klt_tracker_.init(3, true, 30);

  // Initialize the depth image to all NaNs
  depth_image_ = cv::Mat(640, 480, CV_32FC1, cv::Scalar(NAN));

  // Initialize the measurement noise covariance matrices
  depth_R_ << 0.1;
  feat_R_ << 1.0, 0.0, 0.0, 1.0;
  acc_R_ << 0.5, 0.0, 0.0, 0.5;
  att_R_ << 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01;
  pos_R_ << 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01;
  vel_R_ << 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1;
}

VIEKF_ROS::~VIEKF_ROS()
{}

void VIEKF_ROS::imu_callback(const sensor_msgs::ImuConstPtr &msg)
{
  Vector6d u;
  u(0) = msg->linear_acceleration.x;
  u(1) = msg->linear_acceleration.y;
  u(2) = msg->linear_acceleration.z;
  u(3) = msg->angular_velocity.x;
  u(4) = msg->angular_velocity.y;
  u(5) = msg->angular_velocity.z;
  ekf_mtx_.lock();
  ekf_.step(u, msg->header.stamp.toSec());
  ekf_mtx_.unlock();

  Vector2d z_acc = u.block<2,1>(0, 0);
  ekf_mtx_.lock();
  ekf_.update(z_acc, vi_ekf::VIEKF::ACC, acc_R_);
  ekf_mtx_.unlock();

  Vector4d z_att;
  z_att << msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z;
  ekf_mtx_.lock();
  ekf_.update(z_att, vi_ekf::VIEKF::ATT, att_R_, true);
  ekf_mtx_.unlock();
}

void VIEKF_ROS::color_image_callback(const sensor_msgs::ImageConstPtr &msg)
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

  // Track Features in Image
  std::vector<Point2f> features;
  std::vector<int> ids;
  klt_tracker_.load_image(cv_ptr->image, msg->header.stamp.toSec(), features, ids);

  ekf_mtx_.lock();
  ekf_.keep_only_features(ids);
  ekf_mtx_.unlock();
  for (int i = 0; i < features.size(); i++)
  {
    double depth = depth_image_.at<double>(features[i].x,features[i].y);
    if (depth > 1e3)
    {
      depth = NAN;
    }
    Vector2d z_feat;
    z_feat << features[i].x, features[i].y;
    Matrix1d z_depth;
    z_depth << depth;
    ekf_mtx_.lock();
    ekf_.update(z_feat, vi_ekf::VIEKF::FEAT, feat_R_, false, ids[i], depth);
    ekf_.update(z_depth, vi_ekf::VIEKF::DEPTH, depth_R_, depth != depth, ids[i]);
    ekf_mtx_.unlock();
  }
}

void VIEKF_ROS::depth_image_callback(const sensor_msgs::ImageConstPtr &msg)
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
  depth_image_ = cv_ptr->image;
}

void VIEKF_ROS::truth_callback(const geometry_msgs::PoseStampedConstPtr &msg)
{
  Vector3d z_pos;
  z_pos << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;

  Vector4d z_att;
  z_att << msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z;

  ekf_mtx_.lock();
  ekf_.update(z_pos, vi_ekf::VIEKF::POS, pos_R_, false);
  ekf_mtx_.unlock();

  ekf_mtx_.lock();
  ekf_.update(z_att, vi_ekf::VIEKF::ATT, att_R_, false);
  ekf_mtx_.unlock();
}


int main(int argc, char* argv[])
{
  ros::init(argc, argv, "viekf_node");
  VIEKF_ROS ekf;

  ros::spin();
//  ros::AsyncSpinner spinner(0); // Use 4 threads
//  spinner.start();
//  ros::waitForShutdown();

  return 0;
}





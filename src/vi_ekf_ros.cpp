#include "vi_ekf_ros.h"

VIEKF_ROS::VIEKF_ROS() :
  nh_private_("~"),
  it_(nh_)
{
  imu_sub_ = nh_.subscribe("imu/data", 500, &VIEKF_ROS::imu_callback, this);
  truth_sub_ = nh_.subscribe("vrpn/Leo/pose", 10, &VIEKF_ROS::truth_callback, this);
  odometry_pub_ = nh_.advertise<nav_msgs::Odometry>("odom", 1);

  image_sub_ = it_.subscribe("camera/rgb/image_mono", 10, &VIEKF_ROS::color_image_callback, this);
  depth_sub_ = it_.subscribe("camera/depth/image_raw", 10, &VIEKF_ROS::depth_image_callback, this);
  output_pub_ = it_.advertise("tracked", 1);

  std::string log_directory;
  std::string default_log_folder = ros::package::getPath("vi_ekf") + "/logs/" + to_string(ros::Time::now().sec) + "/";
  nh_private_.param<std::string>("log_directory", log_directory, default_log_folder );

  ekf_mtx_.lock();
  ekf_.init(ekf_.get_state(), log_directory, true);
  Vector2d cam_center, focal_len;
  focal_len << 533.013144, 533.503964;
  cam_center << 316.680559, 230.660661;
  Matrix3d R_b_c;
  R_b_c << 0, 1, 0,    0, 0, 1,     1, 0, 0;
  quat::Quaternion q_b_c = quat::Quaternion::from_R(R_b_c);
  Vector3d p_b_c;
  p_b_c << 0.16, -0.05, 0.1;
  ekf_.set_camera_intrinsics(cam_center, focal_len);
  ekf_.set_camera_to_IMU(p_b_c, q_b_c);
  ekf_mtx_.unlock();
  klt_tracker_.init(NUM_FEATURES, true, 30);

  // Initialize the depth image to all NaNs
  depth_image_ = cv::Mat(640, 480, CV_32FC1, cv::Scalar(NAN));
  got_depth_ = false;

  // Initialize the measurement noise covariance matrices
  depth_R_ << 0.1;
  feat_R_ << 1.0, 0.0, 0.0, 1.0;
  acc_R_ << 0.5, 0.0, 0.0, 0.5;
  att_R_ << 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01;
  pos_R_ << 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01;
  vel_R_ << 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1;

  // Wait for truth to initialize pose
  initialized_ = false;

  imu_count_ = 0;
  u_sum_.setZero();
}

VIEKF_ROS::~VIEKF_ROS()
{}

void VIEKF_ROS::imu_callback(const sensor_msgs::ImuConstPtr &msg)
{


  if (!initialized_)
    return;

  imu_count_++;

  u_sum_(0) += msg->linear_acceleration.x;
  u_sum_(1) += msg->linear_acceleration.y;
  u_sum_(2) += msg->linear_acceleration.z;
  u_sum_(3) += msg->angular_velocity.x;
  u_sum_(4) += msg->angular_velocity.y;
  u_sum_(5) += msg->angular_velocity.z;

  if ((msg->header.stamp - last_imu_update_).toSec() > 0.010)
  {
    // moving average over imu readings
    Vector6d u = u_sum_ / imu_count_;

    // Reset counting variables
    imu_count_ = 0;
    u_sum_.setZero();
    last_imu_update_ = msg->header.stamp;

    // Propagate filter
    ekf_mtx_.lock();
    ekf_.step(u, msg->header.stamp.toSec());
    ekf_mtx_.unlock();

    // update accelerometer measurement
    Vector2d z_acc = u.block<2,1>(0, 0);
    ekf_mtx_.lock();
    ekf_.update(z_acc, vi_ekf::VIEKF::ACC, acc_R_, true);
    ekf_mtx_.unlock();
  }

  //  Vector4d z_att;
  //  z_att << msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z;
  //  ekf_mtx_.lock();
  //  //  ekf_.update(z_att, vi_ekf::VIEKF::ATT, att_R_, true);
  //  ekf_mtx_.unlock();
}

void VIEKF_ROS::color_image_callback(const sensor_msgs::ImageConstPtr &msg)
{
  if (!initialized_)
    return;

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

   if (!got_depth_)
    return;

  // Track Features in Image
  std::vector<Point2f> features;
  std::vector<int> ids;
  Mat tracked;
  klt_tracker_.load_image(cv_ptr->image, msg->header.stamp.toSec(), features, ids, tracked);

  // Plot feature locations on depth image
  Mat plot_depth_ft, plot_depth;
  depth_image_.copyTo(plot_depth_ft);
  double min, max;
  cv::minMaxLoc(plot_depth_ft, &min, &max);
  plot_depth_ft -= min;
  plot_depth_ft /= (max-min);
  plot_depth_ft *= 255;
  for (int i = 0; i < features.size(); i++)
  {
    circle(plot_depth, features[i], 5, Scalar(255,255,255), -1);
    putText(plot_depth, to_string(ids[i]), features[i], FONT_HERSHEY_COMPLEX, 0.5, Scalar(255,255,255));
  }

  ekf_mtx_.lock();
  ekf_.keep_only_features(ids);
  ekf_mtx_.unlock();
  for (int i = 0; i < features.size(); i++)
  {
    int x = round(features[i].x);
    int y = round(features[i].y);
    float depth_mm;
    double depth;
    try
    {
      depth_mm = depth_image_.at<float>(y, x);
      depth = ((double)depth_mm) * 1e-3;

    }
    catch (const Exception& e)
    {
      ROS_ERROR_STREAM("depth error" << e.what());
      throw;
    }

    if (depth > 1e3)
    {
      depth = NAN;
    }
    else if (depth < 0.1)
    {
      depth = NAN;
    }
    Vector2d z_feat;
    z_feat << features[i].x, features[i].y;
    Matrix1d z_depth;
    z_depth << depth;
    ekf_mtx_.lock();
    if (!ekf_.update(z_feat, vi_ekf::VIEKF::FEAT, feat_R_, true, ids[i], depth))
      ekf_.update(z_depth, vi_ekf::VIEKF::DEPTH, depth_R_, (depth == depth), ids[i]);
    ekf_mtx_.unlock();

    // Draw depth square on tracked features
    double h = 50.0 /depth;
    rectangle(tracked, Point(x-h, y-h), Point(x+h, y+h), Scalar(0, 255, 0));
  }

  cv::imshow("tracked", tracked);
  waitKey(1);
}

void VIEKF_ROS::depth_image_callback(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_FATAL("cv_bridge exception: %s", e.what());
    return;
  }
  depth_image_ = cv_ptr->image;
  got_depth_ = true;
}

void VIEKF_ROS::truth_callback(const geometry_msgs::PoseStampedConstPtr &msg)
{
  if (!initialized_)
  {
    xVector x0;
    x0.setZero();
    x0.block<3,1>(vi_ekf::VIEKF::xPOS, 0) << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    x0.block<4,1>(vi_ekf::VIEKF::xATT, 0) << msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z;
    x0(vi_ekf::VIEKF::xMU, 0) = 0.2;
    ekf_mtx_.lock();
    ekf_.set_x0(x0);
    initialized_ = true;
    ekf_mtx_.unlock();
    return;
  }
  Vector3d z_pos;
  z_pos << msg->pose.position.z, -msg->pose.position.x, -msg->pose.position.y;

  Vector4d z_att;
  z_att << msg->pose.orientation.w, msg->pose.orientation.z, -msg->pose.orientation.x, -msg->pose.orientation.y;

  ekf_mtx_.lock();
  ekf_.update(z_pos, vi_ekf::VIEKF::POS, pos_R_, true);
  ekf_mtx_.unlock();

  ekf_mtx_.lock();
  ekf_.update(z_att, vi_ekf::VIEKF::ATT, att_R_, true);
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





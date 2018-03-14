#include "vi_ekf_ros.h"
#include "eigen_helpers.h"

VIEKF_ROS::VIEKF_ROS() :
  nh_private_("~"),
  it_(nh_)
{
  imu_sub_ = nh_.subscribe("imu/data", 500, &VIEKF_ROS::imu_callback, this);
  truth_sub_ = nh_.subscribe("vrpn/Leo/pose", 10, &VIEKF_ROS::truth_callback, this);
  odometry_pub_ = nh_.advertise<nav_msgs::Odometry>("odom", 1);

  image_sub_ = it_.subscribe("camera/color/image_raw", 10, &VIEKF_ROS::color_image_callback, this);
  depth_sub_ = it_.subscribe("camera/depth/image_rect_raw", 10, &VIEKF_ROS::depth_image_callback, this);
  output_pub_ = it_.advertise("tracked", 1);

  std::string log_directory;
  std::string default_log_folder = ros::package::getPath("vi_ekf") + "/logs/" + to_string(ros::Time::now().sec) + "/";
  nh_private_.param<std::string>("log_directory", log_directory, default_log_folder );
  

  Eigen::Matrix<double, vi_ekf::VIEKF::xZ, 1> x0;
  Eigen::Matrix<double, vi_ekf::VIEKF::dxZ, 1> P0diag, Qxdiag, lambda;
  uVector Qudiag;
  Vector3d P0feat, Qxfeat, lambdafeat;
  Vector2d cam_center, focal_len;
  Vector4d q_b_c;
  Vector3d p_b_c;
  Vector2d feat_r_diag, acc_r_diag;
  Vector3d att_r_diag, pos_r_diag, vel_r_diag;
  importMatrixFromParamServer(nh_private_, x0, "x0");
  importMatrixFromParamServer(nh_private_, P0diag, "P0");
  importMatrixFromParamServer(nh_private_, Qxdiag, "Qx");
  importMatrixFromParamServer(nh_private_, lambda, "lambda");
  importMatrixFromParamServer(nh_private_, lambdafeat, "lambda_feat");
  importMatrixFromParamServer(nh_private_, Qudiag, "Qu");
  importMatrixFromParamServer(nh_private_, P0feat, "P0_feat");
  importMatrixFromParamServer(nh_private_, Qxfeat, "Qx_feat");
  importMatrixFromParamServer(nh_private_, cam_center, "cam_center");
  importMatrixFromParamServer(nh_private_, focal_len, "focal_len");
  importMatrixFromParamServer(nh_private_, q_b_c, "q_b_c");
  importMatrixFromParamServer(nh_private_, p_b_c, "p_b_c");
  importMatrixFromParamServer(nh_private_, feat_r_diag, "feat_R");
  importMatrixFromParamServer(nh_private_, acc_r_diag, "acc_R");
  importMatrixFromParamServer(nh_private_, att_r_diag, "att_R");
  importMatrixFromParamServer(nh_private_, pos_r_diag, "pos_R");
  importMatrixFromParamServer(nh_private_, vel_r_diag, "vel_R");
  double depth_r, alt_r, min_depth;
  bool partial_update, drag_term, keyframe_reset;
  ROS_FATAL_COND(!nh_private_.getParam("depth_R", depth_r), "you need to specify the 'depth_R' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("alt_R", alt_r), "you need to specify the 'alt_R' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("min_depth", min_depth), "you need to specify the 'min_depth' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("imu_LPF", imu_LPF_), "you need to specify the 'imu_LPF' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("num_features", num_features_), "you need to specify the 'num_features' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("invert_image", invert_image_), "you need to specify the 'invert_image' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("partial_update", partial_update), "you need to specify the 'partial_update' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("drag_term", drag_term), "you need to specify the 'drag_term' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("keyframe_reset", keyframe_reset), "you need to specify the 'keyframe_reset' parameter");

  num_features_ = (num_features_ > NUM_FEATURES) ? NUM_FEATURES : num_features_;

  P0feat(2,0) = 1.0/(16.0 * min_depth * min_depth);

  ekf_mtx_.lock();
  ekf_.init(x0, P0diag, Qxdiag, lambda, Qudiag, P0feat, Qxfeat, lambdafeat,
            cam_center, focal_len, q_b_c, p_b_c, min_depth, log_directory, 
            drag_term, partial_update, keyframe_reset);
  ekf_mtx_.unlock();
  klt_tracker_.init(num_features_, false, 30);

  // Initialize the depth image to all NaNs
  depth_image_ = cv::Mat(640, 480, CV_32FC1, cv::Scalar(NAN));
  got_depth_ = false;

  // Initialize the measurement noise covariance matrices
  depth_R_ << depth_r;
  feat_R_ = feat_r_diag.asDiagonal();
  acc_R_ = acc_r_diag.asDiagonal();
  att_R_ = att_r_diag.asDiagonal();
  pos_R_ = pos_r_diag.asDiagonal();
  vel_R_ = vel_r_diag.asDiagonal();
  alt_R_ << alt_r;

  // Turn on the specified measurements
  ROS_FATAL_COND(!nh_private_.getParam("use_truth", use_truth_), "you need to specify the 'use_truth' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_depth", use_depth_), "you need to specify the 'use_depth' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_features", use_features_), "you need to specify the 'use_features' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_acc", use_acc_), "you need to specify the 'use_acc' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_imu_att", use_imu_att_), "you need to specify the 'use_imu_att' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_alt", use_alt_), "you need to specify the 'use_alt' parameter");

  cout << "truth:" << use_truth_ << "\n";
  cout << "depth:" << use_depth_ << "\n";
  cout << "features:" << use_features_ << "\n";
  cout << "acc:" << use_acc_ << "\n";
  cout << "imu_att:" << use_imu_att_ << "\n";
  cout << "imu_alt:" << use_alt_ << "\n";

  // Wait for truth to initialize pose
  initialized_ = false;

  u_prev_.setZero();

//  initialized_ = true;

  odom_msg_.header.frame_id = "body";
}

VIEKF_ROS::~VIEKF_ROS()
{}

void VIEKF_ROS::imu_callback(const sensor_msgs::ImuConstPtr &msg)
{
  imu_(0) = msg->linear_acceleration.x;
  imu_(1) = msg->linear_acceleration.y;
  imu_(2) = msg->linear_acceleration.z;
  imu_(3) = msg->angular_velocity.x;
  imu_(4) = msg->angular_velocity.y;
  imu_(5) = msg->angular_velocity.z;

  if (got_init_truth_ && !initialized_)
  {
//    Vector3d init_b_a, init_b_g;  
//    init_b_g << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
//    init_b_a << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z + 9.80665;
//    ekf_.set_imu_bias(init_b_g, init_b_a);
    initialized_ = true;
    return;
  }

  // Propagate filter
  ekf_mtx_.lock();
  ekf_.propagate(imu_, msg->header.stamp.toSec());
  ekf_mtx_.unlock();

  // update accelerometer measurement
  Vector2d z_acc = imu_.block<2,1>(0, 0);
  ekf_mtx_.lock();
  ekf_.update(z_acc, vi_ekf::VIEKF::ACC, acc_R_, use_acc_);
  ekf_mtx_.unlock();

  // update attitude measurement
  Vector4d z_att;
  z_att << msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z;
  ekf_mtx_.lock();
  if (use_imu_att_)
    ekf_.update(z_att, vi_ekf::VIEKF::ATT, att_R_, (use_truth_) ? true : use_imu_att_);
  ekf_mtx_.unlock();

  odom_msg_.header.stamp = msg->header.stamp;
  odom_msg_.pose.pose.position.x = ekf_.get_state()(vi_ekf::VIEKF::xPOS,0);
  odom_msg_.pose.pose.position.y = ekf_.get_state()(vi_ekf::VIEKF::xPOS+1,0);
  odom_msg_.pose.pose.position.z = ekf_.get_state()(vi_ekf::VIEKF::xPOS+2,0);
  odom_msg_.pose.pose.orientation.w = ekf_.get_state()(vi_ekf::VIEKF::xATT,0);
  odom_msg_.pose.pose.orientation.x = ekf_.get_state()(vi_ekf::VIEKF::xATT+1,0);
  odom_msg_.pose.pose.orientation.y = ekf_.get_state()(vi_ekf::VIEKF::xATT+2,0);
  odom_msg_.pose.pose.orientation.z = ekf_.get_state()(vi_ekf::VIEKF::xATT+3,0);
  odom_msg_.twist.twist.linear.x = ekf_.get_state()(vi_ekf::VIEKF::xVEL,0);
  odom_msg_.twist.twist.linear.y = ekf_.get_state()(vi_ekf::VIEKF::xVEL+1,0);
  odom_msg_.twist.twist.linear.z = ekf_.get_state()(vi_ekf::VIEKF::xVEL+2,0);
  odom_msg_.twist.twist.angular.x = imu_(3);
  odom_msg_.twist.twist.angular.y = imu_(4);
  odom_msg_.twist.twist.angular.z = imu_(5);
  odometry_pub_.publish(odom_msg_);
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

  Mat img;
  if (invert_image_)
    cv::flip(cv_ptr->image, img, -1);
  else
    cv_ptr->image.copyTo(img);
  
  // Track Features in Image
  std::vector<Point2f> features;
  std::vector<int> ids;
  klt_tracker_.load_image(img, msg->header.stamp.toSec(), features, ids);

  ekf_mtx_.lock();
  ekf_.keep_only_features(ids);
  ekf_mtx_.unlock();

  for (int i = 0; i < features.size(); i++)
  {
    int x = round(features[i].x);
    int y = round(features[i].y);
    float depth_mm;
    double depth;
    depth_mm = depth_image_.at<float>(y, x);
    depth = ((double)depth_mm) * 1e-3;
    if (depth > 1e3)
      depth = NAN;
    else if (depth < 0.1)
      depth = NAN;

    Vector2d z_feat;
    z_feat << features[i].x, features[i].y;
    Matrix1d z_depth;
    z_depth << depth;

    ekf_mtx_.lock();
    if (!ekf_.update(z_feat, vi_ekf::VIEKF::FEAT, feat_R_, use_features_, ids[i], (use_depth_) ? depth : NAN))
      ekf_.update(z_depth, vi_ekf::VIEKF::DEPTH, depth_R_, use_depth_, ids[i]);
    ekf_mtx_.unlock();

    // Draw depth and position of tracked features
    Eigen::Vector2d est_feat = ekf_.get_feat(ids[i]);
    circle(img, features[i], 5, Scalar(0,255,0));
    circle(img, Point(est_feat.x(), est_feat.y()), 5, Scalar(255, 0, 255));
    double h_true = 50.0 /depth;
    double h_est = 50.0 /ekf_.get_depth(ids[i]);
    rectangle(img, Point(x-h_true, y-h_true), Point(x+h_true, y+h_true), Scalar(0, 255, 0));
    rectangle(img, Point(est_feat.x()-h_est, est_feat.y()-h_est), Point(est_feat.x()+h_est, est_feat.y()+h_est), Scalar(255, 0, 255));
  }
  cv::imshow("tracked", img);

  cv::Mat cov;
  cv::eigen2cv(ekf_.get_covariance(), cov);
  cv::imshow("covariance", cov);
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
  if (invert_image_)
    cv::flip(cv_ptr->image, depth_image_, ROTATE_180);
  else
    cv_ptr->image.copyTo(depth_image_);
  got_depth_ = true;
}

void VIEKF_ROS::truth_callback(const geometry_msgs::PoseStampedConstPtr &msg)
{
  if (!got_init_truth_)
  {
    xVector x0;
    x0.setZero();
    x0.block<3,1>(vi_ekf::VIEKF::xPOS, 0) << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    x0.block<4,1>(vi_ekf::VIEKF::xATT, 0) << msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z;
    x0(vi_ekf::VIEKF::xMU, 0) = 0.2;
    ekf_mtx_.lock();
//    ekf_.set_x0(x0);
    ekf_mtx_.unlock();
    got_init_truth_ = true;
    return;
  }
  Vector3d z_pos;
  z_pos << msg->pose.position.z, -msg->pose.position.x, -msg->pose.position.y;

  Vector4d z_att;
  z_att << -msg->pose.orientation.w, -msg->pose.orientation.z, msg->pose.orientation.x, msg->pose.orientation.y;

  Eigen::Matrix<double, 1, 1> z_alt;
  z_alt << msg->pose.position.y;

  ekf_mtx_.lock();
  ekf_.update(z_pos, vi_ekf::VIEKF::POS, pos_R_, use_truth_);
  ekf_.update(z_alt, vi_ekf::VIEKF::ALT, alt_R_, (use_truth_) ? false : use_alt_);
  ekf_mtx_.unlock();

  ekf_mtx_.lock();
  if (!use_imu_att_)
    ekf_.update(z_att, vi_ekf::VIEKF::ATT, att_R_, use_truth_);
  ekf_mtx_.unlock();
}


int main(int argc, char* argv[])
{
  ros::init(argc, argv, "vi_ekf_node");
  VIEKF_ROS ekf;

  ros::spin();
  //  ros::AsyncSpinner spinner(0); // Use 4 threads
  //  spinner.start();
  //  ros::waitForShutdown();

  return 0;
}



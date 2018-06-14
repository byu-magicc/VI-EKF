#include "vi_ekf.h"

namespace vi_ekf
{

VIEKF::VIEKF(){}

void VIEKF::init(Matrix<double, xZ,1> x0, Matrix<double, dxZ,1> &P0, Matrix<double, dxZ,1> &Qx,
                 Matrix<double, dxZ,1> &lambda, uVector &Qu, Vector3d& P0_feat, Vector3d& Qx_feat,
                 Vector3d& lambda_feat, Vector2d &cam_center, Vector2d &focal_len, Vector4d &q_b_c,
                 Vector3d &p_b_c, double min_depth, std::string log_directory, bool use_drag_term, bool partial_update,
                 bool use_keyframe_reset, double keyframe_overlap)
{
  x_.block<(int)xZ, 1>(0,0) = x0;
  P_.block<(int)dxZ, (int)dxZ>(0,0) = P0.asDiagonal();
  Qx_.block<(int)dxZ, (int)dxZ>(0,0) = Qx.asDiagonal();
  lambda_.block<(int)dxZ, 1>(0,0) = lambda;
  
  features_.clear();
  features_.reserve(NUM_FEATURES);
  
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    P_.block<3,3>(dxZ+3*i, dxZ+3*i) = P0_feat.asDiagonal();
    Qx_.block<3,3>(dxZ+3*i, dxZ+3*i) = Qx_feat.asDiagonal();
    lambda_.block<3,1>(dxZ+3*i,0) = lambda_feat;
  }
  
  Qu_ = Qu.asDiagonal();
  P0_feat_ = P0_feat.asDiagonal();
  
  Lambda_ = dx_ones_ * lambda_.transpose() + lambda_*dx_ones_.transpose() - lambda_*lambda_.transpose();
  
  len_features_ = 0;
  next_feature_id_ = 0;
  
  // set cam-to-body
  p_b_c_ = p_b_c;
  q_b_c_ = Quat(q_b_c);
  
  // set camera intrinsics
  cam_center_ = cam_center;
  cam_F_ << focal_len(0), 0, 0,
      0, focal_len(1), 0;
  
  use_drag_term_ = use_drag_term;
  partial_update_ = partial_update;
  keyframe_reset_ = use_keyframe_reset;
  prev_t_ = 0.0;
  
  min_depth_ = min_depth;
  
  current_node_global_pose_.setZero();
  current_node_global_pose_(eATT,0) = 1.0;
  
  keyframe_overlap_threshold_ = keyframe_overlap;
  keyframe_features_.clear();
  edges_.clear();
  keyframe_reset_callback_ = nullptr;
  
  if (log_directory.compare("~") != 0)
    init_logger(log_directory);
  
  K_.setZero();
  H_.setZero();
  imu_sum_.setZero();
  imu_count_ = 0; 
}

void VIEKF::set_x0(const Matrix<double, xZ, 1>& _x0)
{
  x_.topRows(xZ) = _x0;
}

void VIEKF::register_keyframe_reset_callback(std::function<void(void)> cb)
{
  keyframe_reset_callback_ = cb;
}


VIEKF::~VIEKF()
{
  if (log_.stream)
  {
    for (std::vector<std::ofstream>::iterator it=log_.stream->begin(); it!=log_.stream->end(); ++it)
    {
      (*it) << endl;
      (*it).close();
    }
  }
}

void VIEKF::set_imu_bias(const Vector3d& b_g, const Vector3d& b_a)
{
  x_.block<3,1>((int)xB_G,0) = b_g;
  x_.block<3,1>((int)xB_A,0) = b_a;
}


const xVector& VIEKF::get_state() const
{
  return x_;
}

const eVector& VIEKF::get_current_node_global_pose() const
{
  return current_node_global_pose_;
}

const MatrixXd VIEKF::get_covariance() const
{
  MatrixXd ret = P_.topLeftCorner(dxZ+3*len_features_, dxZ+3*len_features_);
  return ret;
}


VectorXd VIEKF::get_depths() const
{
  VectorXd out(len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    out[i] = 1.0/x_((int)xZ + 4 + 5*i);
  }
  return out;
}

MatrixXd VIEKF::get_zetas() const
{
  MatrixXd out(3, len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    Vector4d qzeta = x_.block<4,1>(xZ + 5*i,0);
    out.block<3,1>(0,i) = Quat(qzeta).rota(e_z);
  }
  return out;
}

MatrixXd VIEKF::get_qzetas() const
{
  MatrixXd out(4, len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    out.block<4,1>(0,i) = x_.block<4,1>(xZ + 5*i,0);
  }
  return out;
}

VectorXd VIEKF::get_zeta(const int i) const
{
  Vector4d qzeta_i = x_.block<4,1>(xZ + 5*i,0);
  return Quat(qzeta_i).rota(e_z);
}

double VIEKF::get_depth(const int id) const
{
  int i = global_to_local_feature_id(id);
  return 1.0/x_((int)xZ + 4 + 5*i);
}

Vector2d VIEKF::get_feat(const int id) const
{
  int i = global_to_local_feature_id(id);
  Quat q_zeta(x_.block<4,1>(xZ+i*5, 0));
  Vector3d zeta = q_zeta.rota(e_z);
  double ezT_zeta = e_z.transpose() * zeta;
  return cam_F_ * zeta / ezT_zeta + cam_center_;
}

void VIEKF::propagate_Image()
{
  double dt = prev_t_ - prev_image_t_;
  prev_image_t_ = prev_t_;
  
  // Average IMU over the interval
  imu_sum_ /= (double)imu_count_;
  
  // Update covariance over the interval
//  std::cout << "dt = " << dt << " imu_sum: " << imu_sum_.transpose() << "\n";
  dynamics(x_, imu_sum_, false, true);
  int dx = dxZ+3*len_features_;
  P_.topLeftCorner(dx, dx) += (A_.topLeftCorner(dx, dx) * P_.topLeftCorner(dx, dx) 
                               + P_.topLeftCorner(dx, dx) * A_.topLeftCorner(dx, dx).transpose() 
                               + G_.topRows(dx) * Qu_ * G_.topRows(dx).transpose()
                               + Qx_.topLeftCorner(dx, dx) ) * dt;

  
  // zero out imu counters
  imu_sum_.setZero();
  imu_count_ = 0;
}

void VIEKF::propagate_IMU(const uVector &u, const double t)
{
  double start = now();
  
  if (prev_t_ < 0.0001)
  {
    start_t_ = t;
    prev_t_ = t;
    prev_image_t_ = t;
    return;
  }
  
  double dt = t - prev_t_;
  prev_t_ = t;
  
  // Add the IMU measurements for later covariance update
  imu_sum_ += u;
  imu_count_++;
  
  // update the state, but not the covariance
  dynamics(x_, u, true, false);
  
  NAN_CHECK;
  boxplus(x_, dx_*dt, xp_);
  x_ = xp_;
  
  static int count = 0;
  if (count++ % 5 == 0)
    propagate_Image();
  
  NAN_CHECK;
  
  fix_depth();
  
  NAN_CHECK;
  NEGATIVE_DEPTH;
  
  log_.prop_time = (0.1 * (now() - start)) + 0.9 * log_.prop_time;
  log_.count++;
  
  if (log_.count > 10 && log_.stream)
  {
    (*log_.stream)[LOG_PERF] << t-start_t_ << "\t" << log_.prop_time;
    for (int i = 0; i < 10; i++)
    {
      (*log_.stream)[LOG_PERF] << "\t" << log_.update_times[i];
    }
    (*log_.stream)[LOG_PERF] << "\n";
    log_.count = 0;
  }
  
  log_.prop_log_count++;
  if (log_.stream && log_.prop_log_count > 1)
  {
    log_.prop_log_count = 0;
    (*log_.stream)[LOG_PROP] << t-start_t_ << " " << x_.transpose() << " " <<  P_.diagonal().transpose() << " \n";
    (*log_.stream)[LOG_INPUT] << t-start_t_ << " " << (u - x_.block<6,1>((int)xB_A, 0)).transpose() << "\n";
    (*log_.stream)[LOG_XDOT] << t-start_t_ << " " << dt << " " <<  dx_.transpose() << "\n";
  }
}

}







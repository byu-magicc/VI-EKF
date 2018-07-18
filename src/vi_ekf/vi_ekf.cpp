#include "vi_ekf.h"

namespace vi_ekf
{

VIEKF::VIEKF(){}

void VIEKF::init(Matrix<double, xZ,1>& x0, Matrix<double, dxZ,1> &P0, Matrix<double, dxZ,1> &Qx,
                 Matrix<double, dxZ,1> &lambda, uVector &Qu, Vector3d& P0_feat, Vector3d& Qx_feat,
                 Vector3d& lambda_feat, Vector2d &cam_center, Vector2d &focal_len, Vector4d &q_b_c,
                 Vector3d &p_b_c, double min_depth, std::string log_directory, bool use_drag_term, bool partial_update,
                 bool use_keyframe_reset, double keyframe_overlap)
{
  x_.setZero();
  xp_.setZero();
  P_.setZero();
  Qx_.setZero();
  x_.block<(int)xZ, 1>(0,0) = x0;
  P_.block<(int)dxZ, (int)dxZ>(0,0) = P0.asDiagonal();
  Qx_.block<(int)dxZ, (int)dxZ>(0,0) = Qx.asDiagonal();
  lambda_.block<(int)dxZ, 1>(0,0) = lambda;
  
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
  
  current_feature_ids_.clear();
  
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
  prev_t_ = -0.001;
  
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
  if (log_)
  {
    for (std::vector<std::ofstream>::iterator it=log_->begin(); it!=log_->end(); ++it)
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

const dxMatrix& VIEKF::get_covariance() const
{
  return P_;
}

const dxVector VIEKF::get_covariance_diagonal() const
{
  dxVector ret = P_.diagonal();
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

void VIEKF::propagate_covariance()
{
  double dt = prev_t_ - prev_image_t_;
  prev_image_t_ = prev_t_;
  
  // Average IMU over the interval
  imu_sum_ /= (double)imu_count_;
  
  // Update covariance over the interval
  dynamics(x_, imu_sum_, false, true);   
  
  int dx = dxZ+3*len_features_;
//  A_ = I_big_ + A_ *dt + A_*A_*dt*dt/2.0;
  A_ = (A_*dt).exp();
  P_ = A_ * P_ * A_.transpose() + (G_ * Qu_ * G_.transpose()+ Qx_) * dt;
  
  // zero out imu counters
  imu_sum_.setZero();
  imu_count_ = 0;
}

void VIEKF::propagate_state(const uVector &u, const double t)
{
  double start = now();
  
  if (prev_t_ < 0)
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
  
  NAN_CHECK;
  
  fix_depth();
  
  NAN_CHECK;
  NEGATIVE_DEPTH;
  
  log_state(t, x_, P_.diagonal(), u, dx_);
}

}







#include "vi_ekf.h"

#ifndef DEBUG
#define NAN_CHECK if (NaNsInTheHouse()) cout << "NaNs In The House at line " << __LINE__ << "!!!\n"
#define NEGATIVE_DEPTH if (NegativeDepth()) cout << "Negatiive Depth " << __LINE__ << "!!!\n"
#else
#define NAN_CHECK {}
#define NEGATIVE_DEPTH {}
#endif

using namespace quat;
using namespace std;

namespace vi_ekf
{

VIEKF::VIEKF()
{
  xVector x0;
  x0.setZero();
  x0(VIEKF::xATT) = 1.0;
  x0(VIEKF::xMU) = 0.2;
  init(x0, "~", true);
}

void VIEKF::init(xVector x0, std::string log_directory, bool multirotor)
{
  x_ = x0;
  Eigen::VectorXd diag((int)dxZ);

  diag << 0.0001, 0.0001, 0.0001,  // pos
      0.01, 0.01, 0.01,        // vel
      0.001, 0.001, 0.001,     // att
      1e-2, 1e-2, 1e-3,        // b_acc
      1e-3, 1e-3, 1e-3,        // b_omega
      1e-7;                    // mu
  P_.block<(int)dxZ, (int)dxZ>(0, 0) = diag.asDiagonal();
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    P_(dxZ+3*i,   dxZ+3*i)   = 0.01;
    P_(dxZ+3*i+1, dxZ+3*i+1) = 0.01;
    P_(dxZ+3*i+2, dxZ+3*i+2) = 10.0;
  }

  diag << 0.001, 0.001, 0.001,     // pos
      0.1, 0.1, 0.1,           // vel
      0.005, 0.005, 0.005,     // att
      1e-7, 1e-7, 1e-7,        // b_acc
      1e-8, 1e-8, 1e-8,        // b_omega
      0.0;                     // mu
  Qx_.block<(int)dxZ, (int)dxZ>(0,0) = diag.asDiagonal();
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    Qx_(dxZ+3*i,   dxZ+3*i) = 0.001;
    Qx_(dxZ+3*i+1, dxZ+3*i+1) = 0.001;
    Qx_(dxZ+3*i+2, dxZ+3*i+2) = 0.1;
  }

  diag.resize(6,1);
  diag << 0.05, 0.05, 0.05,        // y_acc
      0.001, 0.001, 0.001;     // y_omega
  Qu_ = diag.asDiagonal();

  diag.resize(3,1);
  diag << 0.01, 0.01, 0.5;
  P0_feat_ = diag.asDiagonal();

  len_features_ = 0;
  next_feature_id_ = 0;

  current_feature_ids_.clear();

  q_b_c_ = quat::Quaternion::Identity();
  p_b_c_ = Eigen::Vector3d::Zero();

  use_drag_term = multirotor;
  prev_t_ = 0.0;
  cam_center_ << 320.0, 240.0;
  cam_F_ << 250.0, 0, 0,
            0, 250.0, 0;

  if (log_directory.compare("~") != 0)
  {
    init_logger(log_directory);
  }
}

void VIEKF::set_x0(const Eigen::VectorXd& _x0)
{
  x_ = _x0;
}


VIEKF::~VIEKF()
{
  if (logger_)
  {
    for (std::map<log_type_t, std::ofstream>::iterator it=logger_->begin(); it!=logger_->end(); ++it)
    {
      it->second.close();
    }
  }
}

void VIEKF::set_camera_intrinsics(const Eigen::Vector2d &center, const Eigen::Vector2d &focal_len)
{
  cam_center_ = center;
  cam_F_ << focal_len(0), 0, 0,
      0, focal_len(1), 0;
}

void VIEKF::set_camera_to_IMU(const Eigen::Vector3d& translation, const quat::Quaternion& rotation)
{
  p_b_c_ = translation;
  q_b_c_ = rotation;
}

void VIEKF::set_imu_bias(const Eigen::Vector3d& b_g, const Eigen::Vector3d& b_a)
{
  x_.block<3,1>((int)xB_G,0) = b_g;
  x_.block<3,1>((int)xB_A,0) = b_a;
}


const xVector& VIEKF::get_state() const
{
  return x_;
}

const dxMatrix& VIEKF::get_covariance() const
{
  return P_;
}


Eigen::VectorXd VIEKF::get_depths() const
{
  Eigen::VectorXd out(len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    out[i] = 1.0/x_((int)xZ + 4 + 5*i);
  }
  return out;
}

Eigen::MatrixXd VIEKF::get_zetas() const
{
  Eigen::MatrixXd out(3, len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    Eigen::Vector4d qzeta = x_.block<4,1>(xZ + 5*i,0);
    out.block<3,1>(0,i) = Quaternion(qzeta).rot(e_z);
  }
  return out;
}

Eigen::MatrixXd VIEKF::get_qzetas() const
{
  Eigen::MatrixXd out(4, len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    out.block<4,1>(0,i) = x_.block<4,1>(xZ + 5*i,0);
  }
  return out;
}

Eigen::VectorXd VIEKF::get_zeta(const int i) const
{
  Eigen::Vector4d qzeta_i = x_.block<4,1>(xZ + 5*i,0);
  return Quaternion(qzeta_i).rot(e_z);
}

double VIEKF::get_depth(const int i) const
{
  return 1.0/x_((int)xZ + 4 + 5*i);
}

void VIEKF::boxplus(const xVector& x, const dxVector& dx, xVector& out) const
{
  out.block<6,1>((int)xPOS, 0) = x.block<6,1>((int)xPOS, 0) + dx.block<6,1>((int)dxPOS, 0);
  out.block<4,1>((int)xATT, 0) = (Quaternion(x.block<4,1>((int)xATT, 0)) + dx.block<3,1>((int)dxATT, 0)).elements();
  out.block<7,1>((int)xB_A, 0) = x.block<7,1>((int)xB_A, 0) + dx.block<7,1>((int)dxB_A, 0);

  for (int i = 0; i < len_features_; i++)
  {
    out.block<4,1>(xZ+i*5,0) = q_feat_boxplus(Quaternion(x.block<4,1>(xZ+i*5,0)), dx.block<2,1>(dxZ+3*i,0)).elements();
    out(xZ+i*5+4) = x(xZ+i*5+4) + dx(dxZ+3*i+2);
  }
}


bool VIEKF::init_feature(const Eigen::Vector2d& l, const int id, const double depth)
{
  // If we already have a full set of features, we can't do anything about this new one
  if (len_features_ >= NUM_FEATURES)
    return false;

  // Adjust lambdas to be with respect to image center
  Eigen::Vector2d l_centered = l - cam_center_;

  // Calculate Quaternion to Feature
  Eigen::Vector3d zeta;
  zeta << l_centered(0), l_centered(1)*(cam_F_(1,1)/cam_F_(0,0)), cam_F_(0,0);
  zeta.normalize();
  Eigen::Vector4d qzeta = Quaternion::from_two_unit_vectors(e_z, zeta).elements();

  // If depth is NAN (default argument)
  double init_depth = depth;
  if (depth != depth)
  {
    if (len_features_ > 0)
      init_depth = get_depths().mean();
    else
      init_depth = default_depth_;
  }

  // Increment feature counters
  current_feature_ids_.push_back(next_feature_id_);
  next_feature_id_ += 1;
  len_features_ += 1;

  //  Initialize the state vector
  int x_max = xZ + 5*len_features_;
  x_.block<4,1>(x_max - 5, 0) = qzeta;
  x_(x_max - 1 ) = 1.0/init_depth;

  // Zero out the cross-covariance and reset the uncertainty on this new feature
  int dx_max = dxZ+3*len_features_;
  P_.block(dx_max-3, 0, 3, dx_max-3).setZero();
  P_.block(0, dx_max-3, dx_max-3, 3).setZero();
  P_.block<3,3>(dx_max-3, dx_max-3) = P0_feat_;

  NAN_CHECK;

  return true;
}


void VIEKF::clear_feature(const int id)
{
    int local_feature_id = global_to_local_feature_id(id);
  int xZETA_i = xZ + 5 * local_feature_id;
  int dxZETA_i = dxZ + 3 * local_feature_id;
  current_feature_ids_.erase(current_feature_ids_.begin() + local_feature_id);
  len_features_ -= 1;
  int dx_max = dxZ+3*len_features_;

  // Remove the right portions of state and covariance and shift everything to the upper-left corner of the matrix
  if (local_feature_id < len_features_)
  {
    x_.block(xZETA_i, 0, (x_.rows() - (xZETA_i+5)), 1) = x_.bottomRows(x_.rows() - (xZETA_i + 5));
    P_.block(dxZETA_i, 0, (P_.rows() - (dxZETA_i+3)), P_.cols()) = P_.bottomRows(P_.rows() - (dxZETA_i+3));
    P_.block(0, dxZETA_i, P_.rows(), (P_.cols() - (dxZETA_i+3))) = P_.rightCols(P_.cols() - (dxZETA_i+3));
  }

  NAN_CHECK;
}


void VIEKF::keep_only_features(const vector<int> features)
{
  std::vector<int> features_to_remove;
  for (int local_id = 0; local_id < current_feature_ids_.size(); local_id++)
  {
    bool keep_feature = false;
    for (int i = 0; i < features.size(); i++)
    {
      if (current_feature_ids_[local_id] == features[i])
      {
        keep_feature = true;
        break;
      }
    }
    if (!keep_feature)
    {
      features_to_remove.push_back(current_feature_ids_[local_id]);
    }
  }
  for (int i = 0; i < features_to_remove.size(); i++)
  {
    clear_feature(features_to_remove[i]);
  }
  NAN_CHECK;
}


void VIEKF::step(const uVector &u, const double t)
{
  propagate(x_, P_, u, t);
}

void VIEKF::propagate(const xVector& x, dxMatrix& P, const uVector &u, const double t)
{
  double start = now();
  if (prev_t_ < 0.0001)
  {
    start_t_ = t;
  }
  else
  {
    double dt = t - prev_t_;

    dynamics(x, u, dx_, A_, G_);
    boxplus(x, dx_*dt, x_);
    NAN_CHECK;
     P_ += (A_ * P_ + P_ * A_.transpose() + G_ * Qu_ * G_.transpose() + Qx_)*dt;
  }
  prev_t_ = t;

  NAN_CHECK;
  NEGATIVE_DEPTH;


  perf_log_.prop_time += 0.1 * (now() - start - perf_log_.prop_time);
  perf_log_.count++;

  if (perf_log_.count > 1000 && logger_)
  {
    (*logger_)[LOG_PERF] << t-start_t_ << "\t" << perf_log_.prop_time;
    for (int i = 0; i < 10; i++)
    {
      (*logger_)[LOG_PERF] << "\t" << perf_log_.update_times[i];
    }
    (*logger_)[LOG_PERF] << "\n";
    perf_log_.count = 0;
  }

  if (logger_)
  {
    (*logger_)[LOG_PROP] << t-start_t_ << " " << x_.transpose() << " " <<  P_.diagonal().transpose() << " \n";
  }
}

void VIEKF::dynamics(const xVector &x, const uVector &u, dxVector &xdot, dxMatrix &dfdx, dxuMatrix &dfdu)
{
  dynamics(x, u);
  xdot = dx_;
  dfdx = A_;
  dfdu = G_;
}


void VIEKF::dynamics(const xVector& x, const uVector &u)
{
  dx_.setZero();
  A_.setZero();
  G_.setZero();

  Eigen::Vector3d vel = x.block<3, 1>((int)xVEL, 0);
  Quaternion q_I_b(x.block<4,1>((int)xATT,0));

  Eigen::Vector3d omega = u.block<3,1>((int)uG, 0) - x.block<3,1>((int)xB_G, 0);
  Eigen::Vector3d acc = u.block<3,1>((int)uA, 0) - x.block<3,1>((int)xB_A, 0);
  Eigen::Vector3d acc_z;
  acc_z << 0, 0, acc(2,0);
  double mu = x((int)xMU);

  Eigen::Vector3d gravity_B = q_I_b.invrot(gravity);
  Eigen::Vector3d vel_I = q_I_b.invrot(vel);
  Eigen::Vector3d vel_xy;
  vel_xy << vel(0), vel(1), 0.0;

  // Calculate State Dynamics
  dx_.block<3,1>((int)dxPOS,0) = vel_I;
  if (use_drag_term)
    dx_.block<3,1>((int)dxVEL,0) = acc_z + gravity_B - mu*vel_xy;
  else
    dx_.block<3,1>((int)dxVEL,0) = acc + q_I_b.rot(gravity);
  dx_.block<3,1>((int)dxATT, 0) = omega;

  // State Jacobian
  A_.block<3,3>((int)dxPOS, (int)dxVEL) = q_I_b.R();
  A_.block<3,3>((int)dxPOS, (int)dxATT) = skew(vel_I);
  if (use_drag_term)
  {
    A_.block<3,3>((int)dxVEL, (int)dxVEL) = -mu * I_2x3.transpose() * I_2x3;
    A_.block<3,3>((int)dxVEL, (int)dxB_A) << 0, 0, 0, 0, 0, 0, 0, 0, -1;
    A_.block<3,1>((int)dxVEL, (int)dxMU) = -vel_xy;
  }
  else
  {
    A_.block<3,3>((int)dxVEL, (int)dxB_A) = -I_3x3;
  }
  A_.block<3,3>((int)dxVEL, (int)dxATT) = skew(gravity_B);
  A_.block<3,3>((int)dxATT, (int)dxB_G) = -I_3x3;

  // Input Jacobian
  if (use_drag_term)
    G_.block<3,3>((int)dxVEL, (int)uA) << 0, 0, 0, 0, 0, 0, 0, 0, 1;
  else
    G_.block<3,3>((int)dxVEL, (int)uA) = I_3x3;
  G_.block<3,3>((int)dxATT, (int)uG) = I_3x3;

  // Camera Dynamics
  Eigen::Vector3d vel_c_i = q_b_c_.invrot(vel - omega.cross(p_b_c_));
  Eigen::Vector3d omega_c_i = q_b_c_.invrot(omega);


  Quaternion q_zeta;
  double rho;
  Eigen::Vector3d zeta;
  Eigen::Matrix<double, 3, 2> T_z;
  Eigen::Matrix3d skew_zeta;
  Eigen::Matrix3d skew_vel_c = skew(vel_c_i);
  Eigen::Matrix3d skew_p_b_c = skew(p_b_c_);
  Eigen::Matrix3d R_b_c = q_b_c_.R();
  int xZETA_i, xRHO_i, dxZETA_i, dxRHO_i;
  for (int i = 0; i < len_features_; i++)
  {
    xZETA_i = (int)xZ+i*5;
    xRHO_i = (int)xZ+5*i+4;
    dxZETA_i = (int)dxZ + i*3;
    dxRHO_i = (int)dxZ + i*3+2;

    q_zeta = (x.block<4,1>(xZETA_i, 0));
    rho = x(xRHO_i);
    zeta = q_zeta.rot(e_z);
    T_z = T_zeta(q_zeta);
    skew_zeta = skew(zeta);

    double rho2 = rho*rho;

    // Feature Dynamics
    dx_.block<2,1>(dxZETA_i,0) = -T_z.transpose() * (omega_c_i + rho * zeta.cross(vel_c_i));
    dx_(dxRHO_i) = rho2 * zeta.dot(vel_c_i);

    // Feature Jacobian
    A_.block<2, 3>(dxZETA_i, (int)dxVEL) = -rho * T_z.transpose() * skew_zeta * R_b_c;
    A_.block<2, 3>(dxZETA_i, (int)dxB_G) = T_z.transpose() * (rho * skew_zeta * R_b_c * skew_p_b_c + R_b_c);
    A_.block<2, 2>(dxZETA_i, dxZETA_i) = -T_z.transpose() * (skew(rho * skew_zeta * vel_c_i + omega_c_i) + (rho * skew_vel_c * skew_zeta)) * T_z;
    A_.block<2, 1>(dxZETA_i, dxRHO_i) = -T_z.transpose() * zeta.cross(vel_c_i);
    A_.block<1, 3>(dxRHO_i, (int)dxVEL) = rho2 * zeta.transpose() * R_b_c;
    A_.block<1, 3>(dxRHO_i, (int)dxB_G) = -rho2 * zeta.transpose() * R_b_c * skew_p_b_c;
    A_.block<1, 2>(dxRHO_i, dxZETA_i) = -rho2 * vel_c_i.transpose() * skew_zeta * T_z;
    A_(dxRHO_i, dxRHO_i) = 2 * rho * zeta.transpose() * vel_c_i;

    // Feature Input Jacobian
    G_.block<2, 3>(dxZETA_i, (int)uG) = -T_z.transpose() * (R_b_c + rho*skew_zeta * R_b_c*skew_p_b_c);
    G_.block<1, 3>(dxRHO_i, (int)uG) = rho2*zeta.transpose() * R_b_c * skew_p_b_c;
  }
}

bool VIEKF::update(const Eigen::VectorXd& z, const measurement_type_t& meas_type,
                   const Eigen::MatrixXd& R, const bool active, const int id, const double depth)
{
  double start = now();

  // If this is a new feature, initialize it
  if (meas_type == FEAT && id >= 0)
  {
    if (std::find(current_feature_ids_.begin(), current_feature_ids_.end(), id) == current_feature_ids_.end())
    {
      init_feature(z, id, depth);
      return true; // Don't do a measurement update this time
    }
  }

  int z_dim = z.rows();

  NAN_CHECK;

  zhat_.setZero();
  H_.setZero();

  (this->*(measurement_functions[meas_type]))(x_, zhat_, H_, id);

  NAN_CHECK;

  zVector residual;
  if (meas_type == QZETA)
  {
    residual.topRows(2) = q_feat_boxminus(Quaternion(z), Quaternion(zhat_));
    z_dim = 2;
  }
  else if (meas_type == ATT)
  {
    residual.topRows(3) = Quaternion(z) - Quaternion(zhat_);
    z_dim = 3;
  }
  else
  {
    residual.topRows(z_dim) = z - zhat_.topRows(z_dim);
  }

  NAN_CHECK;

  if (active)
  {
    K_.leftCols(z_dim) = P_ * H_.topRows(z_dim).transpose() * (R + H_.topRows(z_dim)*P_ * H_.topRows(z_dim).transpose()).inverse();
    NAN_CHECK;
    P_ = (I_big_ - K_.leftCols(z_dim)*H_.topRows(z_dim))*P_;
    xp_ = x_;
    boxplus(xp_, K_.leftCols(z_dim) * residual.topRows(z_dim), x_);
  }

  if (meas_type == DEPTH || meas_type == INV_DEPTH)
  {
    // Apply an Inequality Constrain per
    // "Avoiding Negative Depth in Inverse Depth Bearing-Only SLAM"
    // by Parsley and Julier
    for (int i = 0; i < len_features_; i++)
    {
      int xRHO_i = xZ + 5*i + 4;
      int dxRHO_i = dxZ + 3*i + 2;
      if (x_(xRHO_i, 0) < 0.0)
      {
        double err = 1e-3 - x_(xRHO_i, 0);
        P_(dxRHO_i, dxRHO_i) += err*err;
        x_(xRHO_i, 0) += err;
      }
    }
  }

  NAN_CHECK;
  NEGATIVE_DEPTH;

  if (logger_)
  {
    (*logger_)[LOG_MEAS] << measurement_names[meas_type] << "\t" << prev_t_-start_t_ << "\t"
                         << z.transpose() << "\t" << zhat_.topRows(z_dim).transpose() << "\t" << id << "\n";
  }
  perf_log_.update_times[(int)meas_type] += 0.1 * (now() - start - perf_log_.update_times[(int)meas_type]);
  perf_log_.count++;
  return false;
}

void VIEKF::h_acc(const xVector& x, zVector& h, hMatrix& H, const int id)
{
  (void)id;
  Eigen::Vector3d vel = x.block<3,1>((int)xVEL,0);
  Eigen::Vector3d b_a = x.block<3,1>((int)xB_A,0);
  double mu = x(xMU,0);

  h.topRows(2) = I_2x3 * (-mu * vel + b_a);
  H.setZero();
  H.block<2, 3>(0, (int)dxVEL) = -mu * I_2x3;
  H.block<2, 3>(0, (int)dxB_A) = I_2x3;
  H.block<2, 1>(0, (int)dxMU) = -I_2x3*vel;
}

void VIEKF::h_alt(const xVector& x, zVector& h, hMatrix& H, const int id)
{
  (void)id;
  h.row(0) = -x.block<1,1>(xPOS+2, 0);

  H.setZero();
  H(0, dxPOS+2) = -1.0;
}

void VIEKF::h_att(const xVector& x, zVector& h, hMatrix& H, const int id)
{
  (void)id;
  h = x.block<4,1>((int)xATT, 0);

  H.setZero();
  H.block<3,3>(0, dxATT) = I_3x3;
}

void VIEKF::h_pos(const xVector& x, zVector& h, hMatrix& H, const int id)
{
  (void)id;
  h.topRows(3) = x.block<3,1>((int)xPOS,0);

  H.setZero();
  H.block<3,3>(0, (int)xPOS) = I_3x3;
}

void VIEKF::h_vel(const xVector& x, zVector& h, hMatrix& H, const int id)
{
  (void)id;
  h.topRows(3) = x.block<3,1>((int)xVEL, 0);

  H.setZero();
  H.block<3,3>(0, (int)dxVEL) = I_3x3;
}

void VIEKF::h_qzeta(const xVector& x, zVector& h, hMatrix &H, const int id)
{
  int i = global_to_local_feature_id(id);

  h = x.block<4,1>(xZ+i*5, 0);

  H.setZero();
  H.block<2,2>(0, dxZ + i*3) = I_2x2;
}

void VIEKF::h_feat(const xVector& x, zVector& h, hMatrix& H, const int id)
{
  int i = global_to_local_feature_id(id);
  Quaternion q_zeta(x.block<4,1>(xZ+i*5, 0));
  Eigen::Vector3d zeta = q_zeta.rot(e_z);
  Eigen::Matrix3d sk_zeta = skew(zeta);
  double ezT_zeta = e_z.transpose() * zeta;
  Eigen::MatrixXd T_z = T_zeta(q_zeta);

  h.topRows(2) = cam_F_ * zeta / ezT_zeta + cam_center_;

  H.setZero();
  H.block<2,2>(0, dxZ + i*3) = -cam_F_ * ((sk_zeta * T_z)/ezT_zeta - (zeta * e_z.transpose() * sk_zeta * T_z)/(ezT_zeta*ezT_zeta));
}

void VIEKF::h_depth(const xVector& x, zVector& h, hMatrix& H, const int id)
{
  int i = global_to_local_feature_id(id);
  double rho = x(xZ+i*5+4,0 );

  h(0,0) = 1.0/rho;
  H.setZero();
  H(0, dxZ+3*i+2) = -1.0/(rho*rho);
}

void VIEKF::h_inv_depth(const xVector& x, zVector& h, hMatrix& H, const int id)
{
  int i = global_to_local_feature_id(id);
  h(0,0) = x(xZ+i*5+4,0);

  H.setZero();
  H(0, dxZ+3*i+2) = 1.0;
}

void VIEKF::h_pixel_vel(const xVector& x, zVector& h, hMatrix& H, const int id)
{
  (void)x;
  (void)h;
  (void)H;
  (void)id;
  ///TODO:
}

void VIEKF::init_logger(string root_filename)
{
  logger_ = new std::map<log_type_t, std::ofstream>;

  // Make the directory
  system(("mkdir -p " + root_filename).c_str());

  // A logger for the results of propagation
  (*logger_)[LOG_PROP].open(root_filename + "prop.txt", std::ofstream::out | std::ofstream::trunc);
  (*logger_)[LOG_MEAS].open(root_filename + "meas.txt", std::ofstream::out | std::ofstream::trunc);
  (*logger_)[LOG_PERF].open(root_filename + "perf.txt", std::ofstream::out | std::ofstream::trunc);
}





}

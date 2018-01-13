#include "vi_ekf.h"

using namespace quat;
using namespace std;
using namespace std::placeholders;

namespace vi_ekf
{

VIEKF::VIEKF()
{
  Eigen::VectorXd x0;
  x0.setZero(VIEKF::xZ, 1);
  x0(VIEKF::xATT) = 1.0;
  x0(VIEKF::xMU) = 0.2;
  VIEKF(x0, true);
}

VIEKF::VIEKF(Eigen::MatrixXd x0, bool multirotor)
{
  x_ = x0;
  Eigen::VectorXd diag((int)dxZ);

  diag << 0.0001, 0.0001, 0.0001,  // pos
      0.01, 0.01, 0.01,        // vel
      0.001, 0.001, 0.001,     // att
      1e-2, 1e-2, 1e-3,        // b_acc
      1e-3, 1e-3, 1e-3,        // b_omega
      1e-7;                         // mu
  P_ = diag.asDiagonal();


  diag << 0.001, 0.001, 0.001,     // pos
      0.1, 0.1, 0.1,           // vel
      0.005, 0.005, 0.005,     // att
      1e-7, 1e-7, 1e-7,        // b_acc
      1e-8, 1e-8, 1e-8,        // b_omega
      0.0;                     // mu
  Qx_ = diag.asDiagonal();

  diag.resize(3,1);
  diag << 0.001, 0.001, 0.01; // x, y, 1/depth
  Qx_feat_ = diag.asDiagonal();

  diag.resize(6,1);
  diag << 0.05, 0.05, 0.05,        // y_acc
          0.001, 0.001, 0.001;     // y_omega
  Qu_ = diag.asDiagonal();

  diag.resize(3,1);
  diag << 0.01, 0.01, 10.0; // x, y, 1/depth
  P0_feat_ = diag.asDiagonal();

  len_features_ = 0;
  next_feature_id_ = 0;

  global_to_local_feature_id_.clear();

  q_b_c_ = quat::Quaternion::Identity();
  p_b_c_ = Eigen::Vector3d::Zero();

  A_.resize((int)dxZ, (int)dxZ);
  G_.resize((int)dxZ, 6);
  I_big_ = Eigen::Matrix<double, (int)dxZ, (int)dxZ>::Identity();
  dx_.resize((int)dxZ);

  use_drag_term = multirotor;
  prev_t_ = 0.0;
  cam_center_ << 320.0, 240.0;
  cam_F_ << 250.0, 0, 0,
            0, 250.0, 0;

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


Eigen::VectorXd VIEKF::get_state() const
{
  return x_;
}

Eigen::VectorXd VIEKF::get_covariance() const
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

Eigen::VectorXd VIEKF::boxplus(const Eigen::VectorXd& x, const Eigen::VectorXd& dx) const
{
  Eigen::VectorXd out;
  out.resizeLike(x);

  out.block<3,1>((int)xPOS, 0) = x.block<3,1>((int)xPOS, 0) + dx.block<3,1>((int)dxPOS, 0);
  out.block<3,1>((int)xVEL, 0) = x.block<3,1>((int)xVEL, 0) + dx.block<3,1>((int)dxVEL, 0);
  out.block<4,1>((int)xATT, 0) = (Quaternion(x.block<4,1>((int)xATT, 0)) + dx.block<3,1>((int)dxATT, 0)).elements();
  out.block<3,1>((int)xB_A, 0) = x.block<3,1>((int)xB_A, 0) + dx.block<3,1>((int)dxB_A, 0);
  out.block<3,1>((int)xB_G, 0) = x.block<3,1>((int)xB_G, 0) + dx.block<3,1>((int)dxB_G, 0);
  out((int)xMU) = x((int)xMU) + dx((int)dxMU);

  for (int i = 0; i < len_features_; i++)
  {
    int xFEAT = xZ+i*5;
    int xRHO = xZ+i*5+4;
    int dxFEAT = dxZ+3*i;
    int dxRHO = dxZ+3*i+2;

    out.block<4,1>(xFEAT,0) = q_feat_boxplus(Quaternion(x.block<4,1>(xFEAT,0)), dx.block<2,1>(dxFEAT,0)).elements();
    out(xRHO) = x(xRHO) + dx(dxRHO);
  }
  return out;
}


void VIEKF::init_feature(const Eigen::Vector2d& l, const int id, const double depth)
{
  len_features_ += 1;
  global_to_local_feature_id_[id] = next_feature_id_;
  next_feature_id_ += 1;

  // Adjust lambdas to be with respect to image center
  Eigen::Vector2d l_centered = l - cam_center_;

  // Calculate Quaternion to Feature
  Eigen::Vector3d zeta;
  zeta << l_centered(0), l_centered(1)*(cam_F_(1,1)/cam_F_(0,0)), cam_F_(0,0);
  zeta.normalize();
  Eigen::Vector4d qzeta = Quaternion::from_two_unit_vectors(e_z, zeta).elements();

  // If depth is NAN (default arguement)
  double init_depth = depth;
  if (depth != depth)
  {
    if (len_features_ > 0)
      init_depth = get_depths().mean();
    else
      init_depth = default_depth_;
  }

  //  Add 5 states to state vector
  int x_max = xZ + 5*len_features_;
  x_.conservativeResize(x_max, 1);
  x_.block<4,1>(x_max - 5, 0) = qzeta;
  x_(x_max - 1 ) = 1.0/depth;

  // Add 3 more states to covariance and process noise matrices
  int dx_max = dxZ+3*len_features_;
  Qx_.conservativeResize(dx_max, dx_max);
  Qx_.block<3,3>(dx_max-3, dx_max-3) = Qx_feat_;
  P_.conservativeResize(dx_max, dx_max);
  P_.block<3,3>(dx_max-3, dx_max-3) = P0_feat_;

  // Adjust matrix workspace to fit these new states
  A_.resize(dx_max, dx_max);
  G_.resize(dx_max, 6);
  I_big_.resize(dx_max, dx_max);
  for (int i = 0; i < dx_max; i++) I_big_(i,i) = 1.0;
  dx_.resize(dx_max, 1);
}


void VIEKF::clear_feature(const int id)
{
  int local_feature_id = global_to_local_feature_id_[id];
  int xZETA_i = xZ + 5 * local_feature_id;
  int dxZETA_i = dxZ + 3 * local_feature_id;
  auto it = global_to_local_feature_id_.find(id);
  global_to_local_feature_id_.erase(it);
  len_features_ -= 1;

  // Remove the right portions of state and covariance
  if (local_feature_id < len_features_)
  {
    x_.block(xZETA_i, 0, (x_.rows() - (xZETA_i+5)), 1) = x_.bottomRows(x_.rows() - (xZETA_i + 5));
    x_.conservativeResize(x_.rows() - 5);
  }
  for (int i = dxZETA_i; i < 3; i++)
  {
    P_.block(dxZETA_i, 0, (P_.rows() - (dxZETA_i+3)), P_.cols()) = P_.bottomRows(P_.rows() - (dxZETA_i+3));
    P_.block(0, dxZETA_i, P_.rows(), (P_.cols() - (dxZETA_i+3))) = P_.rightCols(P_.cols() - (dxZETA_i+3));
    P_.conservativeResize(P_.rows() - 3, P_.cols() - 3);
  }

  // Adjust matrix workspace to fit these new states
  int dx_max = dxZ+3*len_features_;
  A_.resize(dx_max, dx_max);
  G_.resize(dx_max, 6);
  I_big_.resize(dx_max, dx_max);
  for (int i = 0; i < dx_max; i++) I_big_(i,i) = 1.0;
  dx_.resize(dx_max, 1);
}


void VIEKF::keep_only_features(const vector<int> features)
{
  for (std::map<int, int>::iterator it=global_to_local_feature_id_.begin(); it!=global_to_local_feature_id_.end(); ++it)
  {
    bool keep_feature = false;
    for (int i = 0; i < features.size(); i++)
    {
      if (it->first == features[i])
      {
        keep_feature = true;
      }
    }
    if (!keep_feature)
    {
      clear_feature(it->first);
    }
  }
}


void VIEKF::step(const Eigen::Matrix<double, 6, 1>& u, const double t)
{
  propagate(x_, P_, u, t);
}

void VIEKF::propagate(Eigen::VectorXd& x, Eigen::MatrixXd& P, const Eigen::Matrix<double, 6, 1> u,
                      const double t)
{
  if (prev_t_ > 0.0001)
  {
    double dt = t - prev_t_;

    dynamics(x, u, dx_, A_, G_);
    Eigen::MatrixXd Pdot = A_ * P_ + P_ * A_.transpose() + G_ * Qu_ * G_.transpose() + Qx_;

    x_ = boxplus(x, dx_*dt);
    P_ += Pdot*dt;
  }
  prev_t_ = t;
}
void VIEKF::dynamics(const Eigen::VectorXd& x, const Eigen::MatrixXd& u, Eigen::VectorXd& xdot,
                     Eigen::MatrixXd& dfdx, Eigen::MatrixXd& dfdu)
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
  Eigen::Vector3d vel_c_i = q_b_c_.invrot(vel - skew(omega) * p_b_c_);
  Eigen::Vector3d omega_c_i = q_b_c_.invrot(omega);

  for (int i = 0; i < len_features_; i++)
  {
    int xZETA_i = (int)xZ+i*5;
    int xRHO_i = (int)xZ+5*i+4;
    int dxZETA_i = (int)dxZ + i*3;
    int dxRHO_i = (int)dxZ + i*3+2;

    Quaternion q_zeta(x.block<4,1>(xZETA_i, 0));
    double rho = x(xRHO_i);
    Eigen::Vector3d zeta = q_zeta.rot(e_z);
    Eigen::Matrix<double, 3, 2> T_z = T_zeta(q_zeta);
    Eigen::Matrix3d skew_zeta = skew(zeta);
    Eigen::Matrix3d skew_vel_c = skew(vel_c_i);
    Eigen::Matrix3d skew_p_b_c = skew(p_b_c_);
    Eigen::Matrix3d R_b_c = q_b_c_.R();
    double rho2 = rho*rho;

    // Feature Dynamics
    dx_.block<2,1>(dxZETA_i,0) = -T_z.transpose() * (omega_c_i + rho * skew_zeta * vel_c_i);
    dx_(dxRHO_i) = rho2 * zeta.dot(vel_c_i);

    // Feature Jacobian
    A_.block<2, 3>(dxZETA_i, (int)dxVEL) = -rho * T_z.transpose() * skew_zeta * R_b_c;
    A_.block<2, 3>(dxZETA_i, (int)dxB_G) = T_z.transpose() * (rho * skew_zeta * R_b_c * skew_p_b_c + R_b_c);
    A_.block<2, 2>(dxZETA_i, dxZETA_i) = -T_z.transpose() * (skew(rho * skew_zeta * vel_c_i + omega_c_i) + (rho * skew_vel_c * skew_zeta)) * T_z;
    A_.block<2, 1>(dxZETA_i, dxRHO_i) = -T_z.transpose() * skew_zeta * vel_c_i;
    A_.block<1, 3>(dxRHO_i, (int)dxVEL) = rho2 * zeta.transpose() * R_b_c;
    A_.block<1, 3>(dxRHO_i, (int)dxB_G) = -rho2 * zeta.transpose() * R_b_c * skew_p_b_c;
    A_.block<1, 2>(dxRHO_i, dxZETA_i) = -rho2 * vel_c_i.transpose() * skew_zeta * T_z;
    A_(dxRHO_i, dxRHO_i) = 2 * rho * zeta.transpose() * vel_c_i;

    // Feature Input Jacobian
    G_.block<2, 3>(dxZETA_i, (int)uG) = -T_z.transpose() * (R_b_c + rho*skew_zeta * R_b_c*skew_p_b_c);
    G_.block<1, 3>(dxRHO_i, (int)uG) = rho2*zeta.transpose() * R_b_c * skew_p_b_c;
  }

  // Copy to outputs
  dfdx = A_;
  dfdu = G_;
  xdot = dx_;
}

Eigen::VectorXd VIEKF::update(Eigen::VectorXd& z, const measurement_type_t meas_type,
                              const Eigen::MatrixXd& R, bool passive, const int id, const double depth)
{
  // If this is a new feature, initialize it
  if (meas_type == FEAT && id > 0)
  {
    if (global_to_local_feature_id_.find(id) == global_to_local_feature_id_.end())
    {
      init_feature(z, id, depth);
    }
  }

  Eigen::VectorXd zhat;
  Eigen::MatrixXd H;
  CALL_PTRMEMBER_FN(this, measurement_functions_[meas_type])(x_, zhat, H, id);

  Eigen::VectorXd residual;
  if (meas_type == QZETA)
  {
    residual = q_feat_boxminus(Quaternion(z), Quaternion(zhat));
  }
  else if (meas_type == ATT)
  {
    residual = Quaternion(z) - Quaternion(zhat);
  }
  else
  {
    residual = z - zhat;
  }

  if (!passive)
  {
    Eigen::MatrixXd K = P_ * H.transpose() * (R + H*P_).inverse() * H.transpose();
    P_ = (I_big_ - K*H)*P_;
    x_ = boxplus(x_, K * residual);
  }
}

void VIEKF::h_acc(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id)
{
  (void)id;
  Eigen::Vector3d vel = x.block<3,1>((int)xVEL,0);
  Eigen::Vector3d b_a = x.block<3,1>((int)xB_A,0);
  double mu = x(xMU,0);

  h = I_2x3 * (-mu * vel + b_a);
  H.setZero(2, dxZ + 3*len_features_);
  H.block<2, 3>(0, (int)dxVEL) = -mu * I_2x3;
  H.block<2, 3>(0, (int)dxB_A) = I_2x3;
  H.block<2, 1>(0, (int)dxMU) = -I_2x3*vel;
}

void VIEKF::h_alt(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id)
{
  (void)id;
  h = -x.block<1,1>(xPOS+2, 0);

  H.setZero(1, dxZ + 3*len_features_);
  H(0, dxPOS+2) = -1.0;
}

void VIEKF::h_att(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id)
{
  (void)id;
  h = x.block<4,1>((int)xATT, 0);

  H.setZero(3, dxZ + 3*len_features_);
  H.block<3,3>(0, dxATT) = I_3x3;
}

void VIEKF::h_pos(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id)
{
  (void)id;
  h = x.block<3,1>((int)xPOS,0);

  H.setZero(3, dxZ+3*len_features_);
  H.block<3,3>(0, (int)xPOS) = I_3x3;
}

void VIEKF::h_vel(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id)
{
  (void)id;
  h = x.block<3,1>((int)xVEL, 0);

  H.setZero(3, dxZ+3*len_features_);
  H.block<3,3>(0, (int)dxVEL) = I_3x3;
}

void VIEKF::h_qzeta(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd &H, const int id)
{
  int i = global_to_local_feature_id_[id];

  h = x.block<4,1>(xZ+i*5, 0);

  H.setZero(2, dxZ+3*len_features_);
  H.block<2,2>(0, dxZ + i*3) = I_2x2;
}

void VIEKF::h_feat(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id)
{
  int i = global_to_local_feature_id_[id];
  Quaternion q_zeta(x.block<4,1>(xZ+i*5, 0));
  Eigen::Vector3d zeta = q_zeta.rot(e_z);
  Eigen::Matrix3d sk_zeta = skew(zeta);
  double ezT_zeta = e_z.transpose() * zeta;
  Eigen::MatrixXd T_z = T_zeta(q_zeta);

  h = cam_F_ * zeta / ezT_zeta + cam_center_;

  H.setZero(2, dxZ + 3*len_features_);
  H.block<2,2>(0, dxZ + i*3) = -cam_F_ * ((sk_zeta * T_z)/ezT_zeta - (zeta * e_z.transpose() * sk_zeta * T_z)/(ezT_zeta*ezT_zeta));
}

void VIEKF::h_depth(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id)
{
  int i = global_to_local_feature_id_[id];
  double rho = x(xZ+i*5+4,0 );
  h.resize(1,1);
  h(0,0) = 1.0/rho;
  H.setZero(1, dxZ+3 * len_features_);
  H(0, dxZ+3*id+2) = -1.0/(rho*rho);
}

void VIEKF::h_inv_depth(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id)
{
  int i = global_to_local_feature_id_[id];
  h = x.block<1,1>(xZ+i*5+4,0);
  H.setZero(1, dxZ+3 * len_features_);
  H(0, dxZ+3*id+2) = 1.0;
}

void VIEKF::h_pixel_vel(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id)
{
  (void)x;
  (void)h;
  (void)H;
  (void)id;
  ///TODO:
}





}

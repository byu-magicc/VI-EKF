#include "vi_ekf.h"

using namespace quat;
using namespace std;
using namespace std::placeholders;

namespace vi_ekf
{

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

  initialized_features_.clear();
  global_to_local_feature_id_.clear();

  q_b_c_ = quat::Quaternion::Identity();
  p_b_c_ = Eigen::Vector3d::Zero();

  measurement_functions_.clear();
  measurement_functions_[ACC] = &VIEKF::h_acc;
  measurement_functions_[ALT] = &VIEKF::h_alt;
  measurement_functions_[ATT] = &VIEKF::h_att;
  measurement_functions_[POS] = &VIEKF::h_pos;
  measurement_functions_[VEL] = &VIEKF::h_vel;
  measurement_functions_[QZETA] = &VIEKF::h_qzeta;
  measurement_functions_[FEAT] = &VIEKF::h_feat;
  measurement_functions_[PIXEL_VEL] = &VIEKF::h_pixel_vel;
  measurement_functions_[DEPTH] = &VIEKF::h_depth;
  measurement_functions_[INV_DEPTH] = &VIEKF::h_inv_depth;

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

Eigen::VectorXd VIEKF::get_depths(){}
Eigen::MatrixXd VIEKF::get_zetas(){}
Eigen::MatrixXd VIEKF::get_qzetas(){}
Eigen::VectorXd VIEKF::get_zeta(const int i){}
double VIEKF::get_depth(const int i){}

void VIEKF::boxplus(const Eigen::VectorXd& x, const Eigen::VectorXd& dx, Eigen::VectorXd& out)
{
  out.resizeLike(x);

  out.block<3,1>((int)xPOS, 0) = x.block<3,1>((int)xPOS, 0) + dx.block<3,1>((int)dxPOS, 0);
  out.block<3,1>((int)xVEL, 0) = x.block<3,1>((int)xVEL, 0) + dx.block<3,1>((int)dxVEL, 0);
  out.block<4,1>((int)xATT, 0) = (Quaternion(x.block<4,1>((int)xPOS, 0)) + dx.block<3,1>((int)dxPOS, 0)).elements();
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
}
void VIEKF::propagate(Eigen::VectorXd& x, Eigen::MatrixXd& P, const Eigen::Matrix<double, 6, 1> u,
                      const double t)
{
  if (prev_t_ > 0.0001)
  {
    double dt = t - prev_t_;

    dynamics(x, u, dx_, A_, G_);
    Eigen::MatrixXd Pdot = A_ * P_ + P_ * A_.transpose() + G_ * Qu_ * G_.transpose() + Qx_;

    boxplus(x, dx_*dt, x_);
    P_ += Pdot*dt;
  }
}

Eigen::VectorXd VIEKF::update(Eigen::VectorXd& z, const measurement_type_t meas_type,
                       const Eigen::MatrixXd& R, bool passive, const int id){}

void VIEKF::set_imu_bias(const Eigen::Vector3d& b_g, const Eigen::Vector3d& b_a){}

int VIEKF::init_feature(const Eigen::Vector4d& q_zeta, const int id, const double depth){}
void VIEKF::clear_feature(const int id){}
void VIEKF::keep_only_features(Eigen::VectorXd features){}
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
  Eigen::Vector3d vel_xy = I_2x3.transpose() * I_2x3 * vel;

  // Calculate State Dynamics
  dx_.block<3,1>((int)dxPOS,0) = vel_I;
  if (use_drag_term)
    dx_.block<3,1>((int)dxVEL,0) = acc_z + gravity_B - mu*vel_xy;
  else
    dx_.block<3,1>((int)dxVEL,0) = acc + gravity_B;
  dx_.block<3,1>((int)dxATT, 0) = omega;

  // State Jacobian
  A_.block<3,3>((int)dxPOS, (int)dxVEL) = q_I_b.R();
  A_.block<3,3>((int)dxPOS, (int)dxATT) = skew(vel_I);
  if (use_drag_term)
  {
    A_.block<3,3>((int)dxVEL, (int)dxVEL) = -mu * I_2x3.transpose() * I_2x3;
    A_.block<3,3>((int)dxVEL, (int)dxB_A) << 0, 0, 0, 0, 0, 0, 0, 0, -1;
    A_.block<3,1>((int)dxPOS, (int)dxMU) = -vel_xy;
  }
  else
  {
    A_.block<3,3>((int)dxVEL, (int)dxB_A) = -I_3x3;
  }

  // Input Jacobian
  if (use_drag_term)
    A_.block<3,3>((int)dxVEL, (int)uA) << 0, 0, 0, 0, 0, 0, 0, 0, 1;
  else
    A_.block<3,3>((int)dxVEL, (int)uA) = I_3x3;
  A_.block<3,3>((int)dxATT, (int)uG) = I_3x3;

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
    A_.block<2, 2>(dxZETA_i, dxZETA_i) = T_z.transpose() * (skew(omega_c_i - rho*skew_zeta*vel_c_i) - (rho*skew_vel_c * skew_zeta)) * T_z;
    A_.block<2, 1>(dxZETA_i, dxRHO_i) = -T_z.transpose() * skew_zeta * vel_c_i;
    A_.block<1, 3>(dxRHO_i, (int)dxVEL) = rho2 * zeta.transpose() * R_b_c;
    A_.block<1, 3>(dxRHO_i, (int)dxB_G) = -rho2 * zeta.transpose() * R_b_c * skew_p_b_c;
    A_.block<1, 2>(dxRHO_i, dxZETA_i) = -rho2 * vel_c_i.transpose() * skew_zeta * T_z;
    A_(dxRHO_i, dxRHO_i) = 2 * rho * zeta.transpose() * vel_c_i;

    // Feature Input Jacobian
    A_.block<2, 3>(dxZETA_i, (int)uG) = -T_z.transpose() * (R_b_c + rho*skew_zeta * R_b_c*skew_p_b_c);
    A_.block<1, 3>(dxRHO_i, (int)uG) = rho2*zeta.transpose() * R_b_c * skew_p_b_c;
  }

  // Copy to outputs
  dfdx = A_;
  dfdu = G_;
  xdot = dx_;
}

void VIEKF::h_acc(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}
void VIEKF::h_alt(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}
void VIEKF::h_att(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}
void VIEKF::h_pos(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}
void VIEKF::h_vel(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}
void VIEKF::h_qzeta(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}
void VIEKF::h_feat(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}
void VIEKF::h_depth(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}
void VIEKF::h_inv_depth(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}
void VIEKF::h_pixel_vel(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::VectorXd& H, const int id){}





}

#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"

#include <set>
#include <map>
#include <functional>
#include <fstream>
#include <chrono>

#include "quat.h"
#include "math_helper.h"



namespace vi_ekf
{

class VIEKF;

typedef void (VIEKF::*measurement_function_ptr)(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);

static const Eigen::Vector3d gravity = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 0, 9.80665;
  return tmp;
}();

class VIEKF
{
public:

  enum : int{
    xPOS = 0,
    xVEL = 3,
    xATT = 6,
    xB_A = 10,
    xB_G = 13,
    xMU = 16,
    xZ = 17
  };

  enum : int{
    uA = 0,
    uG = 3,
    uTOTAL = 6
  };

  enum : int {
    dxPOS = 0,
    dxVEL = 3,
    dxATT = 6,
    dxB_A = 9,
    dxB_G = 12,
    dxMU = 15,
    dxZ = 16
  };

  typedef enum {
    ACC,
    ALT,
    ATT,
    POS,
    VEL,
    QZETA,
    FEAT,
    PIXEL_VEL,
    DEPTH,
    INV_DEPTH
  } measurement_type_t;

private:
  typedef enum {
    LOG_PROP,
    LOG_MEAS,
    LOG_PERF,
  } log_type_t;

  // State and Covariance Matrices
  Eigen::VectorXd x_;
  Eigen::MatrixXd P_;

  // Process Noise and Initialization Matrices
  Eigen::Matrix3d P0_feat_;
  Eigen::MatrixXd Qx_;
  Eigen::Matrix3d Qx_feat_;
  Eigen::Matrix<double, 6, 6> Qu_;

  // Internal bookkeeping variables
  double prev_t_;
  double start_t_;
  int len_features_;
  int next_feature_id_;
  std::map<int, int> global_to_local_feature_id_;

  // Matrix Workspace
  Eigen::MatrixXd A_;
  Eigen::MatrixXd G_;
  Eigen::MatrixXd I_big_;
  Eigen::VectorXd dx_;

  // EKF Configuration Parameters
  bool use_drag_term;
  double default_depth_ = 1.5;

  // Camera Intrinsics and Extrinsics
  Eigen::Vector2d cam_center_;
  Eigen::Matrix<double, 2, 3> cam_F_;
  quat::Quaternion q_b_c_;
  Eigen::Vector3d p_b_c_;

  // Log Stuff
  std::map<log_type_t, std::ofstream>* logger_ = nullptr;

  typedef struct
  {
    double update_times[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double prop_time = 0.0;
    int count = 0;
  } perf_log_t;
  perf_log_t perf_log_;

public:

  VIEKF();
  ~VIEKF();
  void init(Eigen::MatrixXd x0, std::string log_directory, bool multirotor=true);

  void init_logger(std::string root_filename);

  inline double now()
  {
    std::chrono::microseconds now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
    return (double)now.count()*1e-6;
  }

  inline bool NaNsInTheHouse()
  {
    if( ( (x_ - x_).array() != (x_ - x_).array()).all() || ( (P_ - P_).array() != (P_ - P_).array()).all() )
      return true;
    else
      return false;
  }

  void set_camera_to_IMU(const Eigen::Vector3d& translation, const quat::Quaternion& rotation);
  void set_camera_intrinsics(const Eigen::Vector2d& center, const Eigen::Vector2d& focal_len);
  Eigen::VectorXd get_depths() const;
  Eigen::MatrixXd get_zetas() const;
  Eigen::MatrixXd get_qzetas() const;
  Eigen::VectorXd get_zeta(const int i) const;
  Eigen::VectorXd get_state() const;
  Eigen::VectorXd get_covariance() const;
  double get_depth(const int i) const;
  inline int get_len_features() const { return len_features_; }
  void set_imu_bias(const Eigen::Vector3d& b_g, const Eigen::Vector3d& b_a);
  void init_feature(const Eigen::Vector2d &l, const int id, const double depth=-1.0);
  void clear_feature(const int id);
  void keep_only_features(const std::vector<int> features);

  // State Propagation
  Eigen::VectorXd boxplus(const Eigen::VectorXd& x, const Eigen::VectorXd& dx) const;
  void step(const Eigen::Matrix<double, 6, 1>& u, const double t);
  void propagate(Eigen::VectorXd& x, Eigen::MatrixXd& P, const Eigen::Matrix<double, 6, 1> u, const double t);
  void dynamics(const Eigen::VectorXd& x, const Eigen::MatrixXd& u, Eigen::VectorXd& xdot,
                Eigen::MatrixXd& dfdx, Eigen::MatrixXd& dfdu);

  // Measurement Updates
  void update(const Eigen::MatrixXd& z, const measurement_type_t& meas_type, const Eigen::MatrixXd& R, const bool passive=false, const int id=-1, const double depth=NAN);
  void h_acc(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
  void h_alt(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
  void h_att(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
  void h_pos(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
  void h_vel(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
  void h_qzeta(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
  void h_feat(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
  void h_depth(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
  void h_inv_depth(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
  void h_pixel_vel(const Eigen::VectorXd& x, Eigen::VectorXd& h, Eigen::MatrixXd& H, const int id);
};

static std::map<VIEKF::measurement_type_t, std::string> measurement_names = [] {
  std::map<VIEKF::measurement_type_t, std::string> tmp;
  tmp[VIEKF::ACC] = "ACC";
  tmp[VIEKF::ALT] = "ALT";
  tmp[VIEKF::ATT] = "ATT";
  tmp[VIEKF::POS] = "POS";
  tmp[VIEKF::VEL] = "VEL";
  tmp[VIEKF::QZETA] = "QZETA";
  tmp[VIEKF::FEAT] = "FEAT";
  tmp[VIEKF::PIXEL_VEL] = "PIXEL_VEL";
  tmp[VIEKF::DEPTH] = "DEPTH";
  tmp[VIEKF::INV_DEPTH] = "INV_DEPTH";
  return tmp;
}();

static std::map<VIEKF::measurement_type_t, measurement_function_ptr> measurement_functions = [] {
  std::map<VIEKF::measurement_type_t, measurement_function_ptr> tmp;
  tmp[VIEKF::ACC] = &VIEKF::h_acc;
  tmp[VIEKF::ALT] = &VIEKF::h_alt;
  tmp[VIEKF::ATT] = &VIEKF::h_att;
  tmp[VIEKF::POS] = &VIEKF::h_pos;
  tmp[VIEKF::VEL] = &VIEKF::h_vel;
  tmp[VIEKF::QZETA] = &VIEKF::h_qzeta;
  tmp[VIEKF::FEAT] = &VIEKF::h_feat;
  tmp[VIEKF::PIXEL_VEL] = &VIEKF::h_depth;
  tmp[VIEKF::DEPTH] = &VIEKF::h_inv_depth;
  tmp[VIEKF::INV_DEPTH] = &VIEKF::h_pixel_vel;
  return tmp;
}();

}


#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"

#include <deque>
#include <set>
#include <map>
#include <functional>
#include <fstream>
#include <chrono>
#include <iostream>

#include "quat.h"
#include "math_helper.h"


#ifndef NUM_FEATURES
#ifndef NDEBUG
#define NUM_FEATURES 3
#else
#define NUM_FEATURES 35
#endif
#endif

#define MAX_X 17+NUM_FEATURES*5
#define MAX_DX 16+NUM_FEATURES*3
#define AVG_DEPTH 1.5

typedef Eigen::Matrix<double, MAX_X, 1> xVector;
typedef Eigen::Matrix<double, MAX_DX, 1> dxVector;
typedef Eigen::Matrix<double, MAX_X, MAX_X> xMatrix;
typedef Eigen::Matrix<double, MAX_DX, MAX_DX> dxMatrix;
typedef Eigen::Matrix<double, MAX_DX, 6> dxuMatrix;
typedef Eigen::Matrix<double, 6, 1> uVector;
typedef Eigen::Matrix<double, 4, 1> zVector;
typedef Eigen::Matrix<double, 3, MAX_DX> hMatrix;

namespace vi_ekf
{

class VIEKF;

typedef void (VIEKF::*measurement_function_ptr)(const xVector& x, zVector& h, hMatrix& H, const int id) const;

static const Eigen::Vector3d gravity = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 0, 9.80665;
  return tmp;
}();

static const Eigen::Vector3d khat = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 0, 1.0;
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
    LOG_CONF,
  } log_type_t;

  // State and Covariance and Process Noise Matrices
  xVector x_;
  dxMatrix P_;
  dxMatrix Qx_;
  Eigen::Matrix<double, 6, 6> Qu_;

  // Partial Update Gains
  dxVector lambda_;
  dxMatrix Lambda_;
  
  // Initial uncertainty on features
  Eigen::Matrix3d P0_feat_;

  // Internal bookkeeping variables
  double prev_t_;
  double start_t_;
  int len_features_;
  int next_feature_id_;
  std::vector<int> current_feature_ids_;
//  std::vector<int> keyframe_features_;
//  double keyframe_overlap_threshold_;
  
//  typedef struct{
//    Eigen::Vector3d transform;
//    Eigen::Matrix3d cov;    
//  } edge_SE2_t;
//  std::deque<edge_SE2_t> edges_;

  // Matrix Workspace
  dxMatrix A_;
  dxuMatrix G_;
  dxVector dx_;
  const dxMatrix I_big_ = dxMatrix::Identity();
  const dxMatrix Ones_big_ = dxMatrix::Constant(1.0);
  const dxVector dx_ones_ = dxVector::Constant(1.0);
  xVector xp_;
  Eigen::Matrix<double, MAX_DX, 3>  K_;
  zVector zhat_;
  hMatrix H_;

  // EKF Configuration Parameters
  bool use_drag_term_;
//  bool keyframe_reset_;
  bool partial_update_;
  double min_depth_;

  // Camera Intrinsics and Extrinsics
  Eigen::Vector2d cam_center_;
  Eigen::Matrix<double, 2, 3> cam_F_;
  quat::Quaternion q_b_c_;
  Eigen::Vector3d p_b_c_;

  // Log Stuff
  typedef struct
  {
    std::map<log_type_t, std::ofstream>* stream = nullptr;
    double update_times[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int update_count[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double prop_time = 0.0;
    int prop_log_count = 0;
    int count = 0;
  } log_t;
  log_t log_;


public:

  VIEKF();
  ~VIEKF();
  void init(Eigen::Matrix<double, xZ,1> x0, Eigen::Matrix<double, dxZ,1> &P0, Eigen::Matrix<double, dxZ,1> &Qx,
            Eigen::Matrix<double, dxZ,1> &lambda, uVector &Qu, Eigen::Vector3d& P0_feat, Eigen::Vector3d& Qx_feat,
            Eigen::Vector3d& lambda_feat, Eigen::Vector2d& cam_center, Eigen::Vector2d& focal_len,
            Eigen::Vector4d& q_b_c, Eigen::Vector3d &p_b_c, double min_depth, std::string log_directory, bool use_drag_term, 
            bool partial_update, bool keyframe_reset);
  void init_logger(std::string root_filename);

  inline double now() const
  {
    std::chrono::microseconds now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
    return (double)now.count()*1e-6;
  }

  inline bool NaNsInTheHouse() const
  {
    if( ( (x_).array() != (x_).array()).any() || ((P_).array() != (P_).array()).any() )
    {
      std::cout << "x:\n" << x_.block(0,0, xZ + len_features_, 1) << "\n";
      std::cout << "P:\n" << P_.block(0, 0,xZ + len_features_, xZ + len_features_) << "\n";
      return true;
    }
    else
      return false;
  }

  inline bool BlowingUp() const
  {
    if ( ((x_).array() > 1e8).any() || ((P_).array() > 1e8).any())
      return true;
    else
      return false;
  }

  inline bool NegativeDepth() const
  {
    for (int i = 0; i < len_features_; i++)
    {
      int xRHO_i = (int)xZ+5*i+4;
      if (x_(xRHO_i,0) < 0)
        return true;
    }
    return false;
  }

  inline int global_to_local_feature_id(const int global_id) const
  {
    int dist = std::distance(current_feature_ids_.begin(), std::find(current_feature_ids_.begin(), current_feature_ids_.end(), global_id));
    if (dist < current_feature_ids_.size())
    {
      return dist;
    }
    else
    {
         return -1;
    }
  }

  Eigen::VectorXd get_depths() const;
  Eigen::MatrixXd get_zetas() const;
  Eigen::MatrixXd get_qzetas() const;
  Eigen::VectorXd get_zeta(const int i) const;
  Eigen::Vector2d get_feat(const int id) const;
  const xVector& get_state() const;
  const dxMatrix& get_covariance() const;
  double get_depth(const int id) const;
  inline int get_len_features() const { return len_features_; }

  void set_x0(const Eigen::VectorXd& _x0);
  void set_imu_bias(const Eigen::Vector3d& b_g, const Eigen::Vector3d& b_a);

  bool init_feature(const Eigen::Vector2d &l, const int id, const double depth=-1.0);
  void clear_feature(const int id);
  void keep_only_features(const std::vector<int> features);

  // State Propagation
  void boxplus(const xVector &x, const dxVector &dx, xVector &out) const;
  void boxminus(const xVector& x1, const xVector &x2, dxVector& out) const;
  void step(const uVector& u, const double t);
  void propagate(const uVector& u, const double t);
  void dynamics(const xVector &x, const uVector& u, dxVector& xdot, dxMatrix& dfdx, dxuMatrix& dfdu);
  void dynamics(const xVector &x, const uVector& u);

  // Measurement Updates
  bool update(const Eigen::VectorXd& z, const measurement_type_t& meas_type, const Eigen::MatrixXd& R, const bool active=false, const int id=-1, const double depth=NAN);
  void h_acc(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_alt(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_att(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_pos(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_vel(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_qzeta(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_feat(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_depth(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_inv_depth(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_pixel_vel(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  
  // Keyframe Reset
  void keyframe_reset(const xVector &xm, xVector &xp, dxMatrix &N);
  void keyframe_reset();

  // Inequality Constraint on Depth
  void fix_depth();
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
  tmp[VIEKF::DEPTH] = "DEPTH";
  tmp[VIEKF::INV_DEPTH] = "INV_DEPTH";
  tmp[VIEKF::PIXEL_VEL] = "PIXEL_VEL";
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
  tmp[VIEKF::DEPTH] = &VIEKF::h_depth;
  tmp[VIEKF::INV_DEPTH] = &VIEKF::h_inv_depth;
  tmp[VIEKF::PIXEL_VEL] = &VIEKF::h_pixel_vel;
  return tmp;
}();

}




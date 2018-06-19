#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Dense"

#include <deque>
#include <set>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "quat.h"
#include "math_helper.h"

using namespace quat;
using namespace std;
using namespace Eigen;
using namespace cv;

#define NO_NANS(mat) (mat.array() == mat.array()).all()

//#ifndef NDEBUG
#define NAN_CHECK if (NaNsInTheHouse()) { std::cout << "NaNs In The House at line " << __LINE__ << "!!!\n"; exit(0); }
#define NEGATIVE_DEPTH if (NegativeDepth()) std::cout << "Negative Depth " << __LINE__ << "!!!\n"
#define CHECK_MAT_FOR_NANS(mat) if ((K_.array() != K_.array()).any()) { std::cout << "NaN detected in " << #mat << " at line " << __LINE__ << "!!!\n" << mat << "\n"; exit(0); }
#define ASSERT(cond, message) \
  { \
    if (! (cond)) {\
              std::cerr << "Assertion `" #cond "` failed in " << __FILE__ \
                        << " line " << __LINE__ << ": " << message << std::endl; \
              std::terminate(); \
          } \
  }
//#else
//#define NAN_CHECK {}
//#define NEGATIVE_DEPTH {}
//#define CHECK_MAT_FOR_NANS(mat) {}
//#define ASSERT(condition, message) {}
//#endif

#ifndef NUM_FEATURES
#ifndef NDEBUG
#define NUM_FEATURES 3
#else
#define NUM_FEATURES 35 
#endif
#endif

#define MAX_X 17+NUM_FEATURES*5
#define MAX_DX 16+NUM_FEATURES*3
#define PATCH_SIZE 6
#define PYRAMID_LEVELS 2

typedef Matrix<double, MAX_X, 1> xVector;
typedef Matrix<double, MAX_DX, 1> dxVector;
typedef Matrix<double, MAX_X, MAX_X> xMatrix;
typedef Matrix<double, MAX_DX, MAX_DX> dxMatrix;
typedef Matrix<double, MAX_DX, 6> dxuMatrix;
typedef Matrix<double, 6, 1> uVector;
typedef Matrix<double, 7, 1> eVector;
typedef Matrix<double, 4, 1> zVector;
typedef Matrix<double, 3, MAX_DX> hMatrix;

// Use floats for stuff that touches OpenCV
typedef Matrix<float, 2, 1> pixVector;
typedef Matrix<float, PATCH_SIZE, PATCH_SIZE> patchMat;
typedef Matrix<float, PATCH_SIZE*PATCH_SIZE*PYRAMID_LEVELS, 1> multiPatchVectorf;
typedef Matrix<float, PATCH_SIZE, PATCH_SIZE*PYRAMID_LEVELS> multiPatchMatrixf;
typedef Matrix<float, PATCH_SIZE*PATCH_SIZE*PYRAMID_LEVELS, 2> multiPatchJacMatrix;

namespace vi_ekf
{

class VIEKF;

typedef void (VIEKF::*measurement_function_ptr)(const xVector& x, zVector& h, hMatrix& H, const int id) const;

static const Vector3d gravity = [] {
  Vector3d tmp;
  tmp << 0, 0, 9.80665;
  return tmp;
}();

static const Vector3d khat = [] {
  Vector3d tmp;
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
    ePOS = 0,
    eATT = 3
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
    INV_DEPTH,
    TOTAL_MEAS
  } measurement_type_t;
  
  typedef enum{
    OKAY,
    NEW_FEATURE,
    MEASUREMENT_GATED,
    INVALID_MEASUREMENT,
    FEATURE_LOST,
    FEATURE_TRACKED
  } update_return_code_t;
  
  typedef struct{
    eVector transform;
    Matrix3d cov;    
  } edge_SE2_t;

private:
  typedef enum {
    LOG_PROP,
    LOG_MEAS,
    LOG_PERF,
    LOG_INPUT,
    LOG_XDOT,
    LOG_CONF,
    LOG_KF,
    LOG_DEBUG,
    TOTAL_LOGS
  } log_type_t;

  // State and Covariance and Process Noise Matrices
  xVector x_;
  dxMatrix P_;
  dxMatrix Qx_;
  Matrix<double, 6, 6> Qu_;

  // Partial Update Gains
  dxVector lambda_;
  dxMatrix Lambda_;
  
  // Initial uncertainty on features
  Matrix3d P0_feat_;

  // Internal bookkeeping variables
  double prev_t_; // Used for calculating propagation time in IMU updates
  double start_t_; // Records the start time of the filter 
  
  // Feature Tracker Parameters
  typedef struct
  {
    Vector2f pix; // pixel position
    multiPatchVectorf PatchIntensity; // Stacked vector of intensities for all pyramid levels
    double quality; // Measure of combined quality for all pyramid level patches
    uint32_t frames; // Number of frames the feature has existed
    uint32_t global_id; // Feature id relative to beginning of time
  } feature_t;   // This defines a feature object - and all the associated information for a feature
  std::vector<feature_t> features_; // The current features in the filter
  
  uint32_t len_features_; // How many features are in the filter
  uint32_t next_feature_id_; // What the next global id is to assign to a feature
//  std::vector<int> current_feature_ids_; // 
  uint32_t feature_min_radius_; // the minimum distance between features before consolidating them
  uint32_t feature_detect_radius_; // the minimum distance between features at detection
  uint32_t patch_refresh_; // number of frames to refresh feature patch
  
  // Image Processing Variables
  Mat img_[PYRAMID_LEVELS]; // the pyramid of greyscale, distorted images
  Mat mask_; // The distortion mask
  Mat point_mask_; // Space to hold the point mask when looking for new features
  std::vector<cv::KeyPoint> keypoints_; // Container for newly detected keypoints
  std::vector<cv::KeyPoint> good_keypoints_; // Container for sorted keypoints
  std::vector<cv::Point2f> good_features_; // Container for good features converted from keypoints
  cv::Ptr<cv::Feature2D> detector_; // OpenCV feature detector object
  std::vector<std::vector<cv::Point> > contours_; // For drawing undistortable boundary
  std::vector<cv::Vec4i> hierarchy_; // For drawing undistortable boundary
  Mat camera_matrix_; // Camera intrinsic matrix - (K) is 3x3
  Mat dist_coeff_; // Camera distortion parameters - (D) is a column vector of 4, 5, or 8 elements

  // Custom key to sort keypoints by response in descending order
  struct keypoint_sort_key
  {
    inline bool operator() (const cv::KeyPoint &s1, const cv::KeyPoint &s2)
    {
      return (s1.response > s2.response);
    }
  };
  
  // Camera Intrinsics and Extrinsics
  Vector2d cam_center_;
  Matrix<double, 2, 3> cam_F_;
  Quat q_b_c_;
  Vector3d p_b_c_;
  
  // Used for delayed covariance update
  double prev_image_t_; // The time of the previous covariance update
  uVector imu_sum_; // The integrated IMU (input) since the last covariance update
  int imu_count_; // The number of imu messages since the last covariance update (used for finding the average input over the interval)
  
  // Matrix Workspace - dynamically allocating memory for Eigen is super slow.  So pre-allocate as much as possible,
  // and reuse the workspace
  dxMatrix A_;
  dxuMatrix G_;
  dxVector dx_;
  const dxMatrix I_big_ = dxMatrix::Identity();
  const dxMatrix Ones_big_ = dxMatrix::Constant(1.0);
  const dxVector dx_ones_ = dxVector::Constant(1.0);
  multiPatchVectorf Ip_, Im_;
  ColPivHouseholderQR<multiPatchJacMatrix> qrsolver_;
  SelfAdjointEigenSolver<Matrix2f> eigensolver_;
  xVector xp_;
  Matrix<double, MAX_DX, 3>  K_;
  zVector zhat_;
  hMatrix H_;
  std::vector<Vector2f> pix_, pix_copy_;
  std::vector<xVector> xs_;
  Vector2d vec2d_;

  // EKF Configuration Parameters
  bool use_drag_term_;
  bool keyframe_reset_;
  bool partial_update_;
  double min_depth_;
  
  // Keyframing and Global Pose Reconstruction
  std::vector<int> keyframe_features_; // which features are in the current frame, which were also in frame at the last keyframe
  double keyframe_overlap_threshold_; // when to declare a new keyframe
  std::deque<edge_SE2_t> edges_; // The list of transforms between keyframes
  eVector current_node_global_pose_; // The current global pose calculated by concatenating the edges
  std::function<void(void)> keyframe_reset_callback_;

  // Logging
  typedef struct
  {
    std::vector<std::ofstream>* stream = nullptr;
    double update_times[TOTAL_MEAS];
    int update_count[TOTAL_MEAS];
    double prop_time;
    int prop_log_count;
    int count;
  } log_t;
  log_t log_ = {};

public:

  VIEKF();
  ~VIEKF();
  void init(Matrix<double, xZ,1> x0, Matrix<double, dxZ,1> &P0, Matrix<double, dxZ,1> &Qx,
            Matrix<double, dxZ,1> &lambda, uVector &Qu, Vector3d& P0_feat, Vector3d& Qx_feat,
            Vector3d& lambda_feat, Vector2d& cam_center, Vector2d& focal_len,
            Vector4d& q_b_c, Vector3d &p_b_c, double min_depth, std::string log_directory, bool use_drag_term, 
            bool partial_update, bool use_keyframe_reset, double keyframe_overlap, int feature_min_radius, int feature_detect_radius,
            int patch_refresh, Matrix<double, 5, 1> dist_coeff);
  void init_logger(std::string root_filename);

  inline double now() const
  {
    std::chrono::microseconds now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
    return (double)now.count()*1e-6;
  }

  // Errors
  bool NaNsInTheHouse() const;
  bool BlowingUp() const;
  bool NegativeDepth() const;
  
  // Helpers
  int global_to_local_feature_id(const int global_id) const;
  bool inImage(const Vector2f& pt) const;
  void cv2Patch(const Mat& img, const pixVector& eta, patchMat& patch) const;
  void proj(const Quat &qz, Vector2f& eta, Matrix2f& jac, bool calc_jac) const;
  void multiLvlPatch(const pixVector& eta, multiPatchVectorf& patch) const;
  void multiLvlPatchSideBySide(multiPatchVectorf &src, multiPatchMatrixf& dst) const;
  void extractLvlfromPatch(multiPatchVectorf & src, const uint32_t level, patchMat& dst) const;
  void multiLvlPatchToCv(multiPatchVectorf& src, Mat& dst) const;
  
  // Getters and Setters
  VectorXd get_depths() const;
  MatrixXd get_zetas() const;
  MatrixXd get_qzetas() const;
  VectorXd get_zeta(const int i) const;
  Vector2d get_feat(const int id) const;
  int get_global_id(const int local_id) const;
  const eVector &get_current_node_global_pose() const;
  const xVector& get_state() const;
  const MatrixXd get_covariance() const;
  double get_depth(const int id) const;
  inline int get_len_features() const { return len_features_; }

  void set_x0(const Matrix<double, xZ, 1>& _x0);
  void set_imu_bias(const Vector3d& b_g, const Vector3d& b_a);
  void set_drag_term(const bool use_drag_term) {use_drag_term_ = use_drag_term;}
  bool get_drag_term() {return use_drag_term_;}

  // Feature Tracking
  void set_image(const Mat& mat);
  bool init_feature(const Vector2d &z, const double depth);
  void set_image_mask(const Mat& img) { img.copyTo(mask_); }
  void image_update(const Mat& img, const Matrix2d &R, const double t);
  update_return_code_t iterated_feature_update(const int id, const Matrix2d &R);
  void sample_pixels(const Quat& qz, const Matrix2f &cov, std::vector<pixVector>& eta);
  void patch_error(const pixVector &etahat, const multiPatchVectorf &I0, multiPatchVectorf &e, multiPatchJacMatrix &J);
  void clear_feature_state(const int id);
  void manage_features();
  double calculate_quality(const pixVector &eta);
  void choose_keypoints(std::vector<cv::KeyPoint> &keypoints, std::vector<cv::KeyPoint> &good_keypoints, const int &image_width, const int &image_height, const int &num_new_points);
  void create_distortion_mask(const cv::Size &res);
//  void keep_only_features(const std::vector<int> features);

  // State Propagation
  void boxplus(const xVector &x, const dxVector &dx, xVector &out) const;
  void boxminus(const xVector& x1, const xVector &x2, dxVector& out) const;
  void step(const uVector& u, const double t);
  void propagate_state(const uVector& u, const double t);
  void propagate_cov();
  void dynamics(const xVector &x, const uVector& u, dxVector& xdot, dxMatrix& dfdx, dxuMatrix& dfdu);
  void dynamics(const xVector &x, const uVector& u, bool state = true, bool jac = true);

  // Measurement Updates
  update_return_code_t update(const VectorXd& z, const measurement_type_t& meas_type, const MatrixXd& R, bool active=false, const int id=-1, const double depth=NAN);
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
  void register_keyframe_reset_callback(std::function<void(void)> cb);
  
  // Logger
  void log_measurement(const measurement_type_t meas_type, const MatrixXd& z, const bool active, const int id, double time);
  void log_global_position(const eVector truth_global_transform);
  void log_depth(const int id, double zhat, bool active);

  // Inequality Constraint on Depth
  void fix_depth();
};

static std::vector<std::string> measurement_names = [] {
  std::vector<std::string> tmp;
  tmp.resize(VIEKF::TOTAL_MEAS);
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

static std::vector<measurement_function_ptr> measurement_functions = [] {
  std::vector<measurement_function_ptr> tmp;
  tmp.resize(VIEKF::TOTAL_MEAS);
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





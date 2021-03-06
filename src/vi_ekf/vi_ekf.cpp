#include "vi_ekf.h"
#include "multirotor_sim/utils.h"

namespace vi_ekf
{

VIEKF::VIEKF()
{
  init();
}

VIEKF::VIEKF(const string &param_file)
{
  init();
  load(param_file);
}

void VIEKF::init()
{
  x_.resize(LEN_STATE_HIST);
  P_.resize(LEN_STATE_HIST);
  t_.resize(LEN_STATE_HIST);
  for (int i = 0; i < LEN_STATE_HIST; i++)
  {
    x_[i].setConstant(0);
    P_[i].setConstant(0);
    t_[i] = NAN;
  }
  u_.clear();
  zbuf_.clear();
  i_ = 0;
  xp_.setZero();
  Qx_.setZero();
  len_features_ = 0;
  next_feature_id_ = 0;
  current_feature_ids_.clear();
  start_t_ = NAN; // indicate that we need to initialize the filter
  current_node_global_pose_ = Xformd::Identity();
  global_pose_cov_.setZero();
  keyframe_features_.clear();
  edges_.clear();
  keyframe_reset_callback_ = nullptr;
  K_.setZero();
  H_.setZero();


  measurement_functions = [] {
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

void VIEKF::init(Matrix<double, xZ,1> &x0, Matrix<double, dxZ,1> &P0, Matrix<double, dxZ,1> &Qx,
                 Matrix<double, dxZ,1> &lambda, uVector &Qu, Vector3d& P0_feat, Vector3d& Qx_feat,
                 Vector3d& lambda_feat, Vector2d& cam_center, Vector2d& focal_len,
                 Vector4d& q_b_c, Vector3d &p_b_c, double min_depth, bool use_drag_term,
                 bool use_partial_update, bool use_keyframe_reset, double keyframe_overlap_threshold)
{
  x_[i_].block<(int)xZ, 1>(0,0) = x0;
  P_[i_].block<(int)dxZ, (int)dxZ>(0,0) = P0.asDiagonal();
  Qx_.block<(int)dxZ, (int)dxZ>(0,0) = Qx.asDiagonal();
  Qu_ = Qu.asDiagonal();
  lambda_.block<(int)dxZ, 1>(0,0) = lambda;

  for (int i = 0; i < NUM_FEATURES; i++)
  {
    P_[i_].block<3,3>(dxZ+3*i, dxZ+3*i) = P0_feat.asDiagonal();
    Qx_.block<3,3>(dxZ+3*i, dxZ+3*i) = Qx_feat.asDiagonal();
    lambda_.block<3,1>(dxZ+3*i,0) = lambda_feat;
  }

  Lambda_ = dx_ones_ * lambda_.transpose() + lambda_*dx_ones_.transpose() - lambda_*lambda_.transpose();

  // set camera intrinsics
  cam_center_ = cam_center;
  cam_F_ << focal_len(0), 0, 0,
            0, focal_len(1), 0;

  // set cam-to-body
  p_b_c_ = p_b_c;
  q_b_c_ = Quatd(q_b_c);

  min_depth_ = min_depth;
  keyframe_overlap_threshold_ = keyframe_overlap_threshold;
  use_drag_term_ = use_drag_term;
  use_partial_update_ = use_partial_update;
  use_keyframe_reset_ = use_keyframe_reset;
}

void VIEKF::load(const string &param_file)
{
  // Temporaries
  string name;
  Matrix<double, xZ,1> x0;
  Matrix<double, dxZ,1> P0, Qx, lambda;
  Matrix3d Qx_feat;
  Vector3d lambda_feat;
  Vector2d focal_len;
  Vector4d q_b_c, q_b_u;

  // Load parameters from YAML file
  get_yaml_node("name", param_file, name);
  get_yaml_node("min_depth", param_file, min_depth_);
  get_yaml_node("keyframe_overlap_threshold", param_file, keyframe_overlap_threshold_);
  get_yaml_node("use_drag_term", param_file, use_drag_term_);
  get_yaml_node("use_partial_update", param_file, use_partial_update_);
  get_yaml_node("use_keyframe_reset", param_file, use_keyframe_reset_);
  get_yaml_eigen("x0", param_file, x0);
  get_yaml_eigen("P0", param_file, P0);
  get_yaml_eigen("Qx", param_file, Qx);
  get_yaml_diag("Qu", param_file, Qu_);
  get_yaml_eigen("lambda", param_file, lambda);
  get_yaml_diag("P0_feat", param_file, P0_feat_);
  get_yaml_diag("Qx_feat", param_file, Qx_feat);
  get_yaml_eigen("lambda_feat", param_file, lambda_feat);
  get_yaml_eigen("cam_center", param_file, cam_center_);
  get_yaml_eigen("focal_len", param_file, focal_len);
  get_yaml_eigen("q_b_c", param_file, q_b_c);
  get_yaml_eigen("p_b_c", param_file, p_b_c_);
  get_yaml_eigen("q_b_u", param_file, q_b_u);

  // Populate class variables
  x_[i_].block<(int)xZ, 1>(0,0) = x0;
  P_[i_].block<(int)dxZ, (int)dxZ>(0,0) = P0.asDiagonal();
  Qx_.block<(int)dxZ, (int)dxZ>(0,0) = Qx.asDiagonal();
  lambda_.block<(int)dxZ, 1>(0,0) = lambda;
  
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    P_[i_].block<3,3>(dxZ+3*i, dxZ+3*i) = P0_feat_;
    Qx_.block<3,3>(dxZ+3*i, dxZ+3*i) = Qx_feat;
    lambda_.block<3,1>(dxZ+3*i,0) = lambda_feat;
  }
  
  Lambda_ = dx_ones_ * lambda_.transpose() + lambda_*dx_ones_.transpose() - lambda_*lambda_.transpose();

  cam_F_ << focal_len(0), 0, 0,
            0, focal_len(1), 0;
  q_b_c_ = Quatd(q_b_c);
  q_b_u_ = Quatd(q_b_u);
  
  init_logger("/tmp/", name);
  log_state(0, x_[i_], P_[i_].diagonal(), Vector6d::Zero(), dx_);
}

void VIEKF::set_x0(const Matrix<double, xZ, 1>& _x0)
{
  x_[i_].topRows(xZ) = _x0;
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
  x_[i_].block<3,1>((int)xB_G,0) = b_g;
  x_[i_].block<3,1>((int)xB_A,0) = b_a;
}


const xVector& VIEKF::get_state() const
{
  return x_[i_];
}

const Xformd &VIEKF::get_current_node_global_pose() const
{
  return current_node_global_pose_;
}

const dxMatrix& VIEKF::get_covariance() const
{
  return P_[i_];
}

const dxVector VIEKF::get_covariance_diagonal() const
{
  dxVector ret = P_[i_].diagonal();
  return ret;
}



VectorXd VIEKF::get_depths() const
{
  VectorXd out(len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    out[i] = 1.0/x_[i_]((int)xZ + 4 + 5*i);
  }
  return out;
}

MatrixXd VIEKF::get_zetas() const
{
  MatrixXd out(3, len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    Vector4d qzeta = x_[i_].block<4,1>(xZ + 5*i,0);
    out.block<3,1>(0,i) = Quatd(qzeta).rota(e_z);
  }
  return out;
}

MatrixXd VIEKF::get_qzetas() const
{
  MatrixXd out(4, len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    out.block<4,1>(0,i) = x_[i_].block<4,1>(xZ + 5*i,0);
  }
  return out;
}

VectorXd VIEKF::get_zeta(const int i) const
{
  Vector4d qzeta_i = x_[i_].block<4,1>(xZ + 5*i,0);
  return Quatd(qzeta_i).rota(e_z);
}

double VIEKF::get_depth(const int id) const
{
  int i = global_to_local_feature_id(id);
  return 1.0/x_[i_]((int)xZ + 4 + 5*i);
}

Vector2d VIEKF::get_feat(const int id) const
{
  int i = global_to_local_feature_id(id);
  Quatd q_zeta(x_[i_].block<4,1>(xZ+i*5, 0));
  Vector3d zeta = q_zeta.rota(e_z);
  double ezT_zeta = e_z.transpose() * zeta;
  return cam_F_ * zeta / ezT_zeta + cam_center_;
}

void VIEKF::propagate_state(const uVector &u, const double t, bool save_input)
{
  // Rotate IMU measurement into body frame
  Vector6d ub;
  ub.segment<3>(uA) = q_b_u_.rota(u.segment<3>(uA));
  ub.segment<3>(uG) = q_b_u_.rota(u.segment<3>(uG));

  if (save_input)
  {
    u_.push_front(std::pair<double, uVector>{t, ub});
  }

  if (std::isnan(start_t_))
  {
    start_t_ = t;
    t_[i_] = t;
    return;
  }

  double dt = t - t_[i_];
  if (fabs(dt) < 1e-6)
    return;

  if (dt < 0)
  {
    cerr << "Trying to propagate backwards!  I won't let you" <<endl;
    return;
  }


  NAN_CHECK;

  // Calculate Dynamics and Jacobians
  dynamics(x_[i_], ub, true, true);

  NAN_CHECK;
  int ip = (i_ + 1) % LEN_STATE_HIST; // next state history index

  // Propagate State and Covariance
  boxplus(x_[i_], dx_*dt, x_[ip]);
  G_ = (I_big_ + A_*dt/2.0 + A_*A_*dt*dt/6.0)*G_*dt;
  A_ = I_big_ + A_*dt + A_*A_*dt*dt/2.0;
  P_[ip] = A_ * P_[i_]* A_.transpose() + G_ * Qu_ * G_.transpose() + Qx_;
  t_[ip] = t;
  i_ = ip;

  NAN_CHECK;
  
  // Correct any impossible depth states
  fix_depth();
  
  NAN_CHECK;
  NEGATIVE_DEPTH;
  
  if (save_input)
    log_state(t, x_[i_], P_[i_].diagonal(), ub, dx_);
}


}









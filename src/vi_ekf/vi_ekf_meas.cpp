#include "vi_ekf.h"

namespace vi_ekf
{

void VIEKF::handle_measurements(std::vector<int> &gated_feature_ids)
{
  gated_feature_ids.clear();
  double t_now = t_[i_];


  // Find the oldest measurement that hasn't been handled yet
  auto z_it = zbuf_.end() -1;
  while (z_it->handled == true && z_it != zbuf_.begin())
  {
    z_it--;
  }
  // There were no messages to be handled - exit
  if (z_it == zbuf_.begin() && z_it->handled)
    return;


  // Select the input corresponding to this time
  auto u_it = u_.begin();
  while (u_it != u_.end())
  {
    if (u_it->t <= z_it->t)
      break;
    u_it++;
  }

  if (u_it == u_.end() && u_it->t > z_it->t)
  {
    cerr << "not enough history in input buffer to handle measurement" << endl;
    return;
  }


  // Rewind the state to just before the time indicated by the measurement
  int i = LEN_STATE_HIST;
  while (i > 0)
  {
    if (t_[(i_ + i) % LEN_STATE_HIST] <= z_it->t)
    {
      // rewind state to here (by just setting the position in the circular buffer)
      i_ = (i_ + i) % LEN_STATE_HIST;
      break;
    }
    i--;
  }
  if (i == 0)
  {
    cerr << "not enough history in state buffer to handle measurement" << endl;
    return;
  }


  // Process all inputs and measurements to catch back up to where we were before (t_now)
  while (t_[i_] < t_now)
  {
    double t_next = (u_it+1)->t; // Time of next input

    // While the current measurment occurred between the current time step and the next
    while (z_it->t < t_next && z_it->t > u_it->t)
    {
      // Propagate to the point of the measurement
      propagate_state(u_it->u, z_it->t);
      // Perform the measurement
      update(*z_it);
      if (z_it != zbuf_.begin())
        z_it--;
      else
        break;

    }
    // Propagate to the time of the next input
    propagate_state(u_it->u, u_it->t);
  }

  // If the measurement is exactly at this current time step, then just apply it.
  while (z_it->t == t_[i_])
  {
    update(*z_it);
    if (z_it != zbuf_.begin())
      z_it--;
    else
      break;
  }

  // Clear any old measurements in the queue
  while (zbuf_.size() > LEN_MEAS_HIST)
    zbuf_.pop_back();

}


void VIEKF::add_measurement(const double t, const VectorXd& z, const measurement_type_t& meas_type,
                                               const MatrixXd& R, bool active, const int id, const double depth)
{
  // Figure out the measurement that goes just before this one
  auto z_it = zbuf_.begin();
  while (z_it != zbuf_.end())
  {
    if (z_it->t < t)
      break;
    z_it ++;
  }

  // add the measurement to the measurement queue just after the one we just found
  measurement_t meas;
  meas.t = t;
  meas.type = meas_type;
  meas.zdim = z.rows();
  meas.rdim = R.cols();
  meas.R.block(0, 0, meas.rdim, meas.rdim) = R;
  meas.z.segment(0, meas.zdim) = z;
  meas.active = active;
  meas.id = id;
  meas.depth = depth;
  meas.handled = false;
  if (z_it == zbuf_.begin())
  {
    zbuf_.push_front(meas);
    z_it = zbuf_.begin();
  }
  else
    zbuf_.insert(z_it, meas);

}

VIEKF::meas_result_t VIEKF::update(measurement_t& meas)
{  
  if ((meas.z.topRows(meas.zdim).array() != meas.z.topRows(meas.zdim).array()).any())
    return MEAS_NAN;
  
  // If this is a new feature, initialize it
  if (meas.type == FEAT && meas.id >= 0)
  {
    if (std::find(current_feature_ids_.begin(), current_feature_ids_.end(), meas.id) == current_feature_ids_.end())
    {
      init_feature(meas.z.topRows(2), meas.id, meas.depth);
      return MEAS_NEW_FEATURE; // Don't do a measurement update this time
    }
  }
  
  NAN_CHECK;
  
  zhat_.setZero();
  H_.setZero();
  K_.setZero();
  
  (this->*(measurement_functions[meas.type]))(x_[i_], zhat_, H_, meas.id);
  
  NAN_CHECK;
  
  zVector residual;
  if (meas.type == QZETA)
  {
    residual.topRows(2) = q_feat_boxminus(Quat(meas.z), Quat(zhat_));
  }
  else if (meas.type == ATT)
  {
    residual.topRows(3) = Quat(meas.z) - Quat(zhat_);
  }
  else
  {
    residual.topRows(meas.zdim) = meas.z.topRows(meas.zdim) - zhat_.topRows(meas.zdim);
  }

  auto K = K_.leftCols(meas.rdim);
  auto H = H_.topRows(meas.rdim);
  auto res = residual.topRows(meas.rdim);
  auto R = meas.R.block(0, 0, meas.rdim, meas.rdim);


  //  Perform Covariance Gating Check on Residual
  if (meas.active)
  {
    auto innov =  (H * P_[i_] * H.transpose() + R).inverse();

    double mahal = res.transpose() * innov * res;
    if (mahal > 9.0)
    {
//      std::cout << "gating " << measurement_names[meas_type] << " measurement: " << mahal << std::endl;
      return MEAS_GATED;
    }

    K = P_[i_] * H.transpose() * innov;
    NAN_CHECK;
    
    //    CHECK_MAT_FOR_NANS(H_);
    //    CHECK_MAT_FOR_NANS(K_);
    
    if (NO_NANS(K_) && NO_NANS(H_))
    {
      if (partial_update_)
      {
        // Apply Fixed Gain Partial update per
        // "Partial-Update Schmidt-Kalman Filter" by Brink
        // Modified to operate inline and on the manifold 
        boxplus(x_[i_], lambda_.asDiagonal() * K * residual.topRows(meas.rdim), xp_);
        x_[i_] = xp_;
        A_ = (I_big_ - K * H);
        P_[i_] += (Lambda_).cwiseProduct(A_*P_[i_]*A_.transpose() + K * R * K.transpose() - P_[i_]);

      }
      else
      {
        boxplus(x_[i_], K * residual.topRows(meas.rdim), xp_);
        x_[i_] = xp_;
        A_ = (I_big_ - K * H);
        P_[i_] = A_*P_[i_]*A_.transpose() + K * R * K.transpose();
      }
    }
    NAN_CHECK;
  }
  
  fix_depth();
  
  NAN_CHECK;
  NEGATIVE_DEPTH;
  
  log_measurement(meas.type, prev_t_ - start_t_, meas.zdim, meas.z, zhat_, meas.active, meas.id);
  return MEAS_SUCCESS;
}


void VIEKF::h_acc(const xVector& x, zVector& h, hMatrix& H, const int id) const
{
  (void)id;
  H.setZero();
  
  Vector3d b_a = x.block<3,1>((int)xB_A,0);
  
  if (use_drag_term_)
  {
    Vector3d vel = x.block<3,1>((int)xVEL,0);
    double mu = x(xMU,0);
    
    h.topRows(2) = I_2x3 * (-mu * vel + b_a);
    
    H.block<2, 3>(0, (int)dxVEL) = -mu * I_2x3;
    H.block<2, 3>(0, (int)dxB_A) = I_2x3;
    H.block<2, 1>(0, (int)dxMU) = -I_2x3*vel;
  }
  else
  {
    Vector3d gravity_B = Quat(x.block<4,1>((int)xATT, 0)).rotp(gravity); // R_I^b * vel
    h.topRows(3) = b_a - gravity_B;
    H.block<3,3>(0, (int)dxATT) = skew(-1.0 * gravity_B);
    H.block<3,3>(0, (int)dxB_A) = I_3x3;
  }  
}

void VIEKF::h_alt(const xVector& x, zVector& h, hMatrix& H, const int id) const
{
  (void)id;
  h.row(0) = -x.block<1,1>(xPOS+2, 0);
  
  H.setZero();
  H(0, dxPOS+2) = -1.0;
}

void VIEKF::h_att(const xVector& x, zVector& h, hMatrix& H, const int id) const
{
  (void)id;
  h = x.block<4,1>((int)xATT, 0);
  
  H.setZero();
  H.block<3,3>(0, dxATT) = I_3x3;
}

void VIEKF::h_pos(const xVector& x, zVector& h, hMatrix& H, const int id) const
{
  (void)id;
  h.topRows(3) = x.block<3,1>((int)xPOS,0);
  
  H.setZero();
  H.block<3,3>(0, (int)xPOS) = I_3x3;
}

void VIEKF::h_vel(const xVector& x, zVector& h, hMatrix& H, const int id) const
{
  (void)id;
  h.topRows(3) = x.block<3,1>((int)xVEL, 0);
  
  H.setZero();
  H.block<3,3>(0, (int)dxVEL) = I_3x3;
}

void VIEKF::h_qzeta(const xVector& x, zVector& h, hMatrix &H, const int id) const
{
  int i = global_to_local_feature_id(id);
  
  h = x.block<4,1>(xZ+i*5, 0);
  
  H.setZero();
  H.block<2,2>(0, dxZ + i*3) = I_2x2;
}

void VIEKF::h_feat(const xVector& x, zVector& h, hMatrix& H, const int id) const
{
  int i = global_to_local_feature_id(id);
  Quat q_zeta(x.block<4,1>(xZ+i*5, 0));
  Vector3d zeta = q_zeta.rota(e_z);
  Matrix3d sk_zeta = skew(zeta);
  double ezT_zeta = e_z.transpose() * zeta;
  MatrixXd T_z = T_zeta(q_zeta);
  
  h.topRows(2) = cam_F_ * zeta / ezT_zeta + cam_center_;
  
  H.setZero();
  H.block<2,2>(0,dxZ+i*3) = (1.0 / ezT_zeta) * cam_F_ * (I_3x3 - (zeta * e_z.transpose()) / ezT_zeta) * sk_zeta * T_z;
}

void VIEKF::h_depth(const xVector& x, zVector& h, hMatrix& H, const int id) const
{
  int i = global_to_local_feature_id(id);
  double rho = x(xZ+i*5+4,0 );
  
  h(0,0) = 1.0/rho;
  H.setZero();
  H(0, dxZ+3*i+2) = -1.0/(rho*rho);
}

void VIEKF::h_inv_depth(const xVector& x, zVector& h, hMatrix& H, const int id) const
{
  int i = global_to_local_feature_id(id);
  h(0,0) = x(xZ+i*5+4,0);
  
  H.setZero();
  H(0, dxZ+3*i+2) = 1.0;
}

void VIEKF::h_pixel_vel(const xVector& x, zVector& h, hMatrix& H, const int id) const
{
  (void)x;
  (void)h;
  (void)H;
  (void)id;
  ///TODO:
}

}

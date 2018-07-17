#include "vi_ekf.h"

namespace vi_ekf
{

bool VIEKF::update(const VectorXd& z, const measurement_type_t& meas_type,
                   const MatrixXd& R, bool active, const int id, const double depth)
{  
  if ((z.array() != z.array()).any())
    return true;
  
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
  K_.setZero();
  
  (this->*(measurement_functions[meas_type]))(x_, zhat_, H_, id);
  
  NAN_CHECK;
  
  zVector residual;
  if (meas_type == QZETA)
  {
    residual.topRows(2) = q_feat_boxminus(Quat(z), Quat(zhat_));
    z_dim = 2;
  }
  else if (meas_type == ATT)
  {
    residual.topRows(3) = Quat(z) - Quat(zhat_);
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
    
    //    CHECK_MAT_FOR_NANS(H_);
    //    CHECK_MAT_FOR_NANS(K_);
    
    if (NO_NANS(K_) && NO_NANS(H_))
    {
      if (partial_update_)
      {
        // Apply Fixed Gain Partial update per
        // "Partial-Update Schmidt-Kalman Filter" by Brink
        // Modified to operate inline and on the manifold 
        boxplus(x_, lambda_.asDiagonal() * K_.leftCols(z_dim) * residual.topRows(z_dim), xp_);
        x_ = xp_;
        P_ -= (Lambda_).cwiseProduct(K_.leftCols(z_dim) * H_.topRows(z_dim)*P_);
      }
      else
      {
        boxplus(x_, K_.leftCols(z_dim) * residual.topRows(z_dim), xp_);  
        x_ = xp_;
        P_ = (I_big_ - K_.leftCols(z_dim) * H_.topRows(z_dim))*P_;
      }
    }
    NAN_CHECK;
  }
  
  fix_depth();
  
  NAN_CHECK;
  NEGATIVE_DEPTH;
  
  log_measurement(meas_type, prev_t_ - start_t_, z.rows(), z, zhat_, active, id);
  return false;
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
  H.block<2,2>(0, dxZ + i*3) = -cam_F_ * ((sk_zeta * T_z)/ezT_zeta - (zeta * e_z.transpose() * sk_zeta * T_z)/(ezT_zeta*ezT_zeta));
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

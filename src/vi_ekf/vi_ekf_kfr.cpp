#include "vi_ekf.h"

namespace vi_ekf
{

void VIEKF::keyframe_reset(const xVector &xm, xVector &xp, dxMatrix &N)
{
  x_ = xm;
  keyframe_reset();
  xp = x_;
  N = A_;    
}

void VIEKF::propagate_global_covariance(Matrix6d &P_edge, const edge_t &edge)
{
  // Calculate Cholesky Decomposition for global covariance
  LLT<Matrix6d> chol(P_edge);
  Matrix6d L = chol.matrixL();

  // Calculate Chol Decomp for edge covariance
  LLT<Matrix6d> chol2(edge.cov);
  Matrix6d L2 = chol2.matrixL();

  // Sample from global covariance, propagate points while adding noise from edge and re-create the new covariance
  // This is stupid slow, to improve this, we should implement Barfoot's "Associating Uncertainty" Paper
  P_edge.setZero();
  for (int i = 0; i < 1000; i++)
  {
    Vector6d z, z2;
    setNormalRandom(z, normal_, generator_);
    setNormalRandom(z2, normal_, generator_);

    Xform x_i = (Xform::exp(L * z) * edge.transform) + (L2 * z2); // sample in current covariance banana
    Vector6d dx = x_i - edge.transform; // Error of this sample, on manifold
    P_edge += dx * dx.transpose(); // Build up the outer-product sum
  }
  P_edge /= 999.0; // Finally calculate the covariance matrix
}


void VIEKF::keyframe_reset()
{
  // Save off current position into the new edge
  edge_t edge;
  edge.cov = 1e-8 * Matrix6d::Identity();
  edge.transform.t().segment<2>(0) = x_.segment<2>(xPOS,0);
  edge.transform.t()(2,0) = 0.0; // no altitude information in the edge
  edge.cov.block<2,2>((int)xPOS, (int)xPOS) = P_.block<2,2>((int)xPOS, (int)xPOS);
  
  // reset global xy position
  x_(xPOS, 0) = 0;
  x_(xPOS+1, 0) = 0;
  
  Quat qm(x_.block<4,1>((int)xATT, 0)); 
  
  //// James' way to reset z-axis rotation
  // precalculate some things
  Vector3d v = qm.rota(khat);
  Vector3d s = khat.cross(v); s /= s.norm(); // Axis of rotation (without rotation about khat)
  double theta = acos(khat.transpose() * v); // Angle of rotation
  Matrix3d sk_tv = skew(theta*s);
  Matrix3d sk_u = skew(khat);
  Matrix3d qmR = qm.R();
  Quat qp = Quat::exp(theta * s); // q+
  edge.transform.q() = (qm * qp.inverse()).elements();

  // Save off quaternion and covariance /// TODO - do this right
  edge.cov(2,2) = P_(xATT+2, xATT+2);
  
  // reset rotation about z
  x_.block<4,1>((int)(xATT), 0) = qp.elements();
  
  // Adjust covariance  (use A for N, because it is the right size and there is no need to allocate another one)
  A_ = I_big_;
  A_((int)xPOS, (int)xPOS) = 0;
  A_((int)xPOS+1, (int)xPOS+1) = 0;
  A_.block<3,3>((int)dxATT, (int)dxATT) = (I_3x3 + ((1.-cos(theta))*sk_tv)/(theta*theta) + ((theta - sin(theta))*sk_tv*sk_tv)/(theta*theta*theta)).transpose()
          * (-s * (khat.transpose() * qmR.transpose() * sk_u) - theta * sk_u * (qmR.transpose() * sk_u));

  /// Jerel's way to reset z-axis rotation

  
  
  /// Dan's way to reset z-axis rotation
//  double yaw = qm.yaw();
//  double roll = qm.roll();
//  double pitch = qm.pitch();
  
//  // Save off quaternion and covariance
//  edge.transform.block<4,1>((int)eATT, 0) = Quat::from_euler(0, 0, yaw).elements();
//  edge.cov(2,2) = P_(xATT+2, xATT+2); /// TODO - do this right
  
//  x_.block<4,1>((int)(xATT), 0) << Quat::from_euler(roll, pitch, 0.0).elements();
  
//  // Adjust covariance  (use A for N, because it is the right size and there is no need to allocate another one)
//  // RMEKF paper after Eq. 81
  
//  double cp = std::cos(roll);
//  double sp = std::sin(roll);
//  double tt = std::tan(pitch);
//  A_ = I_big_;
//  A_((int)xPOS, (int)xPOS) = 0;
//  A_((int)xPOS+1, (int)xPOS+1) = 0;
//  A_.block<3,3>((int)dxATT, (int)dxATT) << 1, sp*tt, cp*tt,
//      0, cp*cp, -cp*sp,
//      0, -cp*sp, sp*sp;
  
  NAN_CHECK;
  
  
  P_ = A_ * P_ * A_.transpose();
  
  // Build Global Node Frame Position
  current_node_global_pose_ = current_node_global_pose_ * edge.transform;

  // Propagate Global Covariance
  propagate_global_covariance(global_pose_cov_, edge);
  
  NAN_CHECK;
  
  // call callback
  if (keyframe_reset_callback_ != nullptr)
    keyframe_reset_callback_();
}

}

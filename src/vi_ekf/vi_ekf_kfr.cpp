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

Xform VIEKF::get_global_pose() const
{
  // Log Global Position Estimate
  Xform global_pose;
  Xform rel_pose(x_.block<3,1>((int)xPOS, 0), Quat(x_.block<4,1>((int)xATT, 0)));
  global_pose = current_node_global_pose_ * rel_pose;
  return global_pose;
}

Matrix6d VIEKF::get_global_cov() const
{
  Matrix6d cov;
  edge_t rel_pose;
  rel_pose.transform.t() = x_.block<3,1>((int)xPOS, 0);
  rel_pose.transform.q() = Quat(x_.block<4,1>((int)xATT, 0));
  rel_pose.cov.block<3,3>(0, 0) = P_.block<3,3>(xPOS, xPOS);
  rel_pose.cov.block<3,3>(3, 0) = P_.block<3,3>(xATT, xPOS);
  rel_pose.cov.block<3,3>(0, 3) = P_.block<3,3>(xPOS, xATT);
  rel_pose.cov.block<3,3>(3, 3) = P_.block<3,3>(xATT, xATT);
  propagate_global_covariance(global_pose_cov_, rel_pose, cov);
  return cov;
}

Matrix3d brackets(const Matrix3d& A)
{
  return -A.trace()*I_3x3 + A;
}

Matrix3d brackets(const Matrix3d& A, const Matrix3d& B)
{
  return brackets(A)*brackets(B) + brackets(A*B);
}

void VIEKF::propagate_global_covariance(const Matrix6d &P_prev, const edge_t &edge, Matrix6d &P_new) const
{
  Matrix6d Adj = edge.transform.Adj();
  P_new = P_prev + Adj * edge.cov * Adj.transpose();

  /// TODO - look at Barfoot's way of propagating uncertainty
}


void VIEKF::keyframe_reset()
{
  // Save off current position into the new edge
  edge_t edge;
  edge.cov = 1e-8 * Matrix6d::Identity();
  edge.transform.t().segment<2>(0) = x_.segment<2>(xPOS);
  edge.transform.t()(2,0) = 0.0; // no altitude information in the edge
  edge.cov.block<2,2>((int)xPOS, (int)xPOS) = P_.block<2,2>((int)xPOS, (int)xPOS);
  
  // reset global xy position
  x_(xPOS, 0) = 0;
  x_(xPOS+1, 0) = 0;
  
  Quat qm(x_.block<4,1>((int)xATT, 0)); 
  
  //// James' way to reset z-axis rotation
//  // precalculate some things
//  Vector3d v = qm.rota(khat);
//  Vector3d s = khat.cross(v); s /= s.norm(); // Axis of rotation (without rotation about khat)
//  double theta = acos(khat.transpose() * v); // Angle of rotation
//  Matrix3d sk_tv = skew(theta*s);
//  Matrix3d sk_u = skew(khat);
//  Matrix3d qmR = qm.R();
//  Quat qp = Quat::exp(theta * s); // q+
//  edge.transform.q() = (qm * qp.inverse()).elements();

//  // Save off quaternion and covariance /// TODO - do this right
//  edge.cov(2,2) = P_(xATT+2, xATT+2);
  
//  // reset rotation about z
//  x_.block<4,1>((int)(xATT), 0) = qp.elements();
  
//  // Adjust covariance  (use A for N, because it is the right size and there is no need to allocate another one)
//  A_ = I_big_;
//  A_((int)xPOS, (int)xPOS) = 0;
//  A_((int)xPOS+1, (int)xPOS+1) = 0;
//  A_.block<3,3>((int)dxATT, (int)dxATT) = (I_3x3 + ((1.-cos(theta))*sk_tv)/(theta*theta) + ((theta - sin(theta))*sk_tv*sk_tv)/(theta*theta*theta)).transpose()
//          * (-s * (khat.transpose() * qmR.transpose() * sk_u) - theta * sk_u * (qmR.transpose() * sk_u));

  /// Jerel's way to reset z-axis rotation

  
  
  /// Dan's way to reset z-axis rotation
  double yaw = qm.yaw();
  double roll = qm.roll();
  double pitch = qm.pitch();
  
  // Save off quaternion and covariance
  edge.transform.q_ = Quat::from_euler(0, 0, yaw);
  edge.cov(2,2) = P_(xATT+2, xATT+2); /// TODO - do this right
  
  x_.block<4,1>((int)(xATT), 0) << Quat::from_euler(roll, pitch, 0.0).elements();
  
  // Adjust covariance  (use A for N, because it is the right size and there is no need to allocate another one)
  // RMEKF paper after Eq. 81
  
  double cp = std::cos(roll);
  double sp = std::sin(roll);
  double tt = std::tan(pitch);
  A_ = I_big_;
  A_((int)xPOS, (int)xPOS) = 0;
  A_((int)xPOS+1, (int)xPOS+1) = 0;
  A_.block<3,3>((int)dxATT, (int)dxATT) << 1, sp*tt, cp*tt,
      0, cp*cp, -cp*sp,
      0, -cp*sp, sp*sp;
  
  NAN_CHECK;
  
  
  P_ = A_ * P_ * A_.transpose();
  
  // Propagate Global Covariance
  propagate_global_covariance(global_pose_cov_, edge, global_pose_cov_);

  current_node_global_pose_ = current_node_global_pose_ * edge.transform;

  // Calculate Global Node Frame Position
  
  NAN_CHECK;
  
  // call callback
  if (keyframe_reset_callback_ != nullptr)
    keyframe_reset_callback_();
}

}

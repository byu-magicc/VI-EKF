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


void VIEKF::keyframe_reset()
{
  // Save off current position into the new edge
  edge_SE2_t edge;
  edge.transform.block<2,1>(0,0) = x_.block<2,1>(xPOS,0);
  edge.transform(2,0) = 0.0; // no altitude information in the edge
  edge.cov.block<2,2>((int)xPOS, (int)xPOS) = P_.block<2,2>((int)xPOS, (int)xPOS);
  
  // reset global xy position
  x_(xPOS, 0) = 0;
  x_(xPOS+1, 0) = 0;
  
  Quat qm(x_.block<4,1>((int)xATT, 0)); 
  
  
  ////  Cool way to reset z-axis rotation
  //  // Save off quaternion and covariance /// TODO - do this right
  //  edge.transform.block<4,1>((int)eATT,0) = (qm * qp.inverse()).elements();
  //  edge.cov(2,2) = P_(xATT+2, xATT+2);
  
  //  // precalculate some things
  //  Vector3d v = qm.rota(khat);
  //  Vector3d s = khat.cross(v); // Axis of rotation (without rotation about khat)
  //  s /= s.norm();
  //  double theta = khat.transpose() * v; // Angle of rotation
  //  Matrix3d sk_tv = skew(theta*s);
  //  Matrix3d sk_u = skew(khat);
  //  Matrix3d qmR = qm.R();
  //  Quat qp = Quat::exp(std::acos(theta) * s); // q+
  
  //  // reset rotation about z
  //  x_.block<4,1>((int)(xATT), 0) = qp.elements();
  
  //  // Adjust covariance  (use A for N, because it is the right size and there is no need to allocate another one)
  //  A_ = I_big_;
  //  A_((int)xPOS, (int)xPOS) = 0;
  //  A_((int)xPOS+1, (int)xPOS+1) = 0;
  //  A_.block<3,3>((int)dxATT, (int)dxATT) = (I_3x3 + ((1.-cos(theta))*sk_tv)/(theta*theta) + ((theta - sin(theta))*sk_tv*sk_tv)/(theta*theta*theta)).transpose()
  //      * (-s * (khat.transpose() * qmR.transpose() * sk_u) - theta * sk_u * (qmR.transpose() * sk_u));
  
  
  /// Old way to reset z-axis rotation
  double yaw = qm.yaw();
  double roll = qm.roll();
  double pitch = qm.pitch();
  
  // Save off quaternion and covariance
  edge.transform.block<4,1>((int)eATT, 0) = Quat::from_euler(0, 0, yaw).elements();
  edge.cov(2,2) = P_(xATT+2, xATT+2); /// TODO - do this right
  
  x_.block<4,1>((int)(xATT), 0) << Quat::from_euler(roll, pitch, 0.0).elements();
  x_.block<4,1>((int)(xATT), 0) /= x_.block<4,1>((int)(xATT), 0).norm();
  
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
  
  // Build Global Node Frame Position
  concatenate_edges(current_node_global_pose_, edge.transform, current_node_global_pose_);
  
  NAN_CHECK;
  
  // call callback
  if (keyframe_reset_callback_ != nullptr)
    keyframe_reset_callback_();
}

}

#include "quat.h"

#include <Eigen/Core>

static const Eigen::Matrix<double, 2, 3> I_2x3 = [] {
  Eigen::Matrix<double, 2, 3> tmp;
  tmp << 1.0, 0, 0,
         0, 1.0, 0;
  return tmp;
}();

static const Eigen::Vector3d e_x = [] {
  Eigen::Vector3d tmp;
  tmp << 1.0, 0, 0;
  return tmp;
}();

static const Eigen::Vector3d e_y = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 1.0, 0;
  return tmp;
}();

static const Eigen::Vector3d e_z = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 0, 1.0;
  return tmp;
}();


Eigen::Matrix3d skew(const Eigen::Vector3d v)
{
  Eigen::Matrix3d cum_sum;
  cum_sum << 0.0, -v(2), v(1),
      v(2), 0.0, -v(0),
      -v(1), v(0), 0.0;
  return cum_sum;
}

Eigen::Matrix<double, 3, 2> T_zeta(quat::Quaternion q)
{
  return q.rot(I_2x3.transpose());
}

Eigen::Vector2d q_feat_boxminus(quat::Quaternion q0, quat::Quaternion q1)
{
  Eigen::Vector3d zeta0 = q0.rot(e_z);
  Eigen::Vector3d zeta1 = q1.rot(e_z);

  Eigen::Vector2d dq;
  if ((zeta0 - zeta1).norm() > 1e-16)
  {
    Eigen::Vector3d v = zeta1.cross(zeta0);
    v /= v.norm();
    double theta = std::acos(zeta1.dot(zeta0));
    dq = theta * T_zeta(q1).transpose() * v;
  }
  else
  {
    dq.setZero();
  }
  return dq;
}

quat::Quaternion q_feat_boxplus(quat::Quaternion q, Eigen::Vector2d dq)
{
  quat::Quaternion delta_q = quat::Quaternion::exp(T_zeta(q) * dq);
  quat::Quaternion qplus = delta_q * q;
  return qplus;
}

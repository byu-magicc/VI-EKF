#pragma once

#include "math.h"
#include "quat.h"

#include <Eigen/Core>

static const Eigen::Matrix<double, 2, 3> I_2x3 = [] {
  Eigen::Matrix<double, 2, 3> tmp;
  tmp << 1.0, 0, 0,
         0, 1.0, 0;
  return tmp;
}();

static const Eigen::Matrix3d I_3x3 = [] {
  Eigen::Matrix3d tmp = Eigen::Matrix3d::Identity();
  return tmp;
}();

static const Eigen::Matrix2d I_2x2 = [] {
  Eigen::Matrix2d tmp = Eigen::Matrix2d::Identity();
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

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove);
void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove);

inline Eigen::Matrix3d skew(const Eigen::Vector3d v)
{
  Eigen::Matrix3d mat;
  mat << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
         -v(1), v(0), 0.0;
  return mat;
}

inline Eigen::Matrix<double, 3, 2> T_zeta(quat::Quat q)
{
  return q.doublerot(I_2x3.transpose());
}

inline Eigen::Vector2d q_feat_boxminus(quat::Quat q0, quat::Quat q1)
{
  Eigen::Vector3d zeta0 = q0.rota(e_z);
  Eigen::Vector3d zeta1 = q1.rota(e_z);

  Eigen::Vector2d dq;
  if ((zeta0 - zeta1).norm() > 1e-8)
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

inline quat::Quat q_feat_boxplus(quat::Quat q, Eigen::Vector2d dq)
{
  return quat::Quat::exp(T_zeta(q) * dq) * q;
}


void concatenate_SE2(Eigen::Vector3d& T1, Eigen::Vector3d& T2, Eigen::Vector3d& Tout);
void concatenate_edges(const Eigen::Matrix<double,7,1>& T1, const Eigen::Matrix<double,7,1>& T2, Eigen::Matrix<double,7,1>& Tout);
const Eigen::Matrix<double,7,1> invert_edge(const Eigen::Matrix<double,7,1>& T1);
void invert_SE2(Eigen::Vector3d& T, Eigen::Vector3d& Tout);

template <typename T>
int sign(T in)
{
  return (in >= 0) - (in < 0);
}

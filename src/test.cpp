#include "quat.h"
#include "gtest/gtest.h"
#include <iostream>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include "math_helper.h"
#include "vi_ekf.h"

#define EXPECT_QUATERNION_EQUALS(q1, q2) \
  EXPECT_NEAR((q1).w(), (q1).w(), 1e-8); \
  EXPECT_NEAR((q1).x(), (q1).x(), 1e-8); \
  EXPECT_NEAR((q1).y(), (q1).y(), 1e-8); \
  EXPECT_NEAR((q1).z(), (q1).z(), 1e-8)

#define EXPECT_VECTOR3_EQUALS(v1, v2) \
  EXPECT_NEAR((v1)(0,0), (v1)(0,0), 1e-8); \
  EXPECT_NEAR((v1)(1,0), (v1)(1,0), 1e-8); \
  EXPECT_NEAR((v1)(2,0), (v1)(2,0), 1e-8)

#define EXPECT_VECTOR2_EQUALS(v1, v2) \
  EXPECT_NEAR((v1)(0,0), (v1)(0,0), 1e-8); \
  EXPECT_NEAR((v1)(1,0), (v1)(1,0), 1e-8)

#define EXPECT_MATRIX_EQUAL(m1, m2, tol) {\
  for (int row = 0; row < m1.rows(); row++ ) \
{ \
  for (int col = 0; col < m1.cols(); col++) \
{ \
  EXPECT_NEAR((m1)(row, col), (m2)(row, col), tol); \
  } \
  } \
  }


using namespace quat;

TEST(Quaternion, rotation_direction)
{
  // Compare against a known active and passive rotation
  Eigen::Vector3d v, beta, v_active_rotated, v_passive_rotated;
  v << 0, 0, 1;
  v_active_rotated << 0, std::pow(-0.5,0.5), std::pow(0.5,0.5);
  beta << 1, 0, 0;
  Quaternion q_x_45 = Quaternion::from_axis_angle(beta, 45*M_PI/180.0);

  EXPECT_VECTOR3_EQUALS(q_x_45.rot(v), v_active_rotated);

  v_passive_rotated << 0, std::pow(0.5, 0.5), std::pow(0.5, 0.5);
  EXPECT_VECTOR3_EQUALS(q_x_45.rot(v), v_passive_rotated);
}

TEST(Quaternion, rot_invrot_R)
{
  Eigen::Vector3d v;
  Quaternion q1 = Quaternion::Random();
  for (int i = 0; i < 100; i++)
  {
    v.setRandom();
    q1 = Quaternion::Random();

    // Check that rotations are inverses of each other
    EXPECT_VECTOR3_EQUALS(q1.rot(v), q1.R.T * v);
    EXPECT_VECTOR3_EQUALS(q1.invrot(v), q1.R * v);
  }
}

TEST(Quaternion, from_two_unit_vectors)
{
  Eigen::Vector3d v1, v2;
  for (int i = 0; i < 100; i++)
  {
    v1.setRandom();
    v2.setRandom();
    v1 /= v1.norm();
    v2 /= v2.norm();

    EXPECT_VECTOR3_EQUALS(Quaternion::from_two_unit_vectors(v1, v2).rot(v1), v2);
    EXPECT_VECTOR3_EQUALS(Quaternion::from_two_unit_vectors(v2, v1).invrot(v1), v2);
  }
}

TEST(Quaternion, from_R)
{
  Quaternion q1 = Quaternion::Random();
  Eigen::Vector3d v;
  for (int i = 0; i < 100; i++)
  {
    Eigen::Matrix3d R = q1.R();
    Quaternion qR = Quaternion::from_R(R);
    v.setRandom();
    EXPECT_VECTOR3_EQUALS(qR.rot(v), R.T.dot(v));
  }
}

TEST(Quaternion, otimes)
{
  Quaternion q1 = Quaternion::Random();
  Quaternion qI = Quaternion::Identity();
  EXPECT_QUATERNION_EQUALS(q1 * q1.inverse(), qI);
}

TEST(Quaternion, exp_log_axis_angle)
{
  // Check that qexp is right by comparing with matrix exp and axis-angle
  for (int i = 0; i < 100; i++)
  {
    Eigen::Vector3d omega;
    omega.setRandom();
    Eigen::Matrix3d R_omega_exp = Quaternion::skew(omega).exp();
    Quaternion q_R_omega_exp = Quaternion::from_R(R_omega_exp);
    Quaternion q_omega = Quaternion::from_axis_angle(omega/omega.norm(), omega.norm());
    Quaternion q_omega_exp = Quaternion::exp(omega);
    EXPECT_QUATERNION_EQUALS(q_R_omega_exp, q_omega);
    EXPECT_QUATERNION_EQUALS(q_omega_exp, q_omega);

    // Check that exp and log are inverses of each other
    EXPECT_VECTOR3_EQUALS(Quaternion::log(Quaternion::exp(omega)), omega);
    EXPECT_QUATERNION_EQUALS(Quaternion::exp(Quaternion::log(q_omega)), q_omega);
  }
}


TEST(Quaternion, boxplus_and_boxminus)
{
  Eigen::Vector3d delta1, delta2, zeros;
  zeros.setZero();
  for (int i = 0; i < 100; i++)
  {
    Quaternion q = Quaternion::Random();
    Quaternion q2 = Quaternion::Random();
    delta1.setRandom();
    delta2.setRandom();

    EXPECT_QUATERNION_EQUALS(q + zeros, q);
    EXPECT_QUATERNION_EQUALS(q + (q2 - q), q2);
    EXPECT_VECTOR3_EQUALS((q + delta1) - q, delta1);
    EXPECT_LE(((q+delta1)-(q+delta2)).norm(), (delta1-delta2).norm());
  }
}

TEST(Quaternion, inplace_add_and_mul)
{
  Eigen::Vector3d delta1, delta2, zeros;
  zeros.setZero();
  for (int i = 0; i < 100; i++)
  {
    Quaternion q = Quaternion::Random();
    Quaternion q2 = Quaternion::Random();
    Quaternion q_copy = q.copy();
    delta1.setRandom();
    delta2.setRandom();

    q_copy += delta1;
    EXPECT_QUATERNION_EQUALS(q_copy, q+delta1);

    q_copy = q.copy();
    q_copy *= q2;
    EXPECT_QUATERNION_EQUALS(q_copy, q*q2);
  }
}

TEST(math_helper, T_zeta)
{
  Eigen::Vector3d v2;
  for (int i = 0; i < 100; i++)
  {
    v2.setRandom();
    v2 /= v2.norm();
    Quaternion q2 = Quaternion::from_two_unit_vectors(e_z, v2);
    Eigen::Vector2d T_z_v2 = T_zeta(q2).transpose() * v2;
    EXPECT_LE(T_z_v2.norm(), 1e-8);
  }
}

TEST(math_helper, d_dTdq)
{
  for (int j = 0; j < 100; j++)
  {
    Eigen::Matrix2d d_dTdq;
    d_dTdq.setZero();
    Eigen::Vector3d v2;
    v2.setRandom();
    Quaternion q = Quaternion::Random();
    q.setZ(0);
    q.normalize();
    auto T_z = T_zeta(q);
    Eigen::Vector2d x0 = T_z.transpose() * v2;
    double epsilon = 1e-6;
    Eigen::Matrix2d I = Eigen::Matrix2d::Identity() * epsilon;
    Eigen::Matrix2d a_dTdq = -T_z.transpose() * skew(v2) * T_z;
    for (int i = 0; i < 2; i++)
    {
      quat::Quaternion qplus = q_feat_boxplus(q, I.col(i));
      Eigen::Vector2d xprime = T_zeta(qplus).transpose() * v2;
      Eigen::Vector2d dx = xprime - x0;
      d_dTdq.row(i) = (dx) / epsilon;
    }
    EXPECT_MATRIX_EQUAL(d_dTdq, a_dTdq, 1e-6);
  }
}

TEST(math_helper, dqzeta_dqzeta)
{
  for(int j = 0; j < 100; j++)
  {
    Eigen::Matrix2d d_dqdq;
    quat::Quaternion q = quat::Quaternion::Random();
    if (j == 0)
      q = quat::Quaternion::Identity();
    double epsilon = 1e-6;
    Eigen::Matrix2d I = Eigen::Matrix2d::Identity() * epsilon;
    for (int i = 0; i < 2; i++)
    {
      quat::Quaternion q_prime = q_feat_boxplus(q, I.col(i));
      Eigen::Vector2d dq  = q_feat_boxminus(q_prime, q);
      d_dqdq.row(i) = dq /epsilon;
    }
    Eigen::Matrix2d a_dqdq = T_zeta(q).transpose() * T_zeta(q);
    EXPECT_MATRIX_EQUAL(a_dqdq, d_dqdq, 1e-2);
  }
}

TEST(math_helper, manifold_operations)
{
  Eigen::Vector3d omega, omega2;
  Eigen::Vector2d dx, zeros;
  zeros.setZero();
  for (int i = 0; i < 100; i++)
  {
    omega.setRandom();
    omega2.setRandom();
    dx.setRandom();
    dx /= 2.0;
    omega(2) = 0;
    omega2(2) = 0;
    Quaternion x = Quaternion::exp(omega);
    Quaternion y = Quaternion::exp(omega2);

    EXPECT_QUATERNION_EQUALS( q_feat_boxplus(x, zeros), x);
    EXPECT_VECTOR3_EQUALS( q_feat_boxplus( x, q_feat_boxminus(y, x)).rot(e_z), y.rot(e_z));
    EXPECT_VECTOR2_EQUALS( q_feat_boxminus(q_feat_boxplus(x, dx), x), dx);
  }
}

TEST(VI_EKF, jacobians_test)
{
  Eigen::VectorXd x0((int)vi_ekf::VIEKF::xZ);
  vi_ekf::VIEKF ekf(x0);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


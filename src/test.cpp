#include "quat.h"
#include "gtest/gtest.h"
#include <iostream>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#define EXPECT_QUATERNION_EQUALS(q1, q2) \
  EXPECT_NEAR((q1).w(), (q1).w(), 1e-8); \
  EXPECT_NEAR((q1).x(), (q1).x(), 1e-8); \
  EXPECT_NEAR((q1).y(), (q1).y(), 1e-8); \
  EXPECT_NEAR((q1).z(), (q1).z(), 1e-8)

#define EXPECT_VECTOR3_EQUALS(v1, v2) \
  EXPECT_NEAR((v1).x(), (v1).x(), 1e-8); \
  EXPECT_NEAR((v1).y(), (v1).y(), 1e-8); \
  EXPECT_NEAR((v1).z(), (v1).z(), 1e-8)


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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


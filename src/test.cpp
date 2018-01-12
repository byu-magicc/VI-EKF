#include "gtest/gtest.h"
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
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
using namespace vi_ekf;

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

int print_error(std::string row_id, std::string col_id, Eigen::MatrixXd analytical, Eigen::MatrixXd fd);
int check_all(Eigen::MatrixXd analytical, Eigen::MatrixXd fd, std::string name);
VIEKF init_jacobians_test(Eigen::VectorXd& x0, Eigen::VectorXd& u0)
{
  // Configure initial State
  x0.resize(VIEKF::xZ);
  x0.setZero();
  x0(VIEKF::xATT) = 1.0;
  x0(VIEKF::xMU) = 0.2;
  x0.block<3,1>((int)VIEKF::xPOS, 0) += Eigen::Vector3d::Random() * 100.0;
  x0.block<3,1>((int)VIEKF::xVEL, 0) += Eigen::Vector3d::Random() * 10.0;
  x0.block<4,1>((int)VIEKF::xATT, 0) = (Quaternion(x0.block<4,1>((int)VIEKF::xATT, 0)) + Eigen::Vector3d::Random() * 0.5).elements();
  x0.block<3,1>((int)VIEKF::xB_A, 0) += Eigen::Vector3d::Random() * 1.0;
  x0.block<3,1>((int)VIEKF::xB_G, 0) += Eigen::Vector3d::Random() * 0.5;
  x0((int)VIEKF::xMU, 0) += (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX)))*0.05;

  // Create VIEKF
  VIEKF ekf(x0);

  // camera_to_body transform
  Eigen::Vector3d p_b_c = Eigen::Vector3d::Random() * 0.5;
  Quaternion q_b_c = Quaternion::Random();
  ekf.set_camera_to_IMU(p_b_c, q_b_c);

  // Camera Intrinsics
  Eigen::Vector2d cam_center = Eigen::Vector2d::Random();
  Eigen::Vector2d F;
  F << static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/100.0)),
       static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/100.0));
  ekf.set_camera_intrinsics(cam_center, F);

  // Initialize Random Features
  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector2d l;
    l << std::rand()%640, std::rand()%480;
    double depth = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/10.0));
    ekf.init_feature(l, depth);
  }
  // Recover the new state to return
  x0 = ekf.get_state();

  // Initialize Inputs
  u0.resize(VIEKF::uTOTAL);
  u0.setZero();
  u0.block<3,1>((int)VIEKF::uA, 0) += Eigen::Vector3d::Random() * 1.0;
  u0.block<3,1>((int)VIEKF::uG, 0) += Eigen::Vector3d::Random() * 1.0;

  return ekf;
}

TEST(VI_EKF, dfdx_test)
{  
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;
  vi_ekf::VIEKF ekf = init_jacobians_test(x0, u0);

  Eigen::VectorXd dx0;
  Eigen::MatrixXd a_dfdu;
  Eigen::MatrixXd a_dfdx;

  ekf.dynamics(x0, u0, dx0, a_dfdx, a_dfdu);

  Eigen::MatrixXd Idx;
  Idx.resizeLike(a_dfdx);
  Idx.setZero();
  for (int i = 0; i < Idx.cols(); i++)
  {
    Idx(i,i) = 1.0;
  }

  double epsilon = 1e-6;

  Eigen::MatrixXd d_dfdx;
  d_dfdx.resizeLike(a_dfdx);
  d_dfdx.setZero();


  Eigen::MatrixXd dummy1, dummy2;
  Eigen::VectorXd dxprime;
  Eigen::VectorXd xprime;
  for (int i = 0; i < d_dfdx.cols(); i++)
  {
    xprime = ekf.boxplus(x0, (Idx.col(i) * epsilon));
    ekf.dynamics(xprime, u0, dxprime, dummy1, dummy2);
    d_dfdx.col(i) = (dxprime - dx0) / epsilon;
  }

  EXPECT_EQ(print_error("dxPOS", "dxVEL", a_dfdx, d_dfdx), 0);
  EXPECT_EQ(print_error("dxPOS", "dxATT", a_dfdx, d_dfdx), 0);
  EXPECT_EQ(print_error("dxVEL", "dxVEL", a_dfdx, d_dfdx), 0);
  EXPECT_EQ(print_error("dxVEL", "dxPOS", a_dfdx, d_dfdx), 0);
  EXPECT_EQ(print_error("dxVEL", "dxATT", a_dfdx, d_dfdx), 0);
  EXPECT_EQ(print_error("dxVEL", "dxB_A", a_dfdx, d_dfdx), 0);
  EXPECT_EQ(print_error("dxVEL", "dxB_G", a_dfdx, d_dfdx), 0);
  EXPECT_EQ(print_error("dxVEL", "dxMU", a_dfdx, d_dfdx), 0);

  for (int i = 0; i < ekf.get_len_features(); i++)
  {
    std::string zeta_key = "dxZETA_" + std::to_string(i);
    std::string rho_key = "dxRHO_" + std::to_string(i);

    EXPECT_EQ(print_error(zeta_key, "dxVEL", a_dfdx, d_dfdx), 0);
    EXPECT_EQ(print_error(zeta_key, "dxB_G", a_dfdx, d_dfdx), 0);
    EXPECT_EQ(print_error(zeta_key, zeta_key, a_dfdx, d_dfdx), 0);
    EXPECT_EQ(print_error(zeta_key, rho_key, a_dfdx, d_dfdx), 0);
    EXPECT_EQ(print_error(rho_key, "dxVEL", a_dfdx, d_dfdx), 0);
    EXPECT_EQ(print_error(rho_key, "dxB_G", a_dfdx, d_dfdx), 0);
    EXPECT_EQ(print_error(rho_key, zeta_key, a_dfdx, d_dfdx), 0);
    EXPECT_EQ(print_error(rho_key, rho_key, a_dfdx, d_dfdx), 0);
  }
  EXPECT_EQ(check_all(a_dfdx, d_dfdx, "dfdx"), 0);
}

TEST(VI_EKF, dfdu_test)
{
  Eigen::VectorXd x0;
  Eigen::VectorXd u0;
  vi_ekf::VIEKF ekf = init_jacobians_test(x0, u0);

  // Perform Analytical Differentiation
  Eigen::VectorXd dx0;
  Eigen::MatrixXd a_dfdu;
  Eigen::MatrixXd a_dfdx;
  ekf.dynamics(x0, u0, dx0, a_dfdx, a_dfdu);

  Eigen::MatrixXd d_dfdu;
  d_dfdu.resizeLike(a_dfdu);
  d_dfdu.setZero();
  double epsilon = 1e-6;
  Eigen::MatrixXd dummy1, dummy2;
  Eigen::Matrix<double, 6, 6> Iu = Eigen::Matrix<double, 6, 6>::Identity();

  // Perform Numerical Differentiation
  Eigen::VectorXd dxprime;
  for (int i = 0; i < d_dfdu.cols(); i++)
  {
    Eigen::VectorXd uprime = u0 + (Iu.col(i) * epsilon);
    ekf.dynamics(x0, uprime, dxprime, dummy1, dummy2);
    d_dfdu.col(i) = (dxprime - dx0) / epsilon;
  }

  EXPECT_EQ(print_error("dxVEL","uA", a_dfdu, d_dfdu), 0);
  EXPECT_EQ(print_error("dxVEL","uG", a_dfdu, d_dfdu), 0);
  EXPECT_EQ(print_error("dxATT", "uG", a_dfdu, d_dfdu), 0);
  for (int i = 0; i < ekf.get_len_features(); i++)
  {
    std::string zeta_key = "dxZETA_" + std::to_string(i);
    std::string rho_key = "dxRHO_" + std::to_string(i);
    EXPECT_EQ(print_error(zeta_key, "uG", a_dfdu, d_dfdu), 0);
    EXPECT_EQ(print_error(rho_key, "uG", a_dfdu, d_dfdu), 0);
  }
}

#define HEADER "\033[95m"
#define OKBLUE "\033[94m"
#define OKGREEN "\033[92m"
#define WARNING "\033[93m"
#define FONT_FAIL "\033[91m"
#define ENDC "\033[0m"
#define BOLD "\033[1m"
#define UNDERLINE "\033[4m"

static std::map<std::string, std::vector<int>> indexes = [] {
  std::map<std::string, std::vector<int>> tmp;
  tmp["dxPOS"] = std::vector<int> {0,3};
  tmp["dxVEL"] = std::vector<int> {3,3};
  tmp["dxATT"] = std::vector<int> {6,3};
  tmp["dxB_A"] = std::vector<int> {9,3};
  tmp["dxB_G"] = std::vector<int> {12,3};
  tmp["dxMU"] = std::vector<int> {15,1};
  tmp["uA"] = std::vector<int> {0,3};
  tmp["uG"] = std::vector<int> {3,3};
  for (int i = 0; i < 50; i++)
  {
    tmp["dxZETA_" + std::to_string(i)] = {16 + 3*i, 2};
    tmp["dxRHO_" + std::to_string(i)] = {16 + 3*i+2, 1};
  }
  return tmp;
}();

int print_error(std::string row_id, std::string col_id, Eigen::MatrixXd analytical, Eigen::MatrixXd fd)
{
  Eigen::MatrixXd error_mat = analytical - fd;
  std::vector<int> row = indexes[row_id];
  std::vector<int> col = indexes[col_id];
  if ((error_mat.block(row[0], col[0], row[1], col[1]).array().abs() > 1e-3).any())
  {
    std::cout << FONT_FAIL << "Error in Jacobian " << row_id << ", " << col_id << "\n";
    std::cout << "BLOCK ERROR:\n" << error_mat.block(row[0], col[0], row[1], col[1]) << "\n";
    std::cout << "ANALYTICAL:\n" << analytical.block(row[0], col[0], row[1], col[1]) << "\n";
    std::cout << "FD:\n" << fd.block(row[0], col[0], row[1], col[1]) << ENDC << "\n";
    return 1;
  }
  return 0;
}

int check_all(Eigen::MatrixXd analytical, Eigen::MatrixXd fd, std::string name)
{
  Eigen::MatrixXd error_mat = analytical - fd;
  if ((error_mat.array().abs() > 1e-3).any())
  {
    std::cout << FONT_FAIL << "Error in total " << BOLD << name << ENDC << FONT_FAIL << " matrix" << ENDC << "\n";
    for (int row =0; row < error_mat.rows(); row ++)
    {
      for (int col = 0; col < error_mat.cols(); col++)
      {
        if(std::abs(error_mat(row, col)) > 1e-3)
        {
          std::cout << BOLD << "error in (" << row << ", " << col << "):\tERR: " << error_mat(row,col) << "\tA: " << analytical(row, col) << "\tFD: " << fd(row,col) << ENDC << "\n";
        }
      }
    }
    return 1;
  }
  return 0;
}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


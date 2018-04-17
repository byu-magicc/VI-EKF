#include "gtest/gtest.h"
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include "vi_ekf.h"

using namespace quat;
using namespace vi_ekf;
using namespace Eigen;

#define NUM_ITERS 10

#define EXPECT_QUATERNION_EQUALS(q1, q2) \
  EXPECT_NEAR((q1).w(), (q2).w(), 1e-8); \
  EXPECT_NEAR((q1).x(), (q2).x(), 1e-8); \
  EXPECT_NEAR((q1).y(), (q2).y(), 1e-8); \
  EXPECT_NEAR((q1).z(), (q2).z(), 1e-8)

#define EXPECT_VECTOR3_EQUALS(v1, v2) \
  EXPECT_NEAR((v1)(0,0), (v2)(0,0), 1e-8); \
  EXPECT_NEAR((v1)(1,0), (v2)(1,0), 1e-8); \
  EXPECT_NEAR((v1)(2,0), (v2)(2,0), 1e-8)

#define EXPECT_VECTOR2_EQUALS(v1, v2) \
  EXPECT_NEAR((v1)(0,0), (v2)(0,0), 1e-8); \
  EXPECT_NEAR((v1)(1,0), (v2)(1,0), 1e-8)

#define EXPECT_MATRIX_EQUAL(m1, m2, tol) {\
  for (int row = 0; row < m1.rows(); row++ ) \
{ \
  for (int col = 0; col < m1.cols(); col++) \
{ \
  EXPECT_NEAR((m1)(row, col), (m2)(row, col), tol); \
  } \
  } \
  }

#define CALL_MEMBER_FN(objectptr,ptrToMember) ((objectptr).*(ptrToMember))
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


bool check_block(std::string row_id, std::string col_id, MatrixXd analytical, MatrixXd fd, double tolerance=1e-3)
{
  MatrixXd error_mat = analytical - fd;
  std::vector<int> row = indexes[row_id];
  std::vector<int> col = indexes[col_id];
  if ((error_mat.block(row[0], col[0], row[1], col[1]).array().abs() > tolerance).any())
  {
    std::cout << FONT_FAIL << "Error in Jacobian " << row_id << ", " << col_id << "\n";
    std::cout << "BLOCK ERROR:\n" << error_mat.block(row[0], col[0], row[1], col[1]) << "\n";
    std::cout << "ANALYTICAL:\n" << analytical.block(row[0], col[0], row[1], col[1]) << "\n";
    std::cout << "FD:\n" << fd.block(row[0], col[0], row[1], col[1]) << ENDC << "\n";
    return true;
  }
  return false;
}

int check_all(MatrixXd analytical, MatrixXd fd, std::string name, double tol = 1e-3)
{
  MatrixXd error_mat = analytical - fd;
  if ((error_mat.array().abs() > tol).any())
  {
    std::cout << FONT_FAIL << "Error in total " << BOLD << name << ENDC << FONT_FAIL << " matrix" << ENDC << "\n";
    for (int row =0; row < error_mat.rows(); row ++)
    {
      for (int col = 0; col < error_mat.cols(); col++)
      {
        if(std::abs(error_mat(row, col)) > tol)
        {
          std::cout << BOLD << "error in (" << row << ", " << col << "):\tERR: " << error_mat(row,col) << "\tA: " << analytical(row, col) << "\tFD: " << fd(row,col) << ENDC << "\n";
        }
      }
    }
    return true;
  }
  return false;
}

VIEKF init_jacobians_test(xVector& x0, uVector& u0)
{
  // Configure initial State
  x0.setZero();
  x0(VIEKF::xATT) = 1.0;
  x0(VIEKF::xMU) = 0.2;
  x0.block<3,1>((int)VIEKF::xPOS, 0) += Vector3d::Random() * 100.0;
  x0.block<3,1>((int)VIEKF::xVEL, 0) += Vector3d::Random() * 10.0;
  x0.block<4,1>((int)VIEKF::xATT, 0) = (Quat(x0.block<4,1>((int)VIEKF::xATT, 0)) + Vector3d::Random() * 0.5).elements();
  x0.block<3,1>((int)VIEKF::xB_A, 0) += Vector3d::Random() * 1.0;
  x0.block<3,1>((int)VIEKF::xB_G, 0) += Vector3d::Random() * 0.5;
  x0((int)VIEKF::xMU, 0) += (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX)))*0.05;
  
  // Create VIEKF
  VIEKF ekf;
  Matrix<double, vi_ekf::VIEKF::dxZ, 1> P0, Qx, gamma;
  P0.setOnes();
  Qx.setOnes();
  gamma.setOnes();
  uVector Qu;
  Qu.setOnes();
  Vector3d P0feat, Qxfeat, gammafeat;
  P0feat.setOnes();
  Qxfeat.setOnes();
  gammafeat.setOnes();
  Vector2d cam_center = Vector2d::Random();
  cam_center << 320-25+std::rand()%50, 240-25+std::rand()%50;
  Vector2d focal_len;
  focal_len << static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/100.0)),
      static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/100.0));
  Vector4d q_b_c = Quat::Random().elements();
  Vector3d p_b_c = Vector3d::Random() * 0.5;
  ekf.init(x0.block<17, 1>(0,0), P0, Qx, gamma, Qu, P0feat, Qxfeat, gammafeat, cam_center, focal_len, q_b_c, p_b_c, 2.0, "~", true, true, true, 0.0);
  
  // Initialize Random Features
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    Vector2d l;
    l << std::rand()%640, std::rand()%480;
    double depth = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/10.0));
    ekf.init_feature(l, i, depth);
  }
  // Recover the new state to return
  x0 = ekf.get_state();
  
  // Initialize Inputs
  u0.setZero();
  u0.block<3,1>((int)VIEKF::uA, 0) += Vector3d::Random() * 1.0;
  u0.block<3,1>((int)VIEKF::uG, 0) += Vector3d::Random() * 1.0;
  
  return ekf;
}

double random(double max, double min)
{
  double f = (double)rand() / RAND_MAX;
  return min + f * (max - min);
}

int htest(measurement_function_ptr fn, VIEKF& ekf, const VIEKF::measurement_type_t type, const int id, const int dim, double tol=1e-3)
{
  int num_errors = 0;
  xVector x0 = ekf.get_state();
  zVector z0;
  hMatrix a_dhdx;
  a_dhdx.setZero();
  
  // Call the Measurement function
  CALL_MEMBER_FN(ekf, fn)(x0, z0, a_dhdx, id);
  
  hMatrix d_dhdx;
  d_dhdx.setZero(); 
  
  Matrix<double, MAX_DX, MAX_DX> I = Matrix<double, MAX_DX, MAX_DX>::Identity();
  double epsilon = 1e-6;
  
  zVector z_prime;
  hMatrix dummy_H;
  xVector x_prime;
  for (int i = 0; i < a_dhdx.cols(); i++)
  {
    ekf.boxplus(ekf.get_state(), (I.col(i) * epsilon), x_prime);
    
    CALL_MEMBER_FN(ekf, fn)(x_prime, z_prime, dummy_H, id);
    
    if (type == VIEKF::QZETA)
      d_dhdx.block(0, i, dim, 1) = q_feat_boxminus(Quat(z_prime), Quat(z0))/epsilon;
    else if (type == VIEKF::ATT)
      d_dhdx.col(i) = (Quat(z_prime) - Quat(z0))/epsilon;
    else
      d_dhdx.block(0, i, dim, 1) = (z_prime.topRows(dim) - z0.topRows(dim))/epsilon;
  }
  
  MatrixXd error = (a_dhdx - d_dhdx).topRows(dim);
  double err_threshold = std::max(tol * a_dhdx.norm(), tol);
  
  for (std::map<std::string, std::vector<int>>::iterator it=indexes.begin(); it!=indexes.end(); ++it)
  {
    if(it->second[0] + it->second[1] > error.cols())
      continue;
    MatrixXd block_error = error.block(0, it->second[0], error.rows(), it->second[1]);    
    if ((block_error.array().abs() > err_threshold).any())
    {
      num_errors += 1;
      std::cout << FONT_FAIL << "Error in Measurement " << measurement_names[type] << "_" << id << ", " << it->first << ": (thresh = " << err_threshold << " mean = " << a_dhdx.norm() << ")\n";
      std::cout << "ERR:\n" << block_error << "\nA:\n" << a_dhdx.block(0, it->second[0], error.rows(), it->second[1]) << "\n";
      std::cout << "FD:\n" << d_dhdx.block(0, it->second[0], error.rows(), it->second[1]) << ENDC <<  "\n";
    }
  }
  return num_errors;
}

TEST(Quat, rotation_direction)
{
  // Compare against a known active and passive rotation
  Vector3d v, beta, v_active_rotated, v_passive_rotated;
  v << 0, 0, 1;
  v_active_rotated << 0, std::pow(-0.5,0.5), std::pow(0.5,0.5);
  beta << 1, 0, 0;
  Quat q_x_45 = Quat::from_axis_angle(beta, 45*M_PI/180.0);
  
  EXPECT_VECTOR3_EQUALS(q_x_45.rot(v), v_active_rotated);
  
  v_passive_rotated << 0, std::pow(0.5, 0.5), std::pow(0.5, 0.5);
  EXPECT_VECTOR3_EQUALS(q_x_45.rot(v), v_passive_rotated);
}

TEST(Quat, rot_invrot_R)
{
  Vector3d v;
  Quat q1 = Quat::Random();
  for (int i = 0; i < NUM_ITERS; i++)
  {
    v.setRandom();
    q1 = Quat::Random();
    
    // Check that rotations are inverses of each other
    EXPECT_VECTOR3_EQUALS(q1.rot(v), q1.R().transpose() * v);
    EXPECT_VECTOR3_EQUALS(q1.invrot(v), q1.R() * v);
  }
}

TEST(Quat, from_two_unit_vectors)
{
  Vector3d v1, v2;
  for (int i = 0; i < NUM_ITERS; i++)
  {
    v1.setRandom();
    v2.setRandom();
    v1 /= v1.norm();
    v2 /= v2.norm();
    
    EXPECT_VECTOR3_EQUALS(Quat::from_two_unit_vectors(v1, v2).rot(v1), v2);
    EXPECT_VECTOR3_EQUALS(Quat::from_two_unit_vectors(v2, v1).invrot(v1), v2);
  }
}

TEST(Quat, from_R)
{
  Quat q1 = Quat::Random();
  Vector3d v;
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Matrix3d R = q1.R();
    Quat qR = Quat::from_R(R);
    v.setRandom();
    EXPECT_VECTOR3_EQUALS(qR.rot(v), R.transpose() * v);
	EXPECT_VECTOR3_EQUALS(qR.invrot(v), R * v);
  }
}

TEST(Quat, otimes)
{
  Quat q1 = Quat::Random();
  Quat qI = Quat::Identity();
  EXPECT_QUATERNION_EQUALS(q1 * q1.inverse(), qI);
}

TEST(Quat, exp_log_axis_angle)
{
  // Check that qexp is right by comparing with matrix exp and axis-angle
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Vector3d omega;
    omega.setRandom();
    Matrix3d R_omega_exp = Quat::skew(omega).exp();
    Quat q_R_omega_exp = Quat::from_R(R_omega_exp);
    Quat q_omega = Quat::from_axis_angle(omega/omega.norm(), omega.norm());
    Quat q_omega_exp = Quat::exp(omega);
    EXPECT_QUATERNION_EQUALS(q_R_omega_exp, q_omega);
    EXPECT_QUATERNION_EQUALS(q_omega_exp, q_omega);
    
    // Check that exp and log are inverses of each otherprint_error
    EXPECT_VECTOR3_EQUALS(Quat::log(Quat::exp(omega)), omega);
    EXPECT_QUATERNION_EQUALS(Quat::exp(Quat::log(q_omega)), q_omega);
  }
}


TEST(Quat, boxplus_and_boxminus)
{
  Vector3d delta1, delta2, zeros;
  zeros.setZero();
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Quat q = Quat::Random();
    Quat q2 = Quat::Random();
    delta1.setRandom();
    delta2.setRandom();
    
    EXPECT_QUATERNION_EQUALS(q + zeros, q);
    EXPECT_QUATERNION_EQUALS(q + (q2 - q), q2);
    EXPECT_VECTOR3_EQUALS((q + delta1) - q, delta1);
    EXPECT_LE(((q+delta1)-(q+delta2)).norm(), (delta1-delta2).norm());
  }
}

TEST(Quat, inplace_add_and_mul)
{
  Vector3d delta1, delta2, zeros;
  zeros.setZero();
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Quat q = Quat::Random();
    Quat q2 = Quat::Random();
    Quat q_copy = q.copy();
    delta1.setRandom();
    delta2.setRandom();
    
    q_copy += delta1;
    EXPECT_QUATERNION_EQUALS(q_copy, q+delta1);
    
    q_copy = q.copy();
    q_copy *= q2;
    EXPECT_QUATERNION_EQUALS(q_copy, q*q2);
  }
}

TEST(Quat, euler)
{
  for (int i =0; i < NUM_ITERS; i++)
  {
    double roll = random(-M_PI, M_PI);
    double pitch = random(-M_PI/2.0, M_PI/2.0);
    double yaw = random(-M_PI, M_PI);
    Quat q = Quat::from_euler(roll, pitch, yaw);
    EXPECT_NEAR(roll, q.roll(), 1e-8);
    EXPECT_NEAR(pitch, q.pitch(), 1e-8);
    EXPECT_NEAR(yaw, q.yaw(), 1e-8);    
  }
}

TEST(math_helper, T_zeta)
{
  Vector3d v2;
  for (int i = 0; i < NUM_ITERS; i++)
  {
    v2.setRandom();
    v2 /= v2.norm();
    Quat q2 = Quat::from_two_unit_vectors(e_z, v2);
    Vector2d T_z_v2 = T_zeta(q2).transpose() * v2;
    EXPECT_LE(T_z_v2.norm(), 1e-8);
  }
}

TEST(math_helper, d_dTdq)
{
  for (int j = 0; j < NUM_ITERS; j++)
  {
    Matrix2d d_dTdq;
    d_dTdq.setZero();
    Vector3d v2;
    v2.setRandom();
    Quat q = Quat::Random();
    q.setZ(0);
    q.normalize();
    auto T_z = T_zeta(q);
    Vector2d x0 = T_z.transpose() * v2;
    double epsilon = 1e-6;
    Matrix2d I = Matrix2d::Identity() * epsilon;
    Matrix2d a_dTdq = -T_z.transpose() * skew(v2) * T_z;
    for (int i = 0; i < 2; i++)
    {
      quat::Quat qplus = q_feat_boxplus(q, I.col(i));
      Vector2d xprime = T_zeta(qplus).transpose() * v2;
      Vector2d dx = xprime - x0;
      d_dTdq.row(i) = (dx) / epsilon;
    }
    EXPECT_MATRIX_EQUAL(d_dTdq, a_dTdq, 1e-6);
  }
}

TEST(math_helper, dqzeta_dqzeta)
{
  for(int j = 0; j < NUM_ITERS; j++)
  {
    Matrix2d d_dqdq;
    quat::Quat q = quat::Quat::Random();
    if (j == 0)
      q = quat::Quat::Identity();
    double epsilon = 1e-6;
    Matrix2d I = Matrix2d::Identity() * epsilon;
    for (int i = 0; i < 2; i++)
    {
      quat::Quat q_prime = q_feat_boxplus(q, I.col(i));
      Vector2d dq  = q_feat_boxminus(q_prime, q);
      d_dqdq.row(i) = dq /epsilon;
    }
    Matrix2d a_dqdq = T_zeta(q).transpose() * T_zeta(q);
    EXPECT_MATRIX_EQUAL(a_dqdq, d_dqdq, 1e-2);
  }
}

TEST(math_helper, manifold_operations)
{
  Vector3d omega, omega2;
  Vector2d dx, zeros;
  zeros.setZero();
  for (int i = 0; i < NUM_ITERS; i++)
  {
    omega.setRandom();
    omega2.setRandom();
    dx.setRandom();
    dx /= 2.0;
    omega(2) = 0;
    omega2(2) = 0;
    Quat x = Quat::exp(omega);
    Quat y = Quat::exp(omega2);
    
    EXPECT_QUATERNION_EQUALS( q_feat_boxplus(x, zeros), x);
    EXPECT_VECTOR3_EQUALS( q_feat_boxplus( x, q_feat_boxminus(y, x)).rot(e_z), y.rot(e_z));
    EXPECT_VECTOR2_EQUALS( q_feat_boxminus(q_feat_boxplus(x, dx), x), dx);
  }
}

TEST(VI_EKF, dfdx_test)
{  
  xVector x0;
  uVector u0;
  dxMatrix dummydfdx;
  dxuMatrix dummydfdu;
  dxVector dxprime;
  xVector xprime;
  dxMatrix Idx = dxMatrix::Identity();
  double epsilon = 1e-6;
  
  dxMatrix d_dfdx;
  dxVector dx0;
  dxuMatrix a_dfdu;
  dxMatrix a_dfdx;
  for (int j = 0; j < NUM_ITERS; j++)
  {
    vi_ekf::VIEKF ekf = init_jacobians_test(x0, u0);
    
    // analytical differentiation
    ekf.dynamics(x0, u0, dx0, a_dfdx, a_dfdu);
    d_dfdx.setZero();
    
    // numeri
    for (int i = 0; i < d_dfdx.cols(); i++)
    {
      ekf.boxplus(x0, (Idx.col(i) * epsilon), xprime);
      ekf.dynamics(xprime, u0, dxprime, dummydfdx, dummydfdu);
      d_dfdx.col(i) = (dxprime - dx0) / epsilon;
    }
    
    EXPECT_FALSE(check_block("dxPOS", "dxVEL", a_dfdx, d_dfdx));
    EXPECT_FALSE(check_block("dxPOS", "dxATT", a_dfdx, d_dfdx));
    EXPECT_FALSE(check_block("dxVEL", "dxVEL", a_dfdx, d_dfdx));
    EXPECT_FALSE(check_block("dxVEL", "dxPOS", a_dfdx, d_dfdx));
    EXPECT_FALSE(check_block("dxVEL", "dxATT", a_dfdx, d_dfdx));
    EXPECT_FALSE(check_block("dxVEL", "dxB_A", a_dfdx, d_dfdx));
    EXPECT_FALSE(check_block("dxVEL", "dxB_G", a_dfdx, d_dfdx));
    EXPECT_FALSE(check_block("dxVEL", "dxMU", a_dfdx, d_dfdx));
    
    for (int i = 0; i < ekf.get_len_features(); i++)
    {
      std::string zeta_key = "dxZETA_" + std::to_string(i);
      std::string rho_key = "dxRHO_" + std::to_string(i);
      
      EXPECT_FALSE(check_block(zeta_key, "dxVEL", a_dfdx, d_dfdx));
      EXPECT_FALSE(check_block(zeta_key, "dxB_G", a_dfdx, d_dfdx));
      EXPECT_FALSE(check_block(zeta_key, zeta_key, a_dfdx, d_dfdx, 2e-2));
      EXPECT_FALSE(check_block(zeta_key, rho_key, a_dfdx, d_dfdx));
      EXPECT_FALSE(check_block(rho_key, "dxVEL", a_dfdx, d_dfdx, 2e-2));
      EXPECT_FALSE(check_block(rho_key, "dxB_G", a_dfdx, d_dfdx, 1e-2));
      EXPECT_FALSE(check_block(rho_key, zeta_key, a_dfdx, d_dfdx, 5.0));
      EXPECT_FALSE(check_block(rho_key, rho_key, a_dfdx, d_dfdx, 1e-2));
    }
  }
}

TEST(VI_EKF, dfdu_test)
{
  xVector x0;
  uVector u0;
  dxVector dxprime;
  uVector uprime;
  double epsilon = 1e-6;
  dxMatrix dfdx_dummy;
  dxuMatrix dfdu_dummy;
  dxuMatrix d_dfdu;
  Matrix<double, 6, 6> Iu = Matrix<double, 6, 6>::Identity();
  dxVector dx0;
  dxMatrix a_dfdx;
  dxuMatrix a_dfdu;
  for (int j = 0; j < NUM_ITERS; j++)
  {
    vi_ekf::VIEKF ekf = init_jacobians_test(x0, u0);
    
    // Perform Analytical Differentiation
    ekf.dynamics(x0, u0, dx0, a_dfdx, a_dfdu);
    d_dfdu.setZero();
    
    // Perform Numerical Differentiation
    for (int i = 0; i < d_dfdu.cols(); i++)
    {
      uprime = u0 + (Iu.col(i) * epsilon);
      ekf.dynamics(x0, uprime, dxprime, dfdx_dummy, dfdu_dummy);
      d_dfdu.col(i) = (dxprime - dx0) / epsilon;
    }
    
    EXPECT_FALSE(check_block("dxVEL","uA", a_dfdu, d_dfdu));
    EXPECT_FALSE(check_block("dxVEL","uG", a_dfdu, d_dfdu));
    EXPECT_FALSE(check_block("dxATT", "uG", a_dfdu, d_dfdu));
    for (int i = 0; i < ekf.get_len_features(); i++)
    {
      std::string zeta_key = "dxZETA_" + std::to_string(i);
      std::string rho_key = "dxRHO_" + std::to_string(i);
      EXPECT_FALSE(check_block(zeta_key, "uG", a_dfdu, d_dfdu));
      EXPECT_FALSE(check_block(rho_key, "uG", a_dfdu, d_dfdu, 1e-1));
    }
  }
}

TEST(VI_EKF, h_test)
{
  xVector x0;
  uVector u0;
  for (int j = 0; j < NUM_ITERS; j++)
  {
    vi_ekf::VIEKF ekf = init_jacobians_test(x0, u0);
    
    EXPECT_EQ(htest(&VIEKF::h_acc, ekf, VIEKF::ACC, 0, 2), 0);
    EXPECT_EQ(htest(&VIEKF::h_pos, ekf, VIEKF::POS, 0, 3), 0);
    EXPECT_EQ(htest(&VIEKF::h_vel, ekf, VIEKF::VEL, 0, 3), 0);
    EXPECT_EQ(htest(&VIEKF::h_alt, ekf, VIEKF::ALT, 0, 1), 0);
    EXPECT_EQ(htest(&VIEKF::h_att, ekf, VIEKF::ATT, 0, 3), 0);
    for (int i = 0; i < ekf.get_len_features(); i++)
    {
      EXPECT_EQ(htest(&VIEKF::h_feat, ekf, VIEKF::FEAT, i, 2, 5e-1), 0);
      EXPECT_EQ(htest(&VIEKF::h_qzeta, ekf, VIEKF::QZETA, i, 2), 0);
      EXPECT_EQ(htest(&VIEKF::h_depth, ekf, VIEKF::DEPTH, i, 1), 0);
      EXPECT_EQ(htest(&VIEKF::h_inv_depth, ekf, VIEKF::INV_DEPTH, i, 1), 0);
      //        EXPECT_EQ(htest(VIEKF::h_pixel_vel, ekf, VIEKF::PIXEL_VEL, i), 0); // Still needs to be implemented
    }
  }
}


TEST(VI_EKF, KF_reset_test)
{
  uVector u0;
  dxMatrix d_dxpdxm;
  dxMatrix a_dxpdxm;
  xVector xm;
  xVector xp;
  dxMatrix dummy;
  dxMatrix I_dx = dxMatrix::Identity();
  xVector xm_prime;    
  xVector xp_prime;    
  dxVector d_xp;
  
  for (int j = 0; j < NUM_ITERS; j++)
  {
    vi_ekf::VIEKF ekf = init_jacobians_test(xm, u0);
    ekf.keyframe_reset(xm, xp, a_dxpdxm);
    
    d_dxpdxm.setZero();
    double epsilon = 1e-6;

    // Perform Numerical Differentiation
    for (int i = 0; i < d_dxpdxm.cols(); i++)
    {
      ekf.boxplus(xm, (I_dx.col(i) * epsilon), xm_prime);
      ekf.keyframe_reset(xm_prime, xp_prime, dummy);
      ekf.boxminus(xp_prime, xp, d_xp);
      d_dxpdxm.col(i) =  d_xp / epsilon;
    }
    
    EXPECT_FALSE(check_block("dxPOS", "dxPOS", a_dxpdxm, d_dxpdxm));
    EXPECT_FALSE(check_block("dxATT", "dxATT", a_dxpdxm, d_dxpdxm, 1e-1));
  }
  EXPECT_FALSE(check_all(a_dxpdxm, d_dxpdxm, "dfdx", 1e-1));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



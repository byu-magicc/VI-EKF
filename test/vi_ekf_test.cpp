#include <iostream>
#include <chrono>
#include <gtest/gtest.h>
#include "multirotor_sim/utils.h"
#include "multirotor_sim/simulator.h"
#include "multirotor_sim/estimator_wrapper.h"
#include "vi_ekf.h"

using namespace std;


TEST(VIEKF_Test, SimulatedData)
{
  // Initialize EKF
  vi_ekf::VIEKF ekf;

  // Initialize simulation
  Simulator multirotor(true);
  multirotor.load("../params/sim_params.yaml");

  // Register callbacks
  EstimatorWrapper est;

  auto imu_cb = [&ekf](const double& t, const Vector6d& z, const Matrix6d& R)
  {
    ekf.add_measurement(t, z, vi_ekf::VIEKF::ACC, R, true);
  };
  auto feat_cb = [&ekf](const double& t, const multirotor_sim::ImageFeat& z, const Matrix2d& R_pix, const Matrix1d& R_depth)
  {
    for (int i = 0; i < z.pixs.size(); ++i)
      ekf.add_measurement(t, z.pixs[i], vi_ekf::VIEKF::FEAT, R_pix, true, z.feat_ids[i]);
  };
//  auto mocap_cb = [&ekf](const double& t, const Xformd& z, const Matrix6d& R)
//  {
//    ekf.add_measurement(t, z.t(), vi_ekf::VIEKF::POS, R.topLeftCorner<3,3>(), true);
//    ekf.add_measurement(t, z.q().elements(), vi_ekf::VIEKF::ATT, R.bottomRightCorner<3,3>(), true);
//  };

  est.register_imu_cb(imu_cb);
  est.register_feat_cb(feat_cb);
//  est.register_mocap_cb(mocap_cb);
  multirotor.register_estimator(&est);

  // Run the simulation and log information for plotting
  while (multirotor.run())
  {
    //
  }

  // Check error magnitudes
}

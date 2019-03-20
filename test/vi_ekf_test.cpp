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
  vi_ekf::VIEKF ekf("../params/ekf.yaml");

  // Initialize simulation
  Simulator multirotor(true);
  multirotor.load("../params/sim_params.yaml");

  // Register callbacks
  EstimatorWrapper est;

  auto imu_cb = [&ekf](const double& t, const Vector6d& z, const Matrix6d& R)
  {
    ekf.propagate_state(z, t);
  };
  auto feat_cb = [&ekf](const double& t, const multirotor_sim::ImageFeat& z, const Matrix2d& R_pix, const Matrix1d& R_depth)
  {
    for (int i = 0; i < z.pixs.size(); ++i)
      ekf.add_measurement(t, z.pixs[i], vi_ekf::VIEKF::FEAT, R_pix, true, z.feat_ids[i]);
  };

  est.register_imu_cb(imu_cb);
  est.register_feat_cb(feat_cb);
  multirotor.register_estimator(&est);

  // Initialize loggers and log initial truth data
  ofstream true_state_log;
  true_state_log.open("/tmp/multirotor_true_state.log");
  true_state_log.write((char*)&multirotor.t_, sizeof(double));
  true_state_log.write((char*)multirotor.state().arr.data(), multirotor.state().arr.rows() * sizeof(double));

  ofstream true_bias_drag_log;
  true_bias_drag_log.open("/tmp/multirotor_true_imu_biases_drag.log");
  true_bias_drag_log.write((char*)&multirotor.t_, sizeof(double));
  true_bias_drag_log.write((char*)multirotor.accel_bias_.data(), 3 * sizeof(double));
  true_bias_drag_log.write((char*)multirotor.gyro_bias_.data(), 3 * sizeof(double));
  true_bias_drag_log.write((char*)&multirotor.dyn_.get_drag(), sizeof(double));


  // Run the simulation and log truth for plotting
  while (multirotor.run())
  {
    true_state_log.write((char*)&multirotor.t_, sizeof(double));
    true_state_log.write((char*)multirotor.state().arr.data(), multirotor.state().arr.rows() * sizeof(double));

    true_bias_drag_log.write((char*)&multirotor.t_, sizeof(double));
    true_bias_drag_log.write((char*)multirotor.accel_bias_.data(), 3 * sizeof(double));
    true_bias_drag_log.write((char*)multirotor.gyro_bias_.data(), 3 * sizeof(double));
    true_bias_drag_log.write((char*)&multirotor.dyn_.get_drag(), sizeof(double));
  }
  true_state_log.close();
  true_bias_drag_log.close();

  // Check error magnitudes
}

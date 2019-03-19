#include <iostream>
#include <chrono>
#include <gtest/gtest.h>
#include "multirotor_sim/utils.h"
#include "multirotor_sim/simulator.h"
#include "vi_ekf.h"

using namespace std;


TEST(VIEKF_Test, SimulatedData)
{
  // Initialize EKF
  vi_ekf::VIEKF ekf;

  // Initialize simulation
  Simulator multirotor(true);
  multirotor.load("../params/sim_params.yaml");
//  multirotor.register_estimator(this);

  // Run the simulation
  multirotor.run();
}

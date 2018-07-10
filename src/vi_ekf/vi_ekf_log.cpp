#include "vi_ekf.h"

namespace vi_ekf
{


void VIEKF::log_global_position(const eVector truth_global_transform) //Vector3d pos, const Vector4d att)
{ 
  // cut off initial state from truth - for plotting comparison
  //  concatenate_edges(invert_fed );
  
  if (log_.stream)
  {  
    // Log Global Position Estimate
    eVector global_pose;
    eVector rel_pose;
    rel_pose.block<3,1>((int)ePOS, 0) = x_.block<3,1>((int)xPOS, 0);
    rel_pose.block<4,1>((int)eATT, 0) = x_.block<4,1>((int)xATT, 0);
    concatenate_edges(current_node_global_pose_, rel_pose, global_pose);
    
    (*log_.stream)[LOG_MEAS] << "GLOBAL_POS" << "\t" << prev_t_-start_t_ << "\t"
                             << truth_global_transform.topRows(3).transpose() << "\t" << global_pose.topRows(3).transpose() << "\n";
    
    // Log Global Attitude Estimate
    (*log_.stream)[LOG_MEAS] << "GLOBAL_ATT" << "\t" << prev_t_-start_t_ << "\t"
                             << truth_global_transform.bottomRows(4).transpose() << "\t" << global_pose.bottomRows(4).transpose() << "\n";
  }
  //  WRT_DBG;
}

void VIEKF::log_depth(const int id, double zhat, bool active)
{
  int i = global_to_local_feature_id(id);
  double z = x_(xZ+5*i + 4, 0);
  (*log_.stream)[LOG_MEAS] << measurement_names[DEPTH] << "\t" << prev_t_-start_t_ << "\t"
                           << z << "\t" << zhat << "\t";
  (*log_.stream)[LOG_MEAS] << P_(dxZ + 3*i + 2, dxZ + 3*i + 2) << "\t" << id << "\t" << active << "\n";
}

void VIEKF::disable_logger()
{
  for (auto i = log_.stream->begin(); i != log_.stream->end(); i++)
  {
    i->close();
  }
  delete log_.stream;
  log_.stream = NULL;
}

void VIEKF::init_logger(string root_filename)
{
  log_.stream = new std::vector<std::ofstream>;
  (*log_.stream).resize(TOTAL_LOGS);
  
  // Make the directory
  int result = system(("mkdir -p " + root_filename).c_str());
  (void)result;
  
  // A logger for the results of propagation
  (*log_.stream)[LOG_PROP].open(root_filename + "/prop.txt", std::ofstream::out | std::ofstream::trunc);
  (*log_.stream)[LOG_MEAS].open(root_filename + "/meas.txt", std::ofstream::out | std::ofstream::trunc);
  (*log_.stream)[LOG_PERF].open(root_filename + "/perf.txt", std::ofstream::out | std::ofstream::trunc);
  (*log_.stream)[LOG_CONF].open(root_filename + "/conf.txt", std::ofstream::out | std::ofstream::trunc);
  (*log_.stream)[LOG_INPUT].open(root_filename + "/input.txt", std::ofstream::out | std::ofstream::trunc);
  (*log_.stream)[LOG_XDOT].open(root_filename + "/xdot.txt", std::ofstream::out | std::ofstream::trunc);
  (*log_.stream)[LOG_KF].open(root_filename + "/kf.txt", std::ofstream::out | std::ofstream::trunc);
  (*log_.stream)[LOG_DEBUG].open(root_filename + "/debug.txt", std::ofstream::out | std::ofstream::trunc);
  
  // Save configuration
  (*log_.stream)[LOG_CONF] << "Test Num: " << root_filename << "\n";
  (*log_.stream)[LOG_CONF] << "x0" << x_.block<(int)xZ, 1>(0,0).transpose() << "\n";
  (*log_.stream)[LOG_CONF] << "P0: " << P_.diagonal().block<(int)xZ, 1>(0,0).transpose() << "\n";
  (*log_.stream)[LOG_CONF] << "P0_feat: " << P0_feat_.diagonal().transpose() << "\n";
  (*log_.stream)[LOG_CONF] << "Qx: " << Qx_.diagonal().block<(int)dxZ, 1>(0,0).transpose() << "\n";
  (*log_.stream)[LOG_CONF] << "Qx_feat: " << Qx_.diagonal().block<3, 1>((int)dxZ,0).transpose() << "\n";
  (*log_.stream)[LOG_CONF] << "Qu: " << Qu_.diagonal().transpose() << "\n";
  (*log_.stream)[LOG_CONF] << "lambda: " << lambda_.block<(int)dxZ,1>(0,0).transpose() << "\n";
  (*log_.stream)[LOG_CONF] << "lambda_feat: " << lambda_.block<3,1>((int)dxZ,0).transpose() << "\n";  
  (*log_.stream)[LOG_CONF] << "partial_update: " << partial_update_ << "\n";
  (*log_.stream)[LOG_CONF] << "keyframe reset: " << keyframe_reset_ << "\n";
  (*log_.stream)[LOG_CONF] << "Using Drag Term: " << use_drag_term_ << "\n";
  (*log_.stream)[LOG_CONF] << "keyframe overlap: " << keyframe_overlap_threshold_ << "\n";
  (*log_.stream)[LOG_CONF] << "num features: " << NUM_FEATURES << "\n";
  (*log_.stream)[LOG_CONF] << "min_depth: " << min_depth_ << std::endl;
  
  // Start Performance Log
  (*log_.stream)[LOG_PERF] << "time\tprop\t";
  for (int i = 0; i < TOTAL_MEAS; i++)
  {
    log_.update_times[i] = 0;
    log_.update_count[i] = 0;
    (*log_.stream)[LOG_PERF] << measurement_names[i] << "\t";
  }
  (*log_.stream)[LOG_PERF] << endl;
}

}

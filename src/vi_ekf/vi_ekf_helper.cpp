#include "vi_ekf.h"

#ifdef MC_SIM
#include "utils.h"
#endif

namespace vi_ekf
{

#ifdef MC_SIM
void VIEKF::load(std::string filename)
{
  Matrix<double, xMU, 1> x;
  double mu0;
  get_yaml_eigen("x0", filename, x);
  get_yaml_node("mu0", filename, mu0);
  Matrix<double, xZ,1> x0;
  x0.block<xMU,1>(0,0) = x;
  x0(xMU, 0) = mu0;
  
  Matrix<double, dxZ,1> P0, Qx, lambda;
  get_yaml_eigen("P0", filename, P0);
  get_yaml_eigen("Qx", filename, Qx);
  get_yaml_eigen("lambda", filename, lambda);
  
  uVector Qu;
  Vector3d P0_feat, Qx_feat, lambda_feat, p_b_c;
  Vector2d cam_center, focal_len;
  Vector4d q_b_c;
  get_yaml_eigen("Qu", filename, Qu);
  get_yaml_eigen("P0_feat", filename, P0_feat);
  get_yaml_eigen("Qx_feat", filename, Qx_feat);
  get_yaml_eigen("lambda_feat", filename, lambda_feat);
  get_yaml_eigen("p_b_c", filename, p_b_c);
  get_yaml_eigen("cam_center", filename, cam_center);
  get_yaml_eigen("focal_len", filename, focal_len);
  get_yaml_eigen("q_b_c", filename, q_b_c);
  
  std::string log_directory;
  double min_depth, keyframe_overlap;
  bool use_drag_term, partial_update, keyframe_reset;
  get_yaml_node("min_depth", filename, min_depth);
  get_yaml_node("log_directory", filename, log_directory);
  get_yaml_node("use_drag_term", filename, use_drag_term);
  get_yaml_node("partial_update", filename, partial_update);
  get_yaml_node("keyframe_reset", filename, keyframe_reset);
  get_yaml_node("keyframe_overlap", filename, keyframe_overlap);
  
  init(x0, P0, Qx, lambda, Qu, P0_feat, Qx_feat, lambda_feat, cam_center,
       focal_len, q_b_c, p_b_c, min_depth, log_directory, use_drag_term, 
       partial_update, keyframe_reset, keyframe_overlap);
}
#endif


void VIEKF::boxplus(const xVector& x, const dxVector& dx, xVector& out) const
{
  out.block<6,1>((int)xPOS, 0) = x.block<6,1>((int)xPOS, 0) + dx.block<6,1>((int)dxPOS, 0);
  out.block<4,1>((int)xATT, 0) = (Quat(x.block<4,1>((int)xATT, 0)) + dx.block<3,1>((int)dxATT, 0)).elements();
  out.block<7,1>((int)xB_A, 0) = x.block<7,1>((int)xB_A, 0) + dx.block<7,1>((int)dxB_A, 0);
  for (int i = 0; i < len_features_; i++)
  {
    out.block<4,1>(xZ+i*5,0) = q_feat_boxplus(Quat(x.block<4,1>(xZ+i*5,0)), dx.block<2,1>(dxZ+3*i,0)).elements();
    out(xZ+i*5+4) = x(xZ+i*5+4) + dx(dxZ+3*i+2);
  }
}

void VIEKF::boxminus(const xVector &x1, const xVector &x2, dxVector &out) const
{
  out.block<6,1>((int)dxPOS, 0) = x1.block<6,1>((int)xPOS, 0) - x2.block<6,1>((int)xPOS, 0);
  out.block<3,1>((int)dxATT, 0) = (Quat(x1.block<4,1>((int)xATT, 0)) - Quat(x2.block<4,1>((int)xATT, 0)));
  out.block<7,1>((int)dxB_A, 0) = x1.block<7,1>((int)xB_A, 0) - x2.block<7,1>((int)xB_A, 0);
  
  for (int i = 0; i < len_features_; i++)
  {
    out.block<2,1>(dxZ+i*3,0) = q_feat_boxminus(Quat(x1.block<4,1>(xZ+i*5,0)), Quat(x2.block<4,1>(xZ+i*5,0)));
    out(dxZ+i*3+2) = x1(xZ+i*5+4) - x2(xZ+i*5+4);
  }
}


int VIEKF::global_to_local_feature_id(const int global_id) const
{
  int dist = std::distance(current_feature_ids_.begin(), std::find(current_feature_ids_.begin(), current_feature_ids_.end(), global_id));
  if (dist < current_feature_ids_.size())
  {
    return dist;
  }
  else
  {
    return -1;
  }
}

void VIEKF::fix_depth()
{
  // Apply an Inequality Constraint per
  // "Avoiding Negative Depth in Inverse Depth Bearing-Only SLAM"
  // by Parsley and Julier
  for (int i = 0; i < len_features_; i++)
  {
    int xRHO_i = xZ + 5*i + 4;
    int dxRHO_i = dxZ + 3*i + 2;
    if (x_(xRHO_i, 0) != x_(xRHO_i, 0))
    {
      // if a depth state has gone NaN, reset it
      x_(xRHO_i, 0) = 1.0/(2.0*min_depth_);
    }
    if (x_(xRHO_i, 0) < 0.0)
    {
      // If the state has gone negative, reset it
      double err = 1.0/(2.0*min_depth_) - x_(xRHO_i, 0);
      P_(dxRHO_i, dxRHO_i) += err*err;
      x_(xRHO_i, 0) = 1.0/(2.0*min_depth_);
    }
    else if (x_(xRHO_i, 0) > 1e2)
    {
      // If the state has grown unreasonably large, reset it
      P_(dxRHO_i, dxRHO_i) = P0_feat_(2,2);
      x_(xRHO_i, 0) = 1.0/(2.0*min_depth_);
    }
  }
}

}

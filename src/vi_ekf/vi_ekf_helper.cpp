#include "vi_ekf.h"

namespace vi_ekf
{


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
  int i = 0;
  while ( i != features_.size() && features_[i].global_id != global_id)
  {
    i++;
  }
  if (i < features_.size())
  {
    return i;
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

void VIEKF::set_image(const Mat &grey)
{
  // Build multilevel image
  img_[0] = grey;
  for (int i = 1; i < PYRAMID_LEVELS; i++)
  {
    pyrDown( img_[i-1], img_[i], Size( img_[i-1].cols/2, img_[i-1].rows/2 ) );
  }
}

void VIEKF::multiLvlPatch(const pixVector &eta, multiPatchVectorI& dst) const
{
  int x = std::round(eta(0,0));
  int y = std::round(eta(1,0));
  Size sz(PATCH_SIZE, PATCH_SIZE);
  for (int i = 0; i < PYRAMID_LEVELS; i++)
  {
    Point tl(x - PATCH_SIZE/2, y - PATCH_SIZE/2);
    x /= 2;
    y /= 2;
    Mat ROI = img_[i](Rect(tl, sz));
    
    // Convert to Eigen, but just the part that we want to update
    const Mat _dst(PATCH_SIZE, PATCH_SIZE, traits::Type<uint8_t>::value, 
                   dst.data() + i*PATCH_SIZE*PATCH_SIZE, PATCH_SIZE);
    if( ROI.type() == _dst.type() )
        transpose(ROI, _dst);
    else
    {
        ROI.convertTo(_dst, _dst.type());
        transpose(_dst, _dst);
    }
  }
}

void VIEKF::multiLvlPatchSideBySide(multiPatchVectorI& src, multiPatchMatrixI& dst) const
{
   dst = Eigen::Map<multiPatchMatrixI>(src.data());
}

void VIEKF::multiLvlPatchToCv(multiPatchVectorI& src, Mat& dst) const
{
  multiPatchMatrixI side_by_side;
  multiLvlPatchSideBySide(src, side_by_side);
  eigen2cv(side_by_side, dst);
}

}
